#!/usr/bin/env python3
"""
Export OpenAI Whisper model to ExecuTorch (.pte) format.

Usage:
    python export_whisper_to_pte.py --model base --output models/whisper
    python export_whisper_to_pte.py --model small --output models/whisper
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from whisper.tokenizer import get_tokenizer

# ExecuTorch imports
from torch.export import export, Dim
from executorch.exir import to_edge, EdgeCompileConfig
from executorch.exir.backend.backend_api import to_backend


class WhisperEncoder(nn.Module):
    """Wrapper for Whisper encoder - ExecuTorch compatible."""
    
    def __init__(self, whisper_model):
        super().__init__()
        self.conv1 = whisper_model.encoder.conv1
        self.conv2 = whisper_model.encoder.conv2
        # Register as buffer for proper export
        self.register_buffer(
            'positional_embedding',
            whisper_model.encoder.positional_embedding.clone()
        )
        self.blocks = whisper_model.encoder.blocks
        self.ln_post = whisper_model.encoder.ln_post
        
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: Mel spectrogram (batch, n_mels=80, n_frames=3000)
        Returns:
            Audio features (batch, n_ctx=1500, n_state)
        """
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (batch, n_ctx, n_state)
        
        # Use explicit slicing for ExecuTorch compatibility
        seq_len = x.shape[1]
        pos_emb = self.positional_embedding[:seq_len, :]
        x = x + pos_emb
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_post(x)
        return x


class WhisperDecoder(nn.Module):
    """Whisper decoder for ExecuTorch - supports dynamic sequence length."""
    
    def __init__(self, whisper_model, max_seq_len: int = 448):
        super().__init__()
        self.token_embedding = whisper_model.decoder.token_embedding
        # Register positional embedding as buffer
        self.register_buffer(
            'positional_embedding',
            whisper_model.decoder.positional_embedding.clone()
        )
        self.blocks = nn.ModuleList(whisper_model.decoder.blocks)
        self.ln = whisper_model.decoder.ln
        self.n_vocab = whisper_model.dims.n_vocab
        self.max_seq_len = max_seq_len
        
        # Store attention head configuration
        if len(self.blocks) > 0:
            self.n_head = self.blocks[0].attn.n_head
            self.d_head = whisper_model.dims.n_text_state // self.n_head
        
    def forward(self, tokens: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Token IDs (batch, seq_len)
            audio_features: Encoder output (batch, 1500, n_state)
        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        seq_len = tokens.shape[1]
        
        # Token + positional embeddings
        x = self.token_embedding(tokens) + self.positional_embedding[:seq_len, :]
        
        # Create causal mask dynamically for current sequence length
        # This avoids the guard issue with pre-computed mask slicing
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=tokens.device, dtype=x.dtype),
            diagonal=1
        ) * -10000.0
        
        # Transformer blocks
        for block in self.blocks:
            x = self._block_forward(block, x, audio_features, mask)
        
        x = self.ln(x)
        
        # Project to vocabulary (weight tying with token embeddings)
        logits = F.linear(x, self.token_embedding.weight)
        
        return logits
    
    def _block_forward(self, block, x, audio_features, mask):
        """Forward pass through a single transformer block."""
        # Self-attention
        x_ln = block.attn_ln(x)
        attn_out = self._self_attention(block.attn, x_ln, mask)
        x = x + attn_out
        
        # Cross-attention
        x_ln = block.cross_attn_ln(x)
        cross_out = self._cross_attention(block.cross_attn, x_ln, audio_features)
        x = x + cross_out
        
        # MLP
        x = x + block.mlp(block.mlp_ln(x))
        
        return x
    
    def _self_attention(self, attn, x, mask):
        """Self-attention with causal mask."""
        batch_size, seq_len, n_state = x.shape
        n_head = attn.n_head
        d_head = n_state // n_head
        
        q = attn.query(x)
        k = attn.key(x)
        v = attn.value(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, n_head, d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, n_head, d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, n_head, d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = d_head ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        # mask shape is (seq_len, seq_len), broadcast to (batch, n_head, seq_len, seq_len)
        attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_state)
        out = attn.out(out)
        return out
    
    def _cross_attention(self, attn, x, xa):
        """Cross-attention with audio features."""
        batch_size, seq_len, n_state = x.shape
        _, audio_len, _ = xa.shape
        n_head = attn.n_head
        d_head = n_state // n_head
        
        q = attn.query(x)
        k = attn.key(xa)
        v = attn.value(xa)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, n_head, d_head).transpose(1, 2)
        k = k.view(batch_size, audio_len, n_head, d_head).transpose(1, 2)
        v = v.view(batch_size, audio_len, n_head, d_head).transpose(1, 2)
        
        # Scaled dot-product attention (no mask for cross-attention)
        scale = d_head ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_state)
        out = attn.out(out)
        return out


def export_tokenizer(model_name: str, output_dir: Path):
    """Export tokenizer vocabulary and special tokens to JSON."""
    multilingual = model_name not in ["tiny.en", "base.en", "small.en", "medium.en"]
    tokenizer = get_tokenizer(multilingual=multilingual)
    
    # Get the encoding
    encoding = tokenizer.encoding
    
    # Build vocabulary (token_id -> string)
    vocab = {}
    for token, token_id in encoding._mergeable_ranks.items():
        try:
            vocab[str(token_id)] = token.decode('utf-8', errors='replace')
        except:
            vocab[str(token_id)] = f"<byte_{token_id}>"
    
    # Add special tokens
    for token, token_id in encoding._special_tokens.items():
        vocab[str(token_id)] = token
    
    tokenizer_data = {
        "vocab": vocab,
        "special_tokens": {
            "sot": tokenizer.sot,
            "eot": tokenizer.eot,
            "transcribe": tokenizer.transcribe,
            "translate": tokenizer.translate,
            "sot_prev": tokenizer.sot_prev,
            "sot_lm": tokenizer.sot_lm,
            "no_speech": tokenizer.no_speech,
            "no_timestamps": tokenizer.no_timestamps,
        },
        "language_tokens": dict(zip(tokenizer.all_language_codes, 
                                   [int(t) for t in tokenizer.all_language_tokens]))
        if hasattr(tokenizer, 'all_language_codes') else {"en": 50259},
    }
    
    output_path = output_dir / "tokenizer.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    print(f"Tokenizer saved to: {output_path}")
    return tokenizer_data


def export_encoder_executorch(encoder: nn.Module, output_path: Path, n_mels: int):
    """Export encoder to ExecuTorch .pte format."""
    print("\n" + "=" * 50)
    print("Exporting Encoder to ExecuTorch")
    print("=" * 50)
    
    encoder.eval()
    
    # Example input - fixed size for encoder (30 seconds of audio)
    example_mel = torch.randn(1, n_mels, 3000)
    
    # Test forward pass
    print("Testing encoder forward pass...")
    with torch.no_grad():
        test_out = encoder(example_mel)
    print(f"  Input shape: {example_mel.shape}")
    print(f"  Output shape: {test_out.shape}")
    
    # Export using torch.export
    print("\nExporting with torch.export...")
    with torch.no_grad():
        exported_encoder = export(
            encoder,
            (example_mel,),
            strict=False,  # Allow some graph breaks for complex models
        )
    
    print("Converting to Edge program...")
    edge_encoder = to_edge(
        exported_encoder,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,  # Skip IR validation for complex models
        ),
    )
    
    print("Converting to ExecuTorch program...")
    et_encoder = edge_encoder.to_executorch()
    
    # Save to file
    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        f.write(et_encoder.buffer)
    
    print(f"Encoder saved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return output_path


def export_decoder_executorch(decoder: nn.Module, output_path: Path, n_state: int):
    """Export decoder to ExecuTorch .pte format."""
    print("\n" + "=" * 50)
    print("Exporting Decoder to ExecuTorch")
    print("=" * 50)
    
    decoder.eval()
    
    # Example inputs - use a moderate sequence length for tracing
    # Batch size is always 1 for inference
    example_tokens = torch.tensor([[50258, 50259, 50359, 50363]], dtype=torch.long)
    example_audio_features = torch.randn(1, 1500, n_state)
    
    # Test forward pass
    print("Testing decoder forward pass...")
    with torch.no_grad():
        test_out = decoder(example_tokens, example_audio_features)
    print(f"  Token input shape: {example_tokens.shape}")
    print(f"  Audio features shape: {example_audio_features.shape}")
    print(f"  Output shape: {test_out.shape}")
    
    # Define dynamic dimension for sequence length only
    # Batch is always 1 for inference, so we don't make it dynamic
    seq_dim = Dim("seq_len", min=1, max=447)  # max < 448 to avoid guard issue
    
    # Export using torch.export with dynamic shapes
    print("\nExporting with torch.export...")
    with torch.no_grad():
        exported_decoder = export(
            decoder,
            (example_tokens, example_audio_features),
            dynamic_shapes={
                "tokens": {1: seq_dim},  # Only seq_len is dynamic, batch=1 is fixed
                "audio_features": {},     # Fixed shape
            },
            strict=False,
        )
    
    print("Converting to Edge program...")
    edge_decoder = to_edge(
        exported_decoder,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
        ),
    )
    
    print("Converting to ExecuTorch program...")
    et_decoder = edge_decoder.to_executorch()
    
    # Save to file
    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        f.write(et_decoder.buffer)
    
    print(f"Decoder saved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return output_path


def export_torchscript(encoder: nn.Module, decoder: nn.Module, output_path: Path, n_mels: int, n_state: int):
    """Export to TorchScript (.pt) format for LibTorch C++ inference."""
    print("\n" + "=" * 50)
    print("Exporting to TorchScript (.pt) for LibTorch")
    print("=" * 50)
    
    encoder.eval()
    decoder.eval()
    
    # Example inputs
    example_mel = torch.randn(1, n_mels, 3000)
    example_tokens = torch.tensor([[50258, 50259, 50359, 50363]], dtype=torch.long)
    example_audio_features = torch.randn(1, 1500, n_state)
    
    # Trace encoder
    print("Tracing encoder...")
    with torch.no_grad():
        traced_encoder = torch.jit.trace(encoder, example_mel)
    
    # Trace decoder  
    print("Tracing decoder...")
    with torch.no_grad():
        traced_decoder = torch.jit.trace(decoder, (example_tokens, example_audio_features))
    
    # Save
    encoder_pt = output_path.parent / (output_path.stem + ".encoder.pt")
    decoder_pt = output_path.parent / (output_path.stem + ".decoder.pt")
    
    torch.jit.save(traced_encoder, str(encoder_pt))
    print(f"Encoder saved: {encoder_pt}")
    
    torch.jit.save(traced_decoder, str(decoder_pt))
    print(f"Decoder saved: {decoder_pt}")
    
    return encoder_pt, decoder_pt


def export_model(model_name: str, output_path: Path):
    """Export Whisper model to both ExecuTorch and TorchScript formats."""
    
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name, device="cpu")
    model.eval()
    
    # Get model dimensions
    n_mels = model.dims.n_mels
    n_audio_ctx = model.dims.n_audio_ctx
    n_audio_state = model.dims.n_audio_state
    n_text_state = model.dims.n_text_state
    n_vocab = model.dims.n_vocab
    n_text_ctx = model.dims.n_text_ctx
    
    print(f"\nModel dimensions:")
    print(f"  n_mels:        {n_mels}")
    print(f"  n_audio_ctx:   {n_audio_ctx}")
    print(f"  n_audio_state: {n_audio_state}")
    print(f"  n_text_state:  {n_text_state}")
    print(f"  n_vocab:       {n_vocab}")
    print(f"  n_text_ctx:    {n_text_ctx}")
    
    # Create wrappers
    encoder = WhisperEncoder(model)
    decoder = WhisperDecoder(model, max_seq_len=n_text_ctx)
    
    # Export paths for ExecuTorch
    encoder_pte = output_path.parent / (output_path.stem + ".encoder.pte")
    decoder_pte = output_path.parent / (output_path.stem + ".decoder.pte")
    
    # Export to ExecuTorch (.pte) - for mobile/edge deployment
    export_encoder_executorch(encoder, encoder_pte, n_mels)
    export_decoder_executorch(decoder, decoder_pte, n_audio_state)
    
    # Export to TorchScript (.pt) - for LibTorch C++ desktop inference
    export_torchscript(encoder, decoder, output_path, n_mels, n_audio_state)
    
    # Save model info
    model_info = {
        "model_name": model_name,
        "n_mels": n_mels,
        "n_audio_ctx": n_audio_ctx,
        "n_vocab": n_vocab,
        "n_text_ctx": n_text_ctx,
        "n_audio_state": n_audio_state,
        "n_text_state": n_text_state,
        "files": {
            "executorch": {
                "encoder": output_path.stem + ".encoder.pte",
                "decoder": output_path.stem + ".decoder.pte",
            },
            "torchscript": {
                "encoder": output_path.stem + ".encoder.pt",
                "decoder": output_path.stem + ".decoder.pt",
            }
        }
    }
    
    info_path = output_path.parent / (output_path.stem + ".json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"\nModel info saved to: {info_path}")
    
    return encoder_pte, decoder_pte


def main():
    parser = argparse.ArgumentParser(description="Export Whisper to ExecuTorch")
    parser.add_argument("--model", type=str, default="tiny", 
                        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", 
                                "medium", "medium.en", "large", "large-v2", "large-v3"],
                        help="Whisper model size")
    parser.add_argument("--output", type=str, default="whisper.pte",
                        help="Output path for the model files")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export tokenizer
    print("=" * 50)
    print("Exporting tokenizer...")
    print("=" * 50)
    export_tokenizer(args.model, output_dir)
    
    # Export model
    print("\n" + "=" * 50)
    print(f"Exporting Whisper {args.model} model to ExecuTorch...")
    print("=" * 50)
    export_model(args.model, output_path)
    
    print("\n" + "=" * 50)
    print("Export complete!")
    print("=" * 50)
    print("\nGenerated files:")
    for f in sorted(output_dir.glob(f"{output_path.stem}*")):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
