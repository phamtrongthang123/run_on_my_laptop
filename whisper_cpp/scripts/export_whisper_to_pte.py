#!/usr/bin/env python3
"""
Export OpenAI Whisper model to TorchScript (.pt) format.

Usage:
    python export_whisper_to_pte.py --model base --output whisper.pte
    python export_whisper_to_pte.py --model small --output whisper.pte
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import whisper
from whisper.tokenizer import get_tokenizer


class WhisperEncoder(nn.Module):
    """Wrapper for Whisper encoder."""
    
    def __init__(self, whisper_model):
        super().__init__()
        self.conv1 = whisper_model.encoder.conv1
        self.conv2 = whisper_model.encoder.conv2
        self.positional_embedding = whisper_model.encoder.positional_embedding
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
        
        x = x + self.positional_embedding[:x.shape[1]]
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_post(x)
        return x


class WhisperDecoderStep(nn.Module):
    """Whisper decoder for single-step inference (no KV cache for simplicity)."""
    
    def __init__(self, whisper_model):
        super().__init__()
        self.token_embedding = whisper_model.decoder.token_embedding
        self.positional_embedding = whisper_model.decoder.positional_embedding
        self.blocks = whisper_model.decoder.blocks
        self.ln = whisper_model.decoder.ln
        self.n_vocab = whisper_model.dims.n_vocab
        
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
        x = self.token_embedding(tokens) + self.positional_embedding[:seq_len]
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=tokens.device), diagonal=1).bool()
        mask = mask.float().masked_fill(mask, float('-inf'))
        
        # Transformer blocks
        for block in self.blocks:
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
        
        x = self.ln(x)
        
        # Project to vocabulary (use weight tying)
        logits = x @ self.token_embedding.weight.T
        
        return logits
    
    def _self_attention(self, attn, x, mask):
        """Simple self-attention without KV cache."""
        n_head = attn.n_head
        d_head = x.shape[-1] // n_head
        
        qkv = attn.query(x), attn.key(x), attn.value(x)
        q, k, v = [t.view(*t.shape[:-1], n_head, d_head).transpose(1, 2) for t in qkv]
        
        # Scaled dot-product attention
        scale = d_head ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = attn_weights + mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(*x.shape[:-1], -1)
        out = attn.out(out)
        return out
    
    def _cross_attention(self, attn, x, xa):
        """Cross-attention with audio features."""
        n_head = attn.n_head
        d_head = x.shape[-1] // n_head
        
        q = attn.query(x)
        k = attn.key(xa)
        v = attn.value(xa)
        
        q = q.view(*q.shape[:-1], n_head, d_head).transpose(1, 2)
        k = k.view(*k.shape[:-1], n_head, d_head).transpose(1, 2)
        v = v.view(*v.shape[:-1], n_head, d_head).transpose(1, 2)
        
        scale = d_head ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(*x.shape[:-1], -1)
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


def export_model(model_name: str, output_path: Path):
    """Export Whisper model to TorchScript format."""
    
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name, device="cpu")
    model.eval()
    
    # Get model dimensions
    n_mels = model.dims.n_mels
    n_audio_ctx = model.dims.n_audio_ctx
    n_state = model.dims.n_audio_state
    
    print(f"Model dimensions: n_mels={n_mels}, n_audio_ctx={n_audio_ctx}, n_state={n_state}")
    
    # Create wrappers
    encoder = WhisperEncoder(model)
    encoder.eval()
    
    decoder = WhisperDecoderStep(model)
    decoder.eval()
    
    # Example inputs
    example_mel = torch.randn(1, n_mels, 3000)
    example_tokens = torch.tensor([[50258, 50259, 50359, 50363]], dtype=torch.long)
    
    # Test forward pass
    print("\nTesting encoder...")
    with torch.no_grad():
        audio_features = encoder(example_mel)
    print(f"  Encoder output shape: {audio_features.shape}")
    
    print("Testing decoder...")
    with torch.no_grad():
        logits = decoder(example_tokens, audio_features)
    print(f"  Decoder output shape: {logits.shape}")
    
    # Trace models
    print("\nTracing encoder...")
    with torch.no_grad():
        traced_encoder = torch.jit.trace(encoder, example_mel)
    
    print("Tracing decoder...")
    with torch.no_grad():
        traced_decoder = torch.jit.trace(decoder, (example_tokens, audio_features))
    
    # Verify traced models
    print("\nVerifying traced models...")
    with torch.no_grad():
        test_features = traced_encoder(example_mel)
        test_logits = traced_decoder(example_tokens, test_features)
    print(f"  Traced encoder output: {test_features.shape}")
    print(f"  Traced decoder output: {test_logits.shape}")
    
    # Save models
    encoder_path = output_path.parent / (output_path.stem + ".encoder.pt")
    decoder_path = output_path.parent / (output_path.stem + ".decoder.pt")
    
    torch.jit.save(traced_encoder, encoder_path)
    print(f"\nEncoder saved to: {encoder_path}")
    
    torch.jit.save(traced_decoder, decoder_path)
    print(f"Decoder saved to: {decoder_path}")
    
    # Save model info
    model_info = {
        "model_name": model_name,
        "n_mels": n_mels,
        "n_audio_ctx": n_audio_ctx,
        "n_vocab": model.dims.n_vocab,
        "n_text_ctx": model.dims.n_text_ctx,
        "n_audio_state": model.dims.n_audio_state,
        "n_text_state": model.dims.n_text_state,
    }
    
    info_path = output_path.parent / (output_path.stem + ".json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"Model info saved to: {info_path}")
    
    return encoder_path, decoder_path


def main():
    parser = argparse.ArgumentParser(description="Export Whisper to TorchScript")
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
    print(f"Exporting Whisper {args.model} model...")
    print("=" * 50)
    export_model(args.model, output_path)
    
    print("\n" + "=" * 50)
    print("Export complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
