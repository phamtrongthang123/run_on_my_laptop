import torch

print("Testing NVIDIA Parakeet TDT 0.6B v3...")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Load Parakeet TDT model using NeMo
model_id = "nvidia/parakeet-tdt-0.6b-v3"

print(f"\nLoading Parakeet TDT model from {model_id}...")
print("This may take a while on first run (downloading model)...")

try:
    import nemo.collections.asr as nemo_asr

    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=model_id
    )

    print("\nParakeet TDT model loaded successfully!")
    print(f"Model: {model_id}")
    print(f"Device: {asr_model.device if hasattr(asr_model, 'device') else 'unknown'}")

    # Example usage (commented out):
    # output = asr_model.transcribe(['audio_file.wav'])
    # print(output[0].text)

except Exception as e:
    print(f"\nError loading Parakeet TDT: {e}")
    print("Make sure you have enough VRAM and internet connection for model download")
