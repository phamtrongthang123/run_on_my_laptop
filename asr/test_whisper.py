import whisper
import torch

print("Testing OpenAI Whisper...")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Load model (use 'base' or 'small' for 4GB VRAM)
print("\nLoading Whisper base model...")
model = whisper.load_model("base")

print(f"Whisper model loaded successfully!")
print(f"Model: base")
print(f"Model device: {next(model.parameters()).device}")

# Example usage:
# result = model.transcribe("audio.mp3")
# print(result["text"])
