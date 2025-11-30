import torch
import sys

print("Testing SAM 3 (Segment Anything Model 3)...")
print(f"CUDA Available: {torch.cuda.is_available()}")

try:
    # Attempt to import SAM 3
    import sam3
    from sam3 import build_sam3, SamPredictor

    print("\nSAM 3 is installed!")
    print(f"SAM 3 version: {sam3.__version__ if hasattr(sam3, '__version__') else 'unknown'}")

    # Note: SAM 3 requires downloading checkpoints from Hugging Face
    # You need to request access at: https://huggingface.co/facebook/sam3
    # Then authenticate: huggingface-cli login

    print("\nNOTE: To use SAM 3, you need to:")
    print("1. Request access to checkpoints at: https://huggingface.co/facebook/sam3")
    print("2. Authenticate with: huggingface-cli login")
    print("3. Download and load a checkpoint (e.g., 'sam3_vit_b')")

except ImportError:
    print("\nSAM 3 is not installed.")
    print("\nSAM 3 has complex installation requirements:")
    print("1. PyTorch 2.7+ (currently you have 2.6.0+cu124)")
    print("2. Python 3.12+")
    print("3. CUDA 12.6+")
    print("4. Additional system dependencies (BLAS library, etc.)")
    print("\nInstallation steps:")
    print("  git clone https://github.com/facebookresearch/sam3.git")
    print("  cd sam3")
    print("  pip install -e .")
    print("\nNote: On Windows, this may require additional build tools.")
except Exception as e:
    print(f"\nError with SAM 3: {e}")

print("\n" + "="*60)
print("System Information:")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
