import subprocess
import sys
from pathlib import Path

# Get project root (parent of tests folder)
project_root = Path(__file__).parent.parent

tests = [
    ("ASR: OpenAI Whisper", project_root / "asr" / "test_whisper.py"),
    ("ASR: NVIDIA Parakeet TDT", project_root / "asr" / "test_parakeet.py"),
    ("Segmentation: SAM 3", project_root / "segmentation" / "test_sam3.py"),
]

print("="*60)
print("Running All Model Tests")
print("="*60)

for test_name, test_path in tests:
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"Path: {test_path}")
    print('='*60)
    try:
        result = subprocess.run([sys.executable, str(test_path)], check=True, capture_output=False)
        print(f"\n[OK] {test_name} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Error running {test_name}: Exit code {e.returncode}")
    except FileNotFoundError:
        print(f"\n[ERROR] Test file not found: {test_path}")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error running {test_name}: {e}")

print("\n" + "="*60)
print("All tests completed")
print("="*60)
