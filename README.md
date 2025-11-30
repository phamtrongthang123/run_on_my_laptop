# SoTA Models Testing on Dell XPS 15 9520

Testing state-of-the-art AI models on a Dell XPS 15 9520 laptop.

**Environment:**
- PyTorch: 2.6.0+cu124
- CUDA: 12.4
- Python: 3.12/3.13
- OS: Windows 11

## System Specifications

- **Model:** Dell XPS 15 9520
- **CPU:** 12th Gen Intel(R) Core(TM) i7-12700H
- **RAM:** 16 GB (16,055 MB)
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- **iGPU:** Intel(R) Iris(R) Xe Graphics
- **Package Manager:** uv (no conda)

## Prerequisites

Make sure you have [uv](https://github.com/astral-sh/uv) installed:

```bash
# Install uv if not already installed
# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Quick Start

```bash
# Install all dependencies (this will install PyTorch with CUDA 12.4)
uv sync

# Test PyTorch CUDA
uv run python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Run all model tests
uv run python tests/run_all_tests.py

# Or run individual tests
uv run python asr/test_whisper.py
uv run python asr/test_parakeet.py
uv run python segmentation/test_sam3.py
```

## Detailed Setup

### 1. Initialize Project with PyTorch

The project is pre-configured with PyTorch CUDA support in `pyproject.toml`. Simply run:

```bash
uv sync
```

This will install PyTorch 2.6.0 with CUDA 12.4 support, compatible with your RTX 3050.

**Manual installation (if needed):**
```bash
uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
```

### 2. Install OpenAI Whisper

Already included in dependencies:

```bash
uv add openai-whisper
```

### 3. Install NVIDIA Parakeet TDT 0.6B v3

**Note:** Parakeet TDT requires NeMo toolkit, which has **limited Windows support**. The model may not work properly on Windows due to POSIX-specific dependencies.

```bash
uv add nemo-toolkit[all]
```

**Known Issues:**
- NeMo toolkit uses `signal.SIGKILL` which is not available on Windows
- Best used on Linux systems

### 4. SAM 3 Status

**Note:** SAM 3 is **not compatible** with this system:
- Requires PyTorch 2.7+ (current: 2.6.0)
- Requires CUDA 12.6+ (current: 12.4)
- Requires complex build dependencies (BLAS, etc.)
- Installation on Windows is challenging

The test script `segmentation/test_sam3.py` checks system compatibility and provides installation guidance when requirements are met.

## References

- [PyTorch](https://pytorch.org/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [NVIDIA Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [SAM 3](https://github.com/facebookresearch/sam3)
- [uv Package Manager](https://github.com/astral-sh/uv)
