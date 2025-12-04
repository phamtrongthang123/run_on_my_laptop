# Whisper Standalone

A simple standalone Whisper speech-to-text implementation using **ExecuTorch** for optimized edge deployment.

**Goal:** `whisper.exe --input a.wav --output a.json --model whisper.pte`

## Why ExecuTorch?

ExecuTorch is PyTorch's solution for deploying ML models to edge devices. Compared to TorchScript:

- ✅ **Better optimization** - Uses `torch.export` which captures more accurate graph
- ✅ **Smaller runtime** - Minimal dependencies for deployment
- ✅ **Hardware acceleration** - Built-in support for CoreML, XNNPACK, Vulkan
- ✅ **Mobile ready** - First-class iOS and Android support
- ✅ **Dynamic shapes** - Better handling of variable-length sequences

## Quick Start

```bash
# Install dependencies
uv sync

# Export model to ExecuTorch format (.pte)
uv run python whisper_cpp/scripts/export_whisper_to_pte.py --model tiny --output models/whisper.pte

# Build C++ executable (auto-detects ExecuTorch from pip)
cd whisper_cpp && mkdir build && cd build
cmake -DUSE_EXECUTORCH=ON -DEXECUTORCH_ROOT=/mnt/c/Users/phamt/MainHome/Research/run_on_my_laptop/.venv/lib/python3.12/site-packages/executorch -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release

# Run
./bin/whisper -i audio.wav -m ../../models/whisper.pte -o result.json
```

## Project Structure

```
.
├── pyproject.toml          # Python dependencies (for model export)
├── whisper_cpp/            # C++ standalone executable
│   ├── CMakeLists.txt      # Build configuration
│   ├── README.md           # Detailed build instructions
│   ├── include/            # Headers
│   ├── src/                # Source files
│   └── scripts/            # Model export script
└── models/                 # Exported models (generated)
    ├── whisper.encoder.pte # ExecuTorch encoder
    ├── whisper.decoder.pte # ExecuTorch decoder
    └── tokenizer.json      # Vocabulary
```

## Model Export

The export script converts OpenAI Whisper models to ExecuTorch format:

```bash
# Export different model sizes
uv run python whisper_cpp/scripts/export_whisper_to_pte.py --model tiny --output models/whisper.pte
uv run python whisper_cpp/scripts/export_whisper_to_pte.py --model base --output models/whisper.pte
uv run python whisper_cpp/scripts/export_whisper_to_pte.py --model small --output models/whisper.pte
```

The script uses `torch.export` → `to_edge` → `to_executorch` pipeline for optimal results.

## Build Options

| Option | Backend | Model Format | Use Case |
|--------|---------|--------------|----------|
| `USE_EXECUTORCH=ON` | ExecuTorch | `.pte` | Edge deployment, mobile |
| `USE_EXECUTORCH=OFF` | LibTorch | `.pt` | Desktop, development |

See `whisper_cpp/README.md` for detailed build instructions.

## License

MIT
