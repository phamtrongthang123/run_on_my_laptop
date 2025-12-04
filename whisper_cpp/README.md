# Whisper C++ - Standalone Speech-to-Text

A simple standalone C++ executable for OpenAI Whisper speech recognition.

**Goal:** Single executable + model files = complete transcription system.

```bash
whisper -i audio.wav -m models/whisper.pte -o result.json
```

## Features

- 🎯 **Simple CLI**: Just input, output, and model paths
- 🚀 **ExecuTorch Backend**: Optimized for edge deployment with `.pte` format
- 📦 **LibTorch Fallback**: Also supports TorchScript `.pt` format
- 🎵 **Full Audio Support**: Automatically processes long audio in 30-second chunks
- 💻 **Cross-platform**: Windows, Linux, macOS

## Quick Start

### 1. Export Whisper Model

First, export the Whisper model to ExecuTorch format:

```bash
# Install dependencies
uv sync  # or pip install -e .

# Export the model to ExecuTorch format
uv run python scripts/export_whisper_to_pte.py --model base --output models/whisper.pte
```

This creates in `models/`:
- `whisper.encoder.pte` - Audio encoder (ExecuTorch)
- `whisper.decoder.pte` - Text decoder (ExecuTorch)
- `whisper.json` - Model configuration
- `tokenizer.json` - Token vocabulary

Available model sizes:
- `tiny` / `tiny.en` - 39M parameters
- `base` / `base.en` - 74M parameters (recommended for testing)
- `small` / `small.en` - 244M parameters
- `medium` / `medium.en` - 769M parameters
- `large` / `large-v2` / `large-v3` - 1.5B parameters

### 2. Build the Executable

You can build with either ExecuTorch (recommended) or LibTorch backend.

#### Option A: Build with ExecuTorch (Recommended)

##### Prerequisites
- CMake 3.18+
- C++17 compatible compiler
- ExecuTorch installation

##### Install ExecuTorch

```bash
# Clone and build ExecuTorch
git clone https://github.com/pytorch/executorch.git
cd executorch
git submodule sync
git submodule update --init

# Install dependencies
pip install cmake
./install_requirements.sh

# Build ExecuTorch
mkdir cmake-out && cd cmake-out
cmake -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
      -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
      -DCMAKE_INSTALL_PREFIX=install \
      ..
cmake --build . --target install -j$(nproc)
```

##### Build Whisper with ExecuTorch

```bash
mkdir build && cd build

cmake -DUSE_EXECUTORCH=ON \
      -DEXECUTORCH_ROOT=/path/to/executorch/cmake-out/install \
      -DCMAKE_BUILD_TYPE=Release \
      ..

cmake --build . --config Release
```

#### Option B: Build with LibTorch (Fallback)

If you prefer TorchScript models, you can still use LibTorch:

##### Download LibTorch

**Linux:**
```bash
# CPU only
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.4.0+cpu.zip

# With CUDA 12.4
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.4.0+cu124.zip
```

**Windows:**
```powershell
# CPU only
Invoke-WebRequest -Uri "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.4.0%2Bcpu.zip" -OutFile libtorch.zip
Expand-Archive libtorch.zip -DestinationPath .
```

**macOS:**
```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.4.0.zip
unzip libtorch-macos-arm64-2.4.0.zip
```

##### Build with LibTorch

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

### 3. Run Transcription

```bash
# With ExecuTorch models (.pte)
./build/bin/whisper -i audio.wav -m models/whisper.pte -o result.json

# With LibTorch models (.pt)
./build/bin/whisper -i audio.wav -m models/whisper.pt -o result.json

# With verbose output
./build/bin/whisper -i audio.wav -m models/whisper.pte -o result.json -v

# Show help
./build/bin/whisper --help
```

## Usage

```
Whisper - Speech-to-Text Transcription

Usage: whisper [options]

Options:
  -i, --input PATH      Input audio file (WAV format)
  -o, --output PATH     Output JSON file (default: <input>.json)
  -m, --model PATH      Model base path (e.g., models/whisper.pte)
  -t, --tokenizer PATH  Tokenizer JSON file (default: auto-detect)
  -l, --language LANG   Language code (default: en)
  --task TASK           Task: transcribe or translate (default: transcribe)
  --timestamps          Include word timestamps
  -v, --verbose         Verbose output
  -h, --help            Show this help message

Example:
  whisper -i audio.wav -m models/whisper.pte -o result.json
```

## Output Format

```json
{
  "text": "Good morning. This is the stock market news for today...",
  "language": "en",
  "duration": 157.08,
  "processing_time": 23.42
}
```

## ExecuTorch vs LibTorch

| Feature | ExecuTorch (.pte) | LibTorch (.pt) |
|---------|-------------------|----------------|
| Format | Portable, optimized | TorchScript |
| Binary size | Smaller | Larger |
| Edge deployment | ✅ Designed for | ⚠️ Heavy |
| Mobile support | ✅ iOS/Android | ❌ Desktop only |
| Hardware acceleration | CoreML, XNNPACK, etc. | CUDA only |
| Export method | `torch.export` | `torch.jit.trace` |

## How It Works

### Chunked Processing

For audio longer than 30 seconds, the program automatically:
1. Splits the audio into 30-second chunks
2. Processes each chunk through the encoder and decoder
3. Combines all transcriptions into the final output

### Architecture

```
Audio (WAV) → Mel Spectrogram → Encoder → Audio Features
                                              ↓
                               Decoder (autoregressive)
                                              ↓
                               Tokens → Text
```

## Project Structure

```
whisper_cpp/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── include/
│   ├── audio.h             # Audio processing header
│   └── tokenizer.h         # Tokenizer header
├── src/
│   ├── whisper.cpp         # Main executable
│   ├── audio.cpp           # Audio processing (WAV, mel spectrogram, chunking)
│   └── tokenizer.cpp       # Token decoding
├── scripts/
│   └── export_whisper_to_pte.py  # Model export script
└── models/                 # Exported models (create this)
    ├── whisper.encoder.pte # ExecuTorch encoder
    ├── whisper.decoder.pte # ExecuTorch decoder
    ├── whisper.json        # Model config
    └── tokenizer.json      # Vocabulary
```

## Supported Audio Formats

Currently supports:
- **WAV** (PCM 16-bit, 8-bit, 32-bit float)
- Mono or stereo (automatically converted to mono)
- Any sample rate (automatically resampled to 16kHz)
- Any duration (automatically chunked into 30-second segments)

For other formats (MP3, FLAC, etc.), convert to WAV first:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Performance

Approximate transcription times:

| Model | Parameters | 30s Audio (CPU) | 157s Audio (CPU) |
|-------|------------|-----------------|------------------|
| tiny  | 39M        | ~2s             | ~10s             |
| base  | 74M        | ~4s             | ~23s             |
| small | 244M       | ~10s            | ~55s             |

*Tested on Intel i7-12700H, CPU only*

## Troubleshooting

### ExecuTorch Build Issues

**"Cannot find executorch"**
Make sure `EXECUTORCH_ROOT` points to the install directory:
```bash
cmake -DUSE_EXECUTORCH=ON -DEXECUTORCH_ROOT=/path/to/executorch/cmake-out/install ..
```

**Export fails with "unsupported operation"**
Some operations may not be supported by ExecuTorch. Try:
- Using `strict=False` in the export (already set in the script)
- Checking the [ExecuTorch supported ops](https://pytorch.org/executorch/stable/ir-ops-set-definition.html)

### LibTorch Build Issues

**"Cannot find Torch"**
Set the LibTorch path:
```bash
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
```

**"Missing DLLs" (Windows)**
Copy the DLLs from `libtorch/lib/` to the same directory as `whisper.exe`.

### Model Issues

**"Encoder model not found"**
Use the **base** path without `.encoder`:
```bash
# Wrong:
./whisper -m models/whisper.encoder.pte ...

# Correct:
./whisper -m models/whisper.pte ...
```

### Audio Issues

Convert your audio to standard WAV format:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -acodec pcm_s16le output.wav
```

## License

MIT License. This project uses:
- OpenAI Whisper model weights (MIT License)
- PyTorch/ExecuTorch (BSD License)
