# Whisper C++ - Standalone Speech-to-Text

A simple standalone C++ executable for OpenAI Whisper speech recognition.

**Goal:** Single executable + model files = complete transcription system.

```bash
whisper -i audio.wav -m models/whisper.pt -o result.json
```

## Features

- 🎯 **Simple CLI**: Just input, output, and model paths
- 📦 **TorchScript Models**: Export Whisper to `.pt` format
- 🚀 **Fast**: Native C++ with LibTorch backend
- 🎵 **Full Audio Support**: Automatically processes long audio in 30-second chunks
- 💻 **Cross-platform**: Windows, Linux, macOS

## Quick Start

### 1. Export Whisper Model

First, export the Whisper model to TorchScript format:

```bash
# Install dependencies (if not already installed)
pip install openai-whisper torch

# Export the model (creates whisper.encoder.pt, whisper.decoder.pt, tokenizer.json)
python scripts/export_whisper_to_pte.py --model base --output models/
```

This creates three files in `models/`:
- `whisper.encoder.pt` - Audio encoder
- `whisper.decoder.pt` - Text decoder  
- `tokenizer.json` - Token vocabulary

Available model sizes:
- `tiny` / `tiny.en` - 39M parameters
- `base` / `base.en` - 74M parameters (recommended for testing)
- `small` / `small.en` - 244M parameters
- `medium` / `medium.en` - 769M parameters
- `large` / `large-v2` / `large-v3` - 1.5B parameters

### 2. Build the Executable

#### Prerequisites

- CMake 3.18+
- C++17 compatible compiler
- LibTorch (PyTorch C++ distribution)

#### Download LibTorch

**Linux:**
```bash
# CPU only
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip

# With CUDA 12.1
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip
```

**Windows:**
```powershell
# CPU only
Invoke-WebRequest -Uri "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.1.0%2Bcpu.zip" -OutFile libtorch.zip
Expand-Archive libtorch.zip -DestinationPath .

# With CUDA 12.1
Invoke-WebRequest -Uri "https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.1.0%2Bcu121.zip" -OutFile libtorch.zip
Expand-Archive libtorch.zip -DestinationPath .
```

**macOS:**
```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.1.0.zip
unzip libtorch-macos-arm64-2.1.0.zip
```

#### Build

```bash
# Create build directory
mkdir build && cd build

# Download LibTorch into build directory (or specify path)
# ... (see above)

# Configure (adjust path to libtorch)
cmake -DCMAKE_PREFIX_PATH=$(pwd)/libtorch -DCMAKE_BUILD_TYPE=Release ..

# Build
cmake --build . --config Release

# The executable will be in build/bin/whisper
```

**Windows with Visual Studio:**
```powershell
mkdir build; cd build
cmake -DCMAKE_PREFIX_PATH=C:\path\to\libtorch -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

### 3. Run Transcription

```bash
# Basic usage - transcribes entire audio file
./build/bin/whisper -i audio.wav -m models/whisper.pt -o result.json

# With verbose output
./build/bin/whisper -i audio.wav -m models/whisper.pt -o result.json -v

# Show help
./build/bin/whisper --help
```

**Important:** The `-m` option takes the base model path (`models/whisper.pt`), and the program automatically loads both `whisper.encoder.pt` and `whisper.decoder.pt`.

## Usage

```
Whisper - Speech-to-Text Transcription

Usage: whisper [options]

Options:
  -i, --input PATH      Input audio file (WAV format)
  -o, --output PATH     Output JSON file (default: <input>.json)
  -m, --model PATH      Model base path (e.g., models/whisper.pt)
  -t, --tokenizer PATH  Tokenizer JSON file (default: auto-detect)
  -l, --language LANG   Language code (default: en)
  --task TASK           Task: transcribe or translate (default: transcribe)
  --timestamps          Include word timestamps
  -v, --verbose         Verbose output
  -h, --help            Show this help message

Example:
  whisper -i audio.wav -m models/whisper.pt -o result.json
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

## How It Works

### Chunked Processing

For audio longer than 30 seconds, the program automatically:
1. Splits the audio into 30-second chunks
2. Processes each chunk through the encoder and decoder
3. Combines all transcriptions into the final output

Example output for a 157-second audio file:
```
Loading audio: Stockmarketnews.wav
  Sample rate: 24000 Hz
  Channels: 1
  Duration: 157.08 seconds
  Resampling to 16000 Hz...
  Processing 6 chunk(s) of 30 seconds each...
  Chunk 1/6 [0s - 30s] mel: 80 x 2998
  Chunk 2/6 [30s - 60s] mel: 80 x 2998
  ...
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
    ├── whisper.encoder.pt
    ├── whisper.decoder.pt
    └── tokenizer.json
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

### "Cannot find Torch"
Set the LibTorch path:
```bash
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
```

### "Missing DLLs" (Windows)
Copy the DLLs from `libtorch/lib/` to the same directory as `whisper.exe`, or add `libtorch/lib` to your PATH.

### "Encoder model not found: models/whisper.encoder.encoder.pt"
You're specifying the wrong model path. Use the **base** path without `.encoder`:
```bash
# Wrong:
./whisper -m models/whisper.encoder.pt ...

# Correct:
./whisper -m models/whisper.pt ...
```

### "Model not found"
Make sure you have all three model files:
```bash
ls models/
# Should show: whisper.encoder.pt, whisper.decoder.pt, tokenizer.json
```

### Audio processing issues
Convert your audio to standard WAV format:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -acodec pcm_s16le output.wav
```

### Repetitive output in transcription
This can happen with greedy decoding when the model gets "stuck". Solutions:
- Use a larger model (`small` or `medium` instead of `base`)
- The audio chunk may contain silence or noise

## License

MIT License. This project uses:
- OpenAI Whisper model weights (MIT License)
- PyTorch/LibTorch (BSD License)
