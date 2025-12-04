# Whisper Standalone

A simple standalone Whisper speech-to-text implementation.

**Goal:** `whisper.exe --input a.wav --output a.json --model whisper.pte`

## Quick Start

```bash
# Install dependencies
uv sync

# Export model to portable format
uv run whisper_cpp/scripts/export_whisper_to_pte.py --model large --output models/whisper

# Build C++ executable (see whisper_cpp/README.md)
cd whisper_cpp && mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release

# Run
./whisper -i audio.wav -m ../models/whisper.pte -o result.json
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
```

See `whisper_cpp/README.md` for detailed build instructions.
