# Whisper Runner
This is the real README to run whisper on my laptop. Originally from ExecuTorch repo [here](https://github.com/pytorch/executorch/blob/main/examples/models/whisper/README.md)

## Build
First, git clone so that the cmake link and all other paths are ready 
```bash
git clone --recursive https://github.com/pytorch/executorch.git
cd executorch
```
The best way to build the Whisper runner is to use the git repo of ExecuTorch itself. They do use many other submodules. 

Then we need conda to install cuda toolkit and other dependencies.
```bash
conda create -n executorch python=3.11 -y
conda activate executorch
conda install cmake -y 
conda install cuda-toolkit -c nvidia/label/cuda-12.8.0 -y 
```

Then install the python dependencies. This is purely for exporting the model and preparing the audio. The runner itself is in C++. 

```bash
pip install uv 

# Go into the install_dev and change the nightly into this. If you don't, the new support of cuda won't work. 
# def install_torch_nightly_deps():
#     """Install torch related dependencies from pinned nightly"""
#     EXECUTORCH_NIGHTLY_VERSION = "dev20251204"
#     TORCHAO_NIGHTLY_VERSION = "dev20251104"
#     # Torch nightly is aligned with pinned nightly in https://github.com/pytorch/executorch/blob/main/torch_pin.py#L2
#     TORCH_NIGHTLY_VERSION = "dev20251104"
#     subprocess.check_call(
#         [
#             sys.executable,
#             "-m",
#             "uv",
#             "pip",
#             "install",
#             f"torch==2.10.0.{TORCH_NIGHTLY_VERSION}",
#             f"torchvision==0.25.0.{TORCH_NIGHTLY_VERSION}",
#             f"torchaudio==2.10.0.{TORCH_NIGHTLY_VERSION}",
#             f"torchao==0.15.0.{TORCHAO_NIGHTLY_VERSION}",
#             "--extra-index-url",
#             "https://download.pytorch.org/whl/nightly/cu128",
#         ]
#     )
#     subprocess.check_call(
#         [
#             sys.executable,
#             "-m",
#             "pip",
#             "install",
#             f"executorch==1.1.0.{EXECUTORCH_NIGHTLY_VERSION}",
#             "--extra-index-url",
#             "https://download.pytorch.org/whl/nightly/cpu",
#         ]
#     )
git clone https://github.com/huggingface/optimum-executorch.git
cd optimum-executorch
pip install '.[dev]'
python install_dev.py
cd .. 
```


For CUDA:
```
make whisper-cuda
```


## Usage

### Export Whisper Model

Use [Optimum-ExecuTorch](https://github.com/huggingface/optimum-executorch) to export a Whisper model from Hugging Face:

#### CUDA backend:
Keep the environment above activated, then run (pardon me the long list of exports):
```bash
export CUDA_HOME=/home/ptthang/miniforge3/envs/executorch/
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CPATH:/home/ptthang/miniforge3/envs/executorch/targets/x86_64-linux/include
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/home/ptthang/miniforge3/envs/executorch/targets/x86_64-linux/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/home/ptthang/miniforge3/envs/executorch/targets/x86_64-linux/include
optimum-cli export executorch \
    --model openai/whisper-small \
    --task automatic-speech-recognition \
    --recipe cuda \
    --dtype bfloat16 \
    --device cuda \
    --output_dir ./
```

This command generates:
- `model.pte` — Compiled Whisper model
- `aoti_cuda_blob.ptd` — Weight data file for CUDA backend


### Preprocessor

Export a preprocessor to convert raw audio to mel-spectrograms:

```bash
# Use --feature_size 128 for whisper-large-v3 and whisper-large-v3-turbo
python -m executorch.extension.audio.mel_spectrogram \
    --feature_size 80 \
    --stack_output \
    --max_audio_len 300 \
    --output_file whisper_preprocessor.pte
```

### Quantization

Export quantized models to reduce size and improve performance (Not enabled for Metal yet):

```bash
# 4-bit tile packed quantization for encoder
optimum-cli export executorch \
    --model openai/whisper-small \
    --task automatic-speech-recognition \
    --recipe cuda \
    --dtype bfloat16 \
    --device cuda \
    --qlinear 4w \
    --qlinear_encoder 4w \
    --qlinear_packing_format tile_packed_to_4d \
    --qlinear_encoder_packing_format tile_packed_to_4d \
    --output_dir ./
```


### Download Tokenizer

Download the tokenizer files required for inference according to your model version:

**For Whisper Small:**
```bash
curl -L https://huggingface.co/openai/whisper-small/resolve/main/tokenizer.json -o tokenizer.json
curl -L https://huggingface.co/openai/whisper-small/resolve/main/tokenizer_config.json -o tokenizer_config.json
curl -L https://huggingface.co/openai/whisper-small/resolve/main/special_tokens_map.json -o special_tokens_map.json
```

**For Whisper Large v2:**
```bash
curl -L https://huggingface.co/openai/whisper-large-v2/resolve/main/tokenizer.json -o tokenizer.json
curl -L https://huggingface.co/openai/whisper-large-v2/resolve/main/tokenizer_config.json -o tokenizer_config.json
curl -L https://huggingface.co/openai/whisper-large-v2/resolve/main/special_tokens_map.json -o special_tokens_map.json
```

### Prepare Audio

Generate test audio or use an existing WAV file. The model expects 16kHz mono audio.

```bash
# Generate sample audio using librispeech dataset
python -c "from datasets import load_dataset; import soundfile as sf; sample = load_dataset('distil-whisper/librispeech_long', 'clean', split='validation')[0]['audio']; sf.write('output.wav', sample['array'][:sample['sampling_rate']*30], sample['sampling_rate'])"
```

### Run Inference

After building the runner (see [Build](#build) section), execute it with the exported model and audio:

#### CUDA backend:

```bash
# Run the Whisper runner
cmake-out/examples/models/whisper/whisper_runner \
    --model_path ../model.pte \
    --data_path ../aoti_cuda_blob.ptd \
    --tokenizer_path ../ \
    --audio_path ../output.wav \
    --processor_path ../whisper_preprocessor.pte \
    --temperature 0
```
