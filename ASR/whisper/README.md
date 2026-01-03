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


### Download Tokenizer

Download the tokenizer files required for inference according to your model version:

**For Whisper Small:**
```bash
curl -L https://huggingface.co/openai/whisper-small/resolve/main/tokenizer.json -o tokenizer.json
curl -L https://huggingface.co/openai/whisper-small/resolve/main/tokenizer_config.json -o tokenizer_config.json
curl -L https://huggingface.co/openai/whisper-small/resolve/main/special_tokens_map.json -o special_tokens_map.json
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
# If you run with the same environment, you don't need those lines below. But if you, like me, copy the built whisper_runner somewhere else (laptop),
# you need to set the CUDA paths again. And remember to download libaoti_cuda_shims.so and put it in the same folder as whisper_runner
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./
export CUDA_HOME=/home/ptthang/miniconda3/envs/executorch/
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH # it really depends, sometimes lib64

cmake-out/examples/models/whisper/whisper_runner \
    --model_path ../model.pte \
    --data_path ../aoti_cuda_blob.ptd \
    --tokenizer_path ../ \
    --audio_path ../output.wav \
    --processor_path ../whisper_preprocessor.pte \
    --temperature 0
```

And hopefully you will see the transcription output:
```bash
(executorch) ptthang@ProfThang:/mnt/c/Users/phamt/Downloads/whisper$ ./whisper_runner     --model_path ./model.pte     --data_path ./aoti_cuda_blob.ptd     --tokenizer_path ./     --audio_path output.wav     --processor_path whisper_preproc
essor.pte     --temperature 0
I tokenizers:regex.cpp:27] Registering override fallback regex
I tokenizers:hf_tokenizer.cpp:142] Setting up normalizer...
I tokenizers:hf_tokenizer.cpp:148] Normalizer field is null, skipping
I tokenizers:hf_tokenizer.cpp:160] Setting up pretokenizer...
I tokenizers:hf_tokenizer.cpp:164] Pretokenizer set up
I tokenizers:hf_tokenizer.cpp:180] Loading BPE merges...
I tokenizers:hf_tokenizer.cpp:215] Loaded 0 BPE merge rules
I tokenizers:hf_tokenizer.cpp:227] Built merge ranks map with 0 entries
I tokenizers:hf_tokenizer.cpp:256] Loaded tokens from special_tokens_map.json: bos='<|endoftext|>', eos='<|endoftext|>'
<|en|><|transcribe|><|0.00|> Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.<|6.60|><|6.60|> Nor is Mr. Quilter's manner less interesting than his matter.<|11.30|><|11.30|> He tells us that at this festive season of the year, with Christmas and roast beef looming<|16.86|><|16.86|> before us, simile is drawn from eating and its results occur most readily to the mind.<|23.80|><|23.80|> He has grave doubts whether Sir Frederick Layton's work is really Greek after all, and can discover<|29.98|><|29.98|><|endoftext|>
PyTorchObserver {"prompt_tokens":0,"generated_tokens":109,"model_load_start_ms":1764907778897,"model_load_end_ms":1764907783864,"inference_start_ms":1764907783864,"inference_end_ms":1764907785270,"prompt_eval_end_ms":1764907784100,"first_token_ms":1764907784127,"aggregate_sampling_time_ms":0,"SCALING_FACTOR_UNITS_PER_SECOND":1000}
```

The size is ~500MB in total. 
```bash
(executorch) ptthang@ProfThang:/mnt/c/Users/phamt/Downloads/whisper$ ls
aoti_cuda_blob.ptd     model.pte   output2.wav              tokenizer.json         whisper_preprocessor.pte
libaoti_cuda_shims.so  output.wav  special_tokens_map.json  tokenizer_config.json  whisper_runner

(executorch) ptthang@ProfThang:/mnt/c/Users/phamt/Downloads/whisper$ ls -lh
total 522M
-rwxrwxrwx 1 ptthang ptthang 498M Dec  4 21:46 aoti_cuda_blob.ptd
-rwxrwxrwx 1 ptthang ptthang  72K Dec  4 21:50 libaoti_cuda_shims.so
-rwxrwxrwx 1 ptthang ptthang 2.6M Dec  4 21:42 model.pte
-rwxrwxrwx 1 ptthang ptthang 938K Dec  4 22:09 output.wav
-rwxrwxrwx 1 ptthang ptthang 938K Dec  4 21:46 output2.wav
-rwxrwxrwx 1 ptthang ptthang 2.2K Dec  4 21:46 special_tokens_map.json
-rwxrwxrwx 1 ptthang ptthang 2.4M Dec  4 21:46 tokenizer.json
-rwxrwxrwx 1 ptthang ptthang 277K Dec  4 21:46 tokenizer_config.json
-rwxrwxrwx 1 ptthang ptthang  78K Dec  4 21:46 whisper_preprocessor.pte
-rwxrwxrwx 1 ptthang ptthang  18M Dec  4 21:47 whisper_runner
```