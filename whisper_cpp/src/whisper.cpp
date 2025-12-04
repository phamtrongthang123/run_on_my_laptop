/**
 * @file whisper.cpp
 * @brief Main Whisper executable using ExecuTorch/LibTorch
 * 
 * Usage:
 *   whisper --input audio.wav --output result.json --model whisper.pte
 *   whisper -i audio.wav -o result.json -m whisper.pte
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <cstring>

#include "audio.h"
#include "tokenizer.h"

#ifdef USE_EXECUTORCH
    // ExecuTorch headers
    #include <executorch/extension/module/module.h>
    #include <executorch/extension/tensor/tensor.h>
    #include <executorch/runtime/core/exec_aten/exec_aten.h>
    #define INFERENCE_BACKEND "ExecuTorch"
    
    namespace et = executorch::extension;
    namespace runtime = executorch::runtime;
#else
    // LibTorch headers
    #include <torch/script.h>
    #include <torch/torch.h>
    #define INFERENCE_BACKEND "LibTorch"
#endif

namespace fs = std::filesystem;

// Configuration
struct Config {
    std::string input_path;
    std::string output_path;
    std::string model_path;
    std::string tokenizer_path;
    std::string language = "en";
    std::string task = "transcribe";
    bool timestamps = false;
    bool verbose = false;
    int max_tokens = 224;
};

// Result structure
struct TranscriptionResult {
    std::string text;
    std::string language;
    double duration_seconds;
    double processing_time_seconds;
    std::vector<std::pair<float, float>> segments;  // (start, end) times
};

// Command line argument parser
Config parse_args(int argc, char* argv[]) {
    Config config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            config.input_path = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            config.output_path = argv[++i];
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if ((arg == "-t" || arg == "--tokenizer") && i + 1 < argc) {
            config.tokenizer_path = argv[++i];
        } else if ((arg == "-l" || arg == "--language") && i + 1 < argc) {
            config.language = argv[++i];
        } else if (arg == "--task" && i + 1 < argc) {
            config.task = argv[++i];
        } else if (arg == "--timestamps") {
            config.timestamps = true;
        } else if ((arg == "-v" || arg == "--verbose")) {
            config.verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Whisper - Speech-to-Text Transcription\n\n"
                     << "Usage: whisper [options]\n\n"
                     << "Options:\n"
                     << "  -i, --input PATH      Input audio file (WAV format)\n"
                     << "  -o, --output PATH     Output JSON file\n"
                     << "  -m, --model PATH      Model file (.pte or .pt)\n"
                     << "  -t, --tokenizer PATH  Tokenizer JSON file (default: auto-detect)\n"
                     << "  -l, --language LANG   Language code (default: en)\n"
                     << "  --task TASK           Task: transcribe or translate (default: transcribe)\n"
                     << "  --timestamps          Include word timestamps\n"
                     << "  -v, --verbose         Verbose output\n"
                     << "  -h, --help            Show this help message\n\n"
                     << "Example:\n"
                     << "  whisper -i audio.wav -m models/whisper.pte -o result.json\n";
            exit(0);
        }
    }
    
    // Validate required arguments
    if (config.input_path.empty()) {
        std::cerr << "Error: Input file required (-i/--input)" << std::endl;
        exit(1);
    }
    if (config.model_path.empty()) {
        std::cerr << "Error: Model file required (-m/--model)" << std::endl;
        exit(1);
    }
    
    // Default output path
    if (config.output_path.empty()) {
        fs::path input(config.input_path);
        config.output_path = input.stem().string() + ".json";
    }
    
    // Auto-detect tokenizer path
    if (config.tokenizer_path.empty()) {
        fs::path model_dir = fs::path(config.model_path).parent_path();
        fs::path tokenizer_path = model_dir / "tokenizer.json";
        if (fs::exists(tokenizer_path)) {
            config.tokenizer_path = tokenizer_path.string();
        } else {
            // Try same directory as executable
            config.tokenizer_path = "tokenizer.json";
        }
    }
    
    return config;
}

// Write JSON output
void write_json_output(const std::string& path, const TranscriptionResult& result) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot write output file: " << path << std::endl;
        return;
    }
    
    // Escape JSON string
    auto escape_json = [](const std::string& s) -> std::string {
        std::string escaped;
        for (char c : s) {
            switch (c) {
                case '"': escaped += "\\\""; break;
                case '\\': escaped += "\\\\"; break;
                case '\n': escaped += "\\n"; break;
                case '\r': escaped += "\\r"; break;
                case '\t': escaped += "\\t"; break;
                default: escaped += c; break;
            }
        }
        return escaped;
    };
    
    file << "{\n"
         << "  \"text\": \"" << escape_json(result.text) << "\",\n"
         << "  \"language\": \"" << result.language << "\",\n"
         << "  \"duration\": " << result.duration_seconds << ",\n"
         << "  \"processing_time\": " << result.processing_time_seconds;
    
    if (!result.segments.empty()) {
        file << ",\n  \"segments\": [";
        for (size_t i = 0; i < result.segments.size(); ++i) {
            if (i > 0) file << ", ";
            file << "{\"start\": " << result.segments[i].first 
                 << ", \"end\": " << result.segments[i].second << "}";
        }
        file << "]";
    }
    
    file << "\n}\n";
    
    std::cout << "Output written to: " << path << std::endl;
}

#ifdef USE_EXECUTORCH
// ExecuTorch inference implementation
class WhisperModelET {
public:
    WhisperModelET() = default;
    ~WhisperModelET() = default;
    
    bool load(const std::string& encoder_path, const std::string& decoder_path) {
        std::cout << "Loading encoder: " << encoder_path << std::endl;
        encoder_ = std::make_unique<et::Module>(encoder_path);
        if (!encoder_->is_loaded()) {
            std::cerr << "Failed to load encoder: " << encoder_path << std::endl;
            return false;
        }
        
        std::cout << "Loading decoder: " << decoder_path << std::endl;
        decoder_ = std::make_unique<et::Module>(decoder_path);
        if (!decoder_->is_loaded()) {
            std::cerr << "Failed to load decoder: " << decoder_path << std::endl;
            return false;
        }
        
        loaded_ = true;
        return true;
    }
    
    // Encode audio mel spectrogram to features
    std::vector<float> encode(const std::vector<float>& mel_data, int n_mels, int n_frames) {
        // Create input tensor: shape (1, n_mels, n_frames)
        std::vector<int32_t> input_shape = {1, n_mels, n_frames};
        auto input_tensor = et::from_blob(
            const_cast<float*>(mel_data.data()),
            input_shape,
            runtime::ScalarType::Float
        );
        
        // Run encoder
        auto result = encoder_->forward({input_tensor});
        if (!result.ok()) {
            std::cerr << "Encoder forward failed" << std::endl;
            return {};
        }
        
        // Extract output tensor
        auto& outputs = result.get();
        if (outputs.empty()) {
            std::cerr << "Encoder returned no outputs" << std::endl;
            return {};
        }
        
        auto& output_evalue = outputs[0];
        if (!output_evalue.isTensor()) {
            std::cerr << "Encoder output is not a tensor" << std::endl;
            return {};
        }
        
        auto output_tensor = output_evalue.toTensor();
        
        // Copy output data
        size_t output_size = output_tensor.numel();
        std::vector<float> features(output_size);
        const float* output_data = output_tensor.const_data_ptr<float>();
        std::memcpy(features.data(), output_data, output_size * sizeof(float));
        
        // Store dimensions for later use
        n_audio_ctx_ = output_tensor.size(1);
        n_audio_state_ = output_tensor.size(2);
        
        return features;
    }
    
    // Decode tokens autoregressively
    std::vector<int> decode(const std::vector<float>& audio_features,
                           const std::vector<int>& initial_tokens,
                           const whisper::Tokenizer& tokenizer,
                           int max_tokens) {
        std::vector<int> tokens = initial_tokens;
        int eot_token = tokenizer.special_tokens().eot;
        
        // Create audio features tensor: shape (1, n_audio_ctx, n_audio_state)
        std::vector<int32_t> audio_shape = {1, n_audio_ctx_, n_audio_state_};
        auto audio_tensor = et::from_blob(
            const_cast<float*>(audio_features.data()),
            audio_shape,
            runtime::ScalarType::Float
        );
        
        for (int i = 0; i < max_tokens; ++i) {
            // Create token tensor: shape (1, seq_len)
            std::vector<int64_t> token_data(tokens.begin(), tokens.end());
            std::vector<int32_t> token_shape = {1, static_cast<int32_t>(tokens.size())};
            auto token_tensor = et::from_blob(
                token_data.data(),
                token_shape,
                runtime::ScalarType::Long
            );
            
            // Run decoder
            auto result = decoder_->forward({token_tensor, audio_tensor});
            if (!result.ok()) {
                std::cerr << "Decoder forward failed at step " << i << std::endl;
                break;
            }
            
            auto& outputs = result.get();
            if (outputs.empty() || !outputs[0].isTensor()) {
                std::cerr << "Invalid decoder output at step " << i << std::endl;
                break;
            }
            
            auto logits_tensor = outputs[0].toTensor();
            
            // Get logits for the last token: shape (1, seq_len, vocab_size) -> (vocab_size,)
            int seq_len = logits_tensor.size(1);
            int vocab_size = logits_tensor.size(2);
            const float* logits_data = logits_tensor.const_data_ptr<float>();
            
            // Find argmax of last position
            const float* last_logits = logits_data + (seq_len - 1) * vocab_size;
            int next_token = 0;
            float max_val = last_logits[0];
            for (int j = 1; j < vocab_size; ++j) {
                if (last_logits[j] > max_val) {
                    max_val = last_logits[j];
                    next_token = j;
                }
            }
            
            // Check for end of text
            if (next_token == eot_token) {
                break;
            }
            
            tokens.push_back(next_token);
        }
        
        return tokens;
    }
    
    bool is_loaded() const { return loaded_; }

private:
    std::unique_ptr<et::Module> encoder_;
    std::unique_ptr<et::Module> decoder_;
    bool loaded_ = false;
    int n_audio_ctx_ = 1500;
    int n_audio_state_ = 512;  // Will be updated after encode()
};

#else
// LibTorch inference implementation
class WhisperModelPT {
public:
    bool load(const std::string& encoder_path, const std::string& decoder_path) {
        try {
            std::cout << "Loading encoder: " << encoder_path << std::endl;
            encoder_ = torch::jit::load(encoder_path);
            encoder_.eval();
            
            std::cout << "Loading decoder: " << decoder_path << std::endl;
            decoder_ = torch::jit::load(decoder_path);
            decoder_.eval();
            
            loaded_ = true;
            return true;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return false;
        }
    }
    
    torch::Tensor encode(const torch::Tensor& mel) {
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs = {mel};
        return encoder_.forward(inputs).toTensor();
    }
    
    std::vector<int> decode(const torch::Tensor& audio_features,
                           const std::vector<int>& initial_tokens,
                           const whisper::Tokenizer& tokenizer,
                           int max_tokens) {
        torch::NoGradGuard no_grad;
        
        std::vector<int> tokens = initial_tokens;
        int eot_token = tokenizer.special_tokens().eot;
        
        for (int i = 0; i < max_tokens; ++i) {
            // Create token tensor
            auto token_tensor = torch::tensor(tokens, torch::kLong).unsqueeze(0);
            
            // Forward pass
            std::vector<torch::jit::IValue> inputs = {token_tensor, audio_features};
            auto logits = decoder_.forward(inputs).toTensor();
            
            // Get last token logits
            auto last_logits = logits.index({0, -1});
            
            // Greedy decoding: argmax
            int next_token = last_logits.argmax().item<int>();
            
            // Check for end of text
            if (next_token == eot_token) {
                break;
            }
            
            tokens.push_back(next_token);
        }
        
        return tokens;
    }
    
    bool is_loaded() const { return loaded_; }

private:
    torch::jit::script::Module encoder_;
    torch::jit::script::Module decoder_;
    bool loaded_ = false;
};
#endif

int main(int argc, char* argv[]) {
    std::cout << "Whisper Transcription Tool (Backend: " << INFERENCE_BACKEND << ")\n" << std::endl;
    
    // Parse arguments
    Config config = parse_args(argc, argv);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Load tokenizer
    whisper::Tokenizer tokenizer;
    if (!tokenizer.load(config.tokenizer_path)) {
        std::cerr << "Warning: Could not load tokenizer from: " << config.tokenizer_path << std::endl;
        std::cerr << "Using default token IDs" << std::endl;
    }
    
    // Determine model paths
    fs::path model_path(config.model_path);
    std::string encoder_path, decoder_path;
    
#ifdef USE_EXECUTORCH
    // ExecuTorch uses .pte files
    if (model_path.extension() == ".pte") {
        encoder_path = (model_path.parent_path() / (model_path.stem().string() + ".encoder.pte")).string();
        decoder_path = (model_path.parent_path() / (model_path.stem().string() + ".decoder.pte")).string();
    } else {
        // Assume base path provided
        encoder_path = config.model_path + ".encoder.pte";
        decoder_path = config.model_path + ".decoder.pte";
    }
#else
    // LibTorch uses .pt files
    if (model_path.extension() == ".pt") {
        encoder_path = (model_path.parent_path() / (model_path.stem().string() + ".encoder.pt")).string();
        decoder_path = (model_path.parent_path() / (model_path.stem().string() + ".decoder.pt")).string();
    } else if (model_path.extension() == ".pte") {
        // Try .pt versions if .pte requested but using LibTorch
        encoder_path = (model_path.parent_path() / (model_path.stem().string() + ".encoder.pt")).string();
        decoder_path = (model_path.parent_path() / (model_path.stem().string() + ".decoder.pt")).string();
    } else {
        encoder_path = config.model_path + ".encoder.pt";
        decoder_path = config.model_path + ".decoder.pt";
    }
#endif
    
    if (!fs::exists(encoder_path)) {
        std::cerr << "Error: Encoder model not found: " << encoder_path << std::endl;
        return 1;
    }
    if (!fs::exists(decoder_path)) {
        std::cerr << "Error: Decoder model not found: " << decoder_path << std::endl;
        return 1;
    }
    
    // Load model
#ifdef USE_EXECUTORCH
    WhisperModelET model;
#else
    WhisperModelPT model;
#endif
    
    if (!model.load(encoder_path, decoder_path)) {
        std::cerr << "Error: Failed to load model" << std::endl;
        return 1;
    }
    
    // Process audio - get all chunks
    whisper::AudioProcessor audio_processor;
    auto mel_chunks = audio_processor.preprocess_all_chunks(config.input_path);
    
    if (mel_chunks.empty()) {
        std::cerr << "Error: Failed to process audio" << std::endl;
        return 1;
    }
    
    float total_audio_duration = audio_processor.get_last_audio_duration();
    
    std::string full_transcription;
    
    // Process each chunk
    for (size_t chunk_idx = 0; chunk_idx < mel_chunks.size(); ++chunk_idx) {
        auto& mel = mel_chunks[chunk_idx];
        
        std::cout << "\n[Chunk " << (chunk_idx + 1) << "/" << mel_chunks.size() << "]" << std::endl;
        
        if (config.verbose) {
            std::cout << "Mel shape: (" << mel.n_mels << ", " << mel.n_frames << ")" << std::endl;
        }
        
#ifdef USE_EXECUTORCH
        // ExecuTorch path
        std::cout << "Encoding audio..." << std::endl;
        auto audio_features = model.encode(mel.data, mel.n_mels, mel.n_frames);
        
        if (audio_features.empty()) {
            std::cerr << "Error: Encoding failed" << std::endl;
            continue;
        }
        
        if (config.verbose) {
            std::cout << "Audio features size: " << audio_features.size() << std::endl;
        }
        
        // Get initial tokens
        auto initial_tokens = tokenizer.get_initial_tokens(
            config.language, config.task, config.timestamps
        );
        
        if (config.verbose) {
            std::cout << "Initial tokens: ";
            for (int t : initial_tokens) std::cout << t << " ";
            std::cout << std::endl;
        }
        
        // Decode
        std::cout << "Decoding..." << std::endl;
        auto output_tokens = model.decode(audio_features, initial_tokens, tokenizer, config.max_tokens);
#else
        // LibTorch path
        // Convert mel spectrogram to tensor (batch, n_mels, n_frames)
        auto mel_tensor = torch::from_blob(
            mel.data.data(),
            {1, mel.n_mels, mel.n_frames},
            torch::kFloat32
        ).clone();
        
        if (config.verbose) {
            std::cout << "Mel tensor shape: " << mel_tensor.sizes() << std::endl;
        }
        
        // Encode audio
        std::cout << "Encoding audio..." << std::endl;
        auto audio_features = model.encode(mel_tensor);
        
        if (config.verbose) {
            std::cout << "Audio features shape: " << audio_features.sizes() << std::endl;
        }
        
        // Get initial tokens
        auto initial_tokens = tokenizer.get_initial_tokens(
            config.language, config.task, config.timestamps
        );
        
        if (config.verbose) {
            std::cout << "Initial tokens: ";
            for (int t : initial_tokens) std::cout << t << " ";
            std::cout << std::endl;
        }
        
        // Decode
        std::cout << "Decoding..." << std::endl;
        auto output_tokens = model.decode(audio_features, initial_tokens, tokenizer, config.max_tokens);
#endif
        
        if (config.verbose) {
            std::cout << "Output tokens (" << output_tokens.size() << "): ";
            for (int t : output_tokens) std::cout << t << " ";
            std::cout << std::endl;
        }
        
        // Decode tokens to text
        std::string chunk_text = tokenizer.decode(output_tokens, true);
        
        std::cout << "Chunk transcription: " << chunk_text << std::endl;
        
        // Append to full transcription
        if (!full_transcription.empty() && !chunk_text.empty()) {
            full_transcription += " ";
        }
        full_transcription += chunk_text;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Prepare result
    TranscriptionResult result;
    result.text = full_transcription;
    result.language = config.language;
    result.duration_seconds = total_audio_duration;
    result.processing_time_seconds = duration.count() / 1000.0;
    
    // Output
    std::cout << "\n========================================" << std::endl;
    std::cout << "Transcription:" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << result.text << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Processing time: " << result.processing_time_seconds << "s" << std::endl;
    
    // Write JSON output
    write_json_output(config.output_path, result);
    
    return 0;
}
