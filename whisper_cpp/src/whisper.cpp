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

#include "audio.h"
#include "tokenizer.h"

// Try to use ExecuTorch, fallback to LibTorch
#ifdef USE_EXECUTORCH
    #include <executorch/runtime/executor/program.h>
    #include <executorch/runtime/executor/method.h>
    #include <executorch/runtime/core/evalue.h>
    #include <executorch/extension/data_loader/file_data_loader.h>
    #define INFERENCE_BACKEND "ExecuTorch"
#else
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
                     << "  whisper -i audio.wav -o result.json -m whisper.pte\n";
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
// ExecuTorch inference
class WhisperModelET {
public:
    bool load(const std::string& encoder_path, const std::string& decoder_path) {
        // Load encoder
        auto encoder_loader = executorch::extension::FileDataLoader::from(encoder_path.c_str());
        if (!encoder_loader.ok()) {
            std::cerr << "Failed to load encoder: " << encoder_path << std::endl;
            return false;
        }
        
        auto encoder_program = executorch::runtime::Program::load(&encoder_loader.get());
        if (!encoder_program.ok()) {
            std::cerr << "Failed to create encoder program" << std::endl;
            return false;
        }
        encoder_program_ = std::move(encoder_program.get());
        
        // Load decoder
        auto decoder_loader = executorch::extension::FileDataLoader::from(decoder_path.c_str());
        if (!decoder_loader.ok()) {
            std::cerr << "Failed to load decoder: " << decoder_path << std::endl;
            return false;
        }
        
        auto decoder_program = executorch::runtime::Program::load(&decoder_loader.get());
        if (!decoder_program.ok()) {
            std::cerr << "Failed to create decoder program" << std::endl;
            return false;
        }
        decoder_program_ = std::move(decoder_program.get());
        
        return true;
    }
    
    std::vector<float> encode(const std::vector<float>& mel, int n_mels, int n_frames) {
        // Implementation would use ExecuTorch API
        // This is a placeholder
        std::vector<float> features(1500 * 512);  // Example dimensions
        return features;
    }
    
    std::vector<int> decode(const std::vector<float>& features, 
                           const std::vector<int>& initial_tokens,
                           int max_tokens) {
        // Implementation would use ExecuTorch API
        // This is a placeholder
        return initial_tokens;
    }

private:
    std::unique_ptr<executorch::runtime::Program> encoder_program_;
    std::unique_ptr<executorch::runtime::Program> decoder_program_;
};

#else
// LibTorch inference
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
    
    if (model_path.extension() == ".pte") {
        // ExecuTorch format
        encoder_path = model_path.parent_path() / (model_path.stem().string() + ".encoder.pte");
        decoder_path = model_path.parent_path() / (model_path.stem().string() + ".decoder.pte");
    } else if (model_path.extension() == ".pt") {
        // TorchScript format
        encoder_path = model_path.parent_path() / (model_path.stem().string() + ".encoder.pt");
        decoder_path = model_path.parent_path() / (model_path.stem().string() + ".decoder.pt");
    } else {
        // Assume base path provided
        encoder_path = config.model_path + ".encoder.pt";
        decoder_path = config.model_path + ".decoder.pt";
    }
    
    // Check if encoder exists, try alternate paths
    if (!fs::exists(encoder_path)) {
        // Try with .pt extension
        encoder_path = model_path.parent_path() / (model_path.stem().string() + ".encoder.pt");
        decoder_path = model_path.parent_path() / (model_path.stem().string() + ".decoder.pt");
    }
    
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
    
#ifndef USE_EXECUTORCH
    std::string full_transcription;
    
    // Process each chunk
    for (size_t chunk_idx = 0; chunk_idx < mel_chunks.size(); ++chunk_idx) {
        auto& mel = mel_chunks[chunk_idx];
        
        std::cout << "\n[Chunk " << (chunk_idx + 1) << "/" << mel_chunks.size() << "]" << std::endl;
        
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
    
    std::string transcription = full_transcription;
#else
    // ExecuTorch path (placeholder)
    std::string transcription = "[ExecuTorch inference not fully implemented]";
#endif
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Prepare result
    TranscriptionResult result;
    result.text = transcription;
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
