/**
 * @file audio.cpp
 * @brief Audio processing implementation for Whisper
 */

#include "audio.h"

#include <fstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace whisper {

// Mel scale conversion
float hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

// Hann window
std::vector<float> hann_window(int size) {
    std::vector<float> window(size);
    for (int i = 0; i < size; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (size - 1)));
    }
    return window;
}

// Simple DFT for small sizes (used when FFT not available)
void dft(const std::vector<float>& input, std::vector<float>& real, std::vector<float>& imag) {
    int N = input.size();
    real.resize(N);
    imag.resize(N);
    
    for (int k = 0; k < N; ++k) {
        real[k] = 0.0f;
        imag[k] = 0.0f;
        for (int n = 0; n < N; ++n) {
            float angle = 2.0f * M_PI * k * n / N;
            real[k] += input[n] * std::cos(angle);
            imag[k] -= input[n] * std::sin(angle);
        }
    }
}

// Cooley-Tukey FFT
void fft(std::vector<float>& real, std::vector<float>& imag) {
    int N = real.size();
    if (N <= 1) return;
    
    // Bit reversal
    int bits = 0;
    while ((1 << bits) < N) bits++;
    
    for (int i = 0; i < N; ++i) {
        int j = 0;
        for (int k = 0; k < bits; ++k) {
            if (i & (1 << k)) j |= (1 << (bits - 1 - k));
        }
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }
    
    // FFT
    for (int len = 2; len <= N; len *= 2) {
        float angle = -2.0f * M_PI / len;
        float wpr = std::cos(angle);
        float wpi = std::sin(angle);
        
        for (int i = 0; i < N; i += len) {
            float wr = 1.0f, wi = 0.0f;
            for (int j = 0; j < len / 2; ++j) {
                int u = i + j;
                int v = i + j + len / 2;
                
                float tr = wr * real[v] - wi * imag[v];
                float ti = wr * imag[v] + wi * real[v];
                
                real[v] = real[u] - tr;
                imag[v] = imag[u] - ti;
                real[u] += tr;
                imag[u] += ti;
                
                float temp = wr;
                wr = wr * wpr - wi * wpi;
                wi = temp * wpi + wi * wpr;
            }
        }
    }
}

// Helper to get next power of 2
int next_power_of_2(int n) {
    int p = 1;
    while (p < n) p *= 2;
    return p;
}

// STFT implementation
std::vector<std::vector<float>> stft(
    const std::vector<float>& samples,
    int n_fft,
    int hop_length
) {
    auto window = hann_window(n_fft);
    int n_frames = (samples.size() - n_fft) / hop_length + 1;
    int n_freqs = n_fft / 2 + 1;
    
    // FFT requires power-of-2 size, so pad if needed
    int fft_size = next_power_of_2(n_fft);
    
    std::vector<std::vector<float>> magnitudes(n_freqs, std::vector<float>(n_frames));
    
    std::vector<float> real(fft_size);
    std::vector<float> imag(fft_size);
    
    for (int frame = 0; frame < n_frames; ++frame) {
        int start = frame * hop_length;
        
        // Apply window and zero-pad to fft_size
        for (int i = 0; i < fft_size; ++i) {
            if (i < n_fft && start + i < static_cast<int>(samples.size())) {
                real[i] = samples[start + i] * window[i];
            } else {
                real[i] = 0.0f;
            }
            imag[i] = 0.0f;
        }
        
        // FFT
        fft(real, imag);
        
        // Magnitude (only need first n_freqs bins)
        for (int freq = 0; freq < n_freqs; ++freq) {
            magnitudes[freq][frame] = std::sqrt(real[freq] * real[freq] + imag[freq] * imag[freq]);
        }
    }
    
    return magnitudes;
}

// Create mel filterbank
std::vector<std::vector<float>> create_mel_filterbank(
    int n_mels,
    int n_fft,
    int sample_rate
) {
    int n_freqs = n_fft / 2 + 1;
    
    // Frequency points
    float f_min = 0.0f;
    float f_max = sample_rate / 2.0f;
    
    float mel_min = hz_to_mel(f_min);
    float mel_max = hz_to_mel(f_max);
    
    // Mel points
    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
    }
    
    // Convert to Hz
    std::vector<float> hz_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }
    
    // Convert to FFT bins
    std::vector<int> bin_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        bin_points[i] = static_cast<int>(std::floor((n_fft + 1) * hz_points[i] / sample_rate));
    }
    
    // Create filterbank
    std::vector<std::vector<float>> filterbank(n_mels, std::vector<float>(n_freqs, 0.0f));
    
    for (int m = 0; m < n_mels; ++m) {
        int f_start = bin_points[m];
        int f_center = bin_points[m + 1];
        int f_end = bin_points[m + 2];
        
        // Rising slope
        for (int f = f_start; f < f_center && f < n_freqs; ++f) {
            if (f_center != f_start) {
                filterbank[m][f] = static_cast<float>(f - f_start) / (f_center - f_start);
            }
        }
        
        // Falling slope
        for (int f = f_center; f < f_end && f < n_freqs; ++f) {
            if (f_end != f_center) {
                filterbank[m][f] = static_cast<float>(f_end - f) / (f_end - f_center);
            }
        }
    }
    
    return filterbank;
}

// AudioProcessor implementation
class AudioProcessor::Impl {
public:
    std::vector<std::vector<float>> mel_filterbank_;
    bool filterbank_initialized_ = false;
    float last_audio_duration_ = 0.0f;
    
    void init_filterbank() {
        if (!filterbank_initialized_) {
            mel_filterbank_ = create_mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE);
            filterbank_initialized_ = true;
        }
    }
};

AudioProcessor::AudioProcessor() : impl_(std::make_unique<Impl>()) {}
AudioProcessor::~AudioProcessor() = default;

AudioData AudioProcessor::load_wav(const std::string& path) {
    AudioData audio;
    
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << path << std::endl;
        return audio;
    }
    
    // Read RIFF header
    char riff[4];
    file.read(riff, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0) {
        std::cerr << "Error: Not a valid WAV file (missing RIFF)" << std::endl;
        return audio;
    }
    
    uint32_t file_size;
    file.read(reinterpret_cast<char*>(&file_size), 4);
    
    char wave[4];
    file.read(wave, 4);
    if (std::strncmp(wave, "WAVE", 4) != 0) {
        std::cerr << "Error: Not a valid WAV file (missing WAVE)" << std::endl;
        return audio;
    }
    
    // Parse chunks
    uint16_t audio_format = 1;
    uint16_t num_channels = 1;
    uint32_t sample_rate = 16000;
    uint16_t bits_per_sample = 16;
    
    while (file.good()) {
        char chunk_id[4];
        uint32_t chunk_size;
        
        file.read(chunk_id, 4);
        if (!file.good()) break;
        
        file.read(reinterpret_cast<char*>(&chunk_size), 4);
        if (!file.good()) break;
        
        if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
            file.read(reinterpret_cast<char*>(&audio_format), 2);
            file.read(reinterpret_cast<char*>(&num_channels), 2);
            file.read(reinterpret_cast<char*>(&sample_rate), 4);
            
            uint32_t byte_rate;
            file.read(reinterpret_cast<char*>(&byte_rate), 4);
            
            uint16_t block_align;
            file.read(reinterpret_cast<char*>(&block_align), 2);
            
            file.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            
            // Skip extra format bytes if any
            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, std::ios::cur);
            }
        } else if (std::strncmp(chunk_id, "data", 4) == 0) {
            // Read audio data
            int bytes_per_sample = bits_per_sample / 8;
            int num_samples = chunk_size / (bytes_per_sample * num_channels);
            
            audio.samples.reserve(num_samples * num_channels);
            
            if (bits_per_sample == 16) {
                std::vector<int16_t> raw_samples(num_samples * num_channels);
                file.read(reinterpret_cast<char*>(raw_samples.data()), chunk_size);
                
                for (int16_t sample : raw_samples) {
                    audio.samples.push_back(static_cast<float>(sample) / 32768.0f);
                }
            } else if (bits_per_sample == 32) {
                // 32-bit float
                audio.samples.resize(num_samples * num_channels);
                file.read(reinterpret_cast<char*>(audio.samples.data()), chunk_size);
            } else if (bits_per_sample == 8) {
                std::vector<uint8_t> raw_samples(num_samples * num_channels);
                file.read(reinterpret_cast<char*>(raw_samples.data()), chunk_size);
                
                for (uint8_t sample : raw_samples) {
                    audio.samples.push_back((static_cast<float>(sample) - 128.0f) / 128.0f);
                }
            }
            
            break;  // Got data, done
        } else {
            // Skip unknown chunk
            file.seekg(chunk_size, std::ios::cur);
        }
    }
    
    audio.sample_rate = sample_rate;
    audio.num_channels = num_channels;
    audio.duration_seconds = static_cast<float>(audio.samples.size()) / 
                            (sample_rate * num_channels);
    
    return audio;
}

AudioData AudioProcessor::resample(const AudioData& audio, int target_rate) {
    if (audio.sample_rate == target_rate) {
        return audio;
    }
    
    AudioData resampled;
    resampled.sample_rate = target_rate;
    resampled.num_channels = audio.num_channels;
    
    float ratio = static_cast<float>(target_rate) / audio.sample_rate;
    int new_length = static_cast<int>(audio.samples.size() * ratio);
    
    resampled.samples.resize(new_length);
    
    // Linear interpolation resampling
    for (int i = 0; i < new_length; ++i) {
        float src_idx = i / ratio;
        int idx0 = static_cast<int>(src_idx);
        int idx1 = std::min(idx0 + 1, static_cast<int>(audio.samples.size()) - 1);
        float frac = src_idx - idx0;
        
        resampled.samples[i] = audio.samples[idx0] * (1.0f - frac) + 
                               audio.samples[idx1] * frac;
    }
    
    resampled.duration_seconds = static_cast<float>(new_length) / target_rate;
    
    return resampled;
}

AudioData AudioProcessor::to_mono(const AudioData& audio) {
    if (audio.num_channels == 1) {
        return audio;
    }
    
    AudioData mono;
    mono.sample_rate = audio.sample_rate;
    mono.num_channels = 1;
    
    int num_samples = audio.samples.size() / audio.num_channels;
    mono.samples.resize(num_samples);
    
    for (int i = 0; i < num_samples; ++i) {
        float sum = 0.0f;
        for (int ch = 0; ch < audio.num_channels; ++ch) {
            sum += audio.samples[i * audio.num_channels + ch];
        }
        mono.samples[i] = sum / audio.num_channels;
    }
    
    mono.duration_seconds = static_cast<float>(num_samples) / audio.sample_rate;
    
    return mono;
}

AudioData AudioProcessor::pad_or_trim(const AudioData& audio, int target_samples) {
    AudioData result;
    result.sample_rate = audio.sample_rate;
    result.num_channels = audio.num_channels;
    result.samples.resize(target_samples, 0.0f);
    
    int copy_length = std::min(static_cast<int>(audio.samples.size()), target_samples);
    std::copy(audio.samples.begin(), audio.samples.begin() + copy_length, 
              result.samples.begin());
    
    result.duration_seconds = static_cast<float>(target_samples) / audio.sample_rate;
    
    return result;
}

MelSpectrogram AudioProcessor::compute_mel_spectrogram(const AudioData& audio) {
    impl_->init_filterbank();
    
    MelSpectrogram mel;
    mel.n_mels = N_MELS;
    
    // Compute STFT
    auto magnitudes = stft(audio.samples, N_FFT, HOP_LENGTH);
    mel.n_frames = magnitudes[0].size();
    
    // Apply mel filterbank
    mel.data.resize(mel.n_mels * mel.n_frames);
    
    for (int m = 0; m < mel.n_mels; ++m) {
        for (int f = 0; f < mel.n_frames; ++f) {
            float sum = 0.0f;
            for (int freq = 0; freq < static_cast<int>(magnitudes.size()); ++freq) {
                sum += impl_->mel_filterbank_[m][freq] * magnitudes[freq][f] * magnitudes[freq][f];
            }
            // Log mel spectrogram
            mel.data[m * mel.n_frames + f] = std::log10(std::max(sum, 1e-10f));
        }
    }
    
    // Normalize (Whisper-style: clamp to -8, then scale to [0,1] range approx)
    float max_val = *std::max_element(mel.data.begin(), mel.data.end());
    for (float& val : mel.data) {
        val = std::max(val, max_val - 8.0f);
        val = (val + 4.0f) / 4.0f;  // Approximate normalization
    }
    
    return mel;
}

MelSpectrogram AudioProcessor::preprocess(const std::string& path) {
    std::cout << "Loading audio: " << path << std::endl;
    auto audio = load_wav(path);
    
    if (!audio.is_valid()) {
        std::cerr << "Failed to load audio" << std::endl;
        return MelSpectrogram{};
    }
    
    impl_->last_audio_duration_ = audio.duration_seconds;
    
    std::cout << "  Sample rate: " << audio.sample_rate << " Hz" << std::endl;
    std::cout << "  Channels: " << audio.num_channels << std::endl;
    std::cout << "  Duration: " << audio.duration_seconds << " seconds" << std::endl;
    
    // Convert to mono
    if (audio.num_channels > 1) {
        std::cout << "  Converting to mono..." << std::endl;
        audio = to_mono(audio);
    }
    
    // Resample to 16kHz
    if (audio.sample_rate != SAMPLE_RATE) {
        std::cout << "  Resampling to " << SAMPLE_RATE << " Hz..." << std::endl;
        audio = resample(audio, SAMPLE_RATE);
    }
    
    // Pad or trim to 30 seconds
    std::cout << "  Padding/trimming to " << CHUNK_LENGTH << " seconds..." << std::endl;
    audio = pad_or_trim(audio, N_SAMPLES);
    
    // Compute mel spectrogram
    std::cout << "  Computing mel spectrogram..." << std::endl;
    auto mel = compute_mel_spectrogram(audio);
    
    std::cout << "  Mel spectrogram: " << mel.n_mels << " x " << mel.n_frames << std::endl;
    
    return mel;
}

std::vector<MelSpectrogram> AudioProcessor::preprocess_all_chunks(const std::string& path, float chunk_overlap_seconds) {
    std::cout << "Loading audio: " << path << std::endl;
    auto audio = load_wav(path);
    
    if (!audio.is_valid()) {
        std::cerr << "Failed to load audio" << std::endl;
        return {};
    }
    
    impl_->last_audio_duration_ = audio.duration_seconds;
    
    std::cout << "  Sample rate: " << audio.sample_rate << " Hz" << std::endl;
    std::cout << "  Channels: " << audio.num_channels << std::endl;
    std::cout << "  Duration: " << audio.duration_seconds << " seconds" << std::endl;
    
    // Convert to mono
    if (audio.num_channels > 1) {
        std::cout << "  Converting to mono..." << std::endl;
        audio = to_mono(audio);
    }
    
    // Resample to 16kHz
    if (audio.sample_rate != SAMPLE_RATE) {
        std::cout << "  Resampling to " << SAMPLE_RATE << " Hz..." << std::endl;
        audio = resample(audio, SAMPLE_RATE);
    }
    
    // Calculate chunk parameters
    int chunk_samples = N_SAMPLES;  // 30 seconds worth of samples
    int overlap_samples = static_cast<int>(chunk_overlap_seconds * SAMPLE_RATE);
    int step_samples = chunk_samples - overlap_samples;
    
    int total_samples = audio.samples.size();
    int num_chunks = (total_samples + step_samples - 1) / step_samples;
    if (num_chunks == 0) num_chunks = 1;
    
    std::cout << "  Processing " << num_chunks << " chunk(s) of " << CHUNK_LENGTH << " seconds each..." << std::endl;
    
    std::vector<MelSpectrogram> chunks;
    chunks.reserve(num_chunks);
    
    for (int i = 0; i < num_chunks; ++i) {
        int start_sample = i * step_samples;
        int end_sample = std::min(start_sample + chunk_samples, total_samples);
        
        // Extract chunk
        AudioData chunk;
        chunk.sample_rate = audio.sample_rate;
        chunk.num_channels = 1;
        chunk.samples.assign(
            audio.samples.begin() + start_sample,
            audio.samples.begin() + end_sample
        );
        
        // Pad if needed (for the last chunk)
        if (static_cast<int>(chunk.samples.size()) < chunk_samples) {
            chunk.samples.resize(chunk_samples, 0.0f);
        }
        
        chunk.duration_seconds = static_cast<float>(chunk.samples.size()) / SAMPLE_RATE;
        
        // Compute mel spectrogram for this chunk
        auto mel = compute_mel_spectrogram(chunk);
        
        float chunk_start_time = static_cast<float>(start_sample) / SAMPLE_RATE;
        float chunk_end_time = static_cast<float>(end_sample) / SAMPLE_RATE;
        std::cout << "  Chunk " << (i + 1) << "/" << num_chunks 
                  << " [" << chunk_start_time << "s - " << chunk_end_time << "s]"
                  << " mel: " << mel.n_mels << " x " << mel.n_frames << std::endl;
        
        chunks.push_back(std::move(mel));
    }
    
    return chunks;
}

float AudioProcessor::get_last_audio_duration() const {
    return impl_->last_audio_duration_;
}

}  // namespace whisper
