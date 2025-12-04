/**
 * @file audio.h
 * @brief Audio processing utilities for Whisper
 * 
 * Handles WAV file loading and mel spectrogram computation.
 */

#ifndef WHISPER_AUDIO_H
#define WHISPER_AUDIO_H

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace whisper {

// Whisper audio parameters
constexpr int SAMPLE_RATE = 16000;
constexpr int N_FFT = 400;
constexpr int HOP_LENGTH = 160;
constexpr int N_MELS = 80;
constexpr int CHUNK_LENGTH = 30;  // seconds
constexpr int N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE;  // 480000 samples
constexpr int N_FRAMES = N_SAMPLES / HOP_LENGTH;  // 3000 frames

/**
 * @brief WAV file header structure
 */
struct WavHeader {
    char riff[4];           // "RIFF"
    uint32_t file_size;     // File size - 8
    char wave[4];           // "WAVE"
    char fmt[4];            // "fmt "
    uint32_t fmt_size;      // Format chunk size
    uint16_t audio_format;  // Audio format (1 = PCM)
    uint16_t num_channels;  // Number of channels
    uint32_t sample_rate;   // Sample rate
    uint32_t byte_rate;     // Bytes per second
    uint16_t block_align;   // Block alignment
    uint16_t bits_per_sample;  // Bits per sample
};

/**
 * @brief Audio data container
 */
struct AudioData {
    std::vector<float> samples;  // Audio samples normalized to [-1, 1]
    int sample_rate;
    int num_channels;
    float duration_seconds;
    
    bool is_valid() const { return !samples.empty() && sample_rate > 0; }
};

/**
 * @brief Mel spectrogram result
 */
struct MelSpectrogram {
    std::vector<float> data;  // Flattened mel spectrogram (n_mels * n_frames)
    int n_mels;
    int n_frames;
    
    bool is_valid() const { return !data.empty() && n_mels > 0 && n_frames > 0; }
    
    // Get value at (mel_bin, frame)
    float at(int mel, int frame) const {
        return data[mel * n_frames + frame];
    }
};

/**
 * @brief Audio processor class
 * 
 * Handles loading audio files and computing mel spectrograms
 * compatible with OpenAI Whisper.
 */
class AudioProcessor {
public:
    AudioProcessor();
    ~AudioProcessor();
    
    /**
     * @brief Load audio from WAV file
     * @param path Path to the WAV file
     * @return AudioData structure with loaded samples
     */
    AudioData load_wav(const std::string& path);
    
    /**
     * @brief Resample audio to target sample rate
     * @param audio Input audio data
     * @param target_rate Target sample rate (default: 16000)
     * @return Resampled audio data
     */
    AudioData resample(const AudioData& audio, int target_rate = SAMPLE_RATE);
    
    /**
     * @brief Convert stereo to mono
     * @param audio Input audio data
     * @return Mono audio data
     */
    AudioData to_mono(const AudioData& audio);
    
    /**
     * @brief Pad or trim audio to exact length
     * @param audio Input audio data
     * @param target_samples Target number of samples
     * @return Padded/trimmed audio data
     */
    AudioData pad_or_trim(const AudioData& audio, int target_samples = N_SAMPLES);
    
    /**
     * @brief Compute mel spectrogram from audio
     * @param audio Input audio data (should be 16kHz mono)
     * @return MelSpectrogram structure
     */
    MelSpectrogram compute_mel_spectrogram(const AudioData& audio);
    
    /**
     * @brief Full preprocessing pipeline: load, resample, mono, pad, mel
     * @param path Path to audio file
     * @return MelSpectrogram ready for Whisper
     */
    MelSpectrogram preprocess(const std::string& path);
    
    /**
     * @brief Load and preprocess audio, returning all chunks
     * @param path Path to audio file
     * @param chunk_overlap_seconds Overlap between chunks in seconds (for smoother transitions)
     * @return Vector of MelSpectrograms, one per chunk
     */
    std::vector<MelSpectrogram> preprocess_all_chunks(const std::string& path, float chunk_overlap_seconds = 0.0f);
    
    /**
     * @brief Get total duration of loaded audio
     * @return Duration in seconds, or 0 if no audio loaded
     */
    float get_last_audio_duration() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Apply STFT (Short-Time Fourier Transform)
 * @param samples Audio samples
 * @param n_fft FFT size
 * @param hop_length Hop length between frames
 * @return Complex STFT result (magnitudes)
 */
std::vector<std::vector<float>> stft(
    const std::vector<float>& samples,
    int n_fft = N_FFT,
    int hop_length = HOP_LENGTH
);

/**
 * @brief Create mel filterbank
 * @param n_mels Number of mel bins
 * @param n_fft FFT size
 * @param sample_rate Sample rate
 * @return Mel filterbank matrix
 */
std::vector<std::vector<float>> create_mel_filterbank(
    int n_mels = N_MELS,
    int n_fft = N_FFT,
    int sample_rate = SAMPLE_RATE
);

/**
 * @brief Convert frequency to mel scale
 */
float hz_to_mel(float hz);

/**
 * @brief Convert mel scale to frequency
 */
float mel_to_hz(float mel);

}  // namespace whisper

#endif  // WHISPER_AUDIO_H
