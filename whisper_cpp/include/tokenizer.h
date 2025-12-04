/**
 * @file tokenizer.h
 * @brief Whisper tokenizer for encoding/decoding text
 * 
 * Handles loading vocabulary from JSON and decoding token IDs to text.
 */

#ifndef WHISPER_TOKENIZER_H
#define WHISPER_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace whisper {

/**
 * @brief Special token IDs for Whisper
 */
struct SpecialTokens {
    int sot = 50258;           // <|startoftranscript|>
    int eot = 50257;           // <|endoftext|>
    int transcribe = 50359;    // <|transcribe|>
    int translate = 50358;     // <|translate|>
    int sot_prev = 50361;      // <|startofprev|>
    int sot_lm = 50360;        // <|startlm|>
    int no_speech = 50362;     // <|nospeech|>
    int no_timestamps = 50363; // <|notimestamps|>
    
    // Language tokens (subset)
    int en = 50259;  // English
};

/**
 * @brief Whisper tokenizer class
 * 
 * Loads vocabulary from JSON and provides encoding/decoding.
 */
class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();
    
    /**
     * @brief Load tokenizer from JSON file
     * @param path Path to tokenizer.json
     * @return true if loaded successfully
     */
    bool load(const std::string& path);
    
    /**
     * @brief Decode single token ID to string
     * @param token_id Token ID
     * @return Decoded string
     */
    std::string decode_token(int token_id) const;
    
    /**
     * @brief Decode sequence of token IDs to string
     * @param token_ids Vector of token IDs
     * @param skip_special Skip special tokens (default: true)
     * @return Decoded text
     */
    std::string decode(const std::vector<int>& token_ids, bool skip_special = true) const;
    
    /**
     * @brief Get vocabulary size
     */
    size_t vocab_size() const;
    
    /**
     * @brief Check if token is a special token
     */
    bool is_special_token(int token_id) const;
    
    /**
     * @brief Check if token is a timestamp token
     */
    bool is_timestamp_token(int token_id) const;
    
    /**
     * @brief Get special tokens
     */
    const SpecialTokens& special_tokens() const { return special_tokens_; }
    
    /**
     * @brief Get initial tokens for transcription
     * @param language Language code (e.g., "en")
     * @param task Task: "transcribe" or "translate"
     * @param timestamps Include timestamps
     * @return Vector of initial token IDs
     */
    std::vector<int> get_initial_tokens(
        const std::string& language = "en",
        const std::string& task = "transcribe",
        bool timestamps = false
    ) const;

private:
    std::unordered_map<int, std::string> vocab_;
    std::unordered_map<std::string, int> language_tokens_;
    SpecialTokens special_tokens_;
    bool loaded_ = false;
};

/**
 * @brief Simple JSON parser for tokenizer file
 * 
 * Parses the tokenizer.json file exported from Python.
 */
class JsonParser {
public:
    /**
     * @brief Parse JSON file
     * @param path Path to JSON file
     * @param vocab Output vocabulary map
     * @param special Output special tokens
     * @param language_tokens Output language token map
     * @return true if parsed successfully
     */
    static bool parse_tokenizer_json(
        const std::string& path,
        std::unordered_map<int, std::string>& vocab,
        SpecialTokens& special,
        std::unordered_map<std::string, int>& language_tokens
    );
};

}  // namespace whisper

#endif  // WHISPER_TOKENIZER_H
