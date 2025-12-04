/**
 * @file tokenizer.cpp
 * @brief Whisper tokenizer implementation
 */

#include "tokenizer.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <regex>

namespace whisper {

// Simple JSON parser (minimal implementation for tokenizer.json)
bool JsonParser::parse_tokenizer_json(
    const std::string& path,
    std::unordered_map<int, std::string>& vocab,
    SpecialTokens& special,
    std::unordered_map<std::string, int>& language_tokens
) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open tokenizer file: " << path << std::endl;
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();
    
    // Parse vocab section
    // Format: "vocab": {"token_id": "token_string", ...}
    size_t vocab_start = json.find("\"vocab\"");
    if (vocab_start != std::string::npos) {
        size_t brace_start = json.find('{', vocab_start);
        if (brace_start != std::string::npos) {
            int brace_count = 1;
            size_t pos = brace_start + 1;
            
            while (brace_count > 0 && pos < json.length()) {
                // Find key (token_id as string)
                size_t key_start = json.find('"', pos);
                if (key_start == std::string::npos) break;
                
                size_t key_end = json.find('"', key_start + 1);
                if (key_end == std::string::npos) break;
                
                std::string key_str = json.substr(key_start + 1, key_end - key_start - 1);
                
                // Skip to colon
                pos = json.find(':', key_end);
                if (pos == std::string::npos) break;
                pos++;
                
                // Skip whitespace
                while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')) {
                    pos++;
                }
                
                // Find value (string)
                if (json[pos] == '"') {
                    size_t val_start = pos;
                    pos++; // Skip opening quote
                    
                    std::string value;
                    while (pos < json.length() && json[pos] != '"') {
                        if (json[pos] == '\\' && pos + 1 < json.length()) {
                            pos++;
                            switch (json[pos]) {
                                case 'n': value += '\n'; break;
                                case 'r': value += '\r'; break;
                                case 't': value += '\t'; break;
                                case '\\': value += '\\'; break;
                                case '"': value += '"'; break;
                                case 'u': {
                                    // Unicode escape
                                    if (pos + 4 < json.length()) {
                                        std::string hex = json.substr(pos + 1, 4);
                                        int codepoint = std::stoi(hex, nullptr, 16);
                                        // Simple UTF-8 encoding for BMP
                                        if (codepoint < 0x80) {
                                            value += static_cast<char>(codepoint);
                                        } else if (codepoint < 0x800) {
                                            value += static_cast<char>(0xC0 | (codepoint >> 6));
                                            value += static_cast<char>(0x80 | (codepoint & 0x3F));
                                        } else {
                                            value += static_cast<char>(0xE0 | (codepoint >> 12));
                                            value += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                                            value += static_cast<char>(0x80 | (codepoint & 0x3F));
                                        }
                                        pos += 4;
                                    }
                                    break;
                                }
                                default: value += json[pos]; break;
                            }
                        } else {
                            value += json[pos];
                        }
                        pos++;
                    }
                    pos++; // Skip closing quote
                    
                    // Parse key as int
                    try {
                        int token_id = std::stoi(key_str);
                        vocab[token_id] = value;
                    } catch (...) {
                        // Skip invalid keys
                    }
                }
                
                // Skip to comma or end
                while (pos < json.length() && json[pos] != ',' && json[pos] != '}') {
                    pos++;
                }
                
                if (json[pos] == '}') {
                    brace_count--;
                } else if (json[pos] == ',') {
                    pos++;
                }
            }
        }
    }
    
    // Parse special_tokens section
    auto parse_special_token = [&](const std::string& name, int& target) {
        std::string pattern = "\"" + name + "\":";
        size_t pos = json.find(pattern);
        if (pos != std::string::npos) {
            pos += pattern.length();
            while (pos < json.length() && !std::isdigit(json[pos]) && json[pos] != '-') {
                pos++;
            }
            size_t end = pos;
            while (end < json.length() && (std::isdigit(json[end]) || json[end] == '-')) {
                end++;
            }
            try {
                target = std::stoi(json.substr(pos, end - pos));
            } catch (...) {}
        }
    };
    
    parse_special_token("sot", special.sot);
    parse_special_token("eot", special.eot);
    parse_special_token("transcribe", special.transcribe);
    parse_special_token("translate", special.translate);
    parse_special_token("sot_prev", special.sot_prev);
    parse_special_token("sot_lm", special.sot_lm);
    parse_special_token("no_speech", special.no_speech);
    parse_special_token("no_timestamps", special.no_timestamps);
    
    // Parse language_tokens section
    size_t lang_start = json.find("\"language_tokens\"");
    if (lang_start != std::string::npos) {
        size_t brace_start = json.find('{', lang_start);
        if (brace_start != std::string::npos) {
            size_t brace_end = json.find('}', brace_start);
            std::string lang_section = json.substr(brace_start, brace_end - brace_start + 1);
            
            // Parse key-value pairs
            size_t pos = 1;
            while (pos < lang_section.length()) {
                size_t key_start = lang_section.find('"', pos);
                if (key_start == std::string::npos) break;
                
                size_t key_end = lang_section.find('"', key_start + 1);
                if (key_end == std::string::npos) break;
                
                std::string lang = lang_section.substr(key_start + 1, key_end - key_start - 1);
                
                pos = lang_section.find(':', key_end);
                if (pos == std::string::npos) break;
                pos++;
                
                while (pos < lang_section.length() && !std::isdigit(lang_section[pos])) {
                    pos++;
                }
                
                size_t num_end = pos;
                while (num_end < lang_section.length() && std::isdigit(lang_section[num_end])) {
                    num_end++;
                }
                
                try {
                    int token_id = std::stoi(lang_section.substr(pos, num_end - pos));
                    language_tokens[lang] = token_id;
                } catch (...) {}
                
                pos = num_end;
            }
        }
    }
    
    return !vocab.empty();
}

// Tokenizer implementation
Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;

bool Tokenizer::load(const std::string& path) {
    loaded_ = JsonParser::parse_tokenizer_json(path, vocab_, special_tokens_, language_tokens_);
    
    if (loaded_) {
        std::cout << "Tokenizer loaded: " << vocab_.size() << " tokens" << std::endl;
    }
    
    return loaded_;
}

std::string Tokenizer::decode_token(int token_id) const {
    auto it = vocab_.find(token_id);
    if (it != vocab_.end()) {
        return it->second;
    }
    return "";
}

std::string Tokenizer::decode(const std::vector<int>& token_ids, bool skip_special) const {
    std::string result;
    
    for (int token_id : token_ids) {
        // Skip special tokens if requested
        if (skip_special && is_special_token(token_id)) {
            continue;
        }
        
        // Skip timestamp tokens
        if (is_timestamp_token(token_id)) {
            continue;
        }
        
        result += decode_token(token_id);
    }
    
    // Clean up BPE artifacts (GPT-2 style tokenizer uses Ġ for space)
    // Replace Ġ (U+0120) with space
    std::string cleaned;
    for (size_t i = 0; i < result.length(); ++i) {
        if (i + 1 < result.length() && 
            static_cast<unsigned char>(result[i]) == 0xC4 && 
            static_cast<unsigned char>(result[i + 1]) == 0xA0) {
            cleaned += ' ';
            i++;
        } else {
            cleaned += result[i];
        }
    }
    
    // Trim leading/trailing whitespace
    size_t start = cleaned.find_first_not_of(" \t\n\r");
    size_t end = cleaned.find_last_not_of(" \t\n\r");
    if (start != std::string::npos && end != std::string::npos) {
        cleaned = cleaned.substr(start, end - start + 1);
    }
    
    return cleaned;
}

size_t Tokenizer::vocab_size() const {
    return vocab_.size();
}

bool Tokenizer::is_special_token(int token_id) const {
    return token_id == special_tokens_.sot ||
           token_id == special_tokens_.eot ||
           token_id == special_tokens_.transcribe ||
           token_id == special_tokens_.translate ||
           token_id == special_tokens_.sot_prev ||
           token_id == special_tokens_.sot_lm ||
           token_id == special_tokens_.no_speech ||
           token_id == special_tokens_.no_timestamps ||
           (token_id >= 50259 && token_id <= 50357);  // Language tokens
}

bool Tokenizer::is_timestamp_token(int token_id) const {
    // Timestamp tokens are >= 50364
    return token_id >= 50364;
}

std::vector<int> Tokenizer::get_initial_tokens(
    const std::string& language,
    const std::string& task,
    bool timestamps
) const {
    std::vector<int> tokens;
    
    // Start of transcript
    tokens.push_back(special_tokens_.sot);
    
    // Language token
    auto it = language_tokens_.find(language);
    if (it != language_tokens_.end()) {
        tokens.push_back(it->second);
    } else {
        tokens.push_back(special_tokens_.en);  // Default to English
    }
    
    // Task token
    if (task == "translate") {
        tokens.push_back(special_tokens_.translate);
    } else {
        tokens.push_back(special_tokens_.transcribe);
    }
    
    // Timestamps
    if (!timestamps) {
        tokens.push_back(special_tokens_.no_timestamps);
    }
    
    return tokens;
}

}  // namespace whisper
