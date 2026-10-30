#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "third_party/json.hpp"

static bool utf8_next_codepoint(
    const std::string& s,
    size_t& i,
    char32_t& cp
) {
    if (i >= s.size()) {
        return false;
    }

    unsigned char c = static_cast<unsigned char>(s[i]);

    int len = 1;

    if (c >= 0xF0) len = 4;
    else if (c >= 0xE0) len = 3;
    else if (c >= 0xC0) len = 2;
    else len = 1;

    if (i + len > s.size()) {
        return false;
    }

    char32_t value = 0;

    if (len == 4) value = c & 0x07;
    else if (len == 3) value = c & 0x0F;
    else if (len == 2) value = c & 0x1F;
    else value = c;

    for (int k = 1; k < len; ++k) {
        unsigned char next = static_cast<unsigned char>(s[i + k]);

        if ((next & 0xC0) != 0x80) {
            return false;
        }

        value = (value << 6) | (next & 0x3F);
    }

    cp = value;
    i += len;

    return true;
}

static void utf8_append_codepoint(
    std::string& out,
    char32_t cp
) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

struct Tokenizer {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<std::string, int> merge_rank;

    std::unordered_map<char32_t, unsigned char> byte_decoder;
    std::unordered_map<unsigned char, std::string> byte_encoder;

    std::vector<std::string> special_tokens;

    bool byte_level = false;
    size_t vocab_size = 0;

    static bool json_contains_byte_level(const nlohmann::json& j) {
        if (j.is_object()) {
            if (j.contains("type") && j["type"].is_string()) {
                if (j["type"].get<std::string>() == "ByteLevel") {
                    return true;
                }
            }

            for (auto it = j.begin(); it != j.end(); ++it) {
                if (json_contains_byte_level(it.value())) {
                    return true;
                }
            }
        } else if (j.is_array()) {
            for (const auto& item : j) {
                if (json_contains_byte_level(item)) {
                    return true;
                }
            }
        }

        return false;
    }

    void build_byte_maps() {
        std::vector<bool> present(256, false);

        for (int b = 33; b <= 126; ++b) present[b] = true;
        for (int b = 161; b <= 172; ++b) present[b] = true;
        for (int b = 174; b <= 255; ++b) present[b] = true;

        std::vector<int> bs;
        std::vector<int> cs;

        for (int b = 0; b < 256; ++b) {
            if (present[b]) {
                bs.push_back(b);
                cs.push_back(b);
            }
        }

        int n = 0;

        for (int b = 0; b < 256; ++b) {
            if (!present[b]) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }

        byte_decoder.clear();
        byte_encoder.clear();

        for (size_t i = 0; i < bs.size(); ++i) {
            char32_t cp = static_cast<char32_t>(cs[i]);
            unsigned char byte = static_cast<unsigned char>(bs[i]);

            byte_decoder[cp] = byte;

            std::string s;
            utf8_append_codepoint(s, cp);
            byte_encoder[byte] = s;
        }
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);

        if (!f) {
            throw std::runtime_error("cannot open tokenizer file: " + path);
        }

        nlohmann::json j;
        f >> j;

        if (!j.contains("model") || !j["model"].contains("vocab")) {
            throw std::runtime_error("tokenizer.json missing model.vocab");
        }

        const auto& vocab = j["model"]["vocab"];

        size_t max_id = 0;

        for (auto it = vocab.begin(); it != vocab.end(); ++it) {
            size_t id = it.value().get<size_t>();
            max_id = std::max(max_id, id + 1);
        }

        if (j.contains("added_tokens")) {
            for (const auto& token : j["added_tokens"]) {
                if (token.contains("id")) {
                    size_t id = token["id"].get<size_t>();
                    max_id = std::max(max_id, id + 1);
                }
            }
        }

        id_to_token.assign(max_id, "");
        token_to_id.clear();

        for (auto it = vocab.begin(); it != vocab.end(); ++it) {
            size_t id = it.value().get<size_t>();
            id_to_token[id] = it.key();
            token_to_id[it.key()] = static_cast<int>(id);
        }

        if (j.contains("added_tokens")) {
            for (const auto& token : j["added_tokens"]) {
                if (!token.contains("id") || !token.contains("content")) {
                    continue;
                }

                size_t id = token["id"].get<size_t>();
                std::string content = token["content"].get<std::string>();

                if (id >= id_to_token.size()) {
                    id_to_token.resize(id + 1);
                }

                id_to_token[id] = content;
                token_to_id[content] = static_cast<int>(id);

                bool special = false;

                if (token.contains("special") && token["special"].is_boolean()) {
                    special = token["special"].get<bool>();
                }

                if (special) {
                    special_tokens.push_back(content);
                }
            }
        }

        std::sort(
            special_tokens.begin(),
            special_tokens.end(),
            [](const std::string& a, const std::string& b) {
                return a.size() > b.size();
            }
        );

        merge_rank.clear();

        if (
            j["model"].contains("merges") &&
            j["model"]["merges"].is_array()
        ) {
            const auto& merges = j["model"]["merges"];

            int rank = 0;

            for (const auto& merge_value : merges) {
                std::string merge;

                if (merge_value.is_string()) {
                    merge = merge_value.get<std::string>();
                } else if (
                    merge_value.is_array() &&
                    merge_value.size() >= 2 &&
                    merge_value[0].is_string() &&
                    merge_value[1].is_string()
                ) {
                    merge =
                        merge_value[0].get<std::string>() +
                        " " +
                        merge_value[1].get<std::string>();
                } else {
                    continue;
                }

                if (merge.empty()) {
                    continue;
                }

                merge_rank[merge] = rank;
                ++rank;
            }
        }

        vocab_size = id_to_token.size();

        byte_level = false;

        if (j.contains("decoder")) {
            byte_level = json_contains_byte_level(j["decoder"]);
        }

        if (byte_level) {
            build_byte_maps();
        }
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string out;

        for (int id : ids) {
            if (id < 0 || static_cast<size_t>(id) >= id_to_token.size()) {
                continue;
            }

            const std::string& token = id_to_token[id];

            if (!byte_level) {
                out += token;
                continue;
            }

            size_t i = 0;

            while (i < token.size()) {
                char32_t cp = 0;

                if (!utf8_next_codepoint(token, i, cp)) {
                    ++i;
                    continue;
                }

                auto it = byte_decoder.find(cp);

                if (it != byte_decoder.end()) {
                    out.push_back(static_cast<char>(it->second));
                }
            }
        }

        return out;
    }

    std::vector<std::string> text_to_symbols(const std::string& text) const {
        std::vector<std::string> symbols;

        if (!byte_level) {
            size_t i = 0;

            while (i < text.size()) {
                char32_t cp = 0;

                if (!utf8_next_codepoint(text, i, cp)) {
                    ++i;
                    continue;
                }

                std::string s;
                utf8_append_codepoint(s, cp);
                symbols.push_back(s);
            }

            return symbols;
        }

        for (unsigned char byte : text) {
            auto it = byte_encoder.find(byte);

            if (it != byte_encoder.end()) {
                symbols.push_back(it->second);
            }
        }

        return symbols;
    }

    std::vector<std::string> bpe(const std::vector<std::string>& input_symbols) const {
        std::vector<std::string> symbols = input_symbols;

        while (symbols.size() >= 2) {
            int best_rank = std::numeric_limits<int>::max();
            std::string best_pair;

            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                std::string pair = symbols[i] + " " + symbols[i + 1];

                auto it = merge_rank.find(pair);

                if (it != merge_rank.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pair = pair;
                }
            }

            if (best_pair.empty()) {
                break;
            }

            size_t space = best_pair.find(' ');
            std::string first = best_pair.substr(0, space);
            std::string second = best_pair.substr(space + 1);

            std::vector<std::string> next_symbols;

            for (size_t i = 0; i < symbols.size(); ) {
                if (
                    i + 1 < symbols.size() &&
                    symbols[i] == first &&
                    symbols[i + 1] == second
                ) {
                    next_symbols.push_back(first + second);
                    i += 2;
                } else {
                    next_symbols.push_back(symbols[i]);
                    ++i;
                }
            }

            symbols = next_symbols;
        }

        return symbols;
    }

    std::vector<int> encode_piece(const std::string& piece) const {
        std::vector<int> ids;

        std::vector<std::string> symbols = text_to_symbols(piece);
        std::vector<std::string> bpe_symbols = bpe(symbols);

        for (const std::string& symbol : bpe_symbols) {
            auto it = token_to_id.find(symbol);

            if (it != token_to_id.end()) {
                ids.push_back(it->second);
                continue;
            }

            if (!byte_level) {
                continue;
            }

            size_t i = 0;

            while (i < symbol.size()) {
                char32_t cp = 0;

                if (!utf8_next_codepoint(symbol, i, cp)) {
                    ++i;
                    continue;
                }

                auto byte_it = byte_decoder.find(cp);

                if (byte_it == byte_decoder.end()) {
                    continue;
                }

                auto enc_it = byte_encoder.find(byte_it->second);

                if (enc_it == byte_encoder.end()) {
                    continue;
                }

                auto vocab_it = token_to_id.find(enc_it->second);

                if (vocab_it != token_to_id.end()) {
                    ids.push_back(vocab_it->second);
                }
            }
        }

        return ids;
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> ids;

        size_t i = 0;

        while (i < text.size()) {
            bool matched_special = false;

            for (const std::string& special : special_tokens) {
                if (special.empty()) {
                    continue;
                }

                if (text.compare(i, special.size(), special) == 0) {
                    auto it = token_to_id.find(special);

                    if (it != token_to_id.end()) {
                        ids.push_back(it->second);
                    }

                    i += special.size();
                    matched_special = true;

                    break;
                }
            }

            if (matched_special) {
                continue;
            }

            size_t next_special = text.size();

            for (const std::string& special : special_tokens) {
                if (special.empty()) {
                    continue;
                }

                size_t pos = text.find(special, i);

                if (pos != std::string::npos) {
                    next_special = std::min(next_special, pos);
                }
            }

            std::string piece = text.substr(i, next_special - i);

            if (!piece.empty()) {
                std::vector<int> piece_ids = encode_piece(piece);
                ids.insert(ids.end(), piece_ids.begin(), piece_ids.end());
            }

            i = next_special;
        }

        return ids;
    }
};