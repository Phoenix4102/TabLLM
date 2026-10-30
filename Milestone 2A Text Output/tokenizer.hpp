#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
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

struct Tokenizer {
    std::vector<std::string> id_to_token;
    std::unordered_map<char32_t, unsigned char> byte_decoder;
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

    void build_byte_decoder() {
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

        for (size_t i = 0; i < bs.size(); ++i) {
            byte_decoder[static_cast<char32_t>(cs[i])] =
                static_cast<unsigned char>(bs[i]);
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

        for (auto it = vocab.begin(); it != vocab.end(); ++it) {
            size_t id = it.value().get<size_t>();
            id_to_token[id] = it.key();
        }

        if (j.contains("added_tokens")) {
            for (const auto& token : j["added_tokens"]) {
                if (token.contains("id") && token.contains("content")) {
                    size_t id = token["id"].get<size_t>();

                    if (id < id_to_token.size()) {
                        id_to_token[id] = token["content"].get<std::string>();
                    }
                }
            }
        }

        vocab_size = id_to_token.size();

        byte_level = false;

        if (j.contains("decoder")) {
            byte_level = json_contains_byte_level(j["decoder"]);
        }

        if (byte_level) {
            build_byte_decoder();
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
};