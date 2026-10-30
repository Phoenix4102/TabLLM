#include "..\Baseline\tokenizer.hpp"
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// -----------------------------------------------------------------------------
// Model file format
// -----------------------------------------------------------------------------
//
// Header is exactly 64 bytes.
//
// All tensors are fp32, row-major, in this order:
//
//   embed_tokens.weight                     [vocab_size, hidden_size]
//
// For each layer:
//   input_layernorm.weight                  [hidden_size]
//   self_attn.q_proj.weight                 [hidden_size, hidden_size]
//   self_attn.k_proj.weight                 [kv_dim, hidden_size]
//   self_attn.v_proj.weight                 [kv_dim, hidden_size]
//   self_attn.o_proj.weight                 [hidden_size, hidden_size]
//   post_attention_layernorm.weight         [hidden_size]
//   mlp.gate_proj.weight                    [ffn_hidden_size, hidden_size]
//   mlp.up_proj.weight                      [ffn_hidden_size, hidden_size]
//   mlp.down_proj.weight                    [hidden_size, ffn_hidden_size]
//
// final_norm.weight                         [hidden_size]
// lm_head.weight                            [vocab_size, hidden_size] if not tied

struct Header {
    char magic[4];              // "TLLM"
    uint32_t version;
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t ffn_hidden_size;
    uint32_t ctx_len;
    float rope_theta;
    uint32_t dtype;             // 0 = fp32
    uint32_t tie_lm_head;       // 1 = lm_head shares embedding weight
    uint64_t tensor_bytes;
    uint64_t reserved;
};

static_assert(sizeof(Header) == 64, "Header must be 64 bytes");

// -----------------------------------------------------------------------------
// Memory-mapped read-only model file
// -----------------------------------------------------------------------------

struct MmapFile {
    int fd = -1;
    void* data = nullptr;
    size_t size = 0;

    MmapFile(const char* path) {
        fd = ::open(path, O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("cannot open model file");
        }

        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error("fstat failed");
        }

        size = static_cast<size_t>(st.st_size);
        if (size < sizeof(Header)) {
            ::close(fd);
            throw std::runtime_error("model file too small");
        }

        data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
            ::close(fd);
            throw std::runtime_error("mmap failed");
        }

        ::madvise(data, size, MADV_NORMAL);
    }

    ~MmapFile() {
        if (data != nullptr && data != MAP_FAILED) {
            ::munmap(data, size);
        }
        if (fd >= 0) {
            ::close(fd);
        }
    }

    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;
};

// -----------------------------------------------------------------------------
// Basic math kernels
// -----------------------------------------------------------------------------

static float dot(const float* a, const float* b, size_t n) {
#if defined(__ARM_NEON)
    size_t i = 0;
    float32x4_t sum = vdupq_n_f32(0.0f);

    for (; i + 4 <= n; i += 4) {
        sum = vmlaq_f32(sum, vld1q_f32(a + i), vld1q_f32(b + i));
    }

    float s = vaddvq_f32(sum);

    for (; i < n; ++i) {
        s += a[i] * b[i];
    }

    return s;
#else
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
#endif
}

static void matvec(
    float* out,
    const float* W,
    const float* x,
    size_t out_features,
    size_t in_features
) {
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(out_features); ++i) {
        out[i] = dot(W + static_cast<size_t>(i) * in_features, x, in_features);
    }
}

static float fp16_to_float(uint16_t h) {
    uint32_t sign = ((uint32_t)(h >> 15)) & 1u;
    uint32_t exponent = ((uint32_t)(h >> 10)) & 0x1Fu;
    uint32_t mantissa = ((uint32_t)h) & 0x3FFu;

    uint32_t f;

    if (exponent == 0) {
        if (mantissa == 0) {
            f = sign << 31;
        } else {
            uint32_t shift = 0;

            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }

            mantissa &= 0x3FFu;

            uint32_t exp = 127u - 14u - shift;

            f = (sign << 31) | (exp << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        f = (sign << 31) | 0x7F800000u | (mantissa << 13);
    } else {
        f = (sign << 31) | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }

    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

static void convert_f16_to_f32(
    float* dst,
    const uint16_t* src,
    size_t n
) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        float16x8_t h = vld1q_f16(reinterpret_cast<const float16_t*>(src + i));

        vst1q_f32(dst + i, vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(h)));
    }

    for (; i < n; ++i) {
        dst[i] = fp16_to_float(src[i]);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        dst[i] = fp16_to_float(src[i]);
    }
#endif
}

static float dot_f16_f32(
    const uint16_t* a,
    const float* b,
    size_t n
) {
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    size_t i = 0;

    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);

    for (; i + 8 <= n; i += 8) {
        float16x8_t wh = vld1q_f16(reinterpret_cast<const float16_t*>(a + i));

        float32x4_t w0 = vcvt_f32_f16(vget_low_f16(wh));
        float32x4_t w1 = vcvt_f32_f16(vget_high_f16(wh));

        sum0 = vfmaq_f32(sum0, vld1q_f32(b + i), w0);
        sum1 = vfmaq_f32(sum1, vld1q_f32(b + i + 4), w1);
    }

    float s = vaddvq_f32(sum0) + vaddvq_f32(sum1);

    for (; i < n; ++i) {
        s += fp16_to_float(a[i]) * b[i];
    }

    return s;
#else
    float s = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        s += fp16_to_float(a[i]) * b[i];
    }

    return s;
#endif
}

static void matvec_f16(
    float* out,
    const uint16_t* W,
    const float* x,
    size_t out_features,
    size_t in_features
) {
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(out_features); ++i) {
        out[i] = dot_f16_f32(
            W + static_cast<size_t>(i) * in_features,
            x,
            in_features
        );
    }
}

static void matmul_f32_colmajor(
    float* Y,
    const float* W,
    const float* X,
    size_t O,
    size_t H,
    size_t S
) {
    // Y[O][S] = W[O][H] * X[H][S]

#pragma omp parallel for schedule(static)
    for (std::int64_t oi = 0; oi < static_cast<std::int64_t>(O); ++oi) {
        size_t o = static_cast<size_t>(oi);

        float* y = Y + o * S;
        std::fill(y, y + S, 0.0f);

        const float* w = W + o * H;

        for (size_t k = 0; k < H; ++k) {
            const float wk = w[k];
            const float* x = X + k * S;

            size_t s = 0;

#if defined(__ARM_NEON)
            float32x4_t wv = vdupq_n_f32(wk);

            for (; s + 4 <= S; s += 4) {
                float32x4_t yv = vld1q_f32(y + s);
                float32x4_t xv = vld1q_f32(x + s);

                yv = vmlaq_f32(yv, wv, xv);

                vst1q_f32(y + s, yv);
            }
#endif

            for (; s < S; ++s) {
                y[s] += wk * x[s];
            }
        }
    }
}

static void matmul_f16_colmajor(
    float* Y,
    const uint16_t* W,
    const float* X,
    size_t O,
    size_t H,
    size_t S
) {
    // Y[O][S] = W[O][H] * X[H][S]

#pragma omp parallel for schedule(static)
    for (std::int64_t oi = 0; oi < static_cast<std::int64_t>(O); ++oi) {
        size_t o = static_cast<size_t>(oi);

        float* y = Y + o * S;
        std::fill(y, y + S, 0.0f);

        const uint16_t* w = W + o * H;

        for (size_t k = 0; k < H; ++k) {
            const float wk = fp16_to_float(w[k]);
            const float* x = X + k * S;

            size_t s = 0;

#if defined(__ARM_NEON)
            float32x4_t wv = vdupq_n_f32(wk);

            for (; s + 4 <= S; s += 4) {
                float32x4_t yv = vld1q_f32(y + s);
                float32x4_t xv = vld1q_f32(x + s);

                yv = vmlaq_f32(yv, wv, xv);

                vst1q_f32(y + s, yv);
            }
#endif

            for (; s < S; ++s) {
                y[s] += wk * x[s];
            }
        }
    }
}

static void rmsnorm_batch(
    float* Y,
    const float* X,
    const float* w,
    size_t H,
    size_t S
) {
#pragma omp parallel for schedule(static)
    for (std::int64_t si = 0; si < static_cast<std::int64_t>(S); ++si) {
        size_t s = static_cast<size_t>(si);

        float ss = 0.0f;

        for (size_t h = 0; h < H; ++h) {
            float v = X[h * S + s];
            ss += v * v;
        }

        float scale = 1.0f / std::sqrt(ss / static_cast<float>(H) + 1e-5f);

        for (size_t h = 0; h < H; ++h) {
            Y[h * S + s] = X[h * S + s] * scale * w[h];
        }
    }
}

static float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static void add_arrays(
    float* a,
    const float* b,
    size_t n
) {
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        a[i] += b[i];
    }
}

static void silu_mul(
    float* out,
    const float* gate,
    const float* up,
    size_t n
) {
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        out[i] = silu(gate[i]) * up[i];
    }
}

static void apply_rope_batch(
    float* x,
    size_t heads,
    size_t head_dim,
    size_t start_pos,
    size_t S,
    float theta
) {
    if (head_dim < 2 || theta <= 0.0f) return;

    const size_t half = head_dim / 2;
    const float log_theta = std::log(theta);

    for (size_t s = 0; s < S; ++s) {
        size_t position = start_pos + s;

        for (size_t h = 0; h < heads; ++h) {
            for (size_t i = 0; i < half; ++i) {
                float freq = std::exp(
                    -(2.0f * static_cast<float>(i) / static_cast<float>(head_dim)) * log_theta
                );

                float angle = static_cast<float>(position) * freq;
                float c = std::cos(angle);
                float ss = std::sin(angle);

                float* p0 = x + (h * head_dim + i) * S + s;
                float* p1 = x + (h * head_dim + i + half) * S + s;

                float x0 = *p0;
                float x1 = *p1;

                *p0 = x0 * c - x1 * ss;
                *p1 = x1 * c + x0 * ss;
            }
        }
    }
}

static void rmsnorm(
    float* y,
    const float* x,
    const float* w,
    size_t n,
    float eps = 1e-5f
) {
    float ss = dot(x, x, n);
    float scale = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);

    for (size_t i = 0; i < n; ++i) {
        y[i] = x[i] * scale * w[i];
    }
}

static void softmax(float* x, size_t n) {
    if (n == 0) return;

    float m = x[0];
    for (size_t i = 1; i < n; ++i) {
        m = std::max(m, x[i]);
    }

    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - m);
        sum += x[i];
    }

    float inv = 1.0f / (sum == 0.0f ? 1.0f : sum);

    for (size_t i = 0; i < n; ++i) {
        x[i] *= inv;
    }
}

// // // //

static void apply_rope(
    float* x,
    size_t heads,
    size_t head_dim,
    size_t position,
    float theta
) {
    if (head_dim < 2 || theta <= 0.0f) return;

    const size_t half = head_dim / 2;
    const float log_theta = std::log(theta);

    for (size_t h = 0; h < heads; ++h) {
        float* p = x + h * head_dim;

        for (size_t i = 0; i < half; ++i) {
            float freq = std::exp(
                -(2.0f * static_cast<float>(i) / static_cast<float>(head_dim)) * log_theta
            );

            float angle = static_cast<float>(position) * freq;
            float c = std::cos(angle);
            float s = std::sin(angle);

            float x0 = p[i];
            float x1 = p[i + half];

            p[i] = x0 * c - x1 * s;
            p[i + half] = x1 * c + x0 * s;
        }
    }
}

static int argmax(const float* x, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; ++i) {
        if (x[i] > x[best]) {
            best = i;
        }
    }
    return static_cast<int>(best);
}

static int sample(
    const float* logits,
    size_t vocab,
    float temperature,
    int top_k,
    std::mt19937& rng
) {
    if (vocab == 0) return -1;

    if (temperature <= 0.0f) {
        return argmax(logits, vocab);
    }

    std::vector<float> work(logits, logits + vocab);

    if (top_k > 0 && static_cast<size_t>(top_k) < vocab) {
        std::vector<size_t> idx(vocab);
        std::iota(idx.begin(), idx.end(), 0);

        std::nth_element(
            idx.begin(),
            idx.begin() + top_k,
            idx.end(),
            [&](size_t a, size_t b) {
                return work[a] > work[b];
            }
        );

        float threshold = work[idx[top_k - 1]];

        for (size_t i = 0; i < vocab; ++i) {
            if (work[i] < threshold) {
                work[i] = -INFINITY;
            }
        }
    }

    float max = work[0];
    for (size_t i = 1; i < vocab; ++i) {
        max = std::max(max, work[i]);
    }

    if (!std::isfinite(max)) {
        return argmax(logits, vocab);
    }

    float sum = 0.0f;

    for (size_t i = 0; i < vocab; ++i) {
        work[i] = std::exp((work[i] - max) / temperature);
        sum += work[i];
    }

    if (!(sum > 0.0f) || !std::isfinite(sum)) {
        return argmax(logits, vocab);
    }

    std::uniform_real_distribution<float> dist(0.0f, sum);
    float r = dist(rng);

    float acc = 0.0f;
    for (size_t i = 0; i < vocab; ++i) {
        acc += work[i];
        if (r <= acc) {
            return static_cast<int>(i);
        }
    }

    return static_cast<int>(vocab - 1);
}

// -----------------------------------------------------------------------------
// Engine
// -----------------------------------------------------------------------------

struct Engine {
    struct Layer {
        std::vector<float> norm1;
        std::vector<float> norm2;

        const void* q = nullptr;
        const void* k = nullptr;
        const void* v = nullptr;
        const void* o = nullptr;

        const void* gate = nullptr;
        const void* up = nullptr;
        const void* down = nullptr;
    };

    MmapFile file;
    Header cfg{};

    int dtype = 0;
    size_t element_size = 4;

    size_t head_dim = 0;
    size_t kv_dim = 0;

    const void* emb = nullptr;
    std::vector<Layer> layers;
    std::vector<float> final_norm;
    const void* lm_head = nullptr;

    std::vector<float> k_cache;
    std::vector<float> v_cache;

    std::vector<float> hidden;
    std::vector<float> normed;

    std::vector<float> q_buf;
    std::vector<float> k_buf;
    std::vector<float> v_buf;

    std::vector<float> attn_out;
    std::vector<float> attn_acc;
    std::vector<float> proj_buf;

    std::vector<float> gate_buf;
    std::vector<float> up_buf;
    std::vector<float> ffn_act;
    std::vector<float> ffn_out;

    std::vector<float> logits;
    std::vector<float> scores;

    std::vector<float> hidden_b;
    std::vector<float> normed_b;
    std::vector<float> q_b;
    std::vector<float> k_b;
    std::vector<float> v_b;
    std::vector<float> attn_b;
    std::vector<float> proj_b;
    std::vector<float> gate_b;
    std::vector<float> up_b;
    std::vector<float> ffn_act_b;
    std::vector<float> ffn_out_b;

    size_t pos = 0;

    Engine(const char* path) : file(path) {
        std::memcpy(&cfg, file.data, sizeof(Header));

        if (std::strncmp(cfg.magic, "TLLM", 4) != 0) {
            throw std::runtime_error("bad magic");
        }

        if (cfg.version != 1) {
            throw std::runtime_error("unsupported version");
        }

        if (cfg.dtype != 0 && cfg.dtype != 2) {
            throw std::runtime_error("only dtype 0 fp32 and dtype 2 fp16 are supported");
        }

        dtype = static_cast<int>(cfg.dtype);
        element_size = dtype == 2 ? 2 : 4;

        if (
            cfg.ctx_len < 2 ||
            cfg.vocab_size < 256 ||
            cfg.hidden_size == 0 ||
            cfg.num_layers == 0 ||
            cfg.num_heads == 0 ||
            cfg.num_kv_heads == 0 ||
            cfg.ffn_hidden_size == 0
        ) {
            throw std::runtime_error("invalid model config");
        }

        if (cfg.hidden_size % cfg.num_heads != 0) {
            throw std::runtime_error("hidden_size must be divisible by num_heads");
        }

        if (cfg.num_heads % cfg.num_kv_heads != 0) {
            throw std::runtime_error("GQA requires num_heads % num_kv_heads == 0");
        }

        if (cfg.rope_theta <= 0.0f) {
            throw std::runtime_error("rope_theta must be positive");
        }

        head_dim = static_cast<size_t>(cfg.hidden_size) / cfg.num_heads;
        kv_dim = static_cast<size_t>(cfg.num_kv_heads) * head_dim;

        if (static_cast<uint64_t>(file.size) != sizeof(Header) + cfg.tensor_bytes) {
            throw std::runtime_error("tensor_bytes does not match file size");
        }

        const char* p = reinterpret_cast<const char*>(file.data) + sizeof(Header);
        size_t remaining = static_cast<size_t>(cfg.tensor_bytes);

        auto take = [&](size_t elements) -> const void* {
            const size_t bytes = elements * element_size;

            if (bytes > remaining) {
                throw std::runtime_error("truncated model file");
            }

            const void* ptr = reinterpret_cast<const void*>(p);

            p += bytes;
            remaining -= bytes;

            return ptr;
        };

        auto fill_norm = [&](std::vector<float>& dst, const void* src, size_t n) {
            dst.resize(n);

            if (dtype == 0) {
                std::memcpy(dst.data(), src, n * sizeof(float));
            } else {
                convert_f16_to_f32(
                    dst.data(),
                    reinterpret_cast<const uint16_t*>(src),
                    n
                );
            }
        };

        const size_t H = cfg.hidden_size;
        const size_t F = cfg.ffn_hidden_size;
        const size_t V = cfg.vocab_size;
        const size_t C = cfg.ctx_len;

        emb = take(V * H);

        layers.resize(cfg.num_layers);

        for (size_t i = 0; i < cfg.num_layers; ++i) {
            Layer& L = layers[i];

            const void* norm1_src = take(H);
            fill_norm(L.norm1, norm1_src, H);

            L.q = take(H * H);
            L.k = take(kv_dim * H);
            L.v = take(kv_dim * H);
            L.o = take(H * H);

            const void* norm2_src = take(H);
            fill_norm(L.norm2, norm2_src, H);

            L.gate = take(F * H);
            L.up = take(F * H);
            L.down = take(H * F);
        }

        const void* final_norm_src = take(H);
        fill_norm(final_norm, final_norm_src, H);

        if (cfg.tie_lm_head) {
            lm_head = emb;
        } else {
            lm_head = take(V * H);
        }

        if (remaining != 0) {
            throw std::runtime_error("extra bytes at end of model file");
        }

        k_cache.assign(static_cast<size_t>(cfg.num_layers) * C * kv_dim, 0.0f);
        v_cache.assign(static_cast<size_t>(cfg.num_layers) * C * kv_dim, 0.0f);

        hidden.assign(H, 0.0f);
        normed.assign(H, 0.0f);

        q_buf.assign(H, 0.0f);
        k_buf.assign(kv_dim, 0.0f);
        v_buf.assign(kv_dim, 0.0f);

        attn_out.assign(H, 0.0f);
        attn_acc.assign(H, 0.0f);
        proj_buf.assign(H, 0.0f);

        gate_buf.assign(F, 0.0f);
        up_buf.assign(F, 0.0f);
        ffn_act.assign(F, 0.0f);
        ffn_out.assign(H, 0.0f);

        logits.assign(V, 0.0f);
        scores.assign(C, 0.0f);
    }

    void linear(
        float* out,
        const void* W,
        const float* x,
        size_t out_features,
        size_t in_features
    ) const {
        if (dtype == 0) {
            matvec(
                out,
                reinterpret_cast<const float*>(W),
                x,
                out_features,
                in_features
            );
        } else {
            matvec_f16(
                out,
                reinterpret_cast<const uint16_t*>(W),
                x,
                out_features,
                in_features
            );
        }
    }

    void embed_token(int token, float* hidden_out) const {
        const size_t H = cfg.hidden_size;

        if (dtype == 0) {
            const float* e =
                reinterpret_cast<const float*>(emb) +
                static_cast<size_t>(token) * H;

            std::memcpy(hidden_out, e, H * sizeof(float));
        } else {
            const uint16_t* e =
                reinterpret_cast<const uint16_t*>(emb) +
                static_cast<size_t>(token) * H;

            convert_f16_to_f32(hidden_out, e, H);
        }
    }

    void attention(
        size_t layer_idx,
        size_t position,
        const float* x,
        float* out
    ) {
        const Layer& L = layers[layer_idx];

        linear(q_buf.data(), L.q, x, cfg.hidden_size, cfg.hidden_size);
        linear(k_buf.data(), L.k, x, kv_dim, cfg.hidden_size);
        linear(v_buf.data(), L.v, x, kv_dim, cfg.hidden_size);

        apply_rope(q_buf.data(), cfg.num_heads, head_dim, position, cfg.rope_theta);
        apply_rope(k_buf.data(), cfg.num_kv_heads, head_dim, position, cfg.rope_theta);

        const size_t layer_offset =
            layer_idx * static_cast<size_t>(cfg.ctx_len) * kv_dim;

        float* kd = k_cache.data() + layer_offset + position * kv_dim;
        float* vd = v_cache.data() + layer_offset + position * kv_dim;

        std::memcpy(kd, k_buf.data(), kv_dim * sizeof(float));
        std::memcpy(vd, v_buf.data(), kv_dim * sizeof(float));

        std::fill(attn_acc.begin(), attn_acc.end(), 0.0f);

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        for (size_t h = 0; h < cfg.num_heads; ++h) {
            const size_t kvh = h * cfg.num_kv_heads / cfg.num_heads;
            const float* qp = q_buf.data() + h * head_dim;

            for (size_t t = 0; t <= position; ++t) {
                const float* kp =
                    k_cache.data() +
                    layer_offset +
                    t * kv_dim +
                    kvh * head_dim;

                scores[t] = dot(qp, kp, head_dim) * scale;
            }

            softmax(scores.data(), position + 1);

            float* op = attn_acc.data() + h * head_dim;

            for (size_t t = 0; t <= position; ++t) {
                const float* vp =
                    v_cache.data() +
                    layer_offset +
                    t * kv_dim +
                    kvh * head_dim;

                const float w = scores[t];

                for (size_t d = 0; d < head_dim; ++d) {
                    op[d] += w * vp[d];
                }
            }
        }

        std::memcpy(out, attn_acc.data(), cfg.hidden_size * sizeof(float));
    }

    void ffn(
        size_t layer_idx,
        const float* x,
        float* out
    ) {
        const Layer& L = layers[layer_idx];

        linear(gate_buf.data(), L.gate, x, cfg.ffn_hidden_size, cfg.hidden_size);
        linear(up_buf.data(), L.up, x, cfg.ffn_hidden_size, cfg.hidden_size);

        for (size_t i = 0; i < cfg.ffn_hidden_size; ++i) {
            ffn_act[i] = silu(gate_buf[i]) * up_buf[i];
        }

        linear(out, L.down, ffn_act.data(), cfg.hidden_size, cfg.ffn_hidden_size);
    }

    const float* forward(int token) {
        if (token < 0 || static_cast<size_t>(token) >= cfg.vocab_size) {
            throw std::runtime_error("token out of range");
        }

        if (pos >= cfg.ctx_len) {
            throw std::runtime_error("context full");
        }

        embed_token(token, hidden.data());

        for (size_t l = 0; l < cfg.num_layers; ++l) {
            rmsnorm(
                normed.data(),
                hidden.data(),
                layers[l].norm1.data(),
                cfg.hidden_size
            );

            attention(l, pos, normed.data(), attn_out.data());

            linear(
                proj_buf.data(),
                layers[l].o,
                attn_out.data(),
                cfg.hidden_size,
                cfg.hidden_size
            );

            for (size_t i = 0; i < cfg.hidden_size; ++i) {
                hidden[i] += proj_buf[i];
            }

            rmsnorm(
                normed.data(),
                hidden.data(),
                layers[l].norm2.data(),
                cfg.hidden_size
            );

            ffn(l, normed.data(), ffn_out.data());

            for (size_t i = 0; i < cfg.hidden_size; ++i) {
                hidden[i] += ffn_out[i];
            }
        }

        rmsnorm(
            normed.data(),
            hidden.data(),
            final_norm.data(),
            cfg.hidden_size
        );

        linear(
            logits.data(),
            lm_head,
            normed.data(),
            cfg.vocab_size,
            cfg.hidden_size
        );

        ++pos;

        return logits.data();
    }

        void linear_batch(
        float* Y,
        const void* W,
        const float* X,
        size_t O,
        size_t H,
        size_t S
    ) const {
        if (dtype == 0) {
            matmul_f32_colmajor(
                Y,
                reinterpret_cast<const float*>(W),
                X,
                O,
                H,
                S
            );
        } else {
            matmul_f16_colmajor(
                Y,
                reinterpret_cast<const uint16_t*>(W),
                X,
                O,
                H,
                S
            );
        }
    }

    void embed_batch(
        const std::vector<int>& tokens,
        float* X
    ) const {
        const size_t S = tokens.size();
        const size_t H = cfg.hidden_size;

#pragma omp parallel for schedule(static)
        for (std::int64_t si = 0; si < static_cast<std::int64_t>(S); ++si) {
            size_t s = static_cast<size_t>(si);
            int token = tokens[s];

            if (dtype == 0) {
                const float* e =
                    reinterpret_cast<const float*>(emb) +
                    static_cast<size_t>(token) * H;

                for (size_t h = 0; h < H; ++h) {
                    X[h * S + s] = e[h];
                }
            } else {
                const uint16_t* e =
                    reinterpret_cast<const uint16_t*>(emb) +
                    static_cast<size_t>(token) * H;

                for (size_t h = 0; h < H; ++h) {
                    X[h * S + s] = fp16_to_float(e[h]);
                }
            }
        }
    }

    void attention_prefill(
        size_t layer_idx,
        size_t start_pos,
        size_t S
    ) {
        const size_t layer_offset =
            layer_idx * static_cast<size_t>(cfg.ctx_len) * kv_dim;

        // Store K and V into cache.
        for (size_t s = 0; s < S; ++s) {
            size_t position = start_pos + s;

            float* kd = k_cache.data() + layer_offset + position * kv_dim;
            float* vd = v_cache.data() + layer_offset + position * kv_dim;

            for (size_t d = 0; d < kv_dim; ++d) {
                kd[d] = k_b[d * S + s];
                vd[d] = v_b[d * S + s];
            }
        }

        std::fill(attn_b.begin(), attn_b.end(), 0.0f);

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

#pragma omp parallel for schedule(dynamic)
        for (std::int64_t si = 0; si < static_cast<std::int64_t>(S); ++si) {
            size_t s = static_cast<size_t>(si);
            size_t position = start_pos + s;

            std::vector<float> local_scores(position + 1);
            std::vector<float> qtmp(head_dim);
            std::vector<float> head_out(head_dim);

            for (size_t h = 0; h < cfg.num_heads; ++h) {
                const size_t kvh = h * cfg.num_kv_heads / cfg.num_heads;

                for (size_t d = 0; d < head_dim; ++d) {
                    qtmp[d] = q_b[(h * head_dim + d) * S + s];
                }

                for (size_t t = 0; t <= position; ++t) {
                    const float* kp =
                        k_cache.data() +
                        layer_offset +
                        t * kv_dim +
                        kvh * head_dim;

                    local_scores[t] = dot(qtmp.data(), kp, head_dim) * scale;
                }

                softmax(local_scores.data(), position + 1);

                std::fill(head_out.begin(), head_out.end(), 0.0f);

                for (size_t t = 0; t <= position; ++t) {
                    const float* vp =
                        v_cache.data() +
                        layer_offset +
                        t * kv_dim +
                        kvh * head_dim;

                    const float w = local_scores[t];

                    for (size_t d = 0; d < head_dim; ++d) {
                        head_out[d] += w * vp[d];
                    }
                }

                for (size_t d = 0; d < head_dim; ++d) {
                    attn_b[(h * head_dim + d) * S + s] = head_out[d];
                }
            }
        }
    }

    const float* prefill(const std::vector<int>& tokens) {
        const size_t S = tokens.size();

        if (S == 0) {
            throw std::runtime_error("prefill requires at least one token");
        }

        if (pos + S > cfg.ctx_len) {
            throw std::runtime_error("context full");
        }

        for (int token : tokens) {
            if (token < 0 || static_cast<size_t>(token) >= cfg.vocab_size) {
                throw std::runtime_error("token out of range");
            }
        }

        const size_t H = cfg.hidden_size;
        const size_t F = cfg.ffn_hidden_size;

        hidden_b.resize(H * S);
        normed_b.resize(H * S);

        q_b.resize(H * S);
        k_b.resize(kv_dim * S);
        v_b.resize(kv_dim * S);

        attn_b.resize(H * S);
        proj_b.resize(H * S);

        gate_b.resize(F * S);
        up_b.resize(F * S);
        ffn_act_b.resize(F * S);
        ffn_out_b.resize(H * S);

        const size_t start_pos = pos;

        embed_batch(tokens, hidden_b.data());

        for (size_t l = 0; l < cfg.num_layers; ++l) {
            const Layer& L = layers[l];

            rmsnorm_batch(
                normed_b.data(),
                hidden_b.data(),
                L.norm1.data(),
                H,
                S
            );

            linear_batch(
                q_b.data(),
                L.q,
                normed_b.data(),
                H,
                H,
                S
            );

            linear_batch(
                k_b.data(),
                L.k,
                normed_b.data(),
                kv_dim,
                H,
                S
            );

            linear_batch(
                v_b.data(),
                L.v,
                normed_b.data(),
                kv_dim,
                H,
                S
            );

            apply_rope_batch(
                q_b.data(),
                cfg.num_heads,
                head_dim,
                start_pos,
                S,
                cfg.rope_theta
            );

            apply_rope_batch(
                k_b.data(),
                cfg.num_kv_heads,
                head_dim,
                start_pos,
                S,
                cfg.rope_theta
            );

            attention_prefill(l, start_pos, S);

            linear_batch(
                proj_b.data(),
                L.o,
                attn_b.data(),
                H,
                H,
                S
            );

            add_arrays(
                hidden_b.data(),
                proj_b.data(),
                H * S
            );

            rmsnorm_batch(
                normed_b.data(),
                hidden_b.data(),
                L.norm2.data(),
                H,
                S
            );

            linear_batch(
                gate_b.data(),
                L.gate,
                normed_b.data(),
                F,
                H,
                S
            );

            linear_batch(
                up_b.data(),
                L.up,
                normed_b.data(),
                F,
                H,
                S
            );

            silu_mul(
                ffn_act_b.data(),
                gate_b.data(),
                up_b.data(),
                F * S
            );

            linear_batch(
                ffn_out_b.data(),
                L.down,
                ffn_act_b.data(),
                H,
                F,
                S
            );

            add_arrays(
                hidden_b.data(),
                ffn_out_b.data(),
                H * S
            );
        }

        rmsnorm_batch(
            normed_b.data(),
            hidden_b.data(),
            final_norm.data(),
            H,
            S
        );

        std::vector<float> last_hidden(H);

        for (size_t h = 0; h < H; ++h) {
            last_hidden[h] = normed_b[h * S + (S - 1)];
        }

        linear(
            logits.data(),
            lm_head,
            last_hidden.data(),
            cfg.vocab_size,
            H
        );

        pos = start_pos + S;

        return logits.data();
    }
};

static std::vector<int> read_tokens_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);

    if (!f) {
        throw std::runtime_error("cannot open prompt token file: " + path);
    }

    f.seekg(0, std::ios::end);
    std::streamoff bytes = f.tellg();
    f.seekg(0, std::ios::beg);

    if (bytes <= 0) {
        return {};
    }

    if (bytes % static_cast<std::streamoff>(sizeof(int32_t)) != 0) {
        throw std::runtime_error("prompt token file must contain int32 tokens");
    }

    size_t n = static_cast<size_t>(bytes / static_cast<std::streamoff>(sizeof(int32_t)));

    std::vector<int32_t> raw(n);

    f.read(
        reinterpret_cast<char*>(raw.data()),
        static_cast<std::streamsize>(bytes)
    );

    if (!f) {
        throw std::runtime_error("failed reading prompt token file");
    }

    return std::vector<int>(raw.begin(), raw.end());
}

static void write_tokens_bin(
    const std::string& path,
    const std::vector<int>& tokens
) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);

    if (!f) {
        throw std::runtime_error("cannot open output token file: " + path);
    }

    std::vector<int32_t> raw(tokens.begin(), tokens.end());

    if (!raw.empty()) {
        f.write(
            reinterpret_cast<const char*>(raw.data()),
            static_cast<std::streamsize>(raw.size() * sizeof(int32_t))
        );
    }

    if (!f) {
        throw std::runtime_error("failed writing output token file");
    }
}

static bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

static std::string build_chat_prompt(const std::string& user_text) {
    std::string prompt;

    prompt += "<|im_start|>user\n";
    prompt += user_text;
    prompt += "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n";

    return prompt;
}
// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(
            stderr,
            "Usage: %s model.tllm \"prompt\"|@prompt_tokens.bin "
            "[max_tokens] [temperature] [top_k] [out_tokens.bin] [stop_token] [tokenizer.json]\n",
            argv[0]
        );
        return 1;
    }

    try {
        Engine engine(argv[1]);

        const std::string prompt_arg = argv[2];
        const bool token_prompt = !prompt_arg.empty() && prompt_arg[0] == '@';

        std::string model_path = argv[1];
        std::string tokenizer_path = "";

        if (argc > 8) {
            tokenizer_path = argv[8];
        } else {
            std::string candidate_model_tokenizer =
                model_path + ".tokenizer/tokenizer.json";

            std::string candidate_local = "tokenizer.json";

            if (file_exists(candidate_model_tokenizer)) {
                tokenizer_path = candidate_model_tokenizer;
            } else if (file_exists(candidate_local)) {
                tokenizer_path = candidate_local;
            }
        }

        Tokenizer tokenizer;
        bool has_tokenizer = false;

        if (!tokenizer_path.empty()) {
            tokenizer.load(tokenizer_path);
            has_tokenizer = true;
        }

        int max_tokens = argc > 3 ? std::atoi(argv[3]) : 128;
        float temperature = argc > 4 ? static_cast<float>(std::atof(argv[4])) : 0.8f;
        int top_k = argc > 5 ? std::atoi(argv[5]) : 40;

        std::string out_tokens_path = argc > 6 ? argv[6] : "";
        int stop_token = argc > 7 ? std::atoi(argv[7]) : -1;

        if (max_tokens < 0) max_tokens = 0;
        if (top_k < 0) top_k = 0;

        std::random_device rd;
        std::mt19937 rng(rd());

        std::vector<int> input;

        if (token_prompt) {
            input = read_tokens_bin(prompt_arg.substr(1));
        } else {
            if (!has_tokenizer) {
                throw std::runtime_error(
                    "tokenizer.json not found; cannot encode raw text prompt"
                );
            }

            bool chat_mode = false;

            const char* chat_env = std::getenv("TLLM_CHAT");

            if (chat_env != nullptr && std::string(chat_env) == "1") {
                chat_mode = true;
            }

            std::string final_prompt = prompt_arg;

            if (chat_mode) {
                final_prompt = build_chat_prompt(prompt_arg);
            }

            input = tokenizer.encode(final_prompt);
        }

        if (input.empty()) {
            throw std::runtime_error("no input tokens");
        }

        if (input.size() >= engine.cfg.ctx_len) {
            input.resize(engine.cfg.ctx_len - 1);
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        const float* logits = nullptr;

        if (input.size() == 1) {
            logits = engine.forward(input[0]);
        } else {
            logits = engine.prefill(input);
        }

        auto t_prompt_end = std::chrono::high_resolution_clock::now();

        int next_token = sample(
            logits,
            engine.cfg.vocab_size,
            temperature,
            top_k,
            rng
        );

        std::vector<int> generated;

        for (int step = 0; step < max_tokens && engine.pos < engine.cfg.ctx_len; ++step) {
            if (next_token < 0) break;

            if (stop_token >= 0 && next_token == stop_token) {
                break;
            }

            generated.push_back(next_token);

            if (!has_tokenizer) {
                if (token_prompt) {
                    std::printf("%d ", next_token);
                } else if (next_token < 256) {
                    unsigned char c = static_cast<unsigned char>(next_token);
                    if (c != 0) {
                        std::putchar(c);
                    }
                } else {
                    std::printf("[%d]", next_token);
                }
            }

            std::fflush(stdout);

            if (engine.pos >= engine.cfg.ctx_len) {
                break;
            }

            logits = engine.forward(next_token);

            next_token = sample(
                logits,
                engine.cfg.vocab_size,
                temperature,
                top_k,
                rng
            );
        }

        if (!out_tokens_path.empty()) {
            write_tokens_bin(out_tokens_path, generated);
        }

        if (has_tokenizer) {
            std::string text = tokenizer.decode(generated);
            std::fwrite(text.data(), 1, text.size(), stdout);
        }

        std::printf("\n");

        //Eval Info print
        auto t_end = std::chrono::high_resolution_clock::now();

        long long prompt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_prompt_end - t_start).count();
        long long gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_prompt_end).count();

        double prompt_s = prompt_ms / 1000.0;
        double gen_s = gen_ms / 1000.0;

        std::fprintf(stderr, "\n");
        std::fprintf(stderr, "prompt eval time: %lld ms / %zu tokens = %.2f ms per token\n",
                     prompt_ms, input.size(), input.size() > 0 ? (double)prompt_ms / input.size() : 0.0);
        std::fprintf(stderr, "      eval time: %lld ms / %zu tokens = %.2f tokens per second\n",
                     gen_ms, generated.size(), gen_s > 0.0 ? (double)generated.size() / gen_s : 0.0);
        std::fprintf(stderr, "     total time: %lld ms\n", prompt_ms + gen_ms);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    return 0;
}