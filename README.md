# TabLLM: From-Scratch C++ LLM Inference Engine for Android

A lightweight, high-performance, **from-scratch C++ inference engine** for decoder-only Transformer language models, designed specifically for mobile hardware.

This project contains **zero external machine learning frameworks** (no PyTorch, TensorFlow, ONNX, or llama.cpp). It is built entirely in C++20 with ARM NEON optimizations, and includes a native Android application featuring live token streaming, dynamic model switching, and runtime hyperparameter tuning.

---

##  Project Overview

This project documents the complete journey of building a local LLM inference engine from absolute zero, culminating in a native Android application with live streaming output, dynamic model switching, and runtime hyperparameter control.

**No ML frameworks were used.** Every kernel, every attention mask, every quantization block, and every tokenizer merge was written by hand in C++20 with ARM NEON intrinsics.

**Test Device**: Samsung Galaxy Tab S10+ (MediaTek Dimensity 9300+, ARMv9, Cortex-X4/A720 cores)

---

## Feature Summary

| Category | Implementation |
|---|---|
| **Engine** | Pure C++20, no external ML dependencies |
| **Architecture** | Llama-style: RMSNorm, RoPE, GQA, SwiGLU, KV Cache |
| **Precision** | FP32 → FP16 → Q4 (mixed-precision) |
| **Quantization** | 4-bit block quantization (18 bytes / 32 weights) |
| **Kernels** | ARM NEON microkernels for FP32, FP16, Q4 |
| **Prefill** | Batched GEMM (all prompt tokens in one pass) |
| **Generation** | Autoregressive GEMV with KV cache |
| **Tokenizer** | Native C++ BPE encoder + decoder |
| **Android** | JNI bridge, Kotlin UI, live streaming |
| **Model Format** | Custom `.tllm` binary (memory-mapped) |

---

## Architecture

### Custom Binary Format (`.tllm`)

The engine uses a custom, memory-mapped binary format designed for zero-copy loading on Android:

```text
┌─────────────────────────────────────────────────┐
│ Header (64 bytes)                               │
│   ├── Magic ("TLLM"), Version                   │
│   ├── vocab_size, hidden_size, num_layers       │
│   ├── num_heads, num_kv_heads, ffn_hidden_size  │
│   ├── ctx_len, rope_theta                       │
│   ├── dtype (0=fp32, 2=fp16, 4=q4)              │
│   ├── tie_lm_head                               │
│   └── tensor_bytes                              │
├─────────────────────────────────────────────────┤
│ Embedding Table (FP16 in Q4 mode)               │
├─────────────────────────────────────────────────┤
│ Per Layer:                                      │
│   ├── input_layernorm.weight (FP16)             │
│   ├── q_proj.weight (Q4 blocks)                 │
│   ├── k_proj.weight (Q4 blocks)                 │
│   ├── v_proj.weight (Q4 blocks)                 │
│   ├── o_proj.weight (Q4 blocks)                 │
│   ├── post_attention_layernorm.weight (FP16)    │
│   ├── gate_proj.weight (Q4 blocks)              │
│   ├── up_proj.weight (Q4 blocks)                │
│   └── down_proj.weight (Q4 blocks)              │
├─────────────────────────────────────────────────┤
│ Final Norm (FP16)                               │
├─────────────────────────────────────────────────┤
│ LM Head (Q4 or tied FP16)                       │
└─────────────────────────────────────────────────┘
```

---

## Development Milestones
This project was built incrementally, with each milestone adding a critical capability.

---

### Milestone 1 — Core Engine + External Tokenization

**Objective**: Prove the C++ engine can run real open-source weights end-to-end.

**Deliverables**:
- Custom `.tllm` binary format with 64-byte header (magic, dimensions, dtype, tie flag, tensor offset).
- Full Transformer forward pass: embedding lookup → per-layer RMSNorm → GQA attention with RoPE → residual → SwiGLU FFN → final norm → LM head.
- KV cache with causal masking for autoregressive generation.
- FP32 matrix-vector kernels with OpenMP threading.
- Hugging Face → `.tllm` export pipeline (Python + PyTorch).
- Byte-level tokenizer bridge: Python encodes prompt → `prompt_tokens.bin` → C++ generates → `out_tokens.bin` → Python decodes.

**Workflow**:

[Python encode] → prompt_tokens.bin
↓
[C++ Engine forward]
↓
out_tokens.bin → [Python decode] → text


**Limitations**: Two Python round-trips per inference. Not standalone.

---

### Milestone 2A — Native C++ Token Decoding

**Objective**: Eliminate the Python decode step. Generate text directly in the shell.

**Deliverables**:
- C++ `Tokenizer` class that parses `tokenizer.json` at runtime.
- Vocab table (token string → ID, ID → token string).
- Byte-level decoder: maps BPE token IDs back to UTF-8 byte sequences.
- Special token handling (EOS, BOS, chat markers).
- Engine prints decoded text directly to stdout — no `out_tokens.bin`, no Python.

**Workflow**:

[Python encode] → prompt_tokens.bin
↓
[C++ Engine forward + decode]
↓
text (stdout)


**Result**: Output path fully native. Only prompt encoding still required Python.

---

### Milestone 2B — Native C++ Token Encoding (Full Standalone)

**Objective**: Eliminate Python entirely. Pure text-in → text-out in C++.

**Deliverables**:
- BPE encoder in C++: UTF-8 text → byte-level symbols → greedy merge loop using merge ranks → token IDs.
- Merge rank table parsed from `tokenizer.json`.
- Special token pre-splitting (chat template tokens handled before BPE).
- Chat template formatting built into the engine (`<|im_start|>user\n...`).
- CLI accepts raw string prompts directly.

**Workflow**:

"./tabllm model.tllm "What is the capital of France?""
↓
[C++ encode → forward → decode]
↓
text (stdout)


**Result**: Zero Python dependencies at inference time. Fully standalone binary.

---

### Milestone 3 — Performance Optimization Pipeline

Starting from the baseline FP32 engine, four sequential optimizations were applied. Each was measured independently on-device.

#### Baseline (FP32, single-token prefill, no quantization)

| Metric | Value |
|---|---|
| Model | SmolLM2-135M-Instruct |
| Precision | FP32 (32-bit float) |
| Model Size | ~538 MB |
| Prompt Eval | ~39 ms/token |
| Generation | ~62 tokens/sec |

**Observation**: Engine is memory-bandwidth bound. CPU spends most cycles waiting for weights to arrive from RAM.

---

#### Optimization 1 — FP16 Weights

**Change**: Store all weights as IEEE 754 half-precision (16-bit). Convert to FP32 on-the-fly during dot products using NEON `vcvt_f32_f16`.

| Metric | FP32 (Baseline) | FP16 | Change |
|---|---|---|---|
| Model Size | 538 MB | ~269 MB | **-50%** |
| Prompt Eval | 39 ms/tok | 9.25 ms/tok | **4.2× faster** |
| Generation | 62 tok/s | 121 tok/s | **1.95× faster** |

**Why it works**: Halving weight size halves memory bandwidth pressure. NEON FP16→FP32 conversion is nearly free on Cortex-X4.

---

#### Optimization 2 — Batched Prompt Prefill

**Change**: Replace per-token GEMV prefill with batched GEMM. All prompt tokens are embedded into a `[H × S]` column-major matrix and processed through all layers in one pass.

**Implementation**:
- Column-major activation layout `[H × S]` for cache-friendly GEMM.
- Batched RMSNorm across all positions simultaneously.
- Batched RoPE with per-position angle computation.
- Causal attention computed per query position against cached K/V.
- LM head applied only to the final position's hidden state.

| Metric | Single-Token Prefill | Batched Prefill | Change |
|---|---|---|---|
| 16-token prompt | 628 ms | 148 ms | **4.2× faster** |
| 36-token prompt | ~1,400 ms | 192 ms | **7.3× faster** |
| Per-token cost | ~39 ms | ~5.3 ms | **7.4× cheaper** |

**Why it works**: Weights are fetched from RAM once and applied to all S positions, instead of S separate fetches.

---

#### Optimization 3 — 4-Bit Block Quantization (Q4)

**Change**: Quantize linear layer weights (Q, K, V, O, Gate, Up, Down) to 4-bit integers in blocks of 32. Norms and embeddings remain FP16 (mixed precision).

**Block format**:
```cpp
struct BlockQ4 {
    uint16_t d;       // FP16 scale factor
    uint8_t qs[16];   // 32 × 4-bit weights packed into 16 bytes
};  // 18 bytes per 32 weights = 7.1× compression vs FP32
```
**Quantization formula:**
```text
scale = max(|weights|) / 7.0
q = round(w / scale) + 8        → range [0, 15]
w_dequant = (q - 8) * scale
```

**Mixed-precision strategy (critical for quality):**

- `embed_tokens.weight` → FP16 (embedding lookup is precision-sensitive)
- `*.norm.weight` → FP16 (normalization is precision-sensitive)
- `q/k/v/o/gate/up/down_proj.weight` → Q4 (bulk of parameters, tolerant to quantization)
- `lm_head.weight` → Q4 if untied, FP16 if tied to embeddings

| Metric | FP16 | Q4 Mixed | Change |
|---|---|---|---|
| Model Size | ~269 MB | ~85 MB | **-68%** |
| Generation | 121 tok/s | ~54 tok/s | 0.45x (scalar unpack) |
| Text Quality | Baseline | Near-identical | ✓ |

**Note:** Generation speed dropped because the scalar Q4 unpacking loop (bitwise AND, shift, subtract, cast) became compute-bound. This is fixed in Optimization 4.

---

#### Optimization 4 — NEON Q4 Microkernels

**Change:** Replace the scalar Q4 dot-product with a fully vectorized NEON kernel.

**Kernel strategy:**

1. `vld1q_u8` — load 16 packed bytes (32 weights) in one instruction.
1. `vandq_u8` + `vshrq_n_u8` — extract lower and upper 4-bit nibbles.
1. `vzip1q_u8` / `vzip2q_u8` — interleave into correct weight order.
1. `vsubq_s8` — subtract 8 (convert unsigned [0,15] → signed [-8,7]).
1. `vmovl_s8` → `vmovl_s16` → `vcvtq_f32_s32` — widen to FP32.
1. `vfmaq_f32` — fused multiply-accumulate with activations.

| Metric | Q4 Scalar | Q4 NEON | Change |
|---|---|---|---|
| Generation | ~54 tok/s | ~75 tok/s | **1.4x faster** |
| Prompt Eval | ~6 ms/tok | ~4.4 ms/tok | **1.3x faster** |
| Text Quality | Identical | Identical | ✓ |

---

### Milestone 4 — Scale Up: SmolLM2-1.7B-Instruct

**Objective:** Prove the engine handles genuinely intelligent, billion-parameter models within mobile RAM constraints.

**Model:** HuggingFaceTB/SmolLM2-1.7B-Instruct
**Export:** --dtype q4 --ctx 2048

| Metric | Value |
|---|---|
| Model Size (Q4) | ~1.1 GB |
| Prompt Eval | ~60–99 ms/token |
| Generation | ~6.6 tokens/sec |
| Text Quality | Coherent, instruction-following, factual |

**Key insight:** FP16 would require ~3.4 GB for this model — likely triggering OOM or heavy swapping. Q4 fits comfortably in ~1.1 GB, demonstrating exactly why quantization is essential for mobile LLMs.

---

### TabLLMApp — Native Android Application

**Objective:** Ship the engine as a real, installable Android app with UI.

**Stack:**

```text
┌─────────────────────────────────────────┐
│           Kotlin UI (MainActivity)      │
│  ┌───────────────────────────────────┐  │
│  │ Model Spinner (auto-scan .tllm)   │  │
│  │ Hyperparameter Controls           │  │
│  │   Max Tokens / Temperature        │  │
│  │   Top-K / Top-P / Rep. Penalty    │  │
│  │ Prompt Input                      │  │
│  │ Streaming Output TextView         │  │
│  └───────────────────────────────────┘  │
├─────────────────────────────────────────┤
│         JNI Bridge (native-lib.cpp)     │
│   initEngine() / generate()             │
│   StreamCallback → onToken/onComplete   │
├─────────────────────────────────────────┤
│         C++ Engine (engine.cpp)         │
│   Tokenizer / Engine / Sampler          │
│   NEON Kernels / Q4 / FP16 / FP32       │
└─────────────────────────────────────────┘
```

**Features:**
- **Dynamic Model Switching:** Scans app directory for all `.tllm` files, populates a dropdown. No recompilation needed to swap models.
- **Runtime Hyperparameters:** Max Tokens, Temperature, Top-K, Top-P, Repetition Penalty — all adjustable per generation.
- **Live Streaming:** C++ pushes decoded tokens to Kotlin via JNI callback on every generation step. Text appears word-by-word.
- **Repetition Penalty:** Penalizes previously generated tokens to prevent looping (critical for smaller models).
- **Performance Stats:** TTFT, generation time, tokens/sec, and total time appended to every response.
- **16KB Page Alignment:** Linker flag `-Wl,-z,max-page-size=16384` for Android 15+ compatibility.
- **Static OpenMP:** `-static-openmp` eliminates runtime `libomp.so` dependency.

---

###  Build Instructions

**Pre-requisites:**

| Tool | Version | Purpose |
|---|---|---|
| Android NDK | r26+ (tested: 30.0.16248370) | Cross-compile C++ for ARM64 |
| CMake | 3.22+ | Build system |
| Python | 3.8+ | Model export |
| PyTorch + Transformers | Latest | Weight loading & export |
| JDK | 17 | Android app compilation |
| Gradle | 8.2+ | Android app build |
| Android SDK | API 34 | App packaging |

**1. Export a Model**
```bash
pip install torch transformers numpy

python scripts/export_hf_to_tllm.py \
  --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --out models/smollm2-1.7b-q4.tllm \
  --ctx 2048 \
  --dtype q4
```
Supported `--dtype` values: `fp32`, `fp16`, `q4`.

**2. Build the CLI Engine (for adb testing)**

```bash
$NDK = "$env:ANDROID_HOME\ndk\30.0.16248370"
$toolchain = "$NDK\build\cmake\android.toolchain.cmake"
```
After `cd`ing inside each milestone folder:

```bash
cmake -B build-android -S engine -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE="$toolchain" `
  -DANDROID_ABI=arm64-v8a `
  -DANDROID_PLATFORM=android-26 `
  -DANDROID_STL=c++_static `
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-android -j
```

**3. Build the Android App**

```bash
cd TabLLMApp
gradle wrapper
.\gradlew assembleDebug
```

APK output: `app/build/outputs/apk/debug/app-debug.apk`

```bash
# Install app
adb install app/build/outputs/apk/debug/app-debug.apk

# Push models to app-private storage (no permissions needed)
adb push smollm2-1.7b-q4.tllm /sdcard/Android/data/com.example.tabllm/files/
adb push smollm2-135m-q4.tllm /sdcard/Android/data/com.example.tabllm/files/
adb push smollm2-135m-q4.tllm.tokenizer/tokenizer.json /sdcard/Android/data/com.example.tabllm/files/
```
> **Samsung devices:** Add TabLLM to `Settings` → `Battery` → `Background usage limits` → `Never sleeping apps` to prevent One UI from deleting model files.

---

### Performance Benchmarks

**Device:** Samsung Galaxy Tab S10+ — MediaTek Dimensity 9300+ (1× Cortex-X4 @ 3.25 GHz, 3× Cortex-X4 @ 2.85 GHz, 4× Cortex-A720 @ 2.0 GHz)

**SmolLM2-135M-Instruct**

| Config | Model Size | Prompt Eval | Generration |
|---|---|---|---|
| FP32 (Baseline) | 538 MB | 39.25 ms/tok | 62.62 tok/s |
| FP16|269 MB | 9.25 ms/tok | 121.21 tok/s |
| FP16 + Batched Prefill (16 tok) | 269 MB | 5.19 ms/tok | 108.55 tok/s |
| FP16 + Batched Prefill (36 tok) | 269 MB | 4.33 ms/tok | 108.35 tok/s |
| Q4 Scalar | ~85 MB | 5.56 ms/tok | 53.78 tok/s |
| **Q4 + NEON** | **~85 MB** | **4.40 ms/tok** | **75.29 tok/s** |

**SmolLM2-1.7B-Instruct (Q4 + NEON)**

| Metric | Value |
|---|---|
| Model Size | ~1.1 GB |
| Prompt Eval (16–19 tokens) | 61–99 ms/tok |
| Generation | ~6.6–6.7 tok/s |
| Total Time (96 tokens) | ~7.5 s |

**Optimization Speedup Summary (135M model)**

```text
FP32 Baseline          ██████                          62 tok/s
  + FP16               ████████████                   121 tok/s  (1.95×)
  + Batched Prefill    ████████████                   108 tok/s  (prefill 7× faster)
  + Q4 Scalar          █████                           54 tok/s  (0.45× — unpack overhead)
  + Q4 NEON            ████████                        75 tok/s  (1.4× over scalar Q4)

```
> **Note:** Q4 trades ~38% generation speed vs FP16 for a 68% reduction in model size. This tradeoff is what makes 1.7B+ models feasible on mobile RAM.

---

### Runtime Hyperparameters
All parameters are adjustable in-app without recompilation:

| Parameter | Range | Default | Effect |
|---|---|---|---|
| Max Tokens | 1–4096 | 128 | Maximum generation length |
| Temperature | 0.0–2.0 | 0.7 | Randomness (0 = greedy) |
| Top-K | 0–vocab | 40 | Restrict to top K candidates |
| Top-P | 0.0–1.0 | 0.9 | Nucleus sampling threshold |
| Repetition Penalty | 1.0–2.0 | 1.1 | Penalize repeated tokens |
 
---

###  Known Issues & Solutions

| Issue | Cause | Solution |
|---|---|---|
| `library "libomp.so" not found` | OpenMP linked dynamically | Use `-static-openmp` in CMake|
| 16KB alignment error on Android 15+ | Default 4KB ELF alignment | Add `-Wl,-z,max-page-size=16384` |
| Samsung deletes model files | One UI auto-clean of `Android/data` | Whitelist app in Battery settings |
| `Unresolved reference: layout` in Kotlin | Missing `R` import | Add `import com.example.tabllm.R` |
| Repetitive output on 135M model after quantization | Small model capacity | Increase Rep. Penalty to 1.15–1.2 |
| Q4 slower than FP16 | Scalar unpack overhead | Use NEON Q4 microkernels |

---

### Future Work

- Vulkan compute shaders for GPU-accelerated GEMM
- INT8 KV cache (halve context memory)
- Q5 / Q6 quantization tiers
- Speculative decoding (draft model + verify)
- Multi-modal support (vision encoder)
- Model download manager (in-app HuggingFace Hub)
- Conversation memory (multi-turn context)

---

### Acknowledgments
- **Hugging Face** — SmolLM2 model weights and tokenizer
- **ARM** — NEON intrinsic documentation
- **MediaTek** — Dimensity 9300+ silicon
- **Samsung** — Samsung Galaxy Tab S10+
- **Nlohmann JSON** — JSON for Modern C++

---
Built from scratch. No frameworks. No shortcuts. Just C++ and math.
