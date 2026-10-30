#!/usr/bin/env python3

import argparse
import os
import struct
import sys

import numpy as np   #<--- Modification
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="model.tllm")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--no-tokenizer", action="store_true")
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32") #<-- Modification
    args = ap.parse_args()

    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code
    )

    model_type = getattr(config, "model_type", "")
    if model_type not in ("llama", "mistral"):
        print(
            f"Warning: model_type={model_type}; "
            "this exporter assumes Llama-style tensor names.",
            file=sys.stderr,
        )

    vocab_size = int(config.vocab_size)
    hidden = int(config.hidden_size)
    layers = int(config.num_hidden_layers)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads) or heads)
    ffn = getattr(config, "intermediate_size", None)
    rope_theta = float(getattr(config, "rope_theta", 10000.0) or 10000.0)
    tie = bool(getattr(config, "tie_word_embeddings", False))
    # Modification for optim 1
    dtype_id = 0 if args.dtype == "fp32" else 2
    elem_bytes = 4 if dtype_id == 0 else 2

    max_pos = int(getattr(config, "max_position_embeddings", args.ctx) or args.ctx)
    ctx = max(2, min(args.ctx, max_pos))

    if hidden % heads != 0:
        sys.exit("hidden_size is not divisible by num_attention_heads")

    if heads % kv_heads != 0:
        sys.exit("num_attention_heads is not divisible by num_key_value_heads")

    head_dim = hidden // heads
    kv_dim = kv_heads * head_dim

    print(f"Loading {args.model} ...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )

    model.eval()
    sd = model.state_dict()

    bias_keys = [k for k in sd.keys() if k.endswith(".bias")]
    if bias_keys:
        sys.exit(
            "This engine does not support biases. Found: "
            + ", ".join(bias_keys[:5])
        )

    if ffn is None:
        ffn = int(sd["model.layers.0.mlp.gate_proj.weight"].shape[0])

    ffn = int(ffn)

    entries = []

    def add(name, expected_shape):
        if name not in sd:
            sys.exit(f"Missing tensor: {name}")

        shape = tuple(sd[name].shape)

        if shape != expected_shape:
            sys.exit(
                f"Tensor {name} has shape {shape}, expected {expected_shape}"
            )

        entries.append(name)
        return sd[name].numel() * elem_bytes #<--- Modification

    total = 0

    total += add("model.embed_tokens.weight", (vocab_size, hidden))

    for i in range(layers):
        base = f"model.layers.{i}"

        total += add(f"{base}.input_layernorm.weight", (hidden,))
        total += add(f"{base}.self_attn.q_proj.weight", (hidden, hidden))
        total += add(f"{base}.self_attn.k_proj.weight", (kv_dim, hidden))
        total += add(f"{base}.self_attn.v_proj.weight", (kv_dim, hidden))
        total += add(f"{base}.self_attn.o_proj.weight", (hidden, hidden))
        total += add(f"{base}.post_attention_layernorm.weight", (hidden,))
        total += add(f"{base}.mlp.gate_proj.weight", (ffn, hidden))
        total += add(f"{base}.mlp.up_proj.weight", (ffn, hidden))
        total += add(f"{base}.mlp.down_proj.weight", (hidden, ffn))

    total += add("model.norm.weight", (hidden,))

    if not tie:
        if "lm_head.weight" in sd:
            total += add("lm_head.weight", (vocab_size, hidden))
        else:
            print(
                "Warning: lm_head.weight missing; forcing tie_word_embeddings=1",
                file=sys.stderr,
            )
            tie = True

    header = struct.pack(
        "<4s8If2IQQ",
        b"TLLM",
        1,
        vocab_size,
        hidden,
        layers,
        heads,
        kv_heads,
        ffn,
        ctx,
        rope_theta,
        dtype_id, #<--- Modification
        int(tie),
        total,
        0,
    )

    print(f"Writing {args.out} ...")

    with open(args.out, "wb") as f:
        f.write(header)

        for name in entries:
            t = sd[name]
            arr = t.detach().cpu().float().numpy()
            if dtype_id == 2:                               #<---
                arr = arr.astype(np.dtype("<f2"))           #<--- Modifications
            f.write(arr.tobytes())

    print(f"Wrote {args.out}: {os.path.getsize(args.out)} bytes")

    if not args.no_tokenizer:
        try:
            tok = AutoTokenizer.from_pretrained(
                args.model,
                trust_remote_code=args.trust_remote_code
            )

            tok_dir = args.out + ".tokenizer"
            os.makedirs(tok_dir, exist_ok=True)
            tok.save_pretrained(tok_dir)

            import json

            with open(os.path.join(tok_dir, "special_tokens.json"), "w") as f:
                json.dump(
                    {
                        "bos_token_id": tok.bos_token_id,
                        "eos_token_id": tok.eos_token_id,
                        "pad_token_id": tok.pad_token_id,
                    },
                    f,
                    indent=2,
                )

            print(f"Saved tokenizer to {tok_dir}")

        except Exception as e:
            print(f"Warning: tokenizer export failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()