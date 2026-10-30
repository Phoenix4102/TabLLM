#!/usr/bin/env python3

import argparse

import numpy as np
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", default="prompt_tokens.bin")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--bos", action="store_true")
    ap.add_argument("--eos-file", default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code
    )

    if args.chat:
        messages = [
            {
                "role": "user",
                "content": args.prompt,
            }
        ]

        try:
            text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            ids = tok(
                text,
                add_special_tokens=False
            ).input_ids

        except Exception:
            ids = tok(
                args.prompt,
                add_special_tokens=args.bos
            ).input_ids
    else:
        ids = tok(
            args.prompt,
            add_special_tokens=args.bos
        ).input_ids

    arr = np.array(ids, dtype=np.dtype("<i4"))
    arr.tofile(args.out)

    if args.eos_file:
        eos = tok.eos_token_id
        if eos is None:
            eos = -1

        with open(args.eos_file, "w") as f:
            f.write(str(eos))

    print(f"wrote {len(ids)} tokens to {args.out}")

    if tok.eos_token_id is not None:
        print("eos_token_id:", tok.eos_token_id)


if __name__ == "__main__":
    main()