#!/usr/bin/env python3

import argparse

import numpy as np
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--keep-special", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code
    )

    ids = np.fromfile(
        args.input,
        dtype=np.dtype("<i4")
    ).tolist()

    text = tok.decode(
        ids,
        skip_special_tokens=not args.keep_special
    )

    print(text)


if __name__ == "__main__":
    main()