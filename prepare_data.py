"""
Download a slice of FineWeb-Edu, tokenize with Llama-2 tokenizer, write to bin shards.

Usage:
    python prepare_data.py --out data/fineweb_edu --target_tokens 200_000_000   # smoke
    python prepare_data.py --out data/fineweb_edu --target_tokens 30_000_000_000 # main
"""
import argparse
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

SHARD_TOKENS = 100_000_000  # 100M tokens per shard
DTYPE = np.uint16  # Llama-2 vocab=32000 fits in uint16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fineweb_edu")
    ap.add_argument("--target_tokens", type=int, default=200_000_000)
    ap.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-hf",
                    help="HF repo for tokenizer (gated; or pass a local path / NousResearch/Llama-2-7b-hf)")
    ap.add_argument("--subset", default="sample-10BT",
                    help="FineWeb-Edu subset: sample-10BT, sample-100BT, sample-350BT, default")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    eot = tok.eos_token_id
    print(f"tokenizer={args.tokenizer} vocab={tok.vocab_size} eos={eot}")

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=args.subset, split="train", streaming=True)

    shard_idx = 0
    buf = np.empty(SHARD_TOKENS, dtype=DTYPE)
    pos = 0
    total = 0
    pbar = tqdm(total=args.target_tokens, unit="tok", unit_scale=True)

    for ex in ds:
        ids = tok.encode(ex["text"], add_special_tokens=False)
        ids.append(eot)
        ids = np.asarray(ids, dtype=DTYPE)
        i = 0
        while i < len(ids):
            n = min(SHARD_TOKENS - pos, len(ids) - i)
            buf[pos:pos+n] = ids[i:i+n]
            pos += n; i += n; total += n; pbar.update(n)
            if pos == SHARD_TOKENS:
                fp = out / f"shard_{shard_idx:05d}.bin"
                buf.tofile(fp)
                shard_idx += 1; pos = 0
            if total >= args.target_tokens:
                break
        if total >= args.target_tokens:
            break

    if pos > 0:
        fp = out / f"shard_{shard_idx:05d}.bin"
        buf[:pos].tofile(fp)
        shard_idx += 1
    pbar.close()
    print(f"wrote {shard_idx} shards, {total:,} tokens to {out}")


if __name__ == "__main__":
    main()
