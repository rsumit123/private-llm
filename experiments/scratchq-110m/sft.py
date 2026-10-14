"""
Supervised Fine-Tuning (SFT) on instruction data.

Loads a pretrained checkpoint, then trains on (prompt, response) pairs using
the ChatML chat template. Loss is masked on the prompt — only response tokens
contribute to the gradient, so the model learns to *answer* rather than continue.

Usage:
    python sft.py --base out/plan_b_110m/ckpt_0010500.pt --preset plan_b \
                  --out out/plan_b_sft --epochs 2

Input data: HuggingFaceH4/ultrachat_200k (filtered, conversational).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from configs import PRESETS
from model import LLM


# --- ChatML format -----------------------------------------------------------
# Standard format used by many open models. The special tokens delimit roles
# so the model knows where one turn ends and another begins.
SYS = "You are a helpful assistant."
B_USR, B_ASST, EOT = "<|user|>\n", "<|assistant|>\n", "<|end|>"


def build_example(messages, tok, max_len):
    """Tokenize one conversation; return ids + loss mask (1 = train on this token)."""
    ids, mask = [], []
    # System prompt — no loss
    sys = f"<|system|>\n{SYS}{EOT}\n"
    sys_ids = tok.encode(sys, add_special_tokens=False)
    ids += sys_ids; mask += [0] * len(sys_ids)

    for m in messages:
        role, content = m["role"], m["content"]
        if role == "user":
            text = f"{B_USR}{content}{EOT}\n"
            t_ids = tok.encode(text, add_special_tokens=False)
            ids += t_ids; mask += [0] * len(t_ids)
        elif role == "assistant":
            head = tok.encode(B_ASST, add_special_tokens=False)
            body = tok.encode(f"{content}{EOT}\n", add_special_tokens=False)
            ids += head; mask += [0] * len(head)              # role tag: no loss
            ids += body; mask += [1] * len(body)              # response: train here

    ids = ids[:max_len]; mask = mask[:max_len]
    return ids, mask


def collate(batch, pad_id, max_len):
    L = max(len(x[0]) for x in batch)
    L = min(L, max_len)
    x = np.full((len(batch), L), pad_id, dtype=np.int64)
    m = np.zeros((len(batch), L), dtype=np.int64)
    for i, (ids, mask) in enumerate(batch):
        n = min(len(ids), L)
        x[i, :n] = ids[:n]
        m[i, :n] = mask[:n]
    return torch.from_numpy(x), torch.from_numpy(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="path to pretrained ckpt .pt")
    ap.add_argument("--preset", required=True, choices=list(PRESETS.keys()))
    ap.add_argument("--out", default="out/sft")
    ap.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-5)  # 10x smaller than pretrain
    ap.add_argument("--max_examples", type=int, default=20000)
    ap.add_argument("--extra_jsonl", default=None,
                    help="optional jsonl of {messages:[...]} examples to mix with ultrachat")
    ap.add_argument("--ultrachat_count", type=int, default=10000,
                    help="how many ultrachat examples to stream")
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    mcfg, _ = PRESETS[args.preset]
    model = LLM(mcfg).to(device, dtype=dtype)
    state = torch.load(args.base, map_location=device)["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    print(f"loaded base: {args.base} ({model.num_params()/1e6:.1f}M non-emb params)")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    pad_id = tok.eos_token_id

    examples = []

    # Local jsonl first (e.g. our MCQs) — small, deterministic, in-domain
    if args.extra_jsonl:
        print(f"loading extra jsonl: {args.extra_jsonl}")
        with open(args.extra_jsonl) as f:
            for line in f:
                obj = json.loads(line)
                ids, mask = build_example(obj["messages"], tok, args.max_len)
                if sum(mask) >= 4:
                    examples.append((ids, mask))
        print(f"  → {len(examples)} extra examples loaded")

    if args.ultrachat_count > 0:
        print(f"streaming ultrachat_200k (target {args.ultrachat_count})…")
        ds_iter = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        added = 0
        for ex in ds_iter:
            ids, mask = build_example(ex["messages"], tok, args.max_len)
            if sum(mask) >= 16:
                examples.append((ids, mask))
                added += 1
            if added >= args.ultrachat_count:
                break
        print(f"  → {added} ultrachat examples loaded")

    examples = examples[:args.max_examples]
    print(f"prepared {len(examples)} examples total")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                            betas=(0.9, 0.95), fused=True)
    rng = np.random.default_rng(42)
    model.train()

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        order = rng.permutation(len(examples))
        for i in range(0, len(order), args.batch_size):
            batch = [examples[j] for j in order[i:i + args.batch_size]]
            x, m = collate(batch, pad_id, args.max_len)
            x, m = x.to(device), m.to(device)
            y = x.clone()
            y[m == 0] = -100  # CrossEntropy ignore_index — these tokens contribute 0 loss

            with torch.amp.autocast("cuda", dtype=dtype):
                _, loss = model(x[:, :-1], y[:, 1:], grad_checkpoint=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 20 == 0:
                dt = time.time() - t0; t0 = time.time()
                print(f"epoch {epoch} step {step:5d} | loss {loss.item():.4f} | {dt:.1f}s/20step")
            step += 1

    ckpt = out_dir / "ckpt_sft_final.pt"
    torch.save({"model": model.state_dict(), "step": step, "cfg": {"model": mcfg}}, ckpt)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
