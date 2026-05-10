"""
Evaluate an SFT'd model on the held-out MCQ jsonl by likelihood-scoring each
of the 4 options (A/B/C/D) and comparing to the gold label.

This is the "log-likelihood" eval style used by lm-evaluation-harness — much
more robust than parsing free-form output. We compute log-prob of each option
text given the prompt and pick the highest.

Usage:
    python eval_mcq.py --ckpt out/plan_b_sft/ckpt_sft_final.pt --preset plan_b \
        --eval raw_data/mcq_eval.jsonl
"""
import argparse
import json
from collections import defaultdict

import torch
from transformers import AutoTokenizer

from configs import PRESETS
from model import LLM


LETTERS = ["A", "B", "C", "D"]


def score_option(model, tok, device, prompt, option_text):
    """Return log p(option_text | prompt)."""
    full = prompt + option_text
    p_ids = tok.encode(prompt, add_special_tokens=False)
    f_ids = tok.encode(full, add_special_tokens=False)
    new_ids = f_ids[len(p_ids):]
    if not new_ids:
        return -1e9
    inp = torch.tensor([f_ids[:-1]], device=device)
    tgt = torch.tensor([f_ids[1:]], device=device)
    with torch.no_grad():
        logits, _ = model(inp)
    logp = torch.log_softmax(logits.float(), dim=-1)
    # only score the option positions
    start = len(p_ids) - 1  # logits at position i predict token i+1
    sliced = logp[0, start:start + len(new_ids)]
    score = sliced.gather(1, torch.tensor(new_ids, device=device).unsqueeze(1)).sum().item()
    return score / max(1, len(new_ids))  # length-normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    mcfg, _ = PRESETS[args.preset]
    model = LLM(mcfg).to(device, dtype=dtype)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state); model.eval()
    print(f"loaded {args.ckpt} on {device}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    items = [json.loads(l) for l in open(args.eval)]
    if args.limit:
        items = items[:args.limit]

    SYS = "You are a helpful assistant."
    correct = 0
    by_cat = defaultdict(lambda: [0, 0])  # [right, total]

    for i, ex in enumerate(items):
        raw = ex["_raw"]
        opts = raw["options"]
        gold = raw["correct_index"]
        cat = raw.get("category", "other")
        user = ex["messages"][0]["content"]
        prompt = (
            f"<|system|>\n{SYS}<|end|>\n<|user|>\n{user}<|end|>\n"
            f"<|assistant|>\nThe answer is ("
        )
        scores = []
        for L, o in zip(LETTERS, opts):
            cand = f"{L}) {o}"
            scores.append(score_option(model, tok, device, prompt, cand))
        pred = scores.index(max(scores))
        ok = pred == gold
        correct += int(ok)
        by_cat[cat][1] += 1
        by_cat[cat][0] += int(ok)
        if i < 5 or (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(items)}] acc={correct/(i+1):.3f} | last gold={LETTERS[gold]} pred={LETTERS[pred]}")

    print(f"\n=== overall: {correct}/{len(items)} = {correct/len(items):.3f} ===")
    print("\nby category:")
    for c in sorted(by_cat):
        r, t = by_cat[c]
        print(f"  {c:<18} {r:>4}/{t:<4} = {r/t:.3f}")


if __name__ == "__main__":
    main()
