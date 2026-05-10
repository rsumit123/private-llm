"""
Convert a sample of SQuAD-v2 into our SFT messages format.

Each (context, question, answer) becomes:
    user:      Read the passage and answer the question.
               <context>
               Question: <question>
    assistant: <answer>.

This is exactly the format-4 we use in augment_sft.py and at inference time
in gradio with RAG. Training on this directly teaches the model:
    "given a passage that contains the answer, copy the right span."

Usage:
    python build_squad_jsonl.py --out raw_data/squad.jsonl --n 30000
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--max_ctx_chars", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("loading SQuAD-v2 train…")
    ds = load_dataset("rajpurkar/squad_v2", split="train")
    print(f"  total: {len(ds):,}")

    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    out = []
    for i in indices:
        ex = ds[i]
        # SQuAD-v2 has unanswerable questions (empty answers) — skip those for SFT
        answers = ex["answers"]["text"]
        if not answers: continue
        ctx = ex["context"][:args.max_ctx_chars]
        q = ex["question"].strip()
        a = answers[0].strip()
        if len(a) < 1 or len(a) > 200: continue
        msg = {"messages": [
            {"role": "user",
             "content": f"Read the passage and answer the question.\n\n{ctx}\n\nQuestion: {q}"},
            {"role": "assistant", "content": a + "."}
        ]}
        out.append(msg)
        if len(out) >= args.n: break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} SQuAD examples → {args.out}")


if __name__ == "__main__":
    main()
