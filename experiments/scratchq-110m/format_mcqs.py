"""
Convert gk_questions.json (MCQ format) into ChatML SFT training pairs.

Schema in:
    {"id":..., "category":..., "difficulty": 1|2|3,
     "question": str, "options": [4 strings], "correct_index": 0..3}

Output (jsonl, one example per line):
    {"messages": [{"role":"user", "content": "..."},
                  {"role":"assistant", "content": "..."}]}

Usage:
    python format_mcqs.py --src raw_data/mcqs/gk_questions.json \
        --train_out raw_data/mcqs/train.jsonl --eval_out raw_data/mcqs/eval.jsonl \
        --eval_size 1000
"""
import argparse
import json
import random
from pathlib import Path

LETTERS = ["A", "B", "C", "D"]


def to_messages(q):
    opts = q["options"]
    correct = q["correct_index"]
    user = (
        f"Question ({q['category']}): {q['question']}\n"
        + "\n".join(f"({L}) {o}" for L, o in zip(LETTERS, opts))
    )
    assistant = f"The answer is ({LETTERS[correct]}) {opts[correct]}."
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--train_out", required=True)
    ap.add_argument("--eval_out", required=True)
    ap.add_argument("--eval_size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = json.load(open(args.src))
    print(f"loaded {len(raw)} mcqs")

    random.Random(args.seed).shuffle(raw)
    eval_set = raw[:args.eval_size]
    train_set = raw[args.eval_size:]

    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.train_out, "w") as f:
        for q in train_set:
            f.write(json.dumps(to_messages(q), ensure_ascii=False) + "\n")
    with open(args.eval_out, "w") as f:
        for q in eval_set:
            # store both formatted + raw, useful for grading
            ex = to_messages(q)
            ex["_raw"] = q
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"wrote train={len(train_set)} → {args.train_out}")
    print(f"wrote eval={len(eval_set)} → {args.eval_out}")


if __name__ == "__main__":
    main()
