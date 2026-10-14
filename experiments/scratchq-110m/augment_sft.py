"""
Take our MCQ jsonl and produce a multi-format SFT training file. The goal is
to teach the model to handle questions without options (what real users type)
and to produce statement-form answers (what looks coherent), in addition to
the original MCQ format.

Each MCQ becomes 3 training examples:
  1. Original MCQ: "Question + (A)(B)(C)(D)" → "The answer is (X) text."
  2. Open-question: "Q: question" → "answer text."
  3. Statement: "question" → "<one-sentence answer>"

Plus passage-grounded extraction examples from the KB (if available):
  4. "Passage: ... Question: question" → "answer text."

Usage:
    python augment_sft.py --mcq_jsonl raw_data/mcq_train.jsonl \
        --kb_dir kb \
        --out raw_data/sft_v2.jsonl
"""
import argparse
import json
import random
import re
from pathlib import Path


LETTERS = ["A", "B", "C", "D"]


def mcq_format(q):
    """Format-1: original ChatML MCQ → 'The answer is (X) text.'"""
    options = "\n".join(f"({L}) {o}" for L, o in zip(LETTERS, q["options"]))
    user = f"Question ({q['category']}): {q['question']}\n{options}"
    correct = q["correct_index"]
    asst = f"The answer is ({LETTERS[correct]}) {q['options'][correct]}."
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": asst}]


def open_format(q):
    """Format-2: free-form, no options. Just question → answer text."""
    user = q["question"]
    asst = q["options"][q["correct_index"]] + "."
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": asst}]


def statement_format(q):
    """Format-3: try to make a sentence-form answer using simple templates
    based on the question's interrogative."""
    question = q["question"].rstrip("?.").strip()
    answer = q["options"][q["correct_index"]].strip()
    qlow = question.lower()

    # Cheap rule-based interrogative → statement converter
    # "What is X?" → "X is <answer>." (we don't always know X is correct subject — use generic templates)
    # "Who is/was X?" → "X is/was <answer>."
    # When in doubt, use "The answer is <answer>." as fallback.
    templates = []
    if re.match(r"^(what|which) (is|was|are|were) (the|an|a) ", qlow):
        # "What is the largest planet?" → "The largest planet is Jupiter."
        m = re.match(r"^(?:what|which) (is|was|are|were) (.+)$", question, re.I)
        if m:
            verb, subj = m.group(1), m.group(2).rstrip("?.")
            templates.append(f"{subj.capitalize()} {verb} {answer}.")
    if re.match(r"^who (is|was|are|were) ", qlow):
        m = re.match(r"^who (is|was|are|were) (.+)$", question, re.I)
        if m:
            verb, subj = m.group(1), m.group(2).rstrip("?.")
            templates.append(f"{subj.capitalize()} {verb} {answer}.")
    if re.match(r"^(when|where) ", qlow):
        templates.append(f"{answer}.")
    if re.match(r"^(how many|how much) ", qlow):
        templates.append(f"{answer}.")

    # Generic fallback: "Q: ... A: ..."
    asst = templates[0] if templates else f"{answer}."
    user = question + ("?" if not question.endswith("?") else "")
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": asst}]


def passage_format(q, passage):
    """Format-4: passage-grounded extraction. The model sees a Wikipedia
    chunk plus a question, and is trained to answer using info in the chunk."""
    question = q["question"].rstrip("?.").strip() + "?"
    answer = q["options"][q["correct_index"]] + "."
    user = f"Read the passage and answer the question.\n\n{passage}\n\nQuestion: {question}"
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": answer}]


def find_passage(q, kb, encoder, kb_emb, top_k=1):
    """Best-matching KB chunk for the question (using already-embedded KB)."""
    text = q["question"] + " " + q["options"][q["correct_index"]]
    qv = encoder.encode([text], normalize_embeddings=True)[0]
    scores = kb_emb @ qv
    idx = int(scores.argmax())
    return f"{kb[idx]['title']}: {kb[idx]['text']}", float(scores[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcq_jsonl", required=True)
    ap.add_argument("--kb_dir", default="kb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Load MCQ source — the eval-style jsonl has _raw with original MCQ
    raw_items = []
    for line in open(args.mcq_jsonl):
        ex = json.loads(line)
        if "_raw" in ex:
            raw_items.append(ex["_raw"])
        else:
            # extract from messages content if no _raw
            user = ex["messages"][0]["content"]
            asst = ex["messages"][1]["content"]
            # parse back — fragile, only fall back here
            m = re.match(r"Question \(([^)]+)\): (.*?)\n\(A\) (.*?)\n\(B\) (.*?)\n\(C\) (.*?)\n\(D\) (.*?)$", user, re.S)
            ans = re.match(r"The answer is \(([ABCD])\) ", asst)
            if m and ans:
                cat = m.group(1)
                qtext = m.group(2)
                opts = [m.group(i) for i in (3, 4, 5, 6)]
                ci = LETTERS.index(ans.group(1))
                raw_items.append({"question": qtext, "options": opts, "correct_index": ci, "category": cat})
    print(f"loaded {len(raw_items)} MCQs")

    # Optional: KB for passage-grounded examples
    kb = []
    encoder = None
    kb_emb = None
    kb_path = Path(args.kb_dir)
    if (kb_path / "kb.jsonl").exists() and (kb_path / "kb.npy").exists():
        import numpy as np
        from sentence_transformers import SentenceTransformer
        meta = json.load(open(kb_path / "kb.meta.json"))
        kb = [json.loads(l) for l in open(kb_path / "kb.jsonl")]
        kb_emb = np.load(kb_path / "kb.npy")
        encoder = SentenceTransformer(meta["model"])
        print(f"loaded KB: {len(kb)} chunks for passage-grounded examples")

    out = []
    for q in raw_items:
        out.append({"messages": mcq_format(q)})           # format 1
        out.append({"messages": open_format(q)})          # format 2
        out.append({"messages": statement_format(q)})     # format 3

    # Add passage-grounded examples for a subset of MCQs (those with high KB match)
    if kb and encoder is not None:
        passage_count = 0
        for q in raw_items:
            passage, score = find_passage(q, kb, encoder, kb_emb)
            if score > 0.55:  # only keep good matches
                out.append({"messages": passage_format(q, passage)})
                passage_count += 1
        print(f"added {passage_count} passage-grounded examples")

    rng.shuffle(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} examples → {args.out}")
    # show a few samples
    for i in range(3):
        print(f"--- sample {i} ---")
        print(json.dumps(out[i], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
