"""CPU inference benchmark + answer-quality probe for the Space pipeline.

    .venv/bin/python bench_infer.py --threads 2            # mimic HF cpu-basic (2 vCPU)
    .venv/bin/python bench_infer.py --compare-nocache      # also verify cached == uncached output
"""
import argparse
import json
import time

import torch
from transformers import AutoTokenizer

from configs import PRESETS
from gradio_app import EOT, Retriever, build_prompt
from model import LLM

# (question, any-of acceptable substrings, lowercase)
QUESTIONS = [
    ("Who was the first Prime Minister of India?", ["nehru"]),
    ("What is the capital of India?", ["delhi"]),
    ("Who founded the Maratha Empire?", ["shivaji"]),
    ("Who wrote the Indian National Anthem?", ["tagore"]),
    ("Who is Rama's brother?", ["lakshman", "bharat", "shatrughna"]),
    ("Tell me about Hanuman.", ["vanara", "monkey", "devotee", "god"]),
    ("Who was the first President of India?", ["rajendra prasad"]),
    ("Which river is the longest in India?", ["ganga", "ganges"]),
    ("Who built the Taj Mahal?", ["shah jahan"]),
    ("In which year did India gain independence?", ["1947"]),
    ("Who is known as the Father of the Nation in India?", ["gandhi"]),
    ("What is the national animal of India?", ["tiger"]),
    ("Who wrote the Ramayana?", ["valmiki"]),
    ("Who was Ashoka?", ["maurya", "emperor"]),
    ("Which city is known as the Pink City?", ["jaipur"]),
    ("Who was the first woman Prime Minister of India?", ["indira"]),
    ("What is the Mahabharata?", ["epic", "kurukshetra", "pandava"]),
    ("Who composed the Gita Govinda?", ["jayadeva"]),
    ("Who led the Salt March?", ["gandhi"]),
    ("Where is the Golden Temple located?", ["amritsar"]),
]


def load(device="cpu"):
    mcfg, _ = PRESETS["plan_b"]
    model = LLM(mcfg).to(device, dtype=torch.float32)
    state = torch.load("out/plan_b_sft_v4_model.pt", map_location=device, weights_only=False)["model"]
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state.items()})
    model.eval()
    tok = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf", use_fast=True)
    return model, tok


def penalize(logits, seen, rp):
    for tid in set(seen[-200:]):
        v = logits[0, tid]
        logits[0, tid] = v / rp if v > 0 else v * rp


@torch.no_grad()
def generate(model, tok, prompt, max_new=80, rp=1.1, use_cache=True):
    """Greedy + repetition penalty, identical to the Space's defaults."""
    ids = tok.encode(prompt, return_tensors="pt")
    ctx = model.cfg.max_seq_len
    if ids.shape[1] > ctx - max_new:
        ids = ids[:, -(ctx - max_new):]
    out = ids
    cache = model.new_kv_cache() if use_cache else None
    t0 = time.perf_counter()
    logits, _ = model(ids, kv_cache=cache)
    t_prefill = time.perf_counter() - t0
    n = 0
    while n < max_new:
        last = logits[:, -1, :].float()
        penalize(last, out[0].tolist(), rp)
        nxt = last.argmax(-1, keepdim=True)
        if nxt.item() == tok.eos_token_id:
            break
        out = torch.cat([out, nxt], dim=1)
        n += 1
        if EOT in tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True):
            break
        if use_cache:
            logits, _ = model(nxt, kv_cache=cache, start_pos=out.shape[1] - 1)
        else:
            logits, _ = model(out)
    total = time.perf_counter() - t0
    text = tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True).split(EOT)[0].strip()
    return text, out[0, ids.shape[1]:].tolist(), ids.shape[1], n, t_prefill, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--compare-nocache", action="store_true")
    ap.add_argument("--limit", type=int, default=len(QUESTIONS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--kb", default="kb")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    model, tok = load()
    retr = Retriever(args.kb)
    rows, correct = [], 0
    for q, keys in QUESTIONS[:args.limit]:
        prompt, oos = build_prompt([], q, retriever=retr)
        if oos:
            rows.append({"q": q, "answer": "<out of scope>", "ok": False})
            print(f"[OOS] {q}")
            continue
        text, toks, plen, n, tpre, tot = generate(model, tok, prompt)
        ok = any(k in text.lower() for k in keys)
        correct += ok
        row = {"q": q, "answer": text, "ok": ok, "prompt_tokens": plen, "new_tokens": n,
               "prefill_s": round(tpre, 2), "total_s": round(tot, 2),
               "s_per_tok": round((tot - tpre) / max(n, 1), 3)}
        if args.compare_nocache:
            text2, toks2, *_, tot2 = generate(model, tok, prompt, use_cache=False)
            row.update(nocache_total_s=round(tot2, 2), identical=(toks == toks2))
        rows.append(row)
        print(f"[{'ok' if ok else '--'}] {q} -> {text[:90]!r}  ({plen} prompt tok, {n} new, {tot:.1f}s"
              + (f", no-cache {row['nocache_total_s']}s, identical={row['identical']}" if args.compare_nocache else "") + ")")
    print(f"\naccuracy {correct}/{len(rows)}")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
