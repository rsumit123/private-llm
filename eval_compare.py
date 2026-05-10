"""
Compare our model against open baselines on the same MCQ holdout.

Metrics:
  - LL acc: log-likelihood scoring of (A/B/C/D) — same method as MMLU.
            Picks option with highest P(option_text | prompt).
  - Gen acc: free-form generation, then parse first letter (A/B/C/D)
            from the response. This matches what users actually experience
            when chatting.

Models compared:
  - ours-sft  : our SFT'd 110M (custom code via configs.py + model.py)
  - gpt2      : OpenAI GPT-2 small  (124M)
  - gpt2-md   : OpenAI GPT-2 medium (355M)
  - pythia    : EleutherAI Pythia-410M

Usage:
    python eval_compare.py --eval raw_data/mcq_eval.jsonl --limit 200
"""
import argparse
import json
import re
import time
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LETTERS = ["A", "B", "C", "D"]
GEN_PARSE = re.compile(r"\b([ABCD])[\s).:,-]")


# ----- model adapters -----
class HFAdapter:
    """Wrap a HF AutoModelForCausalLM for our two eval metrics."""
    def __init__(self, name, dtype=torch.float32, device="cpu"):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.device = device

    @torch.no_grad()
    def score_option(self, prompt, option):
        full = prompt + option
        p_ids = self.tok.encode(prompt, add_special_tokens=False)
        f_ids = self.tok.encode(full, add_special_tokens=False)
        new_ids = f_ids[len(p_ids):]
        if not new_ids: return -1e9
        inp = torch.tensor([f_ids[:-1]], device=self.device)
        logits = self.model(inp).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        start = len(p_ids) - 1
        sliced = logp[0, start:start + len(new_ids)]
        s = sliced.gather(1, torch.tensor(new_ids, device=self.device).unsqueeze(1)).sum().item()
        return s / max(1, len(new_ids))

    @torch.no_grad()
    def generate_text(self, prompt, max_new=30):
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **ids, max_new_tokens=max_new, do_sample=False,
            pad_token_id=self.tok.pad_token_id,
        )
        new = out[0, ids["input_ids"].shape[1]:].tolist()
        return self.tok.decode(new, skip_special_tokens=True)


class OursAdapter:
    """Our 110M SFT'd model — uses the same scoring math."""
    def __init__(self, ckpt_path, preset, device="cpu"):
        from configs import PRESETS
        from model import LLM
        mcfg, _ = PRESETS[preset]
        self.model = LLM(mcfg).to(device, dtype=torch.float32)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)["model"]
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state); self.model.eval()
        self.tok = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf", use_fast=True)
        self.device = device
        self.name = "ours-sft-110M"

    @torch.no_grad()
    def score_option(self, prompt, option):
        full = prompt + option
        p_ids = self.tok.encode(prompt, add_special_tokens=False)
        f_ids = self.tok.encode(full, add_special_tokens=False)
        new_ids = f_ids[len(p_ids):]
        if not new_ids: return -1e9
        inp = torch.tensor([f_ids[:-1]], device=self.device)
        logits, _ = self.model(inp)
        logp = torch.log_softmax(logits.float(), dim=-1)
        start = len(p_ids) - 1
        sliced = logp[0, start:start + len(new_ids)]
        s = sliced.gather(1, torch.tensor(new_ids, device=self.device).unsqueeze(1)).sum().item()
        return s / max(1, len(new_ids))

    @torch.no_grad()
    def generate_text(self, prompt, max_new=30):
        ids = self.tok.encode(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(ids, max_new_tokens=max_new, temperature=1.0, top_k=1)  # greedy
        return self.tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)


# ----- prompt templates -----
def make_prompts(ex):
    """Build (LL prompt, generation prompt) for one MCQ."""
    raw = ex["_raw"]
    q = raw["question"]
    opts = raw["options"]
    cat = raw.get("category", "")
    options_block = "\n".join(f"({L}) {o}" for L, o in zip(LETTERS, opts))

    # Same simple prompt used by both metrics; standard "Question: ... Answer:" style
    base = f"Question ({cat}): {q}\n{options_block}\nAnswer: ("
    return base


# ----- eval -----
def eval_model(adapter, items, log_label):
    ll_correct = gen_correct = 0
    by_cat_ll = defaultdict(lambda: [0, 0])
    t0 = time.time()
    for i, ex in enumerate(items):
        raw = ex["_raw"]
        gold = raw["correct_index"]
        cat = raw.get("category", "other")
        prompt = make_prompts(ex)

        # LL eval
        scores = []
        for L, o in zip(LETTERS, raw["options"]):
            scores.append(adapter.score_option(prompt, f"{L}) {o}"))
        ll_pred = scores.index(max(scores))
        ll_ok = ll_pred == gold
        ll_correct += int(ll_ok)
        by_cat_ll[cat][0] += int(ll_ok); by_cat_ll[cat][1] += 1

        # Gen eval
        text = adapter.generate_text(prompt, max_new=20)
        m = GEN_PARSE.search(text)
        gen_letter = m.group(1) if m else None
        # also accept "X)" at the very start
        if not gen_letter and text.strip() and text.strip()[0] in LETTERS:
            gen_letter = text.strip()[0]
        gen_pred = LETTERS.index(gen_letter) if gen_letter in LETTERS else -1
        gen_ok = gen_pred == gold
        gen_correct += int(gen_ok)

        if (i + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  [{log_label}] {i+1}/{len(items)} | LL={ll_correct/(i+1):.3f} Gen={gen_correct/(i+1):.3f} | {dt:.0f}s")
    n = len(items)
    return {
        "ll_acc": ll_correct / n,
        "gen_acc": gen_correct / n,
        "n": n,
        "by_cat_ll": {k: (v[0], v[1], v[0]/v[1]) for k, v in by_cat_ll.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--ours_ckpt", default="out/plan_b_sft/ckpt_sft_final.pt")
    ap.add_argument("--ours_preset", default="plan_b")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", nargs="+",
                    default=["ours", "gpt2", "gpt2-medium", "EleutherAI/pythia-410m"])
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.eval)][:args.limit]
    print(f"loaded {len(items)} eval items")

    results = {}
    for spec in args.models:
        print(f"\n=== {spec} ===")
        if spec == "ours":
            adapter = OursAdapter(args.ours_ckpt, args.ours_preset, device=args.device)
        else:
            adapter = HFAdapter(spec, device=args.device,
                                dtype=torch.bfloat16 if args.device == "cuda" else torch.float32)
        params = sum(p.numel() for p in adapter.model.parameters())
        print(f"  params: {params/1e6:.1f}M")
        results[spec] = eval_model(adapter, items, spec)
        results[spec]["params_m"] = params / 1e6
        del adapter
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"Comparison on {len(items)} MCQs (random baseline = 25.0%)")
    print("=" * 70)
    print(f"{'model':<32} {'params':>8}  {'LL acc':>8}  {'Gen acc':>8}")
    print("-" * 70)
    for spec, r in results.items():
        print(f"{spec:<32} {r['params_m']:>7.0f}M  {r['ll_acc']*100:>7.1f}%  {r['gen_acc']*100:>7.1f}%")

    json.dump(results, open("eval_compare_results.json", "w"), indent=2)
    print("\nfull results saved to eval_compare_results.json")


if __name__ == "__main__":
    main()
