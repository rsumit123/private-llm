"""
Quick generation check.

Usage:
    python sample.py out/smoke_10m/ckpt_0002000.pt --prompt "The capital of France is"
"""
import argparse
import torch
from transformers import AutoTokenizer

from configs import PRESETS
from model import LLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--preset", default="smoke", choices=list(PRESETS.keys()))
    ap.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg, _ = PRESETS[args.preset]
    model = LLM(mcfg).to(device, dtype=torch.bfloat16)
    state = torch.load(args.ckpt, map_location=device)["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    ids = tok.encode(args.prompt, return_tensors="pt").to(device)
    out = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
