"""
Browser chat UI for our trained LLM.

Wraps `LLM.generate()` in a Gradio ChatInterface. Streams tokens as they're
generated. Uses ChatML formatting (same as our SFT pipeline), so it works for
both base and SFT'd checkpoints — base models just continue more freely,
SFT'd models actually answer.

Usage (local):
    python gradio_app.py --ckpt out/plan_b_110m/ckpt_step1000_model.pt --preset plan_b

Usage (HuggingFace Spaces):
    Place this script + model.py + configs.py + the .pt file in the Space repo.
    Spaces auto-runs `app.py` (rename gradio_app.py → app.py).

Args:
    --ckpt           path to checkpoint .pt file
    --preset         which model preset (smoke, plan_b, main)
    --tokenizer      HF repo for tokenizer (default: Llama-2 mirror)
    --share          create a temporary 72-hour public URL (Gradio's gradio.live)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import gradio as gr
from transformers import AutoTokenizer

from configs import PRESETS
from model import LLM


# ChatML special markers — must match sft.py
SYS_PROMPT = "You are a helpful assistant."
B_USR, B_ASST, EOT = "<|user|>\n", "<|assistant|>\n", "<|end|>"


class Retriever:
    """Tiny in-memory dense retriever over the KB built by build_kb.py."""
    def __init__(self, kb_dir):
        kb_dir = Path(kb_dir)
        meta = json.load(open(kb_dir / "kb.meta.json"))
        self.chunks = [json.loads(l) for l in open(kb_dir / "kb.jsonl")]
        self.emb = np.load(kb_dir / "kb.npy")  # already normalized
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(meta["model"])
        print(f"loaded KB: {len(self.chunks)} chunks, dim={meta['dim']}")

    def topk(self, query, k=3):
        q = self.encoder.encode([query], normalize_embeddings=True)[0]
        scores = self.emb @ q  # cosine since both normalized
        idx = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in idx]


def build_prompt(history, user_msg, retriever=None, k=5, max_ctx_chars=2400):
    """Render conversation + user message into a ChatML prompt. If a retriever
    is given, retrieved chunks are inlined into the user message — this works
    better than stuffing them into <|system|> because our SFT model never saw
    long system prompts."""
    parts = [f"<|system|>\n{SYS_PROMPT}{EOT}\n"]
    for u, a in history:
        parts.append(f"{B_USR}{u}{EOT}\n")
        parts.append(f"{B_ASST}{a}{EOT}\n")

    if retriever is not None:
        hits = retriever.topk(user_msg, k=k)
        ctx = ""
        for score, c in hits:
            piece = f"{c['title']}: {c['text']}\n\n"
            remaining = max_ctx_chars - len(ctx)
            if remaining <= 0: break
            # Always include at least the start of the first chunk; truncate later ones
            if len(piece) > remaining:
                piece = piece[:remaining].rstrip() + "…\n\n"
                ctx += piece
                break
            ctx += piece
        # Match SFT v2 training format exactly
        user_block = (
            f"Read the passage and answer the question.\n\n"
            f"{ctx.strip()}\n\nQuestion: {user_msg}"
        )
    else:
        user_block = user_msg

    parts.append(f"{B_USR}{user_block}{EOT}\n{B_ASST}")
    return "".join(parts)


def make_chat_fn(model, tok, device, max_new_tokens=80, temperature=1.0,
                 top_k=1, top_p=1.0, repetition_penalty=1.1, raw_mode=False,
                 retriever=None):
    # Defaults are GREEDY (top_k=1, temp=1.0, top_p=1.0) + light repetition penalty.
    # Empirically: a 110M model with retrieved context does best with greedy
    # decoding on factual Q&A. Sampling at any temperature destroys accuracy.
    eos_id = tok.eos_token_id

    def chat(message, history):
        prompt = message if raw_mode else build_prompt(history, message, retriever=retriever)
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        ctx = model.cfg.max_seq_len
        if ids.shape[1] > ctx - max_new_tokens:
            ids = ids[:, -(ctx - max_new_tokens):]

        out = ids
        is_mcq = ("(A)" in message and "(B)" in message)
        with torch.no_grad():
            for step in range(max_new_tokens):
                logits, _ = model(out[:, -ctx:])
                logits = logits[:, -1, :].float()

                # Repetition penalty: shrink logits of tokens already in the output.
                # Standard formulation from CTRL paper: divide if positive, mult if negative.
                if repetition_penalty != 1.0:
                    seen = out[0].tolist()
                    for tid in set(seen[-200:]):
                        v = logits[0, tid]
                        logits[0, tid] = v / repetition_penalty if v > 0 else v * repetition_penalty

                logits = logits / temperature

                # top-k
                if top_k:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")

                # top-p (nucleus): keep smallest set of tokens whose cum-prob >= top_p
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    mask = cum > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_logits[mask] = -float("inf")
                    logits = torch.full_like(logits, -float("inf")).scatter(1, sorted_idx, sorted_logits)

                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                if nxt.item() == eos_id:
                    break
                out = torch.cat([out, nxt], dim=1)

                new_text = tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
                if EOT in new_text:
                    yield new_text.split(EOT)[0].rstrip()
                    return
                if is_mcq and step > 8 and (new_text.rstrip().endswith(".") or "\n" in new_text[10:]):
                    yield new_text.split("\n")[0].rstrip()
                    return
                yield new_text

    return chat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("LLM_CKPT", "out/plan_b_110m/ckpt_step1000_model.pt"))
    ap.add_argument("--preset", default=os.environ.get("LLM_PRESET", "plan_b"))
    ap.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--raw", action="store_true",
                    help="base-model mode: feed user text directly (no chat template). "
                         "Use this until the model has been SFT'd.")
    ap.add_argument("--kb_dir", default=None,
                    help="dir built by build_kb.py — enables RAG when set")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    print(f"loading {args.ckpt} on {device}…")

    mcfg, _ = PRESETS[args.preset]
    model = LLM(mcfg).to(device, dtype=dtype)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state); model.eval()
    print(f"loaded {model.num_params()/1e6:.1f}M params")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    retriever = None
    if args.kb_dir and not args.raw:
        retriever = Retriever(args.kb_dir)

    chat_fn = make_chat_fn(model, tok, device, raw_mode=args.raw, retriever=retriever)

    if args.raw:
        title = "private-llm (base model, ~10% trained) — text continuation only"
        desc = (
            "This is a *base* language model trained from scratch — it has not been "
            "fine-tuned to answer questions. Give it the *start* of some text and it "
            "will try to continue it. Asking 'What is X?' will mostly fail."
        )
        examples = [
            "Once upon a time in a small village,",
            "Photosynthesis is the process by which",
            "The most common planets in our solar system are",
            "The history of India can be traced back to",
        ]
    else:
        title = "private-llm — 110M params, trained from scratch"
        desc = "Tiny custom LLM. Will say wrong things confidently — that's the charm."
        examples = [
            "What is the capital of France?",
            "Write a short story about a robot learning to bake.",
            "Q: Who wrote Hamlet?",
        ]

    gr.ChatInterface(fn=chat_fn, title=title, description=desc, examples=examples).launch(share=args.share)


if __name__ == "__main__":
    main()
