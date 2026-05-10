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
import os
import torch
import gradio as gr
from transformers import AutoTokenizer

from configs import PRESETS
from model import LLM


# ChatML special markers — must match sft.py
SYS_PROMPT = "You are a helpful assistant."
B_USR, B_ASST, EOT = "<|user|>\n", "<|assistant|>\n", "<|end|>"


def build_prompt(history, user_msg):
    """Render conversation history + new user message into a single ChatML prompt."""
    parts = [f"<|system|>\n{SYS_PROMPT}{EOT}\n"]
    for u, a in history:
        parts.append(f"{B_USR}{u}{EOT}\n")
        parts.append(f"{B_ASST}{a}{EOT}\n")
    parts.append(f"{B_USR}{user_msg}{EOT}\n{B_ASST}")
    return "".join(parts)


def make_chat_fn(model, tok, device, max_new_tokens=120, temperature=0.7, top_k=40, raw_mode=False):
    eos_id = tok.eos_token_id  # </s>

    def chat(message, history):
        prompt = message if raw_mode else build_prompt(history, message)
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        ctx = model.cfg.max_seq_len
        if ids.shape[1] > ctx - max_new_tokens:
            ids = ids[:, -(ctx - max_new_tokens):]

        out = ids
        is_mcq = ("(A)" in message and "(B)" in message)  # heuristic: tighten stop for MCQs
        with torch.no_grad():
            for step in range(max_new_tokens):
                logits, _ = model(out[:, -ctx:])
                logits = logits[:, -1, :] / temperature
                if top_k:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                if nxt.item() == eos_id:
                    break
                out = torch.cat([out, nxt], dim=1)

                new_text = tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
                # Stop on ChatML end marker
                if EOT in new_text:
                    yield new_text.split(EOT)[0].rstrip()
                    return
                # For MCQ-style prompts, the answer is one short sentence — stop on
                # first sentence terminator after we've generated some content
                if is_mcq and step > 8 and (new_text.rstrip().endswith(".") or "\n" in new_text[10:]):
                    sent = new_text.split("\n")[0].rstrip()
                    yield sent
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
    chat_fn = make_chat_fn(model, tok, device, raw_mode=args.raw)

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
