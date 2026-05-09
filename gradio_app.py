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


def make_chat_fn(model, tok, device, max_new_tokens=200, temperature=0.8, top_k=40):
    eot_ids = tok.encode(EOT, add_special_tokens=False)

    def chat(message, history):
        prompt = build_prompt(history, message)
        ids = tok.encode(prompt, return_tensors="pt").to(device)
        # Truncate from left if too long for the model's context
        ctx = model.cfg.max_seq_len
        if ids.shape[1] > ctx - max_new_tokens:
            ids = ids[:, -(ctx - max_new_tokens):]

        out = ids
        generated = ""
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = model(out[:, -ctx:])
                logits = logits[:, -1, :] / temperature
                if top_k:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                out = torch.cat([out, nxt], dim=1)

                # decode incrementally and stream
                new_text = tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=False)
                # stop if we emit the EOT marker
                if EOT in new_text:
                    new_text = new_text.split(EOT)[0]
                    yield new_text
                    return
                generated = new_text
                yield generated

    return chat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("LLM_CKPT", "out/plan_b_110m/ckpt_step1000_model.pt"))
    ap.add_argument("--preset", default=os.environ.get("LLM_PRESET", "plan_b"))
    ap.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    ap.add_argument("--share", action="store_true")
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
    chat_fn = make_chat_fn(model, tok, device)

    gr.ChatInterface(
        fn=chat_fn,
        title="private-llm — 110M params, trained from scratch",
        description=(
            "Tiny custom LLM trained on FineWeb-Edu (and possibly fine-tuned on Indian "
            "MCQs/news). Will say wrong things confidently. That's the charm."
        ),
        examples=[
            "What is the capital of France?",
            "Write a short story about a robot learning to bake.",
            "Explain photosynthesis in one paragraph.",
            "Q: Who wrote Hamlet?",
        ],
    ).launch(share=args.share)


if __name__ == "__main__":
    main()
