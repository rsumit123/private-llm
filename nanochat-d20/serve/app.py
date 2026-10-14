"""Chat UI + OpenAI-compatible API for a nanochat d20 model trained from scratch.

Same file runs in two places:
  - Hugging Face Space (CPU): Gradio chat at /            (API disabled unless API_KEY is set)
  - GPU box behind llm-api.skdev.one: Gradio + /v1/models, /v1/chat/completions
Env: MODEL_REPO (weights), API_KEY (enables /v1), APP_PORT, MAX_TOKENS.
"""
import json
import os
import threading
import time
import uuid

import torch

BASE_DIR = os.environ.setdefault("NANOCHAT_BASE_DIR", os.path.abspath("nanochat_data"))
MODEL_REPO = os.environ.get("MODEL_REPO", "sumitkClasses/nanochat-d20-chat")
MODEL_ID = "nanochat-d20"
API_KEY = os.environ.get("API_KEY", "")
DEFAULT_MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))

if not os.path.exists(os.path.join(BASE_DIR, "chatsft_checkpoints")):
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_REPO, local_dir=BASE_DIR, allow_patterns=["chatsft_checkpoints/*", "tokenizer/*"])

from nanochat.checkpoint_manager import load_model  # noqa: E402  (needs NANOCHAT_BASE_DIR first)
from nanochat.common import autodetect_device_type, compute_init  # noqa: E402
from nanochat.engine import Engine  # noqa: E402

device_type = os.environ.get("DEVICE") or autodetect_device_type()
if device_type == "cpu":
    torch.set_num_threads(int(os.environ.get("THREADS", "0")) or os.cpu_count() or 2)
_, _, _, _, device = compute_init(device_type)
model, tokenizer, _ = load_model("sft", device, phase="eval")
engine = Engine(model, tokenizer)
gen_lock = threading.Lock()

BOS = tokenizer.get_bos_token_id()
U0, U1 = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
A0, A1 = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")
MAX_CTX = model.config.sequence_len


def render(messages):
    """OpenAI-style messages -> nanochat token ids, ending with <|assistant_start|>.
    The model was never trained on a system role, so system text is prepended to the first user turn."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system" and m["content"])
    toks, first_user = [BOS], True
    for m in messages:
        text = m["content"] if isinstance(m["content"], str) else " ".join(
            p.get("text", "") for p in m["content"] if isinstance(p, dict))
        if m["role"] == "user":
            if first_user and system:
                text = f"{system}\n\n{text}"
            first_user = False
            toks += [U0, *tokenizer.encode(text), U1]
        elif m["role"] == "assistant":
            toks += [A0, *tokenizer.encode(text), A1]
    toks.append(A0)
    return toks


def stream(messages, max_tokens=DEFAULT_MAX_TOKENS, temperature=0.6, top_k=50):
    """Yield text deltas. Decodes the running sequence so multi-byte characters never split."""
    toks = render(messages)
    max_tokens = max(1, min(int(max_tokens), MAX_CTX - len(toks)))
    if len(toks) >= MAX_CTX:
        yield "(conversation too long for this model's 2048-token context; start a new chat)"
        return
    out, sent = [], ""
    with gen_lock:
        for column, _ in engine.generate(toks, num_samples=1, max_tokens=max_tokens,
                                         temperature=float(temperature), top_k=int(top_k)):
            t = column[0]
            if t in (A1, BOS):
                break
            out.append(t)
            text = tokenizer.decode(out)
            if not text.endswith("�"):
                yield text[len(sent):]
                sent = text


# ------------------------------------------------------------------ Gradio UI
import gradio as gr  # noqa: E402


def chat_fn(message, history, max_tokens, temperature):
    msgs = [{"role": h["role"], "content": h["content"]} for h in history
            if h.get("role") in ("user", "assistant") and isinstance(h.get("content"), str)]
    msgs.append({"role": "user", "content": message})
    reply = ""
    for d in stream(msgs, max_tokens=max_tokens, temperature=temperature):
        reply += d
        yield reply


ABOUT = """
**A 20-layer chat model (~390M transformer + ~500M embedding parameters) trained completely from scratch** — tokenizer, pretraining and chat fine-tuning,
no pretrained weights anywhere — using [karpathy/nanochat](https://github.com/karpathy/nanochat) on a single H100:
**~5 hours of pretraining** on 3.5B tokens plus **47 minutes of chat fine-tuning**.

It writes fluent English, holds a conversation, writes poems and explains things simply, but it **confidently makes up
facts** (it is ~200x smaller than GPT-3.5). This free CPU Space is slow: replies stream at a few words per second.
Benchmarks: ARC-Easy 55.2% · MMLU 35.2% · HumanEval 11.0% · CORE 0.236.  
Weights: [sumitkClasses/nanochat-d20-chat](https://huggingface.co/sumitkClasses/nanochat-d20-chat) · Code, training logs and scripts: [github.com/rsumit123/private-llm](https://github.com/rsumit123/private-llm)
"""

demo = gr.ChatInterface(
    chat_fn,
    type="messages",
    title="nanochat-d20 · a chat model trained from scratch",
    description=ABOUT,
    additional_inputs=[
        gr.Slider(32, 1024, value=min(DEFAULT_MAX_TOKENS, 1024), step=32, label="Max new tokens"),
        gr.Slider(0.1, 1.2, value=0.6, step=0.05, label="Temperature"),
    ],
    examples=[["Write a short poem about the monsoon."],
              ["Explain photosynthesis to a 10 year old."],
              ["Give me three tips for a job interview."],
              ["Why is the sky blue?"]],
    cache_examples=False,
)

# ------------------------------------------------------------------ OpenAI-compatible API
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402

app = FastAPI()


def _authorized(req: Request):
    return API_KEY and req.headers.get("authorization", "") == f"Bearer {API_KEY}"


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_ID, "device": device_type}


@app.get("/v1/models")
def models(req: Request):
    if not _authorized(req):
        return JSONResponse({"error": {"message": "unauthorized"}}, status_code=401)
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "sumitkClasses"}]}


@app.post("/v1/chat/completions")
async def completions(req: Request):
    if not _authorized(req):
        return JSONResponse({"error": {"message": "unauthorized"}}, status_code=401)
    body = await req.json()
    kw = dict(max_tokens=body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_MAX_TOKENS,
              temperature=body.get("temperature", 0.6) or 0.6)
    cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
    msgs = body.get("messages", [])
    if body.get("stream"):
        def sse():
            yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                                         "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}) + "\n\n"
            for d in stream(msgs, **kw):
                yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                                             "choices": [{"index": 0, "delta": {"content": d}, "finish_reason": None}]}) + "\n\n"
            yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                                         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")
    text = await run_in_threadpool(lambda: "".join(stream(msgs, **kw)))
    return {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(render(msgs)), "completion_tokens": len(tokenizer.encode(text)),
                      "total_tokens": len(render(msgs)) + len(tokenizer.encode(text))}}


app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    # On HF Spaces, PORT belongs to the platform; Gradio apps must serve GRADIO_SERVER_PORT (7860).
    port = os.environ.get("GRADIO_SERVER_PORT") or os.environ.get("APP_PORT") or "7860"
    uvicorn.run(app, host="0.0.0.0", port=int(port))
