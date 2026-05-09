# Project Plan & Decisions

Living doc of decisions made while building this LLM. Update as we go.

## Goal
Train a small Llama-style LLM from scratch on a single GPU, get it answering simple questions, and submit to a public benchmark — primarily as a learning exercise.

## Budget
- Total compute budget: ~$10 on vast.ai
- Smoke test already burned ~$0.13 → ~$9.87 remaining

## Pipeline (decided)
1. **Pretrain Plan B**: 110M-param Llama-style model on ~2B FineWeb-Edu tokens. ~10–15h on a 3090, ~$3.
2. **Continued pretraining (domain adapt)** on ~100 Indian Express PDFs (~7M tokens, 1–2 epochs at low LR 5e-5). ~30 min, ~$0.10. Purpose: shift style/vocab toward Indian English; not for new knowledge.
3. **SFT**: mix ~7000 user-supplied Indian-context MCQs with ~10k UltraChat conversations. Hold out ~1000 MCQs for eval. ~45 min, ~$0.20.
4. **Eval**: lm-evaluation-harness (HellaSwag, ARC, WinoGrande, PIQA, BoolQ) + custom MCQ holdout. ~15 min, ~$0.05.
5. **Total cost target**: ~$3.50. Leaves ~$6 buffer for restarts/retries/scaling up.

## Architecture (decided)
Llama-style decoder-only transformer:
- RMSNorm, RoPE, SwiGLU, tied embeddings
- FlashAttention via PyTorch SDPA
- Plan B config: 768 hidden / 12 layers / 12 heads / 2048 ffn / 1024 ctx ≈ 110M params
- Tokenizer: Llama-2 BPE (32k vocab) — `NousResearch/Llama-2-7b-hf` mirror (open)

## Data sources (decided)
- **Pretraining**: `HuggingFaceFW/fineweb-edu` (sample-10BT subset, streamed)
- **SFT base**: `HuggingFaceH4/ultrachat_200k`
- **User-supplied (pending)**:
  - ~100 Indian Express PDFs → continued pretraining (extract via `pymupdf`)
  - ~8000 MCQs (JSON) → SFT + custom eval holdout

## Karpathy tools — decisions
- Stick with our hand-rolled stack for learning value.
- Borrow from `nanochat` later: (a) KV-cache inference, (b) custom BPE tokenizer if the final mix is heavy on Indian text, (c) eval harness wiring.
- Skip `llm.c` (raw CUDA, not learning ML) and `nanoGPT` (already conceptually equivalent to ours).

## LoRA / Unsloth — decisions
Not applicable here. Both are *fine-tuning* accelerators that need a pretrained base. We're training from scratch and our 110M model fits comfortably in full precision on a 3090.

## Vast.ai notes
- SSH key: `~/.ssh/id_ed25519_vast`
- API key file: `.vast-ai-api-key` (gitignored)
- Target offers: RTX 3090, 1 GPU, ≥80GB disk, ≥500 Mbps inet, reliability >0.98, dph <0.20

## Open items / next session pickups
- [x] User pointed to MCQ JSON: `gk_questions.json` (6686 entries, copied to `raw_data/mcqs/`). Confirmed superset of the smaller category files.
- [x] Newspapers: 102 PDFs in `~/Library/Mobile Documents/com~apple~CloudDocs/Downloads/Newspaper /` (487MB).
- [ ] Plan B pretraining run (in progress, started 2026-05-09 ~10:58, ETA ~12-13h, target loss ~3.0).
- [x] PDF extraction script (`extract_pdfs.py`) — written, untested.
- [x] MCQ formatter (`format_mcqs.py`) — written, untested.
- [x] Gradio app (`gradio_app.py`) — written, untested. Uses ChatML formatting, streams tokens.
- [ ] Wire up `lm-evaluation-harness`.
- [ ] Add KV-cache to attention for faster Gradio inference (currently O(n²) per response). Borrow from nanochat. Acceptable for demo-tier use as-is.
- [ ] Deploy Gradio to HuggingFace Spaces after final SFT weights are ready. Plan: rename `gradio_app.py` → `app.py`, push weights to `huggingface.co/rsumit123/private-llm-110m`, point Space at it.

## Deployment recipe (Hugging Face Spaces)
Once weights are final:
1. `huggingface-cli login`
2. `huggingface-cli upload rsumit123/private-llm-110m ckpt_final.pt model.safetensors`
3. Create Space at `huggingface.co/new-space`, SDK=Gradio, Hardware=CPU basic (free)
4. Push: `app.py` (renamed from gradio_app.py), `model.py`, `configs.py`, `requirements.txt`. Have `app.py` download the ckpt from the model repo on startup.
5. Public URL appears at `huggingface.co/spaces/rsumit123/private-llm-chat` — share with anyone.
