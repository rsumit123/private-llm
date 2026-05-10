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

## Status: shipped 🚀 (2026-05-10)

**Live demo:** https://huggingface.co/spaces/sumitkClasses/scratchq-110

**Final spend:** $5.39 of $10 budget. Vast.ai instance destroyed.

### What was built
- ✅ Pretrained 110M Llama-style transformer from scratch on 2B FineWeb-Edu tokens (10,500 steps, final loss 3.10)
- ✅ SFT v4 — two-stage: 22k augmented Indian MCQs (loss 1.81) → 5k SQuAD passes at lr=1e-5 (final loss 0.73)
- ✅ Built Indian-Wikipedia knowledge base — 4,631 chunks across 171 articles, embedded with BAAI/bge-small-en-v1.5
- ✅ Comparative eval vs GPT-2 small / GPT-2 medium / Pythia-410M — we win on LL by 10-19pp, on Generation by 15-20pp
- ✅ Gradio chat UI with RAG (top-5 retrieval), greedy decoding + light repetition penalty, out-of-scope detection (cosine < 0.55)
- ✅ Deployed to HuggingFace Spaces (CPU basic, free tier, permanent URL)

### Resume claim
> *"Pretrained a 110M-parameter Llama-style transformer from scratch on 2B FineWeb-Edu tokens (~$2 of compute), then SFT'd on 6,686 Indian general-knowledge MCQs (augmented to 22k examples across 4 formats) plus SQuAD-style passage extraction. Achieved **36.5% on held-out 4-option MCQs vs GPT-2 medium at 17.5%** — a 19 pp improvement on a model 3× larger. Deployed with retrieval-augmented generation over a 4,600-chunk Indian Wikipedia knowledge base."*

### Open items (post-ship, optional)
- [ ] KV-cache inference for faster CPU generation (currently O(n²) per response).
- [ ] Try fine-tuning Qwen2.5-1.5B on same data as a "production tier" demo alongside the from-scratch one.
- [ ] Submit base model checkpoint to Open LLM Leaderboard for an external benchmark.
