# private-llm

Train a small Llama-style LLM from scratch on a single GPU. Pretrain → SFT → eval.

See [CONCEPTS.md](CONCEPTS.md) for an explanation of the architecture and training process.

## Files
- `model.py` — Llama-style transformer (RMSNorm, RoPE, SwiGLU, FlashAttention via SDPA)
- `configs.py` — model + training presets (`smoke`, `plan_b`, `main`)
- `prepare_data.py` — stream FineWeb-Edu, tokenize, write `.bin` shards
- `data.py` — mmap shard loader
- `train.py` — pretraining loop
- `sft.py` — supervised fine-tuning on UltraChat-200k
- `sample.py` — generation from a checkpoint

## Quick start

```bash
pip install -r requirements.txt

# 1. Smoke test (~hour, 5M params, verifies pipeline)
python prepare_data.py --target_tokens 100000000
python train.py smoke
python sample.py out/smoke_10m/ckpt_0002000.pt --preset smoke --prompt "Hello"

# 2. Real run (Plan B: 110M params, 2B tokens, ~8-15h on a 3090)
python prepare_data.py --target_tokens 2000000000
python train.py plan_b --wandb

# 3. SFT (~30 min)
python sft.py --base out/plan_b_110m/ckpt_0010500.pt --preset plan_b --out out/plan_b_sft

# 4. Chat
python sample.py out/plan_b_sft/ckpt_sft_final.pt --preset plan_b \
  --prompt "<|user|>What is the capital of France?<|end|>\n<|assistant|>\n"
```

## Presets

| Preset | Params | Tokens | ~Time on 3090 | ~Cost |
|--------|--------|--------|---------------|-------|
| `smoke` | 5M | 32M | ~30 min | ~$0.10 |
| `plan_b` | 110M | 2B | ~10-15h | ~$2-3 |
| `main` | 300M | 25B | weeks | $$$ |
