#!/usr/bin/env bash
# End-to-end smoke test: ~hours on a 5090, verifies pipeline before the real run.
set -euo pipefail

# 1. Tokenize ~200M tokens (enough to feed 2k smoke steps with margin)
python prepare_data.py \
  --out data/fineweb_edu \
  --target_tokens 200000000 \
  --tokenizer NousResearch/Llama-2-7b-hf \
  --subset sample-10BT

# 2. Train 10M-param model for 2k steps
python train.py smoke

# 3. Generate from final checkpoint
python sample.py out/smoke_10m/ckpt_0002000.pt --preset smoke \
  --prompt "The capital of France is"
