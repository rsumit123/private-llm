# Multi-head latent attention for nanochat

`mla.patch` applies to [karpathy/nanochat](https://github.com/karpathy/nanochat) at commit `92d63d4`:

```bash
git clone https://github.com/karpathy/nanochat && cd nanochat && git checkout 92d63d4 && git apply ../mla.patch
python -m pytest tests/test_mla.py -q     # cached decoding == full forward for MHA / GQA / MLA; training steps
```

| File | Change |
|---|---|
| `nanochat/gpt.py` | `MLAttention` (latent KV, decoupled RoPE key, QK-norm on expanded keys); `GPTConfig.attn_type`, `mla_kv_dim`, `mla_rope_dim`, `use_value_embeds` |
| `nanochat/engine.py` | `KVCache(latent_dim=...)` stores one compact latent per token per layer; `nbytes()` |
| `scripts/base_train.py` | variant selection via `NANOCHAT_ATTN`, `NANOCHAT_KV_HEADS`, `NANOCHAT_MLA_KV_DIM`, `NANOCHAT_MLA_ROPE_DIM`, `NANOCHAT_NO_VE` (saved in checkpoint metadata) |
| `nanochat/optim.py` | higher `torch._dynamo` recompile limit (MLA adds matrix shapes to the compiled Muon step) |
| `scripts/bench_attn.py` | KV-cache bytes, sequences per memory budget, decode throughput per model |
| `tests/test_mla.py` | correctness tests |

The RoPE key uses every `head_dim / rope_dim`-th frequency of the model's rotary table, which equals a native
`rope_dim`-sized RoPE. Results: see the top-level README, "Architecture experiment".
