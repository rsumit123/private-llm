# Concepts behind this repo

A guided tour of what each piece is and why we use it. Assumes you know basic NN concepts (layers, weights, gradients, backprop).

## 1. The big picture

We're training a **decoder-only Transformer** language model from scratch. Two phases:

1. **Pretraining**: feed the model billions of tokens of web text and have it predict the next token. It learns grammar, facts, basic reasoning — but only as a "text continuer," not a question-answerer. Output: a *base model*.
2. **Fine-tuning (SFT)**: take the base model and train it on `(prompt, response)` pairs so it learns to *respond* to instructions instead of continuing them. Output: an *instruct model*.

That's the whole pipeline. Everything else is engineering to make it fit on one GPU.

## 2. Tokens, not characters

Models don't see text as letters. A **tokenizer** chops text into ~32k unique sub-word "tokens" using BPE (byte-pair encoding):
```
"unbelievable"  →  ["un", "believ", "able"]   →  [443, 9301, 519]
```
We use Llama-2's tokenizer (32k vocab). Each token is mapped to an integer id; integers are what the model actually consumes.

**Why sub-words?** Pure characters → too many timesteps. Pure words → vocabulary explodes (millions of unseen ones). BPE is the sweet spot: common words are 1 token, rare words split into pieces.

## 3. The Transformer architecture (Llama-style)

Our model is in `model.py`. It's a stack of identical **blocks**, each containing two sublayers: attention + feed-forward. We use the modern "Llama" recipe:

```
input ids → token embedding → [Block × N] → norm → linear → logits over vocab
```

### Token embedding (`tok_emb`)
A lookup table: each of 32000 token ids maps to a learned vector of `hidden_size` (768 for Plan B). Think of it as turning each word into a 768-dim point in "meaning space."

### Each Block has:
**RMSNorm → Attention → residual add → RMSNorm → FFN → residual add**

#### Attention (the magic)
Each token looks at every previous token and decides "how much do I care about each of you?" It's a weighted average over past tokens, where weights come from comparing **queries** (what I'm looking for) with **keys** (what I offer) — and we mix in the **values** (what I actually contribute).
- We split the work across `num_heads` parallel "attention heads" — each head learns to attend to a different kind of relationship (one head might track subject-verb, another might track long-range topic).
- **RoPE** (Rotary Position Embeddings): how the model knows token *position*. We rotate query/key vectors by an angle that depends on position — so two tokens at relative distance 5 always have the same rotation between them, regardless of where they are in the sequence. This generalizes to longer contexts than absolute position embeddings.
- **Causal mask**: a token can only see itself and previous tokens (set future positions to −∞ before softmax). This is what makes it a *language model*: you can't peek at the answer.
- We use PyTorch's `scaled_dot_product_attention` which automatically uses **FlashAttention-2** under the hood — a fused-kernel implementation that's ~2x faster and uses way less memory.

#### Feed-Forward (FFN) with SwiGLU
After attention, every token independently goes through a small MLP. The Llama version uses **SwiGLU**:
```
ffn(x) = down(silu(gate(x)) * up(x))
```
Two linear layers (gate, up) compute element-wise modulated activations, then a third (down) projects back. SwiGLU consistently beats vanilla ReLU FFN by ~0.5% loss for the same compute.

#### RMSNorm (instead of LayerNorm)
Normalizes each token's vector to unit RMS, then scales by a learned per-dim weight. Cheaper than LayerNorm (no mean subtraction), no quality loss. Used by Llama, Mistral, etc.

### Residual connections
Each sublayer's output is *added* to its input (`x = x + attn(norm(x))`). This is what lets you stack 12+ blocks without gradients vanishing — the gradient has a "highway" backward through the residuals.

### Tied embeddings
The token embedding (32000 → 768) and the final output projection (768 → 32000) share weights. Saves ~25M params, slightly improves quality.

### Final layer
After the last block we do RMSNorm, then a linear `lm_head` from 768 → 32000. The output is a **logit** for every token in the vocab — the model's score for "how likely is this the next token?" Apply softmax to get a probability distribution.

## 4. Training: predict the next token

The fundamental task: given tokens `[t0, t1, t2, ...]`, predict `t_{i+1}` at each position.
- Input: `[t0, t1, t2, t3]`
- Targets: `[t1, t2, t3, t4]` (shifted by one)
- Loss: cross-entropy between predicted distribution and true next token, averaged over all positions.

Because of the causal mask, we get N predictions from one forward pass — very efficient.

### Optimizer: AdamW
Adam adjusts the learning rate per-parameter based on running estimates of the gradient mean (`β1`) and variance (`β2`). **W**eight decay is regularization that pulls weights toward zero (decoupled in AdamW, hence the name). Standard choice. Betas (0.9, 0.95) are the LLM-tuned values.

### Learning rate schedule: warmup + cosine decay
- **Warmup** (first ~500 steps): linearly ramp LR from 0 → peak. Prevents the model from blowing up before Adam's running averages stabilize.
- **Cosine decay**: after warmup, smoothly decay LR to ~10% of peak following a cosine curve. Empirically the best schedule for one-shot pretraining.

### bf16 mixed precision
Weights stored in fp32; activations and matmuls in bf16 (brain float, same dynamic range as fp32 but half the bits). 2-3x speedup on Ampere/Ada GPUs with negligible quality cost.

### Gradient accumulation
Simulates a bigger batch size than fits in memory. Forward+backward 16 micro-batches, *accumulate* gradients, then step optimizer once. Effective batch = `micro_batch * grad_accum * seq_len`.

### Gradient checkpointing
Trades compute for memory: instead of caching every block's activations for backward, we re-compute them. ~30% slower but lets us fit a bigger model. We turn it on for Plan B.

### `torch.compile`
JIT-compiles the model graph to fused CUDA kernels. ~20-40% faster after a slow first step.

## 5. Data

We use **FineWeb-Edu** — a 1.3T-token subset of CommonCrawl filtered for educational quality by an LLM classifier. It's the cleanest large-scale pretraining corpus you can get for free right now. We tokenize once into binary `uint16` shards (vocab fits in 16 bits) and `mmap` them at training time so they appear as one big array but only the pages we touch are paged into RAM.

## 6. SFT (Supervised Fine-Tuning)

After pretraining you have a *base model* — completes text. To make it answer questions, we fine-tune on instruction data formatted as a chat:
```
<|system|>
You are a helpful assistant.<|end|>
<|user|>
What is the capital of France?<|end|>
<|assistant|>
The capital of France is Paris.<|end|>
```

Two key choices:
1. **Loss masking**: cross-entropy is applied *only* on assistant tokens. The user/system tokens contribute zero loss (we set targets to -100, the `ignore_index`). The model learns to *generate* responses, not memorize prompts.
2. **Lower learning rate** (~2e-5, 10x smaller than pretraining): the base model's weights are already "good," we just want to nudge them toward the response format. Big LR would catastrophically forget pretraining knowledge.

Dataset: `HuggingFaceH4/ultrachat_200k` — 200k synthetic-but-high-quality dialogues distilled from GPT-4.

## 7. What about LoRA / Unsloth?

Both are **fine-tuning accelerators** — not pretraining. They freeze the base model and train tiny low-rank adapters (a few % of params) instead of all of them. Useful when you have a 7B+ base model and want to fine-tune on a single GPU. They don't help us here because (a) we're training from scratch — there's no base model to attach LoRA to, (b) our 110M model already fits comfortably on one 3090 in full precision.

We *could* use LoRA on top of an existing model like Llama-3-8B to fine-tune on your data. That's a different project — much faster path to "answers questions well," but not "trained from scratch."

## 8. How loss numbers translate to quality

| Loss | What it means |
|---|---|
| ~10.4 | Random — uniform over 32k vocab (`ln 32000 ≈ 10.37`) |
| ~6 | Markov-chain-tier; sometimes valid bigrams |
| ~4-5 | English-shaped gibberish (our 5M smoke at 5.5) |
| ~3.5-4 | Grammatical sentences, often nonsensical content |
| ~3.0-3.5 | GPT-2-small quality; coherent paragraphs, weak reasoning |
| ~2.5 | TinyLlama / Pythia-1.4B; useful chatbot after SFT |
| ~2.0 | Frontier 7B+ models |

Loss is per-token cross-entropy in nats. Equivalently, **perplexity = exp(loss)** — the effective number of "options" the model is choosing among. Loss 3 ↔ perplexity 20.

## 9. Evaluation

After training, we'll run `lm-evaluation-harness` on standard benchmarks:
- **HellaSwag**: pick the right sentence completion (commonsense)
- **ARC-easy / ARC-challenge**: grade-school science MCQ
- **WinoGrande**: pronoun resolution / commonsense
- **PIQA**: physical commonsense
- **BoolQ**: yes/no reading comprehension

For a 110M model expect HellaSwag ~30-35% (random=25), ARC-easy ~40%. MMLU is essentially noise at this scale. The Open LLM Leaderboard accepts submissions and gives you a public score.
