"""
Continued (domain-adaptive) pretraining: take the Plan B base ckpt, train it
for a small number of steps at a low LR on the newspaper text. Same loss
function as pretraining (next-token prediction), but starting from learned
weights and on a narrow corpus.

Goal: shift the model's vocab/style/topic distribution toward Indian-English
news, without catastrophically forgetting the FineWeb-Edu pretraining.

Usage:
    python continue_pretrain.py --base out/plan_b_110m/ckpt_0010500.pt \
        --data_dir data/newspapers \
        --out_dir out/plan_b_news \
        --steps 1500 --lr 5e-5 --warmup 50
"""
import argparse
import math
import time
from pathlib import Path

import torch

from configs import PRESETS
from data import ShardDataset
from model import LLM


def lr_at(step, peak, min_lr, warmup, total):
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, progress)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak - min_lr) * coeff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="path to base pretrained ckpt")
    ap.add_argument("--preset", default="plan_b")
    ap.add_argument("--data_dir", required=True, help="dir with shard_*.bin")
    ap.add_argument("--out_dir", default="out/plan_b_news")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-5)        # 6× smaller than pretrain peak
    ap.add_argument("--min_lr", type=float, default=5e-6)
    ap.add_argument("--micro_batch", type=int, default=12)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")

    mcfg, _ = PRESETS[args.preset]
    model = LLM(mcfg).to(device, dtype=dtype)
    print(f"loading base ckpt: {args.base}")
    ck = torch.load(args.base, map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    model.load_state_dict(state)
    model.train()
    print(f"params: {model.num_params()/1e6:.1f}M")

    ds = ShardDataset(args.data_dir, args.seq_len, seed=42)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=True,
    )

    t0 = time.time()
    tokens_per_step = args.micro_batch * args.grad_accum * args.seq_len
    for step in range(args.steps):
        lr = lr_at(step, args.lr, args.min_lr, args.warmup, args.steps)
        for g in opt.param_groups: g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(args.grad_accum):
            x, y = ds.get_batch(args.micro_batch, device)
            with torch.amp.autocast("cuda", dtype=dtype):
                _, loss = model(x, y, grad_checkpoint=True)
            (loss / args.grad_accum).backward()
            loss_acc += loss.item()
        loss_acc /= args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0:
            dt = time.time() - t0; t0 = time.time()
            tok_s = tokens_per_step * args.log_every / dt if step > 0 else 0
            print(f"step {step:5d} | loss {loss_acc:.4f} | lr {lr:.2e} | {tok_s:,.0f} tok/s")

    final = out_dir / "ckpt_news_final.pt"
    torch.save({"model": model.state_dict(), "step": args.steps}, final)
    print(f"saved {final}")


if __name__ == "__main__":
    main()
