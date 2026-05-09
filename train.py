"""
Train a Llama-style LLM on tokenized shards.

Usage:
    python train.py smoke              # 10M smoke run, ~hours on a 5090
    python train.py main               # 300M run, multi-day on a 5090
    python train.py main --resume      # resume from latest checkpoint
"""
import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from configs import PRESETS
from data import ShardDataset
from model import LLM


def lr_at(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * coeff


def save_ckpt(path, model, opt, step, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "cfg": cfg,
    }, path)


def prune_ckpts(out_dir: Path, keep: int):
    cks = sorted(out_dir.glob("ckpt_*.pt"))
    for old in cks[:-keep]:
        old.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", choices=list(PRESETS.keys()))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    mcfg, tcfg = PRESETS[args.preset]
    out_dir = Path(tcfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(tcfg.seed)

    device = "cuda"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[tcfg.dtype]
    torch.set_float32_matmul_precision("high")

    print(f"=== {tcfg.name} ===")
    print(f"model: hidden={mcfg.hidden_size} layers={mcfg.num_layers} heads={mcfg.num_heads}")

    ds = ShardDataset(tcfg.data_dir, tcfg.seq_len, seed=tcfg.seed)
    model = LLM(mcfg).to(device, dtype=dtype)
    print(f"params: {model.num_params()/1e6:.1f}M")

    # Param groups: weight-decay only on 2D weights
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tcfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tcfg.lr, betas=(tcfg.beta1, tcfg.beta2), fused=True,
    )

    start_step = 0
    if args.resume:
        cks = sorted(out_dir.glob("ckpt_*.pt"))
        if cks:
            ck = torch.load(cks[-1], map_location=device)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            start_step = ck["step"] + 1
            print(f"resumed from {cks[-1]} at step {start_step}")

    if tcfg.compile:
        model = torch.compile(model)

    if args.wandb:
        import wandb
        wandb.init(project=tcfg.wandb_project, name=tcfg.name, config={**vars(mcfg), **vars(tcfg)})

    model.train()
    t0 = time.time()
    tokens_per_step = tcfg.micro_batch_size * tcfg.grad_accum_steps * tcfg.seq_len

    for step in range(start_step, tcfg.max_steps):
        lr = lr_at(step, tcfg)
        for g in opt.param_groups: g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(tcfg.grad_accum_steps):
            x, y = ds.get_batch(tcfg.micro_batch_size, device)
            with torch.amp.autocast("cuda", dtype=dtype):
                _, loss = model(x, y, grad_checkpoint=tcfg.grad_checkpoint)
            (loss / tcfg.grad_accum_steps).backward()
            loss_acc += loss.item()
        loss_acc /= tcfg.grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()

        if step % tcfg.log_interval == 0:
            dt = time.time() - t0; t0 = time.time()
            tok_s = tokens_per_step * tcfg.log_interval / dt if step > 0 else 0
            print(f"step {step:6d} | loss {loss_acc:.4f} | lr {lr:.2e} | {tok_s:,.0f} tok/s")
            if args.wandb:
                import wandb; wandb.log({"loss": loss_acc, "lr": lr, "tok_s": tok_s, "step": step})

        if step > 0 and step % tcfg.ckpt_interval == 0:
            save_ckpt(out_dir / f"ckpt_{step:07d}.pt", model, opt, step, {"model": mcfg, "train": tcfg})
            prune_ckpts(out_dir, tcfg.keep_last_n)

    save_ckpt(out_dir / f"ckpt_{tcfg.max_steps:07d}.pt", model, opt, tcfg.max_steps, {"model": mcfg, "train": tcfg})
    print("done.")


if __name__ == "__main__":
    main()
