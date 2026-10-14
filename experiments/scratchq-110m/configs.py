from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 16
    ffn_size: int = 2816
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5


@dataclass
class TrainConfig:
    name: str = "main_300m"
    out_dir: str = "out/main_300m"
    data_dir: str = "data/fineweb_edu"

    # Tokens
    seq_len: int = 2048
    micro_batch_size: int = 8
    grad_accum_steps: int = 32  # effective batch = 8 * 32 * 2048 = 524288 tokens/step

    # Schedule
    max_steps: int = 50_000        # ~26B tokens at 524k tokens/step
    warmup_steps: int = 2_000
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Logging / checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    eval_iters: int = 50
    ckpt_interval: int = 2_000
    keep_last_n: int = 3

    # Runtime
    dtype: str = "bfloat16"
    compile: bool = True
    grad_checkpoint: bool = True
    seed: int = 1337
    wandb_project: str = "private-llm"


# 10M-param smoke config — verifies the whole pipeline end-to-end
SMOKE_MODEL = ModelConfig(
    hidden_size=256,
    num_layers=6,
    num_heads=4,
    num_kv_heads=4,
    ffn_size=704,
    max_seq_len=512,
)

SMOKE_TRAIN = TrainConfig(
    name="smoke_10m",
    out_dir="out/smoke_10m",
    seq_len=512,
    micro_batch_size=16,
    grad_accum_steps=2,
    max_steps=2_000,
    warmup_steps=100,
    eval_interval=200,
    ckpt_interval=500,
    log_interval=10,
    compile=False,  # faster startup for smoke
    grad_checkpoint=False,
)

MAIN_MODEL = ModelConfig()  # 300M defaults
MAIN_TRAIN = TrainConfig()


# Plan B: ~110M params on 2B tokens, fits a 3090 / 4090 in ~8-15h
PLAN_B_MODEL = ModelConfig(
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    num_kv_heads=12,
    ffn_size=2048,
    max_seq_len=1024,  # shorter ctx → bigger batches, faster
)

PLAN_B_TRAIN = TrainConfig(
    name="plan_b_110m",
    out_dir="out/plan_b_110m",
    seq_len=1024,
    micro_batch_size=12,        # ~14 GB VRAM with grad ckpt on 3090
    grad_accum_steps=16,         # effective batch = 12*16*1024 ≈ 196k tok/step
    max_steps=10_500,            # ~2B tokens
    warmup_steps=500,
    lr=3e-4,
    min_lr=3e-5,
    eval_interval=500,
    ckpt_interval=1_000,
    log_interval=10,
    compile=True,
    grad_checkpoint=True,
)


PRESETS = {
    "smoke": (SMOKE_MODEL, SMOKE_TRAIN),
    "plan_b": (PLAN_B_MODEL, PLAN_B_TRAIN),
    "main":  (MAIN_MODEL, MAIN_TRAIN),
}
