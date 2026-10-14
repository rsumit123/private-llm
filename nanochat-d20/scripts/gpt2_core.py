"""CORE (500/task) for HF GPT-2 checkpoints with nanochat's own harness, for an apples-to-apples baseline."""
import sys, torch
from transformers import AutoModelForCausalLM
from nanochat.common import autodetect_device_type, compute_init
from nanochat.tokenizer import RustBPETokenizer
from scripts.base_eval import evaluate_core

class ModelWrapper:  # from nanochat 4186540~1 scripts/base_eval.py
    def __init__(self, model, max_seq_len=None):
        self.model, self.max_seq_len = model, max_seq_len
    def __call__(self, input_ids, targets=None, loss_reduction="mean"):
        return self.model(input_ids).logits
    def get_device(self):
        return next(self.model.parameters()).device

_, _, _, _, device = compute_init(autodetect_device_type())
for name in sys.argv[1:]:
    m = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    res = evaluate_core(ModelWrapper(m, max_seq_len=1024), RustBPETokenizer.from_pretrained("gpt2"), device, max_per_task=500)
    print(f"RESULT {name} CORE={res['core_metric']:.4f}", flush=True)
