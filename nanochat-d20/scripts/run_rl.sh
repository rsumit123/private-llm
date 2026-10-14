#!/bin/bash
# RL (nanochat's simplified GRPO/REINFORCE) on GSM8K, starting from the d20 chat (SFT) checkpoint, on 1x H100.
# Capped at MAX_STEPS optimizer steps (the LR schedule decays to zero over the capped length), then evaluates
# GSM8K (first 400, same subset as the SFT baseline) and ARC/MMLU (regression check), uploads, and self-destroys.
exec >> /root/rl.log 2>&1
export OMP_NUM_THREADS=1 NANOCHAT_BASE_DIR=/root/.cache/nanochat HF_REPO=sumitkClasses/nanochat-d20 PYTHONUNBUFFERED=1
export HF_TOKEN=$(cat /root/.hf_token)
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^CONTAINER_(ID|API_KEY)=" | xargs)
MAX_STEPS=${MAX_STEPS:-170}
mark(){ echo "### $1 at $(date -u +%H:%M:%S)"; }
up(){ /root/nanochat/.venv/bin/python -c "
import os,sys; from huggingface_hub import HfApi
HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj=sys.argv[1], path_in_repo='logs/'+os.path.basename(sys.argv[1]), repo_id=os.environ['HF_REPO'])" "$1" || true; }
push_rl(){ /root/nanochat/.venv/bin/python - <<'PY' || true
import glob, os, re
from huggingface_hub import HfApi
d = os.environ["NANOCHAT_BASE_DIR"] + "/chatrl_checkpoints/d20"
steps = sorted({int(m.group(1)) for f in glob.glob(d + "/model_*.pt") if (m := re.search(r"model_(\d+)\.pt$", f))})
if not steps: print("no rl checkpoint yet"); raise SystemExit
s = steps[-1]
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(repo_id=os.environ["HF_REPO"], folder_path=d, path_in_repo="rl",
                  allow_patterns=[f"model_{s:06d}.pt", f"meta_{s:06d}.json"], delete_patterns=["*.pt", "*.json"])
remote = {f.path.split("/")[-1]: (f.lfs.size if f.lfs else f.size) for f in api.list_repo_tree(os.environ["HF_REPO"], path_in_repo="rl")}
ok = remote.get(f"model_{s:06d}.pt") == os.path.getsize(f"{d}/model_{s:06d}.pt")
print(f"RL_UPLOAD step {s} verified={ok}")
PY
}
destroy(){ mark "self-destroy ($1)"; { set +x; } 2>/dev/null; /root/nanochat/.venv/bin/vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" <<< y; }
( sleep 7920; echo "### HARD CAP"; push_rl; up /root/rl.log; destroy cap ) &

apt-get update -qq && apt-get install -y -qq git curl ca-certificates python3.10-dev >/dev/null
cd /root && git clone -q https://github.com/karpathy/nanochat && cd nanochat && git checkout -q 92d63d4e8b
curl -LsSf https://astral.sh/uv/install.sh | sh; source $HOME/.local/bin/env
uv venv -q && uv sync -q --extra gpu && source .venv/bin/activate && uv pip install -q huggingface_hub vastai
# cap the step count without touching the rest of the recipe
sed -i 's|^num_steps = (len(train_task) // args.examples_per_step) \* args.num_epochs$|&\nif os.environ.get("MAX_STEPS"): num_steps = min(num_steps, int(os.environ["MAX_STEPS"]))|' scripts/chat_rl.py
grep -n "MAX_STEPS" scripts/chat_rl.py
python -c "
import os; from huggingface_hub import snapshot_download
snapshot_download('sumitkClasses/nanochat-d20-chat', local_dir=os.environ['NANOCHAT_BASE_DIR'], allow_patterns=['chatsft_checkpoints/*','tokenizer/*'])"
mark setup

MAX_STEPS=$MAX_STEPS python -m scripts.chat_rl --model-tag=d20 --device-batch-size=16 --eval-every=100 --eval-examples=200 --save-every=25 > /root/rl_train.log 2>&1
mark "rl rc=$?"; push_rl; up /root/rl_train.log

python -m scripts.chat_eval -i rl -a GSM8K -x 400 > /root/rl_eval_gsm8k.log 2>&1; mark "gsm8k rc=$?"; up /root/rl_eval_gsm8k.log
python -m scripts.chat_eval -i rl -a "ARC-Easy|ARC-Challenge|MMLU" -b 32 > /root/rl_eval_mc.log 2>&1; mark "mc rc=$?"; up /root/rl_eval_mc.log
echo RL_DONE; up /root/rl.log
destroy done
