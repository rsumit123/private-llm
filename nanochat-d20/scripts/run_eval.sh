#!/bin/bash
# Benchmarks for nanochat-d20 on any >=10 GB GPU (ran on an RTX A4000, image pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime):
# CORE on the base model (500 examples/task), HumanEval + GSM8K (first 400) on the chat model.
# Logs are uploaded to the Hub as each benchmark finishes; the instance destroys itself at the end or after 55 min.
exec >> /root/eval.log 2>&1
export OMP_NUM_THREADS=1 NANOCHAT_BASE_DIR=/root/nc HF_REPO=sumitkClasses/nanochat-d20 PYTHONUNBUFFERED=1
export HF_TOKEN=$(cat /root/.hf_token)
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^CONTAINER_(ID|API_KEY)=" | xargs)
mark(){ echo "### $1 at $(date -u +%H:%M:%S)"; }
up(){ python -c "
import os,sys; from huggingface_hub import HfApi
HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj=sys.argv[1], path_in_repo='logs/'+os.path.basename(sys.argv[1]), repo_id=os.environ['HF_REPO'])" "$1" || true; }
destroy(){ mark "self-destroy ($1)"; vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" <<< y; }
( sleep 3300; echo "### HARD CAP"; up /root/eval.log; destroy cap ) &

pip install -q --root-user-action=ignore tiktoken rustbpe filelock jinja2 pyarrow pyyaml psutil huggingface_hub vastai
python - <<'PY'
import io, os, tarfile, urllib.request
from huggingface_hub import snapshot_download
tf = tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen("https://codeload.github.com/karpathy/nanochat/tar.gz/92d63d4e8b").read()))
tf.extractall("/root"); os.rename([f"/root/{d}" for d in os.listdir("/root") if d.startswith("nanochat-")][0], "/root/nanochat")
b = os.environ["NANOCHAT_BASE_DIR"]
snapshot_download(os.environ["HF_REPO"], local_dir=b + "/dl", allow_patterns=["tokenizer/*", "sft/*", "base/model_*", "base/meta_*"])
os.makedirs(b + "/base_checkpoints"); os.makedirs(b + "/chatsft_checkpoints")
os.rename(b + "/dl/tokenizer", b + "/tokenizer"); os.rename(b + "/dl/base", b + "/base_checkpoints/d20"); os.rename(b + "/dl/sft", b + "/chatsft_checkpoints/d20")
PY
mark setup
cd /root/nanochat
python -m scripts.base_eval --eval core --model-tag d20 --max-per-task=500 > /root/eval_core.log 2>&1; mark "core rc=$?"; up /root/eval_core.log
python -m scripts.chat_eval -i sft -a HumanEval > /root/eval_humaneval.log 2>&1; mark "humaneval rc=$?"; up /root/eval_humaneval.log
python -m scripts.chat_eval -i sft -a GSM8K -x 400 > /root/eval_gsm8k.log 2>&1; mark "gsm8k rc=$?"; up /root/eval_gsm8k.log
echo EVAL_DONE; up /root/eval.log
destroy done
