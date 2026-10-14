#!/bin/bash
# CORE (500/task) for GPT-2 124M and 355M with nanochat's harness; upload log; self-destroy; 45-min cap.
exec >> /root/gpt2.log 2>&1
export OMP_NUM_THREADS=1 NANOCHAT_BASE_DIR=/root/nc HF_REPO=sumitkClasses/nanochat-d20 PYTHONUNBUFFERED=1
export HF_TOKEN=$(cat /root/.hf_token)
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^CONTAINER_(ID|API_KEY)=" | xargs)
mark(){ echo "### $1 at $(date -u +%H:%M:%S)"; }
up(){ python -c "
import os,sys; from huggingface_hub import HfApi
HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj=sys.argv[1], path_in_repo='logs/'+os.path.basename(sys.argv[1]), repo_id=os.environ['HF_REPO'])" "$1" || true; }
destroy(){ mark "self-destroy ($1)"; vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" <<< y; }
( sleep 2700; echo "### HARD CAP"; up /root/gpt2.log; destroy cap ) &
pip install -q --root-user-action=ignore tiktoken rustbpe filelock jinja2 pyarrow pyyaml psutil wandb transformers huggingface_hub vastai
python - <<'PY'
import io, os, tarfile, urllib.request
tf = tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen("https://codeload.github.com/karpathy/nanochat/tar.gz/92d63d4e8b").read()))
tf.extractall("/root"); os.rename([f"/root/{d}" for d in os.listdir("/root") if d.startswith("nanochat-")][0], "/root/nanochat")
PY
cp /root/gpt2_core.py /root/nanochat/ && cd /root/nanochat && mark setup
python gpt2_core.py gpt2 gpt2-medium; mark "gpt2 rc=$?"
echo GPT2_DONE; up /root/gpt2.log
destroy done
