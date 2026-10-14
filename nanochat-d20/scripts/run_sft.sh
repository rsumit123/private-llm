#!/bin/bash
# Re-run ONLY chat SFT from the uploaded d20 base checkpoint, upload + verify, self-destroy.
exec > /root/sft.log 2>&1
export DEBIAN_FRONTEND=noninteractive OMP_NUM_THREADS=1 NANOCHAT_BASE_DIR=/root/.cache/nanochat
export HF_REPO=sumitkClasses/nanochat-d20
export HF_TOKEN=$(cat /root/.hf_token)
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^CONTAINER_(ID|API_KEY)=" | xargs)
CAP_SECONDS=3840   # 64 min hard cap
set -x
T0=$(date +%s); mark(){ echo "### $1 at +$(( $(date +%s)-T0 ))s"; }
self_destroy(){ mark "self-destroy ($1)"; { set +x; } 2>/dev/null
  /root/nanochat/.venv/bin/vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" <<< y; }
( sleep "$CAP_SECONDS"; echo "### HARD CAP"; self_destroy cap ) &

apt-get update -qq && apt-get install -y -qq git curl ca-certificates python3.10-dev >/dev/null
cd /root && git clone -q https://github.com/karpathy/nanochat && cd nanochat && git checkout -q 92d63d4e8b
curl -LsSf https://astral.sh/uv/install.sh | sh; source $HOME/.local/bin/env
uv venv -q && uv sync -q --extra gpu && source .venv/bin/activate
uv pip install -q huggingface_hub vastai
mark deps
python - <<'PY'
import os
from huggingface_hub import snapshot_download
base = os.environ["NANOCHAT_BASE_DIR"]
snapshot_download(os.environ["HF_REPO"], allow_patterns=["tokenizer/*"], local_dir=base)
snapshot_download(os.environ["HF_REPO"], allow_patterns=["base/*"], local_dir=base + "/tmp_base")
os.makedirs(base + "/base_checkpoints", exist_ok=True)
os.rename(base + "/tmp_base/base", base + "/base_checkpoints/d20")
print(os.listdir(base + "/base_checkpoints/d20"), os.listdir(base + "/tokenizer"))
PY
mark download

python -m scripts.chat_sft --model-tag=d20 --chatcore-every=-1 --eval-every=-1
SFT_RC=$?; mark "sft rc=$SFT_RC"

python - <<'PY'
import glob, os
from huggingface_hub import HfApi
api, repo = HfApi(token=os.environ["HF_TOKEN"]), os.environ["HF_REPO"]
d = os.environ["NANOCHAT_BASE_DIR"] + "/chatsft_checkpoints/d20"
files = sorted(f for f in glob.glob(d + "/*") if f.endswith((".pt", ".json")) and "/optim_" not in f)
print("local:", [(os.path.basename(f), os.path.getsize(f)) for f in files])
api.upload_folder(repo_id=repo, folder_path=d, path_in_repo="sft", allow_patterns=["model_*.pt", "meta_*.json"])
remote = {f.path.split("/")[-1]: (f.lfs.size if f.lfs else f.size) for f in api.list_repo_tree(repo, path_in_repo="sft")}
ok = all(remote.get(os.path.basename(f)) == os.path.getsize(f) for f in files) and files
print("SFT_UPLOAD_VERIFIED" if ok else f"SFT_UPLOAD_MISMATCH {remote}")
PY
mark uploaded
python -c "
import os; from huggingface_hub import HfApi
HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj='/root/sft.log', path_in_repo='logs/sft_rerun.log', repo_id=os.environ['HF_REPO'])"
echo RUN_DONE
self_destroy done
