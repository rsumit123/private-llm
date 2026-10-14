#!/bin/bash
# Attention ablation at depth 12 on 1x H100: MHA vs GQA (2 kv heads) vs MLA (480 latent + 32 rope), equal KV budget for
# GQA/MLA. Identical recipe for all: 1000 steps x 524,288 tokens, bf16, no value embeddings, same seed/data order.
# Then CORE (500/task) + val bpb + inference benchmark. Uploads logs/results, self-destroys, 110-min hard cap.
exec >> /root/abl.log 2>&1
export OMP_NUM_THREADS=1 NANOCHAT_BASE_DIR=/root/.cache/nanochat HF_REPO=sumitkClasses/nanochat-d20 PYTHONUNBUFFERED=1
export HF_TOKEN=$(cat /root/.hf_token)
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^CONTAINER_(ID|API_KEY)=" | xargs)
mark(){ echo "### $1 at $(date -u +%H:%M:%S)"; }
PY=/root/nanochat/.venv/bin/python
up(){ $PY -c "
import os,sys; from huggingface_hub import HfApi
HfApi(token=os.environ['HF_TOKEN']).upload_file(path_or_fileobj=sys.argv[1], path_in_repo='logs/ablation/'+os.path.basename(sys.argv[1]), repo_id=os.environ['HF_REPO'])" "$1" || true; }
destroy(){ mark "self-destroy ($1)"; { set +x; } 2>/dev/null; /root/nanochat/.venv/bin/vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" <<< y; }
( sleep 6600; echo "### HARD CAP"; for f in /root/abl*.log /root/results.jsonl; do up $f; done; destroy cap ) &

apt-get update -qq && apt-get install -y -qq curl ca-certificates python3.10-dev >/dev/null
mkdir -p /root/nanochat && tar -xzf /root/nanochat-mla.tar.gz -C /root/nanochat && cd /root/nanochat
curl -LsSf https://astral.sh/uv/install.sh | sh; source $HOME/.local/bin/env
uv venv -q && uv sync -q --extra gpu && source .venv/bin/activate && uv pip install -q huggingface_hub vastai
python -c "
import os; from huggingface_hub import snapshot_download
snapshot_download('sumitkClasses/nanochat-d20-chat', local_dir=os.environ['NANOCHAT_BASE_DIR'], allow_patterns=['tokenizer/*'])"
python -m nanochat.dataset -n 14
mark setup

COMMON="--depth=12 --num-iterations=1000 --total-batch-size=524288 --device-batch-size=32 --eval-every=250 --core-metric-every=-1 --sample-every=-1 --save-every=-1"
run(){  # tag, then env assignments
  local tag=$1; shift
  env NANOCHAT_NO_VE=1 "$@" python -m scripts.base_train $COMMON --model-tag=$tag > /root/abl_train_$tag.log 2>&1
  mark "train $tag rc=$?"; up /root/abl_train_$tag.log
  python -m scripts.base_eval --eval core,bpb --model-tag=$tag --max-per-task=500 > /root/abl_eval_$tag.log 2>&1
  mark "eval $tag rc=$?"; up /root/abl_eval_$tag.log
}
run a_mha NANOCHAT_KV_HEADS=6
run a_gqa NANOCHAT_KV_HEADS=2
run a_mla NANOCHAT_ATTN=mla NANOCHAT_MLA_KV_DIM=480 NANOCHAT_MLA_ROPE_DIM=32
python -m scripts.bench_attn --tags a_mha,a_gqa,a_mla --prompt 1024 --new 128 --batches 1,32 > /root/abl_bench.log 2>&1
mark "bench rc=$?"; grep "^{" /root/abl_bench.log > /root/results.jsonl; up /root/abl_bench.log; up /root/results.jsonl
echo ABL_DONE; up /root/abl.log
destroy done
