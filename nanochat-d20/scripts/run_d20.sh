#!/bin/bash
# Unattended nanochat d20 run on 1x H100: tokenizer -> pretrain -> SFT -> samples.
# Uploads checkpoints to HF as it goes and destroys its own vast instance at the end
# or at the hard cap. Log: /root/run.log
exec > /root/run.log 2>&1
export DEBIAN_FRONTEND=noninteractive OMP_NUM_THREADS=1 NANOCHAT_BASE_DIR=/root/.cache/nanochat
HF_REPO=sumitkClasses/nanochat-d20; export HF_REPO
export HF_TOKEN=$(cat /root/.hf_token)
CAP_SECONDS=${CAP_SECONDS:-26280}   # 7.3 h hard cap: upload whatever exists, then self-destroy
export $(tr "\0" "\n" < /proc/1/environ | grep -E "^CONTAINER_(ID|API_KEY)=" | xargs)
set -x   # tracing on only after secrets are loaded
T0=$(date +%s); mark(){ echo "### $1 at +$(( $(date +%s)-T0 ))s"; }

self_destroy(){
  mark "self-destroy ($1)"
  sync
  { set +x; } 2>/dev/null
  /root/nanochat/.venv/bin/vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" <<< y
}

# hard cap: upload whatever exists, then destroy, no matter what stage we're in
( sleep "$CAP_SECONDS"; echo "### HARD CAP"; /root/upload.sh final || true; self_destroy cap ) &

apt-get update -qq && apt-get install -y -qq git curl ca-certificates python3.10-dev >/dev/null
cd /root && git clone -q https://github.com/karpathy/nanochat && cd nanochat && git checkout -q 92d63d4e8b && git log -1 --format='%h %cd'
curl -LsSf https://astral.sh/uv/install.sh | sh; source $HOME/.local/bin/env
uv venv -q && uv sync -q --extra gpu && source .venv/bin/activate
uv pip install -q huggingface_hub vastai
{ set +x; } 2>/dev/null; vastai show instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" --raw >/dev/null && echo "SELF_KEY_OK" || echo "SELF_KEY_FAILED"; set -x
mark deps

# tokenizer on the first 8 shards while the rest download (d20 @ ratio 8 needs ~3.5B tokens)
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 90 &
DL=$!
python -m scripts.tok_train && python -m scripts.tok_eval
mark tokenizer
wait $DL; mark data
/root/upload.sh tokenizer

# background uploader: newest base checkpoint -> HF, prune older local ones
( while true; do sleep 600; /root/upload.sh base; done ) &
UPL=$!

python -m scripts.base_train --depth=20 --target-param-data-ratio=8 --fp8 --device-batch-size=32 \
  --save-every=500 --eval-every=500 --core-metric-every=-1 --sample-every=1000 --model-tag=d20
BASE_RC=$?
mark "pretrain rc=$BASE_RC"
kill $UPL 2>/dev/null
sleep 70; /root/upload.sh base
[ $BASE_RC -ne 0 ] && self_destroy "pretrain failed"

python -m scripts.chat_sft --model-tag=d20 --chatcore-every=-1
mark "sft rc=$?"
/root/upload.sh sft

for p in "Hi! Who are you?" "Why is the sky blue?" "Write a short poem about the monsoon in India." \
         "What is the capital of France, and what is it famous for?" "Explain photosynthesis to a 10 year old."; do
  echo "=== PROMPT: $p"; python -m scripts.chat_cli -p "$p" 2>&1 | tail -15
done > /root/samples.txt
cp /root/samples.txt /root/run.log.copy 2>/dev/null; /root/upload.sh final
mark done
echo RUN_DONE
self_destroy done
