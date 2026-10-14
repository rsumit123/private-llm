# nanochat-d20: scripts, serving and results

See the [top-level README](../README.md) for the write-up. This folder holds everything needed to repeat the run.

| Path | What it is |
|---|---|
| `scripts/run_d20.sh` | Unattended run on a rented 1x H100: deps → tokenizer → pretrain (d20, data:param 8, FP8) → chat SFT → sample chats → upload → self-destroy, with a hard time cap |
| `scripts/upload.py`, `upload.sh` | Background uploader: pushes the newest complete checkpoint to the Hub and prunes older local ones |
| `scripts/run_sft.sh` | Re-runs only chat fine-tuning from the uploaded base checkpoint and verifies the upload before shutdown |
| `scripts/run_eval.sh` | CORE (base model) + HumanEval + GSM8K on any >=10 GB GPU |
| `scripts/run_rl.sh` | RL on GSM8K from the chat checkpoint (step-capped, LR decay intact), then GSM8K/ARC/MMLU |
| `scripts/gpt2_core.py`, `run_gpt2.sh` | Scores HF GPT-2 checkpoints with nanochat's CORE harness for a same-settings baseline |
| `serve/app.py` | Gradio chat UI + OpenAI-compatible `/v1` API in one file (deployed to the Hugging Face Space) |
| `results/benchmarks.json` | All scores, including per-task CORE |
| `results/pretraining_curves.png` | Training loss and validation bits-per-byte |
| `results/samples_raw.txt` | Sample chats generated at the end of the run |
| `results/logs/` | Full pretraining, SFT and evaluation logs |

The scripts expect `HF_TOKEN` in `/root/.hf_token` and, for self-destroy, the `CONTAINER_ID` / `CONTAINER_API_KEY`
variables that vast.ai injects into every instance. ARC/MMLU were evaluated with
`python -m scripts.chat_eval -i sft -a "ARC-Easy|ARC-Challenge|MMLU"`; the rest with `run_eval.sh`.
