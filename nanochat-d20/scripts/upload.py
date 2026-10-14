import glob, os, re, sys, time
from huggingface_hub import HfApi

api, repo = HfApi(token=os.environ["HF_TOKEN"]), os.environ["HF_REPO"]
base = os.environ["NANOCHAT_BASE_DIR"]
mode = sys.argv[1]
print(time.strftime("%H:%M:%S"), "upload", mode, flush=True)


def steps(d):
    return sorted({int(m.group(1)) for f in glob.glob(f"{d}/model_*.pt")
                   if (m := re.search(r"model_(\d+)\.pt$", f))})


def complete(d, s, need_optim):
    files = [f"{d}/model_{s:06d}.pt", f"{d}/meta_{s:06d}.json"]
    if need_optim:
        files.append(f"{d}/optim_{s:06d}_rank0.pt")
    return all(os.path.exists(f) and time.time() - os.path.getmtime(f) > 60 for f in files)


def push_latest(local_dir, repo_dir, need_optim):
    ss = [s for s in steps(local_dir) if complete(local_dir, s, need_optim)]
    if not ss:
        print("no complete checkpoint yet"); return
    s = ss[-1]
    marker = f"{local_dir}/.uploaded_{s}"
    if not os.path.exists(marker):
        api.upload_folder(repo_id=repo, folder_path=local_dir, path_in_repo=repo_dir,
                          allow_patterns=[f"*_{s:06d}*"], delete_patterns=["*.pt", "*.json"],
                          commit_message=f"{repo_dir} step {s}")
        open(marker, "w").close()
        print("uploaded", repo_dir, s, flush=True)
    for old in steps(local_dir):
        if old < s:
            for f in glob.glob(f"{local_dir}/*_{old:06d}*"):
                os.remove(f)
            print("pruned local step", old)


if mode == "tokenizer":
    api.upload_folder(repo_id=repo, folder_path=f"{base}/tokenizer", path_in_repo="tokenizer")
elif mode == "base":
    push_latest(f"{base}/base_checkpoints/d20", "base", need_optim=True)
elif mode == "sft":
    d = glob.glob(f"{base}/chatsft_checkpoints/*")
    if d:
        push_latest(sorted(d)[-1], "sft", need_optim=False)
elif mode == "final":
    for f in ["/root/run.log", "/root/samples.txt", "/root/upload.log"]:
        if os.path.exists(f):
            api.upload_file(path_or_fileobj=f, path_in_repo=f"logs/{os.path.basename(f)}", repo_id=repo)
    if os.path.isdir(f"{base}/report"):
        api.upload_folder(repo_id=repo, folder_path=f"{base}/report", path_in_repo="report")
print("ok", mode)
