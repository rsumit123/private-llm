"""
Fetch a specific list of critical Wikipedia articles by direct URL and
append them to an existing KB. Use this to patch articles that the
build_kb.py search/title API kept missing.

Usage:
    python kb_addon.py --kb_dir kb
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests


# Articles we know are critical (often-asked) but kept missing in build_kb.py
MUST_HAVE = [
    "Jawaharlal_Nehru",
    "New_Delhi",
    "Delhi",
    "Rabindranath_Tagore",
    "Mahatma_Gandhi",
    "Indian_independence_movement",
    "Constitution_of_India",
    "Indian_Air_Force",
    "Indian_Space_Research_Organisation",
    "Chandrayaan-3",
    "Jana_Gana_Mana",
    "President_of_India",
    "Mughal_Empire",
    "Rajendra_Prasad",
    "Subhas_Chandra_Bose",
    "Bhagat_Singh",
    "Tipu_Sultan",
    "Maratha_Empire",
    "Indian_Premier_League",
    "Independence_Day_(India)",
    "Republic_Day_(India)",
    "Sundarbans",
    "States_and_union_territories_of_India",
    "Khan_Abdul_Ghaffar_Khan",
    "Sarojini_Naidu",
    "Lal_Bahadur_Shastri",
    "Indira_Gandhi",
    "Atal_Bihari_Vajpayee",
    "Narendra_Modi",
    "Bihar",
    "Tamil_Nadu",
    "Maharashtra",
    "Karnataka",
    "Gujarat",
    "Punjab,_India",
    "Rajasthan",
    "Uttar_Pradesh",
    "West_Bengal",
    "Indian_cuisine",
    "Diwali",
    "Holi",
    "Eid_al-Fitr",
    "Christmas",
]

CHUNK_WORDS = 180
HDRS = {"User-Agent": "private-llm-rag/0.1"}


def fetch_via_rest(slug):
    """Use the REST API by slug — much more reliable than action=query."""
    url = f"https://en.wikipedia.org/w/api.php"
    r = requests.get(url, params={
        "action": "query", "prop": "extracts", "titles": slug.replace("_", " "),
        "explaintext": True, "exsectionformat": "plain",
        "format": "json", "redirects": 1,
    }, headers=HDRS, timeout=30)
    j = r.json()
    pages = j.get("query", {}).get("pages", {})
    for pid, p in pages.items():
        if pid == "-1": return None, None
        title = p.get("title", slug)
        text = p.get("extract", "")
        if text and len(text) > 200:
            return title, text
    return None, None


def chunkify(text, n=CHUNK_WORDS):
    import re
    paras = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks, buf, count = [], [], 0
    for para in paras:
        words = para.split()
        if count + len(words) > n and buf:
            chunks.append(" ".join(buf)); buf, count = [], 0
        buf.extend(words); count += len(words)
        if count >= n:
            chunks.append(" ".join(buf)); buf, count = [], 0
    if buf: chunks.append(" ".join(buf))
    return [c for c in chunks if len(c.split()) >= 30]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb_dir", default="kb")
    args = ap.parse_args()

    kb_dir = Path(args.kb_dir)
    existing = [json.loads(l) for l in open(kb_dir / "kb.jsonl")]
    existing_titles = set(c["title"] for c in existing)
    existing_emb = np.load(kb_dir / "kb.npy")
    meta = json.load(open(kb_dir / "kb.meta.json"))
    print(f"existing KB: {len(existing)} chunks across {len(existing_titles)} articles")

    new_chunks = []
    for slug in MUST_HAVE:
        title, text = fetch_via_rest(slug)
        if not text:
            print(f"  MISS: {slug}")
            continue
        if title in existing_titles:
            print(f"  SKIP (exists): {title}")
            continue
        cs = chunkify(text)
        for j, c in enumerate(cs):
            new_chunks.append({"id": f"{title}#{j}", "title": title, "text": c})
        print(f"  OK: {slug} -> {title}: {len(cs)} chunks")
        time.sleep(0.1)

    print(f"\nadding {len(new_chunks)} new chunks")
    if not new_chunks:
        print("nothing to add"); return

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(meta["model"])
    texts = [f"{c['title']}: {c['text']}" for c in new_chunks]
    new_emb = enc.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    new_emb = np.asarray(new_emb, dtype=np.float32)

    combined_chunks = existing + new_chunks
    combined_emb = np.concatenate([existing_emb, new_emb], axis=0)
    with open(kb_dir / "kb.jsonl", "w") as f:
        for c in combined_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save(kb_dir / "kb.npy", combined_emb)
    meta["count"] = int(combined_emb.shape[0])
    json.dump(meta, open(kb_dir / "kb.meta.json", "w"), indent=2)
    print(f"\nKB now has {len(combined_chunks)} chunks (was {len(existing)})")


if __name__ == "__main__":
    main()
