"""
Fetch a curated list of Indian-topic Wikipedia articles, chunk them into
~200-word passages, embed each chunk with a small sentence-transformer,
and save the resulting knowledge base to disk.

Outputs:
    kb/kb.jsonl   — one chunk per line: {id, title, text}
    kb/kb.npy     — float32 (N, D) embedding matrix
    kb/kb.meta.json — {model, dim, count}

Usage:
    python build_kb.py --out_dir kb
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import requests


# Curated seed topics covering MCQ categories we care about.
# Wikipedia search endpoint will resolve each to the best matching article.
SEEDS = [
    # Hindu mythology / Ramayana / Mahabharata
    "Ramayana", "Rama", "Lakshmana", "Bharata (Ramayana)", "Shatrughna",
    "Sita", "Ravana", "Hanuman", "Vibhishana", "Indrajit",
    "Mahabharata", "Krishna", "Arjuna", "Bhima", "Yudhishthira",
    "Nakula", "Sahadeva", "Karna", "Draupadi", "Duryodhana",
    "Bhagavad Gita", "Vishnu", "Shiva", "Brahma", "Ganesha",
    "Hindu deities", "Avatars of Vishnu",
    # Indian history
    "History of India", "Indian independence movement", "Mahatma Gandhi",
    "Jawaharlal Nehru", "Sardar Vallabhbhai Patel", "B. R. Ambedkar",
    "Subhas Chandra Bose", "Bhagat Singh", "Mughal Empire", "Akbar",
    "Shah Jahan", "Aurangzeb", "Maratha Empire", "Shivaji",
    "British Raj", "Partition of India",
    # Geography
    "Geography of India", "States and union territories of India",
    "Ganges", "Yamuna", "Brahmaputra", "Godavari", "Krishna River",
    "Kaveri", "Indus River", "Himalayas", "Western Ghats", "Eastern Ghats",
    "Thar Desert", "Deccan Plateau",
    # Politics / governance
    "Government of India", "Constitution of India", "President of India",
    "Prime Minister of India", "Parliament of India", "Supreme Court of India",
    "Lok Sabha", "Rajya Sabha", "Indian National Congress", "Bharatiya Janata Party",
    # Defense
    "Indian Armed Forces", "Indian Army", "Indian Air Force", "Indian Navy",
    "Defence Research and Development Organisation",
    # Sports
    "Cricket in India", "Sachin Tendulkar", "Virat Kohli", "M. S. Dhoni",
    "Indian Premier League", "Field hockey in India", "Major Dhyan Chand",
    # Science / general
    "Science and technology in India", "Indian Space Research Organisation",
    "Chandrayaan-3", "C. V. Raman", "A. P. J. Abdul Kalam",
    # Food
    "Indian cuisine", "Biryani", "Curry", "Chai (drink)", "Masala dosa",
    # Entertainment
    "Cinema of India", "Bollywood", "Satyajit Ray", "Amitabh Bachchan",
    "Lata Mangeshkar", "Indian classical music", "Bharatanatyam",
]

WIKI_API = "https://en.wikipedia.org/w/api.php"
CHUNK_WORDS = 180   # ~200-token chunks
HDRS = {"User-Agent": "private-llm-rag/0.1 (research)"}


def fetch_article(title):
    # First resolve to canonical title via search (handles minor name mismatches)
    r = requests.get(WIKI_API, params={
        "action": "query", "list": "search", "srsearch": title,
        "format": "json", "srlimit": 1,
    }, headers=HDRS, timeout=20)
    hits = r.json().get("query", {}).get("search", [])
    if not hits:
        return None, None
    canonical = hits[0]["title"]

    r = requests.get(WIKI_API, params={
        "action": "query", "prop": "extracts",
        "titles": canonical, "explaintext": True, "exsectionformat": "plain",
        "format": "json", "redirects": 1,
    }, headers=HDRS, timeout=30)
    pages = r.json().get("query", {}).get("pages", {})
    for _, p in pages.items():
        text = p.get("extract", "")
        if text:
            return canonical, text
    return canonical, None


def chunkify(text, n_words=CHUNK_WORDS):
    paras = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    chunks, buf, count = [], [], 0
    for para in paras:
        words = para.split()
        if count + len(words) > n_words and buf:
            chunks.append(" ".join(buf)); buf, count = [], 0
        buf.extend(words); count += len(words)
        if count >= n_words:
            chunks.append(" ".join(buf)); buf, count = [], 0
    if buf:
        chunks.append(" ".join(buf))
    return [c for c in chunks if len(c.split()) >= 30]  # drop tiny scraps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="kb")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # 1. Fetch articles
    print(f"fetching {len(SEEDS)} Wikipedia articles…")
    chunks = []
    for i, seed in enumerate(SEEDS):
        canonical, text = fetch_article(seed)
        if not text:
            print(f"  [{i+1}] {seed} -> MISS"); continue
        for j, c in enumerate(chunkify(text)):
            chunks.append({"id": f"{canonical}#{j}", "title": canonical, "text": c})
        print(f"  [{i+1}] {seed} -> {canonical}: {len(text):,} chars")
        time.sleep(0.1)  # be polite to Wikipedia

    print(f"\ntotal chunks: {len(chunks)}")
    with open(out / "kb.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 2. Embed
    from sentence_transformers import SentenceTransformer
    print(f"\nembedding with {args.model}…")
    enc = SentenceTransformer(args.model)
    texts = [f"{c['title']}: {c['text']}" for c in chunks]
    emb = enc.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    np.save(out / "kb.npy", emb)

    meta = {"model": args.model, "dim": int(emb.shape[1]), "count": int(emb.shape[0])}
    json.dump(meta, open(out / "kb.meta.json", "w"), indent=2)
    print(f"saved kb to {out} | {emb.shape[0]} vectors, dim={emb.shape[1]}")


if __name__ == "__main__":
    main()
