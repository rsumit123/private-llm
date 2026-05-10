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


# Curated seed topics. We look these up by direct title with a search fallback.
# Use exact Wikipedia article titles when known.
SEEDS = [
    # Hindu mythology — Ramayana
    "Ramayana", "Rama", "Lakshmana", "Bharata (Ramayana)", "Shatrughna",
    "Sita", "Ravana", "Hanuman", "Vibhishana", "Indrajit", "Kumbhakarna",
    "Dasharatha", "Kausalya", "Kaikeyi", "Sumitra", "Jatayu", "Sugriva",
    "Vali (Ramayana)", "Angada", "Jambavan",
    # Hindu mythology — Mahabharata
    "Mahabharata", "Krishna", "Arjuna", "Bhima", "Yudhishthira",
    "Nakula", "Sahadeva", "Karna", "Draupadi", "Duryodhana", "Dushasana",
    "Bhishma", "Drona", "Kripa", "Ashwatthama", "Shakuni", "Kunti", "Gandhari",
    "Abhimanyu", "Ekalavya",
    # Hindu deities + texts
    "Bhagavad Gita", "Vishnu", "Shiva", "Brahma", "Ganesha", "Lakshmi",
    "Saraswati", "Parvati", "Durga", "Kali", "Indra", "Surya (Hindu deity)",
    "Avatars of Vishnu", "Trimurti",
    # Indian history — pre-modern
    "History of India", "Indus Valley civilisation", "Maurya Empire",
    "Chandragupta Maurya", "Ashoka", "Gupta Empire", "Harsha",
    "Chola dynasty", "Vijayanagara Empire", "Mughal Empire", "Babur",
    "Humayun", "Akbar", "Jahangir", "Shah Jahan", "Aurangzeb",
    "Maratha Empire", "Shivaji", "Bahadur Shah Zafar", "Tipu Sultan",
    # Indian independence
    "Indian independence movement", "Mahatma Gandhi", "Jawaharlal Nehru",
    "Sardar Vallabhbhai Patel", "B. R. Ambedkar", "Subhas Chandra Bose",
    "Bhagat Singh", "Chandra Shekhar Azad", "Lala Lajpat Rai",
    "Bal Gangadhar Tilak", "Sarojini Naidu", "Bipin Chandra Pal",
    "Partition of India", "British Raj", "Quit India Movement",
    "Salt March", "Jallianwala Bagh massacre",
    # Indian leaders post-1947
    "Indira Gandhi", "Lal Bahadur Shastri", "Rajiv Gandhi", "Atal Bihari Vajpayee",
    "Manmohan Singh", "Narendra Modi", "Pratibha Patil", "Rajendra Prasad",
    "Kalpana Chawla", "Sundar Pichai",
    # Geography
    "Geography of India", "Ganges", "Yamuna", "Brahmaputra River",
    "Godavari River", "Krishna River", "Kaveri", "Indus River",
    "Narmada River", "Tapi River", "Himalayas", "Western Ghats",
    "Eastern Ghats", "Thar Desert", "Deccan Plateau", "Sundarbans",
    "New Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru", "Hyderabad",
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Goa", "Kerala",
    # Government
    "Government of India", "Constitution of India", "President of India",
    "Prime Minister of India", "Vice President of India", "Parliament of India",
    "Supreme Court of India", "Lok Sabha", "Rajya Sabha",
    "Indian National Congress", "Bharatiya Janata Party",
    "States and union territories of India",
    # Defense
    "Indian Armed Forces", "Indian Army", "Indian Air Force", "Indian Navy",
    "Defence Research and Development Organisation", "BrahMos", "Tejas (combat aircraft)",
    "INS Vikrant (2013)",
    # Sports
    "Cricket in India", "Sachin Tendulkar", "Virat Kohli", "MS Dhoni",
    "Sourav Ganguly", "Rahul Dravid", "Kapil Dev", "Sunil Gavaskar",
    "Indian Premier League", "Field hockey in India", "Dhyan Chand",
    "Mary Kom", "PT Usha", "PV Sindhu", "Saina Nehwal", "Vishwanathan Anand",
    "Abhinav Bindra", "Neeraj Chopra", "Sania Mirza",
    # Science / Technology
    "Indian Space Research Organisation", "Chandrayaan-3", "Mangalyaan",
    "Aditya-L1", "C. V. Raman", "A. P. J. Abdul Kalam",
    "Homi J. Bhabha", "Vikram Sarabhai", "Srinivasa Ramanujan",
    "Har Gobind Khorana", "Subrahmanyan Chandrasekhar", "Venkatraman Ramakrishnan",
    # Food
    "Indian cuisine", "Biryani", "Curry", "Masala chai", "Masala dosa",
    "Idli", "Samosa", "Naan", "Chapati", "Paneer", "Dal", "Tandoori chicken",
    # Culture / Entertainment
    "Hindi cinema", "Bollywood", "Satyajit Ray", "Amitabh Bachchan",
    "Shah Rukh Khan", "Lata Mangeshkar", "Rabindranath Tagore",
    "Jana Gana Mana", "Vande Mataram", "Bharatanatyam", "Kathak",
    "Indian classical music", "Ravi Shankar (musician)", "Bismillah Khan",
    # Symbols / national
    "National symbols of India", "National Emblem of India", "Flag of India",
    "Bengal tiger", "Indian peafowl", "Lotus", "National Game of India",
]

WIKI_API = "https://en.wikipedia.org/w/api.php"
CHUNK_WORDS = 180   # ~200-token chunks
HDRS = {"User-Agent": "private-llm-rag/0.1 (research)"}


def _api(params, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(WIKI_API, params=params, headers=HDRS, timeout=30)
            if r.status_code != 200:
                time.sleep(1 + attempt); continue
            return r.json()
        except (requests.RequestException, ValueError) as e:
            print(f"    api error ({e.__class__.__name__}), retry {attempt+1}/{retries}")
            time.sleep(1 + attempt * 2)
    return None


def _extract_by_title(title):
    j = _api({"action": "query", "prop": "extracts", "titles": title,
              "explaintext": True, "exsectionformat": "plain",
              "format": "json", "redirects": 1})
    if not j:
        return None, None
    pages = j.get("query", {}).get("pages", {})
    for pid, p in pages.items():
        if pid == "-1":  # missing
            return None, None
        canonical = p.get("title", title)
        text = p.get("extract", "")
        if text and len(text) > 200:
            return canonical, text
    return None, None


def fetch_article(title):
    # 1. Try direct title lookup first — much higher hit rate than search.
    canonical, text = _extract_by_title(title)
    if text: return canonical, text

    # 2. Fallback to search.
    j = _api({"action": "query", "list": "search", "srsearch": title,
              "format": "json", "srlimit": 1})
    if not j: return None, None
    hits = j.get("query", {}).get("search", [])
    if not hits: return None, None
    canonical = hits[0]["title"]
    return _extract_by_title(canonical)


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
