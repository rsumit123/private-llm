"""
Extract text from newspaper PDFs and tokenize into a single .bin shard for
continued pretraining.

Usage:
    python extract_pdfs.py --src '/path/to/pdfs' --out_text raw_data/newspapers/text.txt
    python extract_pdfs.py ... --tokenize --out_bin data/newspapers/shard_00000.bin
"""
import argparse
import re
from pathlib import Path

import numpy as np


def extract_text(pdf_path):
    import fitz  # pymupdf
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        # "blocks" mode preserves column flow better than raw text
        blocks = page.get_text("blocks")
        # blocks: (x0, y0, x1, y1, text, block_no, block_type)
        # sort top-to-bottom, left-to-right within columns
        blocks.sort(key=lambda b: (round(b[1] / 20), b[0]))
        text = "\n".join(b[4] for b in blocks if b[4].strip())
        pages.append(text)
    doc.close()
    return "\n\n".join(pages)


# Heuristics for newspaper junk: page numbers, dateline, ALL-CAPS short headers
JUNK_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),                      # bare page numbers
    re.compile(r"^[A-Z\s]{3,}$"),                    # ALL-CAPS headers
    re.compile(r"^Page \d+ of \d+$", re.I),
    re.compile(r"^© .* Indian Express", re.I),
    re.compile(r"^https?://\S+$"),
]


def clean(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        if len(line) < 4: continue
        if any(p.match(line) for p in JUNK_PATTERNS): continue
        out.append(line)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir containing PDFs")
    ap.add_argument("--out_text", required=True)
    ap.add_argument("--tokenize", action="store_true")
    ap.add_argument("--out_bin", default=None)
    ap.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    args = ap.parse_args()

    src = Path(args.src)
    pdfs = sorted(src.glob("*.pdf"))
    print(f"found {len(pdfs)} PDFs in {src}")

    Path(args.out_text).parent.mkdir(parents=True, exist_ok=True)
    total_chars = 0
    with open(args.out_text, "w") as f:
        for i, p in enumerate(pdfs):
            try:
                raw = extract_text(p)
            except Exception as e:
                print(f"[{i}] {p.name}: ERROR {e}"); continue
            cleaned = clean(raw)
            f.write(cleaned + "\n\n<|end_of_doc|>\n\n")
            total_chars += len(cleaned)
            print(f"[{i+1}/{len(pdfs)}] {p.name}: {len(cleaned):,} chars")
    print(f"total: {total_chars:,} chars to {args.out_text}")

    if args.tokenize:
        assert args.out_bin, "--out_bin required for --tokenize"
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        text = open(args.out_text).read()
        ids = tok.encode(text, add_special_tokens=False)
        ids.append(tok.eos_token_id)
        arr = np.asarray(ids, dtype=np.uint16)
        Path(args.out_bin).parent.mkdir(parents=True, exist_ok=True)
        arr.tofile(args.out_bin)
        print(f"tokenized: {len(arr):,} tokens → {args.out_bin}")


if __name__ == "__main__":
    main()
