#!/usr/bin/env python3
"""
Build a tiny RAG index for the web demo.

Usage:
  python scripts/rag_prep.py \
    --in data/ \
    --out web/public/rag/ \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --chunk-size 600 --chunk-overlap 120

Inputs:
  data/ can contain .txt, .md, or .jsonl ({"id","title","text","source"} per line).

Outputs in web/public/rag/:
  - index.json        (chunk metadata + plain-text chunks for prompt)
  - embeddings.f32    (Float32Array of shape [num_chunks, dim])
  - manifest.json     (stats, model, dims, checksums)
"""

import argparse, json, os, re, sys, time, hashlib
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("Please: pip install sentence-transformers", file=sys.stderr)
    raise

TXT_EXT = {".txt", ".md"}
JSONL_EXT = {".jsonl"}


def read_text_files(root: Path) -> List[Dict[str, Any]]:
    docs = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in TXT_EXT:
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append({
                "id": p.stem,
                "title": p.stem.replace("_", " ").title(),
                "text": text,
                "source": str(p.relative_to(root))
            })
        elif ext in JSONL_EXT:
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                # Expect keys: id, title, text, source
                if "text" in obj:
                    obj.setdefault("id", hashlib.md5(obj["text"].encode()).hexdigest()[:10])
                    obj.setdefault("title", obj["id"])
                    obj.setdefault("source", str(p.relative_to(root)))
                    docs.append(obj)
    return docs


def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").strip()
    # collapse excessive whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def chunk_text(s: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    # simple length-based with paragraph awareness
    s = clean_text(s)
    paras = [p.strip() for p in s.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    cur_len = 0

    def flush():
        nonlocal buf, cur_len
        if buf:
            chunks.append("\n\n".join(buf).strip())
            buf = []
            cur_len = 0

    for p in paras:
        pl = len(p)
        if cur_len + pl + 2 <= chunk_size:
            buf.append(p)
            cur_len += pl + 2
        else:
            if cur_len > 0:
                flush()
            # paragraph itself may be large -> hard wrap
            for i in range(0, len(p), chunk_size):
                sub = p[i:i + chunk_size]
                chunks.append(sub)
            cur_len = 0

    flush()

    # apply overlap (by characters) between adjacent chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, c in enumerate(chunks):
            if i == 0:
                overlapped.append(c)
            else:
                prev = overlapped[-1]
                tail = prev[-chunk_overlap:]
                merged = tail + c
                # prefer not to exceed ~chunk_size+overlap
                overlapped[-1] = prev  # keep prev intact
                overlapped.append(merged)
        chunks = overlapped

    # prune trivial
    return [c.strip() for c in chunks if len(c.strip()) >= 80]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input folder with data")
    ap.add_argument("--out", dest="out", required=True, help="output folder (served as /rag/)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--chunk-size", type=int, default=600)
    ap.add_argument("--chunk-overlap", type=int, default=120)
    args = ap.parse_args()

    in_dir = Path(args.inp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = read_text_files(in_dir)
    if not docs:
        print(f"No documents found under {in_dir}", file=sys.stderr)
        sys.exit(1)

    # chunk
    records = []
    for d in docs:
        chunks = chunk_text(d["text"], args.chunk_size, args.chunk_overlap)
        for idx, ch in enumerate(chunks):
            records.append({
                "id": f"{d['id']}#{idx:04d}",
                "doc_id": d["id"],
                "title": d.get("title", d["id"]),
                "source": d.get("source", d["id"]),
                "offset": idx,
                "text": ch
            })

    texts = [r["text"] for r in records]
    print(f"Embedding {len(texts)} chunks with {args.model} ...")
    model = SentenceTransformer(args.model)
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype=np.float32)  # [N, D]
    n, d = embs.shape

    # write embeddings as raw float32 binary for fast fetch in browser
    emb_path = out_dir / "embeddings.f32"
    embs.tofile(emb_path)

    # write index.json (metadata + chunk text)
    index = {
        "dim": int(d),
        "chunk_count": int(n),
        "doc_count": len(docs),
        "model": args.model,
        "created_utc": int(time.time()),
        "chunks": records,  # contains "text" used for prompt augmentation
    }
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    # manifest
    manifest = {
        "model": args.model,
        "dim": int(d),
        "files": {
            "embeddings.f32": {"sha256": sha256_file(emb_path), "dtype": "float32", "shape": [n, d]},
            "index.json": {"bytes": (out_dir / "index.json").stat().st_size}
        },
        "stats": {
            "docs": len(docs),
            "chunks": n,
            "avg_chunk_chars": int(sum(len(t) for t in texts) / max(1, n)),
        }
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Wrote {n} chunks at dim {d} to {out_dir}")


if __name__ == "__main__":
    main()
