#!/usr/bin/env python3
"""
RAG index builder (Stage 7) with domain metadata.

Usage:
  python scripts/rag_prep.py \
    --in data/ \
    --out web/public/rag/ \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --chunk-size 600 --chunk-overlap 120 \
    --domain-default rail \
    --domains rail,auto,transit,standards

Domain inference:
- From parent dir name if it matches a known domain (e.g., data/auto/*.md -> domain=auto)
- Else from filename prefix before first underscore (e.g., auto_powertrains.md -> domain=auto)
- Else falls back to --domain-default
"""

import argparse, json, os, re, sys, time, hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("Please: pip install sentence-transformers", file=sys.stderr)
    raise

TXT_EXT = {".txt", ".md"}
JSONL_EXT = {".jsonl"}

def infer_domain(path: Path, known: set, default_domain: str) -> str:
    parent = path.parent.name.lower()
    stem = path.stem.lower()
    if parent in known:
        return parent
    # filename prefix before first underscore
    m = re.match(r"([a-z0-9\-]+)_", stem)
    if m and m.group(1) in known:
        return m.group(1)
    return default_domain

def read_text_files(root: Path, known_domains: set, default_domain: str) -> List[Dict[str, Any]]:
    docs = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in TXT_EXT:
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append({
                "id": hashlib.md5(str(p).encode()).hexdigest()[:10],
                "title": p.stem.replace("_", " ").title(),
                "text": text,
                "source": str(p.relative_to(root)),
                "domain": infer_domain(p, known_domains, default_domain)
            })
        elif ext in JSONL_EXT:
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                # Expect keys: id, title, text, source (optional)
                if "text" in obj:
                    obj.setdefault("id", hashlib.md5(obj["text"].encode()).hexdigest()[:10])
                    obj.setdefault("title", obj["id"])
                    obj.setdefault("source", str(p.relative_to(root)))
                    obj.setdefault("domain", infer_domain(p, known_domains, default_domain))
                    docs.append(obj)
    return docs

def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

def chunk_text(s: str, chunk_size: int, chunk_overlap: int) -> List[str]:
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
            for i in range(0, len(p), chunk_size):
                sub = p[i:i + chunk_size]
                chunks.append(sub)
            cur_len = 0
    flush()

    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, c in enumerate(chunks):
            if i == 0:
                overlapped.append(c)
            else:
                prev = overlapped[-1]
                tail = prev[-chunk_overlap:]
                overlapped.append((tail + c).strip())
        chunks = overlapped

    return [c.strip() for c in chunks if len(c.strip()) >= 80]

def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--chunk-size", type=int, default=600)
    ap.add_argument("--chunk-overlap", type=int, default=120)
    ap.add_argument("--domain-default", default="rail")
    ap.add_argument("--domains", default="rail,auto,transit,standards",
                    help="comma-separated list of known domains")
    args = ap.parse_args()

    in_dir = Path(args.inp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    known_domains = set([d.strip().lower() for d in args.domains.split(",") if d.strip()])
    docs = read_text_files(in_dir, known_domains, args.domain_default)
    if not docs:
        print(f"No documents found under {in_dir}", file=sys.stderr)
        sys.exit(1)

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
                "text": ch,
                "meta": {
                    "domain": d.get("domain", args.domain_default)
                }
            })

    texts = [r["text"] for r in records]
    print(f"Embedding {len(texts)} chunks with {args.model} ...")
    model = SentenceTransformer(args.model)
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype=np.float32)  # [N, D]
    n, d = embs.shape

    emb_path = out_dir / "embeddings.f32"
    embs.tofile(emb_path)

    index = {
        "dim": int(d),
        "chunk_count": int(n),
        "doc_count": len(docs),
        "model": args.model,
        "created_utc": int(time.time()),
        "domains": sorted(list(known_domains)),
        "chunks": records,
    }
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

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
            "domains": {r["meta"]["domain"]: 0 for r in records}
        }
    }
    # fill domain counts
    for r in records:
        manifest["stats"]["domains"][r["meta"]["domain"]] += 1

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Wrote {n} chunks at dim {d} to {out_dir}")

if __name__ == "__main__":
    main()
