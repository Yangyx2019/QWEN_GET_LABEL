"""Streaming chunk reader, xlsx question loader, resume helpers."""
from __future__ import annotations
import glob
import json
import os
from typing import Iterator, Dict, Set, List, Union

# `chunks_jsonl` config accepts: single path, list of paths, or glob pattern(s).
ChunkSpec = Union[str, List[str]]


def resolve_chunk_paths(spec: ChunkSpec) -> List[str]:
    """Expand the chunks_jsonl config into a concrete, deduped list of file paths.

    Accepts:
      - "data/chunks.jsonl"            -> single literal path
      - "data/*.jsonl"                 -> glob (any of * ? [)
      - ["data/a.jsonl", "data/b.jsonl"]
      - ["data/儒家/*.jsonl", "data/道家/*.jsonl"]   (mixed)
    """
    if isinstance(spec, str):
        items = [spec]
    elif isinstance(spec, (list, tuple)):
        items = list(spec)
    else:
        raise TypeError(
            f"chunks_jsonl must be str or list, got {type(spec).__name__}"
        )

    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        if not isinstance(item, str) or not item:
            continue
        matches = sorted(glob.glob(item, recursive=True)) \
            if any(ch in item for ch in "*?[") else [item]
        for p in matches:
            ap = os.path.normpath(p)
            if ap not in seen:
                out.append(ap)
                seen.add(ap)
    return out


def iter_chunks(spec: ChunkSpec, resume_ids: Set[str] = None) -> Iterator[Dict]:
    """Yield chunks one-by-one across all matched files, skipping already-processed ids."""
    resume_ids = resume_ids or set()
    for path in resolve_chunk_paths(spec):
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = obj.get("chunk_id")
                if cid is None or cid in resume_ids:
                    continue
                if not isinstance(obj.get("text"), str) or not obj["text"].strip():
                    continue
                yield obj


def count_chunks(spec: ChunkSpec) -> int:
    n = 0
    for path in resolve_chunk_paths(spec):
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
    return n


def load_questions(path: str, column: str = "question") -> List[str]:
    import pandas as pd
    df = pd.read_excel(path)
    col = column if column in df.columns else df.columns[0]
    out = []
    seen = set()
    for x in df[col].dropna().tolist():
        s = str(x).strip()
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def load_resume_ids(path: str) -> Set[str]:
    """Read previously-written chunk_labels.jsonl and return ids already done."""
    if not os.path.exists(path):
        return set()
    ids: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = obj.get("chunk_id")
            if cid:
                ids.add(cid)
    return ids


def load_ontology(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
