"""Streaming chunk reader, xlsx question loader, resume helpers."""
from __future__ import annotations
import json
import os
from typing import Iterator, Dict, Set, List


def iter_chunks(path: str, resume_ids: Set[str] = None) -> Iterator[Dict]:
    """Yield chunks one-by-one, skipping already-processed ids."""
    resume_ids = resume_ids or set()
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


def count_chunks(path: str) -> int:
    if not os.path.exists(path):
        return 0
    n = 0
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
