"""Small helpers: logging, JSON IO, GPU cleanup."""
from __future__ import annotations
import gc
import json
import logging
import os
import sys
from typing import Any


def setup_logger(name: str = "pipeline", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                                     datefmt="%H:%M:%S"))
    logger.addHandler(h)
    logger.propagate = False
    return logger


def dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def dump_jsonl_line(fp, obj: Any) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def free_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def normalize_label(lbl: str) -> str:
    import re
    lbl = (lbl or "").strip().lower()
    lbl = re.sub(r"[^a-z0-9_]+", "_", lbl)
    lbl = re.sub(r"_+", "_", lbl).strip("_")
    return lbl
