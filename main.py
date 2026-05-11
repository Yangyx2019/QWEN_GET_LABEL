"""End-to-end pipeline driver.

Usage:
    python main.py --config config.yaml --stage all     # stage1 + stage2
    python main.py --config config.yaml --stage stage1  # ontology only
    python main.py --config config.yaml --stage stage2  # labeling only

Output files (under cfg.paths.output_dir):
    ontology.json        : {"labels": [...], "descriptions": {...}, ...}
    chunk_labels.jsonl   : one JSON per line {"chunk_id": "...", "labels": [...]}
    stats.json           : counts + label distribution
    failed_chunks.jsonl  : any chunks that errored out
"""
from __future__ import annotations
import argparse
import json
import os
import time
from typing import Optional

import yaml
from tqdm import tqdm

from data_loader import (
    iter_chunks, count_chunks, load_questions, load_resume_ids, load_ontology,
)
from utils import setup_logger, dump_json, free_memory
from embedding_filter import EmbeddingFilter
from llm_engine import VLLMRunner
from label_generator import build_ontology, LABEL_NAMING_SCHEMA
from chunk_labeler import build_guided_schema, build_prompt, parse_labels


# ============ stage 1 ============

def run_stage1(cfg: dict, logger, embedder: EmbeddingFilter,
               runner: VLLMRunner) -> dict:
    """Build ontology from questions. Returns ontology dict and persists to disk."""
    onto_path = os.path.join(cfg["paths"]["output_dir"], "ontology.json")
    if os.path.exists(onto_path):
        logger.info(f"[stage1] ontology already exists at {onto_path}; loading.")
        return load_ontology(onto_path)

    questions = load_questions(
        cfg["paths"]["questions_xlsx"],
        cfg["paths"].get("question_column", "question"),
    )
    logger.info(f"[stage1] loaded {len(questions)} questions")
    if not questions:
        logger.warning("[stage1] no questions found; ontology = seed labels only")

    def embed_fn(texts):
        return embedder.encode_texts(texts)

    def llm_gen(prompts):
        return runner.generate_text(
            prompts,
            max_tokens=80,
            temperature=0.0,
            guided_json=LABEL_NAMING_SCHEMA,
        )

    t0 = time.time()
    onto = build_ontology(
        questions=questions,
        seeds=cfg["ontology"]["seed_labels"],
        embed_fn=embed_fn,
        llm_generate=llm_gen,
        target_labels=cfg["ontology"]["target_labels"],
        min_labels=cfg["ontology"]["min_labels"],
        max_labels=cfg["ontology"]["max_labels"],
    )
    dump_json(onto_path, onto)
    logger.info(f"[stage1] ontology -> {onto_path} "
                f"(n={len(onto['labels'])}, took {time.time()-t0:.1f}s)")
    return onto


# ============ stage 2 ============

def run_stage2(cfg: dict, logger, ontology: dict,
               embedder: EmbeddingFilter, runner: VLLMRunner) -> None:
    chunks_path  = cfg["paths"]["chunks_jsonl"]
    out_dir      = cfg["paths"]["output_dir"]
    labels_path  = os.path.join(out_dir, "chunk_labels.jsonl")
    failed_path  = os.path.join(out_dir, "failed_chunks.jsonl")
    stats_path   = os.path.join(out_dir, "stats.json")

    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"chunks file not found: {chunks_path}")

    # --- prepare embedder side: label matrix
    embedder.fit_labels(
        ontology["labels"],
        [ontology["descriptions"].get(l, l.replace("_", " "))
         for l in ontology["labels"]],
    )

    # --- prepare LLM side: single global guided schema (compiled once by xgrammar)
    schema   = build_guided_schema(ontology["labels"], cfg["inference"]["max_labels_per_chunk"])
    allowed  = set(ontology["labels"])
    top_k    = int(cfg["embedding"]["top_k_labels"])
    thresh   = float(cfg["inference"]["threshold_score"])
    mega_bs  = int(cfg["inference"]["mega_batch"])
    max_lbl  = int(cfg["inference"]["max_labels_per_chunk"])
    max_chr  = int(cfg["inference"]["max_chunk_chars"])
    max_out  = int(cfg["inference"]["max_tokens"])

    resume_ids = load_resume_ids(labels_path)
    total      = count_chunks(chunks_path)
    remaining  = total - len(resume_ids)
    logger.info(f"[stage2] total={total} resume={len(resume_ids)} remaining={remaining}")
    if remaining <= 0:
        logger.info("[stage2] nothing to do.")
        return

    # label distribution counters (continuing from any previous run is fine: we recompute below)
    label_counts: dict = {l: 0 for l in ontology["labels"]}
    n_failed = 0
    n_done   = 0

    fout = open(labels_path, "a", encoding="utf-8")
    ferr = open(failed_path, "a", encoding="utf-8")
    pbar = tqdm(total=remaining, desc="label", unit="chunk", smoothing=0.05)

    buf: list = []
    t0 = time.time()

    def flush():
        nonlocal n_failed, n_done
        if not buf:
            return
        texts = [c["text"][:max_chr] for c in buf]
        # bge-m3 top-K candidates
        topk = embedder.topk_for_texts(texts, k=top_k, threshold=thresh)
        # build prompts
        prompts = []
        for c, (idxs, _scores) in zip(buf, topk):
            cands = [ontology["labels"][i] for i in idxs]
            if not cands:
                cands = ontology["labels"][:top_k]
            descs = {l: ontology["descriptions"].get(l, "") for l in cands}
            prompts.append(build_prompt(c["text"], cands, descs, max_chr))
        # vLLM guided generation
        try:
            outs = runner.generate_text(
                prompts, max_tokens=max_out, temperature=0.0, guided_json=schema,
            )
        except Exception as e:
            logger.exception(f"[stage2] vLLM batch failed: {e}")
            for c in buf:
                ferr.write(json.dumps(
                    {"chunk_id": c.get("chunk_id"), "error": str(e)},
                    ensure_ascii=False) + "\n")
            n_failed += len(buf)
            ferr.flush()
            buf.clear()
            return

        for c, out in zip(buf, outs):
            labs = parse_labels(out, allowed, max_lbl)
            rec = {"chunk_id": c["chunk_id"], "labels": labs}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for l in labs:
                label_counts[l] = label_counts.get(l, 0) + 1
            n_done += 1
        fout.flush()
        pbar.update(len(buf))
        buf.clear()

    try:
        for c in iter_chunks(chunks_path, resume_ids):
            buf.append(c)
            if len(buf) >= mega_bs:
                flush()
        flush()
    except KeyboardInterrupt:
        logger.warning("[stage2] interrupted, flushing & exiting.")
        flush()
    finally:
        fout.close()
        ferr.close()
        pbar.close()

    elapsed = time.time() - t0
    rate = (n_done / elapsed) if elapsed > 0 else 0.0
    logger.info(f"[stage2] done={n_done} failed={n_failed} "
                f"elapsed={elapsed:.1f}s rate={rate:.1f} chunks/s")

    stats = {
        "total_chunks": total,
        "resumed": len(resume_ids),
        "processed_this_run": n_done,
        "failed_this_run": n_failed,
        "ontology_size": len(ontology["labels"]),
        "elapsed_seconds": round(elapsed, 1),
        "chunks_per_second": round(rate, 2),
        "label_distribution": dict(
            sorted(label_counts.items(), key=lambda kv: -kv[1])
        ),
    }
    dump_json(stats_path, stats)
    logger.info(f"[stage2] stats -> {stats_path}")


# ============ entry ============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage",  default="all", choices=["all", "stage1", "stage2"])
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)

    logger = setup_logger("pipeline")
    logger.info(f"stage={args.stage}  model={cfg['llm']['model']}  embed={cfg['embedding']['model']}")

    # bring up bge-m3 first (small), then vLLM (large) so vLLM measures remaining memory
    embedder: Optional[EmbeddingFilter] = None
    runner:   Optional[VLLMRunner] = None

    try:
        embedder = EmbeddingFilter(
            cfg["embedding"]["model"],
            device=cfg["embedding"]["device"],
            batch_size=cfg["embedding"]["batch_size"],
            max_length=cfg["embedding"]["max_length"],
        )
        runner = VLLMRunner(cfg)

        if args.stage in ("all", "stage1"):
            ontology = run_stage1(cfg, logger, embedder, runner)
        else:
            ontology = load_ontology(os.path.join(cfg["paths"]["output_dir"], "ontology.json"))
            logger.info(f"[stage2] loaded ontology with {len(ontology['labels'])} labels")

        if args.stage in ("all", "stage2"):
            run_stage2(cfg, logger, ontology, embedder, runner)
    finally:
        if runner is not None:
            runner.shutdown()
        embedder = None
        free_memory()


if __name__ == "__main__":
    main()
