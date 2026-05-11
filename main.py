"""End-to-end pipeline driver.

Usage:
    python main.py --config config.yaml --stage all     # stage1 + stage2
    python main.py --config config.yaml --stage stage1  # ontology + question labels
    python main.py --config config.yaml --stage stage2  # chunk labeling only

Output files (under cfg.paths.output_dir):
    ontology.json           : {"labels": [...], "descriptions": {...}, ...}
    question_labeled.xlsx   : input xlsx + a `labels` column (pipe-separated)
    chunk_labels.jsonl      : one JSON per line {"chunk_id": "...", "labels": [...]}
    stats.json              : counts + label distribution
    failed_chunks.jsonl     : any chunks that errored out
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
from chunk_labeler import build_guided_schema, build_prompt, parse_labels, OTHER


# ============ stage 1 ============

def _ensure_other(ontology: dict, logger) -> None:
    """In-memory migration: old ontology.json may lack 'other'. Add it on load."""
    if OTHER not in ontology["labels"]:
        ontology["labels"].append(OTHER)
        logger.info(f"[stage1] injected fallback label `{OTHER}`")
    ontology.setdefault("descriptions", {})
    if not ontology["descriptions"].get(OTHER):
        ontology["descriptions"][OTHER] = (
            "USE ALONE: chunk is not related to any listed ethics or cultural concept"
        )


def _label_texts_batch(
    texts: list, ontology: dict, embedder: EmbeddingFilter,
    runner: VLLMRunner, cfg: dict, mode: str = "chunk",
) -> list:
    """One-shot multi-label classification on a list of texts.

    mode='chunk'    -> Stage 2 behavior. Schema enum includes `other`; empty
                       parses fall back to ["other"].
    mode='question' -> Stage 1 question-labeling. Schema enum EXCLUDES `other`
                       (questions BUILT the ontology -> 'other' is a bug);
                       empty parses fall back to bge-m3 top-1 candidate.
    """
    if not texts:
        return []
    embedder.fit_labels(
        ontology["labels"],
        [ontology["descriptions"].get(l, l.replace("_", " "))
         for l in ontology["labels"]],
    )

    allow_other = (mode == "chunk")
    max_lbl  = int(cfg["inference"]["max_labels_per_chunk"])
    schema   = build_guided_schema(ontology["labels"], max_lbl, allow_other=allow_other)
    allowed  = set(ontology["labels"])
    if allow_other:
        allowed.add(OTHER)
    else:
        allowed.discard(OTHER)

    top_k    = int(cfg["embedding"]["top_k_labels"])
    thresh   = float(cfg["inference"]["threshold_score"])
    max_chr  = int(cfg["inference"]["max_chunk_chars"])
    max_out  = int(cfg["inference"]["max_tokens"])

    topk = embedder.topk_for_texts(
        [t[:max_chr] for t in texts], k=top_k, threshold=thresh,
    )
    prompts = []
    for t, (idxs, _scores) in zip(texts, topk):
        cands = [ontology["labels"][i] for i in idxs]
        if not cands:
            cands = ontology["labels"][:top_k]
        descs = {l: ontology["descriptions"].get(l, "") for l in cands}
        prompts.append(build_prompt(t, cands, descs, max_chr, mode=mode))
    outs = runner.generate_text(
        prompts, max_tokens=max_out, temperature=0.0, guided_json=schema,
    )

    fallback = OTHER if allow_other else None
    results = []
    for out, (idxs, _scores) in zip(outs, topk):
        labs = parse_labels(out, allowed, max_lbl, fallback=fallback)
        if not labs:
            # question mode + parse failure -> use bge-m3 top-1 as deterministic backup
            if idxs:
                labs = [ontology["labels"][idxs[0]]]
            elif ontology["labels"]:
                labs = [ontology["labels"][0]]
        results.append(labs)
    return results


def _write_question_labels_xlsx(
    src_xlsx: str, dst_xlsx: str,
    questions: list, labels_per_question: list,
    question_column: str = "question",
    out_column: str = "labels",
    sep: str = "|",
) -> int:
    """Re-read the source xlsx, add a `labels` column (pipe-separated), save to dst.
    Blank/missing question cells get an empty labels cell (NOT `other`, because
    `other` is reserved for chunks)."""
    import pandas as pd
    df = pd.read_excel(src_xlsx)
    col = question_column if question_column in df.columns else df.columns[0]
    q2l = {q.strip(): labs for q, labs in zip(questions, labels_per_question)}
    # if a labels column already exists, drop it -- we are the authoritative source
    if out_column in df.columns:
        df = df.drop(columns=[out_column])

    def lookup(x):
        if not isinstance(x, str):
            x = "" if x is None else str(x)
        labs = q2l.get(x.strip())
        if not labs:
            # blank row or duplicate-but-different-whitespace -> leave empty
            return ""
        return sep.join(labs)

    df[out_column] = df[col].apply(lookup)
    os.makedirs(os.path.dirname(dst_xlsx) or ".", exist_ok=True)
    df.to_excel(dst_xlsx, index=False)
    return len(df)


def run_stage1(cfg: dict, logger, embedder: EmbeddingFilter,
               runner: VLLMRunner) -> dict:
    """1) build ontology from questions  2) label every question with that ontology."""
    out_dir       = cfg["paths"]["output_dir"]
    onto_path     = os.path.join(out_dir, "ontology.json")
    src_xlsx      = cfg["paths"]["questions_xlsx"]
    qcol          = cfg["paths"].get("question_column", "question")
    labeled_xlsx  = os.path.join(out_dir, "question_labeled.xlsx")

    questions = load_questions(src_xlsx, qcol)
    logger.info(f"[stage1] loaded {len(questions)} questions from {src_xlsx}")

    # --- 1) ontology ---
    if os.path.exists(onto_path):
        logger.info(f"[stage1] ontology already exists at {onto_path}; loading.")
        ontology = load_ontology(onto_path)
    else:
        if not questions:
            logger.warning("[stage1] no questions found; ontology = seed labels only")

        def embed_fn(texts):
            return embedder.encode_texts(texts)

        def llm_gen(prompts):
            return runner.generate_text(
                prompts, max_tokens=80, temperature=0.0,
                guided_json=LABEL_NAMING_SCHEMA,
            )

        t0 = time.time()
        ontology = build_ontology(
            questions=questions,
            seeds=cfg["ontology"]["seed_labels"],
            embed_fn=embed_fn,
            llm_generate=llm_gen,
            target_labels=cfg["ontology"]["target_labels"],
            min_labels=cfg["ontology"]["min_labels"],
            max_labels=cfg["ontology"]["max_labels"],
        )
        dump_json(onto_path, ontology)
        logger.info(f"[stage1] ontology -> {onto_path} "
                    f"(n={len(ontology['labels'])}, took {time.time()-t0:.1f}s)")

    _ensure_other(ontology, logger)

    # --- 2) question labeling (mode='question': NO 'other' allowed -- every
    # question must map to a real ontology concept because the ontology was
    # BUILT from these questions; 'other' here would be a self-contradiction.
    if questions:
        t1 = time.time()
        labels_per_q = _label_texts_batch(
            questions, ontology, embedder, runner, cfg, mode="question",
        )
        # Quick health check: count multi-label questions and average label count
        n_multi = sum(1 for ls in labels_per_q if len(ls) >= 2)
        avg_labels = (sum(len(ls) for ls in labels_per_q) / len(labels_per_q)
                      if labels_per_q else 0.0)
        n_rows = _write_question_labels_xlsx(
            src_xlsx, labeled_xlsx, questions, labels_per_q,
            question_column=qcol,
        )
        logger.info(
            f"[stage1] question labels -> {labeled_xlsx} "
            f"(rows={n_rows}, unique_q={len(questions)}, "
            f"multi_label={n_multi}/{len(questions)}, avg_labels={avg_labels:.2f}, "
            f"took {time.time()-t1:.1f}s)"
        )

    return ontology


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

    # Backward-compat: ensure the `other` fallback label is in the ontology.
    # Old ontology.json files (pre-`other`) get migrated in-memory; new runs
    # already include it via config.seed_labels.
    if OTHER not in ontology["labels"]:
        ontology["labels"].append(OTHER)
        logger.info(f"[stage2] injected fallback label `{OTHER}` (not in ontology.json)")
    ontology.setdefault("descriptions", {})
    if not ontology["descriptions"].get(OTHER):
        ontology["descriptions"][OTHER] = (
            "USE ALONE: chunk is not related to any listed ethics or cultural concept"
        )

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
        # how much of the corpus the ontology FAILED to cover -- the lower the better
        "other_count": label_counts.get(OTHER, 0),
        "other_rate": round(label_counts.get(OTHER, 0) / max(n_done, 1), 4),
        "label_distribution": dict(
            sorted(label_counts.items(), key=lambda kv: -kv[1])
        ),
    }
    dump_json(stats_path, stats)
    logger.info(f"[stage2] stats -> {stats_path}  "
                f"(other={stats['other_count']} / {n_done} = {stats['other_rate']:.1%})")


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
