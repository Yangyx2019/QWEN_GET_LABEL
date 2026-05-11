"""Stage 1: query list -> stable ethics ontology.

Pipeline:
  questions  -> bge-m3 embed
             -> Agglomerative cluster (cosine, target ~30 clusters)
             -> centroid-closest reps per cluster
             -> LLM names each cluster (guided JSON, snake_case)
             -> merge with seed labels (seeds always retained, deduped)
             -> build per-label descriptions from cluster reps
"""
from __future__ import annotations
import json
import re
from typing import List, Dict, Callable
import numpy as np

from utils import normalize_label


PROMPT_LABEL_NAMING = """\
You are an expert in ethics, philosophy, and cross-cultural moral theory.

A cluster of user questions on moral / cultural dilemmas is given below.
Your job: assign ONE canonical short label that captures the SPECIFIC ethical concept
of THIS cluster.

Rules:
- Output a single label only.
- Format: snake_case English, 1 to 3 tokens
  (e.g. filial_piety, bodily_autonomy, work_ethic, vigilante_justice, moral_luck,
   counterfactual_ethics, animal_welfare, addiction_recovery).
- Be SPECIFIC. Pick the label that most narrowly captures the cluster's theme.
  * cluster about skipping work / taking time off  -> work_ethic, not duty
  * cluster about drug use / weed                 -> bodily_autonomy, not individual_rights
  * cluster about time-travel / counterfactuals    -> moral_luck or counterfactual_ethics, not social_order
  * cluster about giving away income to charity    -> charity or altruism, not duty
- REUSE a SEED LABEL ONLY when it is a near-perfect semantic match for the cluster
  as a whole. If only part of the cluster matches a seed, propose a NEW label that
  fits the cluster better.
- Forbidden behavior: do NOT default to `duty`, `harm`, `individual_rights`, or
  `social_order` as catch-alls for clusters that have a more precise theme.
- Avoid trivial topic words (e.g. "money", "school", "friend", "job").
- Each cluster should ideally get a label different from neighboring clusters; if
  multiple clusters share the SAME theme, only then reuse the same label.

SEED LABELS (reuse only on near-perfect match; otherwise propose a NEW snake_case label):
{seed_labels}

CLUSTER QUESTIONS:
{questions}

Reply with JSON only, no prose:
{{"label": "<snake_case_label>", "rationale": "<one short sentence why this label fits the cluster>"}}"""


LABEL_NAMING_SCHEMA = {
    "type": "object",
    "properties": {
        "label":     {"type": "string", "pattern": "^[a-z][a-z0-9_]{1,40}$"},
        "rationale": {"type": "string", "maxLength": 200},
    },
    "required": ["label"],
    "additionalProperties": False,
}


# ---------------- clustering ----------------

def cluster_questions(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    n = embeddings.shape[0]
    n_clusters = max(2, min(n_clusters, n - 1))
    clu = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    )
    return clu.fit_predict(embeddings)


def pick_representatives(
    embeddings: np.ndarray, cluster_ids: np.ndarray,
    questions: List[str], max_per_cluster: int = 8,
) -> Dict[int, List[str]]:
    reps: Dict[int, List[str]] = {}
    for c in sorted(set(cluster_ids.tolist())):
        idx = np.where(cluster_ids == c)[0]
        if idx.size == 0:
            continue
        sub = embeddings[idx]
        centroid = sub.mean(axis=0, keepdims=True)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
        sims = (sub @ centroid.T).ravel()
        order = np.argsort(-sims)
        sel = idx[order[:max_per_cluster]]
        reps[int(c)] = [questions[i] for i in sel]
    return reps


# ---------------- LLM naming ----------------

def parse_label_json(text: str) -> str:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return ""
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return ""
    return normalize_label(obj.get("label", ""))


def merge_with_seeds(generated: List[str], seeds: List[str], max_labels: int) -> List[str]:
    out: List[str] = []
    seen: set = set()
    # seeds always first, fixed order
    for s in seeds:
        s_n = normalize_label(s)
        if s_n and s_n not in seen:
            out.append(s_n); seen.add(s_n)
    # then LLM-generated novel labels
    for g in generated:
        g = normalize_label(g)
        if g and g not in seen:
            out.append(g); seen.add(g)
        if len(out) >= max_labels:
            break
    return out[:max_labels]


def build_label_descriptions(
    ontology: List[str], cluster_to_label: Dict[int, str], reps: Dict[int, List[str]],
) -> Dict[str, str]:
    by_label: Dict[str, List[str]] = {}
    for c, lbl in cluster_to_label.items():
        by_label.setdefault(lbl, []).extend(reps.get(c, [])[:3])
    out: Dict[str, str] = {}
    for lbl in ontology:
        if lbl == "other":
            # mandatory fallback: never embed-rank-friendly; described purely by purpose
            out[lbl] = "USE ALONE: chunk is not related to any listed ethics or cultural concept"
            continue
        examples = by_label.get(lbl, [])
        if examples:
            # join up to 3 short example questions
            short = [e[:80] for e in examples[:3]]
            out[lbl] = " | ".join(short)
        else:
            out[lbl] = lbl.replace("_", " ")
    return out


# ---------------- orchestrator ----------------

def build_ontology(
    questions: List[str],
    seeds: List[str],
    embed_fn: Callable[[List[str]], np.ndarray],
    llm_generate: Callable[[List[str]], List[str]],
    target_labels: int = 30,
    min_labels: int = 20,
    max_labels: int = 50,
) -> dict:
    if not questions:
        # degenerate: ontology = seeds
        labels = [normalize_label(s) for s in seeds if s]
        labels = [x for x in labels if x]
        return {
            "labels": labels[:max_labels],
            "descriptions": {l: l.replace("_", " ") for l in labels[:max_labels]},
            "cluster_to_label": {},
            "n_questions": 0,
        }

    embs = embed_fn(questions)

    # decide cluster count: capped by num_questions/2 to avoid singleton clusters dominating
    n_clusters = min(
        max(min_labels, target_labels),
        max_labels,
        max(2, len(questions) // 2),
    )
    cluster_ids = cluster_questions(embs, n_clusters)
    reps = pick_representatives(embs, cluster_ids, questions, max_per_cluster=8)

    prompts: List[str] = []
    cluster_keys: List[int] = []
    for c, qs in reps.items():
        prompts.append(PROMPT_LABEL_NAMING.format(
            seed_labels="\n".join(f"- {s}" for s in seeds),
            questions="\n".join(f"- {q}" for q in qs),
        ))
        cluster_keys.append(c)

    outputs = llm_generate(prompts)

    generated: List[str] = []
    cluster_to_label: Dict[int, str] = {}
    for c, txt in zip(cluster_keys, outputs):
        lbl = parse_label_json(txt)
        if lbl:
            generated.append(lbl)
            cluster_to_label[c] = lbl

    ontology = merge_with_seeds(generated, seeds, max_labels)
    descriptions = build_label_descriptions(ontology, cluster_to_label, reps)

    return {
        "labels": ontology,
        "descriptions": descriptions,
        "cluster_to_label": {str(k): v for k, v in cluster_to_label.items()},
        "n_questions": len(questions),
        "n_clusters": int(n_clusters),
    }
