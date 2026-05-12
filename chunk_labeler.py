"""Multi-label classification used for both chunks AND questions.

Modes:
- `mode="chunk"` (default): includes the `other` fallback in schema + prompt.
  Used in Stage 2 because chunks are external text that may not relate to any
  ontology concept.
- `mode="question"`: EXCLUDES `other` from schema + prompt. Used in Stage 1's
  question-labeling step: questions BUILT the ontology, so every question is
  guaranteed to fit at least one real concept — `other` would be a bug.

Both modes share:
- bge-m3 picks top-K candidate labels (search-space pruning + quality boost)
- vLLM with a SINGLE global guided_json schema (enum + minItems=1)
- xgrammar compiles the schema ONCE; all requests reuse it
- Temperature=0, JSON-only output, deduped by parse_labels
"""
from __future__ import annotations
import json
import re
from typing import Dict, List, Optional

OTHER = "other"


# ---------------- boilerplate pre-filter ----------------
#
# OCR'd PDF corpora carry a lot of non-content chunks (copyright pages, tables of
# contents, reference lists, HTML/MD table fragments, running headers). bge-m3 will
# still pick top-K candidate labels for these, and the LLM, forced to choose one
# from a constrained enum, ends up assigning plausible-but-wrong ethics labels
# (e.g. a TOC line "Ritual 35  The Gentleman 42" -> "ritual").
#
# This heuristic catches the clearest cases and short-circuits them to ["other"]
# without invoking embedder or LLM. Conservative by design — when ambiguous we
# let the LLM decide.

_BOILERPLATE_PATTERNS = [
    # English book front-matter / legal
    re.compile(r"all rights reserved", re.I),
    re.compile(r"copyright\s*[©(]|©\s*\d{4}", re.I),
    re.compile(r"\bISBN[- :]?\d", re.I),
    re.compile(r"\bdoi[:\s]\s*10\.\d{4,9}", re.I),
    re.compile(r"first published\s+\d{4}", re.I),
    re.compile(r"this edition first published", re.I),
    re.compile(r"\bpublished by\b.*\b(press|publisher|publishing)\b", re.I),
    re.compile(r"library of congress cataloging", re.I),
    # Chinese book front-matter
    re.compile(r"图书在版编目|版权所有|不得翻印|印张|开本"),
    re.compile(r"出版社\s*[:：]"),
    re.compile(r"^\s*目\s*录\s*$", re.M),
    # HTML / Markdown table fragments
    re.compile(r"</?(td|tr|th|table)[> ]", re.I),
]


def _looks_like_toc(text: str) -> bool:
    """A run of `<title> ... <page-number>` lines -> table of contents."""
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    if len(lines) < 5:
        return False
    toc_like = sum(1 for l in lines if re.search(r"\s\d{1,4}\s*$", l))
    return toc_like / len(lines) >= 0.45


def _looks_like_refs(text: str) -> bool:
    """Reference list: many year-cites, page-cites, vol numbers per length."""
    if len(text) >= 2000:
        return False
    hits = (
        len(re.findall(r"\bpp?\.\s*\d", text))
        + len(re.findall(r"\(\s*(?:19|20)\d{2}\s*\)", text))
        + len(re.findall(r"\bvol\.\s*\d", text, re.I))
        + len(re.findall(r"\bed\.,?\s", text))
    )
    return hits >= 4


def is_boilerplate(text: str) -> bool:
    """True if the chunk looks like non-content (front matter, refs, tables, etc).

    Such chunks should bypass the LLM and be labeled ["other"] directly. Conservative:
    only fires on clear patterns to avoid eating real ethics content.
    """
    if not text:
        return True
    s = text.strip()
    if len(s) < 40:
        return True
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(s):
            return True
    if _looks_like_toc(s):
        return True
    if _looks_like_refs(s):
        return True
    return False


PROMPT_CHUNK_TMPL = """You are an ethics annotation expert for cross-cultural moral analysis.

Decide which of the CANDIDATE LABELS apply to the TEXT. Be strict:
- Choose a label ONLY if the text clearly expresses, debates, advocates, or critiques the concept.
- Multiple labels are allowed when several concepts genuinely apply.
- If NO ethics label genuinely applies, output {{"labels": ["other"]}} — this is the fallback.
- "other" MUST appear ALONE; never combine "other" with any other label.
- You MUST output at least one label. Empty list is NOT allowed.
- DO NOT invent labels outside the candidate list.

CANDIDATE LABELS:
{candidates}

TEXT:
\"\"\"{text}\"\"\"

Reply with JSON only, no prose:
{{"labels": ["..."]}}"""


PROMPT_QUESTION_TMPL = """You are an ethics annotation expert for cross-cultural moral analysis.

Decide which of the CANDIDATE LABELS apply to the QUESTION. The question is a moral
dilemma; it was used to BUILD this ontology, so at least one concept ALWAYS applies.

Rules:
- Choose every label that the question genuinely raises, debates, advocates, or critiques.
- Multiple labels are allowed when several concepts apply.
- Pick the SINGLE best label if unsure — never output an empty list.
- DO NOT invent labels outside the candidate list.

CANDIDATE LABELS:
{candidates}

QUESTION:
\"\"\"{text}\"\"\"

Reply with JSON only, no prose:
{{"labels": ["..."]}}"""


def build_guided_schema(ontology_labels: List[str], max_labels: int,
                        allow_other: bool = True) -> dict:
    """One global schema per mode.

    `allow_other=True`  -> chunk mode; schema enum includes `other`
    `allow_other=False` -> question mode; schema enum EXCLUDES `other`

    NOTE: `uniqueItems` not used — xgrammar / guidance backends in vLLM ≥0.20
    don't implement it. Dedup is handled in parse_labels.
    """
    labels = list(ontology_labels)
    if allow_other:
        if OTHER not in labels:
            labels.append(OTHER)
    else:
        labels = [l for l in labels if l != OTHER]
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "minItems": 1,
                "maxItems": int(max_labels),
                "items": {"type": "string", "enum": labels},
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }


def ensure_other(candidates: List[str]) -> List[str]:
    """For chunk mode: always append `other` to the candidate list if missing."""
    if OTHER not in candidates:
        return list(candidates) + [OTHER]
    return candidates


def strip_other(candidates: List[str]) -> List[str]:
    """For question mode: defensively remove `other` if it slipped in."""
    return [c for c in candidates if c != OTHER]


def build_prompt(text: str, candidates: List[str],
                 descriptions: Dict[str, str], max_chars: int,
                 mode: str = "chunk") -> str:
    if not text:
        text = ""
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    text = text.replace('"""', '”””')  # neutralize accidental triple-quote

    if mode == "chunk":
        candidates = ensure_other(candidates)
        tmpl = PROMPT_CHUNK_TMPL
    elif mode == "question":
        candidates = strip_other(candidates)
        tmpl = PROMPT_QUESTION_TMPL
    else:
        raise ValueError(f"unknown mode: {mode!r} (expected 'chunk' or 'question')")

    lines = []
    for c in candidates:
        d = (descriptions.get(c) or "").strip()
        if c == OTHER and not d:
            d = "USE ALONE when no other label applies"
        if d:
            lines.append(f"- {c}: {d}")
        else:
            lines.append(f"- {c}")
    return tmpl.format(candidates="\n".join(lines), text=text)


def parse_labels(text: str, allowed: set, max_labels: int,
                 fallback: Optional[str] = OTHER) -> List[str]:
    """Robust JSON parse + filter to `allowed` + dedup, with optional fallback.

    fallback=OTHER  -> empty/junk parses become ["other"]  (chunk mode)
    fallback=None   -> empty/junk parses return []         (question mode: caller
                                                            should plug in bge-m3 top-1)

    Post-rules:
    - if `other` co-occurs with real labels, drop `other` (real signal wins)
    """
    def _miss() -> List[str]:
        return [fallback] if fallback else []

    if not text:
        return _miss()
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            return _miss()
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return _miss()
    labs = obj.get("labels", []) if isinstance(obj, dict) else []
    if not isinstance(labs, list):
        return _miss()

    out: List[str] = []
    seen: set = set()
    for x in labs:
        if not isinstance(x, str):
            continue
        x = x.strip()
        if x in allowed and x not in seen:
            out.append(x); seen.add(x)
        if len(out) >= max_labels:
            break

    if not out:
        return _miss()
    if OTHER in out and len(out) > 1:
        out = [x for x in out if x != OTHER]
    return out
