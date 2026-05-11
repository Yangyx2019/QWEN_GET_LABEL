"""Stage 2: chunk -> multi-label classification.

Design:
- bge-m3 picks top-K candidate labels per chunk (search-space pruning + quality boost)
- vLLM with a SINGLE global guided_json schema (enum over the full ontology)
- xgrammar compiles the schema ONCE, all requests reuse it
- Prompt lists only the top-K candidates so the model focuses there
- Temperature=0, JSON-only output, deduped + filtered by allowed set
"""
from __future__ import annotations
import json
import re
from typing import Dict, List


PROMPT_TMPL = """You are an ethics annotation expert for cross-cultural moral analysis.

Decide which of the CANDIDATE LABELS apply to the TEXT. Be strict:
- Choose a label ONLY if the text clearly expresses, debates, advocates, or critiques the concept.
- Multiple labels are allowed.
- Zero labels are allowed if no concept applies.
- DO NOT invent labels outside the candidate list.

CANDIDATE LABELS:
{candidates}

TEXT:
\"\"\"{text}\"\"\"

Reply with JSON only, no prose:
{{"labels": ["..."]}}"""


def build_guided_schema(ontology_labels: List[str], max_labels: int) -> dict:
    """One global schema used for every request -> xgrammar caches a single FSM."""
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "minItems": 0,
                "maxItems": int(max_labels),
                "uniqueItems": True,
                "items": {"type": "string", "enum": list(ontology_labels)},
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }


def build_prompt(text: str, candidates: List[str],
                 descriptions: Dict[str, str], max_chars: int) -> str:
    if not text:
        text = ""
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    # neutralize accidental triple-quote in chunks
    text = text.replace('"""', '”””')

    lines = []
    for c in candidates:
        d = (descriptions.get(c) or "").strip()
        if d:
            lines.append(f"- {c}: {d}")
        else:
            lines.append(f"- {c}")
    return PROMPT_TMPL.format(candidates="\n".join(lines), text=text)


def parse_labels(text: str, allowed: set, max_labels: int) -> List[str]:
    """Robust parse: strip code fences, extract first JSON object, filter to allowed."""
    if not text:
        return []
    s = text.strip()
    # strip markdown fences if model emitted them
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    labs = obj.get("labels", []) if isinstance(obj, dict) else []
    if not isinstance(labs, list):
        return []
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
    return out
