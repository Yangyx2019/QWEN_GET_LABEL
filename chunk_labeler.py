"""Stage 2: chunk -> multi-label classification.

Design:
- bge-m3 picks top-K candidate labels per chunk (search-space pruning + quality boost)
- vLLM with a SINGLE global guided_json schema (enum over the full ontology)
- xgrammar compiles the schema ONCE, all requests reuse it
- Prompt lists only the top-K candidates so the model focuses there
- Temperature=0, JSON-only output, deduped + filtered by allowed set

Fallback label "other":
- ALWAYS appended to the candidate list shown to the LLM (regardless of bge-m3 top-K)
- ALWAYS in the schema enum
- The prompt teaches the model: pick "other" alone when nothing else fits
- parse_labels enforces "other" is exclusive (cannot co-occur with a real label)
"""
from __future__ import annotations
import json
import re
from typing import Dict, List

OTHER = "other"


PROMPT_TMPL = """You are an ethics annotation expert for cross-cultural moral analysis.

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


def build_guided_schema(ontology_labels: List[str], max_labels: int) -> dict:
    """One global schema used for every request -> xgrammar caches a single FSM.

    minItems=1 forces the model to pick something — if nothing fits, the prompt
    tells it to pick `other`.

    NOTE: `uniqueItems` is intentionally NOT used here — xgrammar / guidance
    backends in vLLM ≥0.20 don't implement it. Dedup is handled in parse_labels.
    """
    labels = list(ontology_labels)
    if OTHER not in labels:
        labels.append(OTHER)
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
    """Always append `other` to the candidate list shown to the LLM if missing.
    Order is preserved; `other` goes last so it's not first-pick by ordering."""
    if OTHER not in candidates:
        return list(candidates) + [OTHER]
    return candidates


def build_prompt(text: str, candidates: List[str],
                 descriptions: Dict[str, str], max_chars: int) -> str:
    if not text:
        text = ""
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    # neutralize accidental triple-quote in chunks
    text = text.replace('"""', '”””')

    candidates = ensure_other(candidates)

    lines = []
    for c in candidates:
        d = (descriptions.get(c) or "").strip()
        if c == OTHER and not d:
            d = "USE ALONE when no other label applies"
        if d:
            lines.append(f"- {c}: {d}")
        else:
            lines.append(f"- {c}")
    return PROMPT_TMPL.format(candidates="\n".join(lines), text=text)


def parse_labels(text: str, allowed: set, max_labels: int) -> List[str]:
    """Robust parse: strip code fences, extract first JSON object, filter to allowed.

    Post-rules:
    - if model returns empty -> output ["other"]  (safety net; schema forbids but be defensive)
    - if "other" appears together with real labels -> drop "other"
      (real signal wins; "other" only matters when it's the sole answer)
    """
    if not text:
        return [OTHER]
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            return [OTHER]
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return [OTHER]
    labs = obj.get("labels", []) if isinstance(obj, dict) else []
    if not isinstance(labs, list):
        return [OTHER]

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
        return [OTHER]
    # exclusivity: if real labels co-occur with "other", drop "other"
    if OTHER in out and len(out) > 1:
        out = [x for x in out if x != OTHER]
    return out
