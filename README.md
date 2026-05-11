# qwen_get_label

Local two-stage labeling pipeline for cultural / ethics RAG experiments.

- **Stage 1 — `query → ontology`**: cluster questions with `bge-m3`, name each cluster with the LLM (guided JSON), merge with seed labels → stable ontology (20–50 labels).
- **Stage 2 — `chunk → labels`**: `bge-m3` picks top-K candidate labels per chunk → vLLM emits multi-label JSON under a single global enum schema (xgrammar, compiled once).
- Designed for **single 24 GB GPU**, **100k–400k chunks in 1–3 h**, no external API.

---

## 0) File layout

```
qwen_get_label/
├── config.yaml             # single source of truth for tuning
├── env.sh                  # HF cache + runtime envs (sourced by every script)
├── requirements.txt
├── run.sh                  # one-shot launcher
├── main.py                 # pipeline driver: stage1 / stage2 / all
├── data_loader.py          # streaming jsonl + xlsx + resume
├── utils.py                # logger / json IO / GPU cleanup / label normalize
├── embedding_filter.py     # bge-m3 wrapper (embed questions, top-K filter)
├── llm_engine.py           # vLLM wrapper (chat template + guided_json + xgrammar)
├── label_generator.py      # Stage 1
├── chunk_labeler.py        # Stage 2 (prompt / schema / parse)
├── tools/
│   ├── make_demo.py            # smoke-test data generator
│   └── download_models.sh      # pre-fetch LLM + bge-m3 into ./models/
├── data/
│   ├── chunks.jsonl        # YOUR INPUT
│   └── question.xlsx       # YOUR INPUT
├── models/                 # auto-created; project-local HF cache (HF_HOME)
└── outputs/                # all artifacts land here
    ├── ontology.json
    ├── chunk_labels.jsonl
    ├── stats.json
    └── failed_chunks.jsonl
```

Direct links:
[main.py](main.py) · [config.yaml](config.yaml) · [label_generator.py](label_generator.py) · [chunk_labeler.py](chunk_labeler.py) · [llm_engine.py](llm_engine.py) · [embedding_filter.py](embedding_filter.py) · [data_loader.py](data_loader.py) · [utils.py](utils.py) · [tools/make_demo.py](tools/make_demo.py)

---

## 1) Install

```bash
# Python 3.10–3.12, CUDA 12.x driver, single 24 GB GPU
pip install -r requirements.txt
```

Key deps: `vllm>=0.6.6`, `xgrammar`, `FlagEmbedding`, `transformers`, `pandas`, `openpyxl`, `scikit-learn`.

---

## 2) Models — cached **inside the project**, no manual download required

All HF cache paths are pinned to `$PROJECT_ROOT/models/` via [env.sh](env.sh), which is sourced by both [run.sh](run.sh) and [tools/download_models.sh](tools/download_models.sh). Auto-download on first run works out of the box; pre-fetching just saves you from a network hiccup mid-pipeline.

Resulting layout:

```
qwen_get_label/
└── models/                                          <-- HF_HOME
    └── hub/
        ├── models--Qwen--Qwen2.5-7B-Instruct-AWQ/
        └── models--BAAI--bge-m3/
```

Cache sizes:

| model                              | size    | role                         |
| ---------------------------------- | ------- | ---------------------------- |
| `Qwen/Qwen2.5-7B-Instruct-AWQ`     | ~5 GB   | LLM (default)                |
| `Qwen/Qwen2.5-14B-Instruct-AWQ`    | ~8 GB   | LLM (higher quality, slower) |
| `BAAI/bge-m3`                      | ~2.2 GB | embedding                    |

**Pre-fetch (recommended):**

```bash
bash tools/download_models.sh         # 7B + bge-m3, ~7 GB
bash tools/download_models.sh 14b     # 14B + bge-m3, ~10 GB
```

After this, `bash run.sh` finds everything locally and never touches the network.

**China mirror** — uncomment the `HF_ENDPOINT` line in [env.sh](env.sh):
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

**Custom location** — edit `HF_HOME` in [env.sh](env.sh) to any absolute path (e.g. `/data/hf_cache`). Both run and download scripts will follow it.

**Switch LLM** — edit `llm.model` in [config.yaml](config.yaml).

---

## 3) Input data formats

`data/chunks.jsonl` (one JSON per line):

```json
{"chunk_id": "doc1_p12_c3", "text": "孟子曰：「老吾老以及人之老...」", "source": "mengzi.pdf", "page": 12}
```

Only `chunk_id` and `text` are required.

`data/question.xlsx`: any sheet with a `question` column (or first column if `question` absent).

---

## 4) Run

```bash
bash run.sh                # full pipeline (stage1 + stage2)
bash run.sh stage1         # build ontology only
bash run.sh stage2         # label chunks using existing outputs/ontology.json
```

Smoke-test in ~5 min before the real run:

```bash
python tools/make_demo.py  # writes 200 chunks + 30 questions to data/
bash run.sh
```

---

## 5) Output

`outputs/ontology.json`:

```json
{
  "labels": ["filial_piety", "social_order", "duty", "harm", ...],
  "descriptions": {"filial_piety": "父母在不远游 | 孝顺 | ...", ...},
  "cluster_to_label": {"0": "filial_piety", "1": "duty", ...},
  "n_questions": 432,
  "n_clusters": 30
}
```

`outputs/chunk_labels.jsonl` (one per chunk, append-only, resume-safe):

```json
{"chunk_id": "doc1_p12_c3", "labels": ["filial_piety", "duty"]}
```

`outputs/stats.json`:

```json
{
  "total_chunks": 250000,
  "resumed": 0,
  "processed_this_run": 250000,
  "failed_this_run": 0,
  "ontology_size": 30,
  "elapsed_seconds": 5430.0,
  "chunks_per_second": 46.04,
  "label_distribution": {"social_order": 41203, "duty": 38110, ...}
}
```

`outputs/failed_chunks.jsonl`: only populated on hard errors. With guided JSON this stays near-empty.

---

## 6) Resume / restart

- `chunk_labels.jsonl` is append-only and `chunk_id`-keyed.
- If the job dies, just rerun `bash run.sh stage2` — `data_loader.load_resume_ids` reads what was already written and skips those ids.
- Worst-case data loss = one `mega_batch` (default 4096 chunks; flushed at every batch boundary).

---

## 7) Configuration knobs

All in [config.yaml](config.yaml). The ones you actually care about:

| key                                | default                            | note                                                                |
| ---------------------------------- | ---------------------------------- | ------------------------------------------------------------------- |
| `llm.model`                        | `Qwen/Qwen2.5-7B-Instruct-AWQ`     | swap to 14B for ≤100k chunks                                        |
| `llm.gpu_memory_utilization`       | 0.82                               | lower to 0.78 if running 14B + bge-m3 on 24 GB                      |
| `llm.max_num_seqs`                 | 256                                | vLLM running batch; main throughput dial                            |
| `llm.max_num_batched_tokens`       | 8192                               | raise to 12288 if you have headroom                                 |
| `embedding.top_k_labels`           | 8                                  | candidates shown to LLM; smaller = faster, larger = higher recall    |
| `inference.mega_batch`             | 4096                               | chunks per `bge-m3 → vLLM` round                                    |
| `inference.max_chunk_chars`        | 800                                | hard truncation; longer chunks slow prefill                         |
| `inference.max_labels_per_chunk`   | 5                                  | upper bound in the JSON schema                                       |
| `inference.threshold_score`        | 0.30                               | cosine cutoff for top-K filter                                       |
| `ontology.target_labels`           | 30                                 | final ontology size target                                          |
| `ontology.seed_labels`             | (20 ethics terms)                  | always retained, fixed order                                         |

---

## 8) Throughput estimates (single 24 GB GPU, AWQ-Marlin)

| chunks   | model   | est. wall-clock     |
| -------- | ------- | ------------------- |
| 100k     | 7B-AWQ  | 30–50 min           |
| 100k     | 14B-AWQ | 70–100 min          |
| 400k     | 7B-AWQ  | 2–2.5 h             |
| 400k     | 14B-AWQ | ~4+ h (not advised) |

`bge-m3` encoding is **not** the bottleneck (~5 min for 400k).

---

## 9) vLLM launch parameters (equivalent server form)

The pipeline uses in-process `LLM(...)` in [llm_engine.py](llm_engine.py). Server-mode equivalent:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq_marlin \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.82 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --guided-decoding-backend xgrammar
```

---

## 10) Performance design highlights

- **Hybrid filter + constrain**: `bge-m3` narrows search space to top-K=8; the LLM still emits under a **single global** enum schema covering the full ontology. xgrammar compiles the FSM **once** and reuses it for every request.
- **Continuous batching**: `LLM.generate(list, ...)` triggers vLLM's continuous batching. No AsyncLLMEngine needed for offline jobs.
- **Co-resident on 24 GB**: `bge-m3` (~0.6 GB fp16) sits alongside 7B-AWQ inside the `gpu_memory_utilization` budget — no model unload between batches.
- **CUDA graphs ON** (`enforce_eager=false`) + `awq_marlin` kernel = ~1.5–2× over the default AWQ kernel.
- **JSON output is tiny** (15–30 tokens) so the bottleneck is **prefill**. Hence `max_chunk_chars=800` + `top_k_labels=8` (short prompts) is the main throughput lever.
- **Incremental jsonl write** with per-batch flush; `chunk_id`-keyed resume → safe to kill at any time.

---

## 11) Troubleshooting

- **CUDA OOM on init** → lower `llm.gpu_memory_utilization` to 0.78, then 0.74.
- **OOM mid-run with 14B** → lower `max_num_seqs` to 128, raise `max_chunk_chars` truncation tighter.
- **Slow first batch** → first call compiles xgrammar FSM + CUDA graphs. Subsequent batches are stable.
- **`failed_chunks.jsonl` not empty** → check the error column. Usually a malformed chunk (empty text); we already filter those, but very long unicode-only sequences may still trip.
- **Ontology has duplicates or low coverage** → raise `ontology.target_labels` to 40, run `bash run.sh stage1` again after deleting `outputs/ontology.json`.
- **Want pure-embedding shortcut (no LLM)** → use `EmbeddingFilter.topk_for_texts(..., threshold=0.45)` directly; skip the LLM step. Drops to ~5 min for 400k chunks at lower quality.

---

## 12) Stage 1 prompt

See [label_generator.py](label_generator.py) — `PROMPT_LABEL_NAMING`. Constrained by:

```json
{
  "type": "object",
  "properties": {
    "label":     {"type": "string", "pattern": "^[a-z][a-z0-9_]{1,40}$"},
    "rationale": {"type": "string", "maxLength": 200}
  },
  "required": ["label"]
}
```

## 13) Stage 2 prompt

See [chunk_labeler.py](chunk_labeler.py) — `PROMPT_TMPL`. Constrained by:

```json
{
  "type": "object",
  "properties": {
    "labels": {
      "type": "array",
      "minItems": 0,
      "maxItems": 5,
      "uniqueItems": true,
      "items": {"type": "string", "enum": ["<all ontology labels>"]}
    }
  },
  "required": ["labels"]
}
```
