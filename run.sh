#!/usr/bin/env bash
set -euo pipefail

# === one-shot launcher ===
# Usage:
#   bash run.sh                # full pipeline
#   bash run.sh stage1         # only ontology
#   bash run.sh stage2         # only labeling (needs outputs/ontology.json)

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1
# pin to single GPU 0 by default; override CUDA_VISIBLE_DEVICES from env
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p outputs logs data

STAGE="${1:-all}"

python -u main.py --config config.yaml --stage "${STAGE}" 2>&1 | tee "logs/run_${STAGE}_$(date +%Y%m%d_%H%M%S).log"
