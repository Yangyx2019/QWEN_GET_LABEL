#!/usr/bin/env bash
set -euo pipefail

# === one-shot launcher ===
# Usage:
#   bash run.sh                # full pipeline
#   bash run.sh stage1         # only ontology
#   bash run.sh stage2         # only labeling (needs outputs/ontology.json)

# Anchor to script dir so `bash run.sh` works from any CWD.
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "${SCRIPT_DIR}"

# Load HF cache + runtime envs (HF_HOME points into ./models)
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

mkdir -p outputs logs data

STAGE="${1:-all}"

# If download_models.sh has finished, run in offline mode so loaders don't
# revisit the hub for optional files (e.g. bge-m3/imgs/) that may 403 on mirrors.
# Override with `HF_HUB_OFFLINE=0 bash run.sh ...` if you really need network.
if [[ -f "${HF_HOME}/.cache_ready" ]]; then
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
else
  echo "[run] WARN: models cache sentinel not found. Run 'bash tools/download_models.sh' first for a stable offline run."
fi

echo "[run] HF_HOME=${HF_HOME}"
echo "[run] HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"
echo "[run] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run] stage=${STAGE}"

python -u main.py --config config.yaml --stage "${STAGE}" 2>&1 | tee "logs/run_${STAGE}_$(date +%Y%m%d_%H%M%S).log"
