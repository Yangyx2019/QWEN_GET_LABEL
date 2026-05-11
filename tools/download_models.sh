#!/usr/bin/env bash
set -euo pipefail
#
# Pre-fetch all models into the project-local HF cache ($PROJECT_ROOT/models).
# Both this script and run.sh source env.sh, so the cache location is identical.
#
# Usage:
#   bash tools/download_models.sh          # downloads default 7B + bge-m3
#   bash tools/download_models.sh 14b      # downloads 14B + bge-m3 instead
#
# After this finishes, `bash run.sh` will find everything locally with no network.

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/env.sh"

MODE="${1:-7b}"
case "${MODE}" in
  7b)  LLM_REPO="Qwen/Qwen2.5-7B-Instruct-AWQ" ;;
  14b) LLM_REPO="Qwen/Qwen2.5-14B-Instruct-AWQ" ;;
  *)   echo "unknown mode: ${MODE} (expected: 7b | 14b)" >&2; exit 1 ;;
esac

EMB_REPO="BAAI/bge-m3"

echo "[download] HF_HOME       = ${HF_HOME}"
echo "[download] LLM repo      = ${LLM_REPO}"
echo "[download] embedding repo= ${EMB_REPO}"

# Make sure huggingface-cli is available; tip user if not.
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found. Installing huggingface_hub[cli]..." >&2
  python -m pip install -U "huggingface_hub[cli]" hf_transfer
fi

# Pull each repo. HF_HOME redirects everything into ./models.
huggingface-cli download "${LLM_REPO}"
huggingface-cli download "${EMB_REPO}"

echo
echo "[download] done."
du -sh "${HF_HOME}" 2>/dev/null || true
echo "[download] models cached under: ${HF_HOME}/hub"
