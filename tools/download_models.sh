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
# We use snapshot_download from python so we can pass `ignore_patterns` --
# the hf-mirror sometimes 403s on docs/.DS_Store inside repos (e.g. bge-m3/imgs/).
# AWQ and bge-m3 use *.safetensors, so we skip the *.bin variants to save ~2GB.
export LLM_REPO EMB_REPO
python - <<'PY'
import os
from huggingface_hub import snapshot_download

llm_repo = os.environ["LLM_REPO"]
emb_repo = os.environ["EMB_REPO"]

print(f"[snapshot] {llm_repo}")
snapshot_download(
    repo_id=llm_repo,
    ignore_patterns=["*.bin", "*.DS_Store", "imgs/*"],
    max_workers=8,
    resume_download=True,
)
print(f"[snapshot] {emb_repo}")
snapshot_download(
    repo_id=emb_repo,
    ignore_patterns=["*.bin", "*.DS_Store", "imgs/*"],
    max_workers=8,
    resume_download=True,
)
PY

# Sentinel so run.sh can know we finished and switch to offline mode safely.
touch "${HF_HOME}/.cache_ready"

echo
echo "[download] done."
du -sh "${HF_HOME}" 2>/dev/null || true
echo "[download] models cached under: ${HF_HOME}/hub"
