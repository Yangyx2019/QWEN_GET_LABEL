# Sourced by run.sh and tools/download_models.sh — keeps cache + runtime envs in one place.

# Anchor paths to THIS file's location, not the caller's CWD.
ENV_SH_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" >/dev/null 2>&1 && pwd )"
export PROJECT_ROOT="${ENV_SH_DIR}"

# ---- HuggingFace cache lives INSIDE the project ----
# Layout:
#   $PROJECT_ROOT/models/
#     hub/                                  # snapshots + blobs (the big stuff)
#       models--Qwen--Qwen2.5-7B-Instruct-AWQ/
#       models--BAAI--bge-m3/
#     ...
export HF_HOME="${PROJECT_ROOT}/models"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
# transformers picks this up too (kept for back-compat)
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

# Faster downloads (hf_transfer); harmless if not installed.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# Optional HF mirror (uncomment if HF Hub is slow from your region).
export HF_ENDPOINT="https://hf-mirror.com"

# Runtime envs used by vLLM / tokenizers.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

# Reduce PyTorch CUDA fragmentation -- helps when bge-m3 + vLLM co-resident on 24GB.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Pin to GPU 0 by default; respect any externally-set value.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${HF_HOME}/hub"
