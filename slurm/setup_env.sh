#!/bin/bash
# Per-job environment setup for TiRex-2 inside the NVIDIA PyTorch container.
# Called at the start of every Slurm job from within the container.
#
# Behaviour (mirrors jaxfibers/slurm/setup_env.sh):
#   * If the container was built via slurm/build_container.sh, the pip deps are
#     already baked in -> we detect that (probe-import tirex2) and skip pip.
#   * Otherwise (fresh nvcr.io pull) we pip install -r requirements_gpu.txt.
#   * HF model cache and the compiled sLSTM kernels live on the mounted project
#     filesystem (NOT in the read-only container) so they persist across jobs and
#     are downloaded/compiled at most once.
#
# Requires PROJECT_ROOT to be set (done by each .sbatch).
set -euo pipefail

echo "[setup] Python: $(python -V)"
echo "[setup] torch : $(python -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)')"

# ── persistent caches on the mounted filesystem (survive across jobs) ──────────
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.hf_cache}"                 # gated TiRex weights
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${PROJECT_ROOT}/.torch_ext}"  # compiled kernels
# Compile the sLSTM kernels for every GPU arch we might land on (a100=8.0, h100=9.0,
# b200=10.0, a30=8.0, a16=8.6). nvcc cross-compiles regardless of the build GPU.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0;10.0}"
mkdir -p "${HF_HOME}" "${TORCH_EXTENSIONS_DIR}"

# Gated NX-AI/TiRex-2 weights: build_container.sh pre-downloads them into HF_HOME.
# If they're already cached, force offline mode — otherwise huggingface_hub makes a
# network call to REVALIDATE the cached files, which 401s for a gated repo unless a
# token is present. Offline mode uses the cache directly (no network, no token needed).
if compgen -G "${HF_HOME}/hub/models--NX-AI--TiRex-2/snapshots/*/model-config.yaml" >/dev/null 2>&1; then
    export HF_HUB_OFFLINE=1
    echo "[setup] Cached TiRex-2 weights found -> HF_HUB_OFFLINE=1 (no network/token needed)."
elif [ -n "${HF_TOKEN:-}" ]; then
    echo "[setup] TiRex-2 weights not cached; will download once with HF_TOKEN into ${HF_HOME}."
else
    echo "[setup] WARNING: TiRex-2 weights not cached and HF_TOKEN not set — gated download will 401."
    echo "        export HF_TOKEN=hf_xxx before submitting, or run slurm/build_container.sh first."
fi

# ── skip pip if the container already has our deps ────────────────────────────
if python -c "import tirex2, vitaldb" 2>/dev/null; then
    echo "[setup] tirex2 + vitaldb already installed in container — skipping pip."
else
    echo "[setup] Installing pip dependencies from requirements_gpu.txt ..."
    pip install --upgrade pip --quiet
    unset PIP_CONSTRAINT   # NVIDIA containers pin numpy; let requirements_gpu.txt win
    pip install --upgrade -r "${PROJECT_ROOT}/requirements_gpu.txt"
fi

echo "[setup] CUDA devices:"
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
echo "[setup] HF_HOME=${HF_HOME}  TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
echo "[setup] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
echo "[setup] Environment ready."
