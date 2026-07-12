#!/bin/bash
#SBATCH -J tirex2-tsfm-setup
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# ONE-TIME: build a Python venv for the zero-shot foundation-model sweep (Chronos, TimesFM,
# Moirai). The venv layers ONLY those extra libraries on top of the container's torch 2.9
# (--system-site-packages), so we never reinstall/clobber torch. It lives on the mounted
# project filesystem and is reused by every slurm/run_zeroshot.sbatch job.
#
#   sbatch slurm/setup_tsfm.sbatch          # run once; ~10-15 min (downloads wheels)
#
# Weights themselves are downloaded lazily on the first run_zeroshot job (all ungated).
set -euo pipefail

# CPU node: skip the enroot NVIDIA hook (nvidia-container-cli is absent on cpu partition).
export NVIDIA_VISIBLE_DEVICES=void
export NVIDIA_DRIVER_CAPABILITIES=void

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: submit with 'sbatch slurm/setup_tsfm.sbatch' (not 'bash')." >&2; exit 1
fi
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
[ -f "${PROJECT_ROOT}/requirements_gpu.txt" ] || [ ! -f "${PROJECT_ROOT}/../requirements_gpu.txt" ] || PROJECT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DEFAULT_SQSH="${HOME}/containers/tirex2.sqsh"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$( [ -f "$DEFAULT_SQSH" ] && echo "$DEFAULT_SQSH" || echo "nvcr.io#nvidia/pytorch:25.09-py3" )}"
mkdir -p "${PROJECT_ROOT}/logs"
echo "Project root : ${PROJECT_ROOT}"; echo "Container : ${CONTAINER_IMAGE}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
VENV='${PROJECT_ROOT}/.tsfm-venv'
echo '[tsfm] container python:' \$(python -V)
python -c 'import torch; print(\"[tsfm] container torch\", torch.__version__)'
python -m venv --system-site-packages \"\$VENV\"
source \"\$VENV/bin/activate\"
python -m pip install --upgrade pip --quiet
echo '[tsfm] torch visible in venv:' \$(python -c 'import torch; print(torch.__version__)')

# Install each model's libs separately so one failure does not block the others.
# --upgrade-strategy only-if-needed keeps the container's torch (never downgraded).
PIP='python -m pip install --upgrade-strategy only-if-needed'
echo '[tsfm] installing chronos-forecasting ...'; \$PIP chronos-forecasting  || echo '[tsfm] chronos FAILED'
echo '[tsfm] installing timesfm[torch] ...';      \$PIP \"timesfm[torch]\"    || echo '[tsfm] timesfm FAILED'
echo '[tsfm] installing uni2ts (Moirai) ...';     \$PIP uni2ts              || echo '[tsfm] uni2ts FAILED'

echo '[tsfm] final torch:' \$(python -c 'import torch; print(torch.__version__, torch.version.cuda)')
echo '[tsfm] import checks:'
python -c 'import chronos; print(\"  chronos OK\")'                 || echo '  chronos import FAILED'
python -c 'import timesfm; print(\"  timesfm OK\")'                 || echo '  timesfm import FAILED'
python -c 'from uni2ts.model.moirai import MoiraiModule; print(\"  uni2ts/moirai OK\")' || echo '  uni2ts import FAILED'
echo '[tsfm] venv ready at '\"\$VENV\"
"
