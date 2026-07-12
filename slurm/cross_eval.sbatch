#!/bin/bash
#SBATCH -J tirex2-xfer
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# GPU job: cross-dataset transfer — apply a checkpoint trained on dataset A (train_baseline.sbatch
# with ALLTRAIN=1) to dataset B's windows. Both datasets must be harmonized to 60 s (see data_h60.yaml).
#
#   CKPT=results/baseline_ckpt_vitaldb60.pt CONFIG=datasets/mover/configs/data.yaml COV=mover_rate \
#       TAG=xfer-tft_vitaldb60TOmover_art  sbatch slurm/cross_eval.sbatch
#   CKPT=results/baseline_ckpt_mover_art.pt CONFIG=datasets/vitaldb/configs/data_h60.yaml COV=rate \
#       TAG=xfer-tft_moverTOvitaldb60      sbatch slurm/cross_eval.sbatch
set -euo pipefail

CKPT="${CKPT:?set CKPT=results/baseline_ckpt_<source>.pt}"
CONFIG="${CONFIG:?set CONFIG=<target dataset config>}"
COV="${COV:?set COV=<target covariate preset>}"
TAG="${TAG:?set TAG=xfer-<model>_<A>TO<B>}"
ARM="${ARM:-M0}"
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: submit with 'sbatch slurm/cross_eval.sbatch'." >&2; exit 1
fi
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
[ -f "${PROJECT_ROOT}/requirements_gpu.txt" ] || [ ! -f "${PROJECT_ROOT}/../requirements_gpu.txt" ] || PROJECT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DEFAULT_SQSH="${HOME}/containers/tirex2.sqsh"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$( [ -f "$DEFAULT_SQSH" ] && echo "$DEFAULT_SQSH" || echo "nvcr.io#nvidia/pytorch:25.09-py3" )}"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results"
export SLURM_STEP_LAUNCH_TIMEOUT=600
echo "CKPT=${CKPT} CONFIG=${CONFIG} COV=${COV} ARM=${ARM} TAG=${TAG}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT='${PROJECT_ROOT}'
source '${PROJECT_ROOT}/slurm/setup_env.sh'
cd '${PROJECT_ROOT}'
echo '[job] cross-eval ${CKPT} -> ${TAG} (arm ${ARM})'
PYTHONPATH=scripts:datasets/vitaldb:datasets/mover python scripts/baselines/cross_eval.py \
    --ckpt ${CKPT} --config ${CONFIG} --cov ${COV} --arm ${ARM} --tag ${TAG} --device cuda
echo '[job] Done.'
"
