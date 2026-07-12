#!/bin/bash
#SBATCH -J tirex2-baseline
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# GPU job: train the matched supervised baseline (TFT) on the SAME windows/splits as the
# zero-shot TiRex-2 run, then score both on the identical held-out test subjects.
# Reuses the tirex2 container (pure-PyTorch model, no extra deps) and the built caches.
#
# Select architecture / covariate / cohort at submit time:
#   MODEL=tft       COV=ce      MATCH=results/ablation_windows_all2873.csv          sbatch slurm/train_baseline.sbatch
#   MODEL=patchtst  COV=ce      MATCH=results/ablation_windows_all2873.csv          sbatch slurm/train_baseline.sbatch
#   MODEL=tft       COV=pressor MATCH=results/ablation_windows_cases115_covpressor.csv \
#       CONFIG=datasets/vitaldb/configs/data_pressor.yaml                            sbatch slurm/train_baseline.sbatch
# 5-fold subject CV (out-of-fold over all cases):
#   MODEL=tft FOLDS=5 MATCH=results/ablation_windows_all2873.csv                     sbatch slurm/train_baseline.sbatch
# MOVER internal CV:
#   MODEL=tft COV=mover_rate CONFIG=datasets/mover/configs/data.yaml FOLDS=5 \
#       MATCH=results/ablation_windows_mover_art.csv                                 sbatch slurm/train_baseline.sbatch
# Cross-dataset transfer SOURCE (train on all, save checkpoint; no CSV/compare):
#   MODEL=tft COV=rate CONFIG=datasets/vitaldb/configs/data_h60.yaml ALLTRAIN=1 \
#       CKPT=results/baseline_ckpt_vitaldb60.pt                                       sbatch slurm/train_baseline.sbatch
#   MODEL=tft COV=mover_rate CONFIG=datasets/mover/configs/data.yaml ALLTRAIN=1 \
#       CKPT=results/baseline_ckpt_mover_art.pt                                       sbatch slurm/train_baseline.sbatch
# compare.py (TiRex vs baseline classification head-to-head) auto-runs only for COV=ce.
set -euo pipefail

MODEL="${MODEL:-tft}"
COV="${COV:-ce}"
CONFIG="${CONFIG:-datasets/vitaldb/configs/data.yaml}"
MATCH="${MATCH:-results/ablation_windows_all2873.csv}"
FOLDS="${FOLDS:-1}"                 # >1 -> K-fold subject CV (out-of-fold over all cases)
ALLTRAIN="${ALLTRAIN:-0}"           # 1 -> train on all cases, save CKPT for cross_eval.py (no CSV)
CKPT="${CKPT:-}"
# cohort selection + mode: all-train uses the full manifest cohort; else lock to the matched windows.
if [ "${ALLTRAIN}" = "1" ]; then
  MODE_ARGS="--all --all-train"; [ -n "${CKPT}" ] && MODE_ARGS="${MODE_ARGS} --save-ckpt ${CKPT}"
else
  MODE_ARGS="--match-tirex ${MATCH} --folds ${FOLDS}"
fi
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: submit this with 'sbatch slurm/train_baseline.sbatch' (not 'bash')." >&2; exit 1
fi
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
[ -f "${PROJECT_ROOT}/requirements_gpu.txt" ] || [ ! -f "${PROJECT_ROOT}/../requirements_gpu.txt" ] || PROJECT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DEFAULT_SQSH="${HOME}/containers/tirex2.sqsh"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$( [ -f "$DEFAULT_SQSH" ] && echo "$DEFAULT_SQSH" || echo "nvcr.io#nvidia/pytorch:25.09-py3" )}"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results"
export SLURM_STEP_LAUNCH_TIMEOUT=600
echo "Project root : ${PROJECT_ROOT}"; echo "Container : ${CONTAINER_IMAGE}"
echo "MODEL=${MODEL} COV=${COV} CONFIG=${CONFIG} FOLDS=${FOLDS} ALLTRAIN=${ALLTRAIN}"
echo "MODE_ARGS=${MODE_ARGS}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT='${PROJECT_ROOT}'
source '${PROJECT_ROOT}/slurm/setup_env.sh'
cd '${PROJECT_ROOT}'
echo '[job] training baseline ${MODEL} cov=${COV} (${MODE_ARGS})'
PYTHONPATH=scripts:datasets/vitaldb:datasets/mover python scripts/baselines/train.py \
    --config ${CONFIG} --model ${MODEL} --cov ${COV} ${MODE_ARGS} \
    --device cuda --batch-size 512 --epochs 60
STEM=\$(basename ${MATCH} | sed 's/ablation_windows_//;s/.csv//')
# compare.py auto-runs only for the VitalDB CE head-to-head (single split); skip for CV/all-train/MOVER.
if [ '${COV}' = 'ce' ] && [ '${ALLTRAIN}' != '1' ] && [ '${FOLDS}' = '1' ]; then
  echo '[job] matched classification comparison (TiRex vs ${MODEL})'
  PYTHONPATH=scripts:datasets/vitaldb:datasets/mover python scripts/baselines/compare.py \
      --tirex \${STEM} --baseline baseline-${MODEL}_\${STEM}
fi
echo '[job] Done.'
"
