#!/bin/bash
#SBATCH -J tirex2-fp
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# GPU job: measure the DEPLOYMENT COMPUTE FOOTPRINT of one model — parameter count, one-off load
# time, per-window inference latency/throughput, and peak GPU memory — timing the SAME forecast
# calls used to produce the paper results (no retraining). Appends one row to
# results/compute_footprint.json keyed by MODEL. Aggregate with scripts/compute_footprint_table.py.
#
# Mirrors slurm/extract_embeddings.sbatch:
#   * chronos/timesfm/moirai  -> per-model isolated venv .tsfm-venv-<model> (run setup_tsfm first)
#   * tirex2 / tft / patchtst -> the tirex2 container env
# tft/patchtst additionally need their trained checkpoint (CKPT); the footprint is architecture-
# driven (d=64, H=15, Lc=30) so any matching baseline checkpoint gives the same numbers.
#
# Run all six on VitalDB (single A100, ~a few min each):
#   for M in tirex2 chronos timesfm moirai; do MODEL=$M sbatch slurm/compute_footprint.sbatch; done
#   MODEL=tft      CKPT=results/baseline_ckpt_vitaldb15_tft.pt      sbatch slurm/compute_footprint.sbatch
#   MODEL=patchtst CKPT=results/baseline_ckpt_vitaldb15_patchtst.pt sbatch slurm/compute_footprint.sbatch
#   # if vitaldb15 checkpoints aren't present, vitaldb60_* give the identical architecture footprint.
# Then aggregate + pull back:
#   PYTHONPATH=scripts python3 scripts/compute_footprint_table.py
#   # copy results/compute_footprint.json + results/tables/TableS_compute_footprint.* to the Mac.
set -euo pipefail

MODEL="${MODEL:-tirex2}"
COV="${COV:-ce}"
CONFIG="${CONFIG:-datasets/vitaldb/configs/data.yaml}"
MATCH="${MATCH:-results/ablation_windows_all2873.csv}"
NWIN="${NWIN:-512}"
BS="${BS:-64}"
CKPT="${CKPT:-}"
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: submit with 'sbatch slurm/compute_footprint.sbatch' (not 'bash')." >&2; exit 1
fi
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
DEFAULT_SQSH="${HOME}/containers/tirex2.sqsh"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$( [ -f "$DEFAULT_SQSH" ] && echo "$DEFAULT_SQSH" || echo "nvcr.io#nvidia/pytorch:25.09-py3" )}"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results"
export SLURM_STEP_LAUNCH_TIMEOUT=600
# Force the enroot NVIDIA hook to inject the allocated GPU (defend against an inherited
# NVIDIA_VISIBLE_DEVICES=void from a login-shell CPU srun). Safe under --gres: the cgroup
# still confines the container to the one allocated A100.
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
echo "Project root : ${PROJECT_ROOT}"; echo "Container : ${CONTAINER_IMAGE}"
echo "MODEL=${MODEL} COV=${COV} CONFIG=${CONFIG} MATCH=${MATCH} NWIN=${NWIN} BS=${BS} CKPT=${CKPT:-<none>}"

# default checkpoints for the trained baselines if CKPT not given
if [ -z "${CKPT}" ]; then
  if [ "${MODEL}" = "tft" ];      then CKPT="results/baseline_ckpt_vitaldb15_tft.pt"; fi
  if [ "${MODEL}" = "patchtst" ]; then CKPT="results/baseline_ckpt_vitaldb15_patchtst.pt"; fi
fi
CKPT_ARG=""; [ -n "${CKPT}" ] && CKPT_ARG="--ckpt ${CKPT}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT='${PROJECT_ROOT}'
source '${PROJECT_ROOT}/slurm/setup_env.sh'
cd '${PROJECT_ROOT}'
if [ '${MODEL}' = 'tirex2' ] || [ '${MODEL}' = 'tft' ] || [ '${MODEL}' = 'patchtst' ]; then
  echo \"[fp] ${MODEL} in container env\"
  PYTHONPATH=scripts:datasets/vitaldb:datasets/mover python scripts/compute_footprint.py \
      --model ${MODEL} --config ${CONFIG} --cov ${COV} --match-tirex ${MATCH} \
      --n-windows ${NWIN} --batch-size ${BS} --device cuda ${CKPT_ARG}
else
  unset HF_HUB_OFFLINE || true
  VENV='${PROJECT_ROOT}/.tsfm-venv-${MODEL}'
  if [ ! -f \"\$VENV/bin/activate\" ]; then
    echo \"[fp] ERROR: \$VENV missing — run: MODELS=${MODEL} sbatch slurm/setup_tsfm.sbatch\" >&2; exit 1
  fi
  source \"\$VENV/bin/activate\"
  PYTHONPATH=scripts:datasets/vitaldb:datasets/mover python scripts/compute_footprint.py \
      --model ${MODEL} --config ${CONFIG} --cov ${COV} --match-tirex ${MATCH} \
      --n-windows ${NWIN} --batch-size ${BS} --device cuda
fi
echo '[fp] Done.'
"
