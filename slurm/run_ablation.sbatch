#!/bin/bash
#SBATCH -J tirex2-ablation
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# GPU job: full-cohort zero-shot ablation for one covariate preset. Reads the caches
# built by build_cache.sbatch; single A100 handles the whole cohort (sLSTM CUDA fast path).
#
# Select the covariate with the COV env-var at submit time (default ce):
#   COV=ce      sbatch slurm/run_ablation.sbatch   # remi+propofol CE, full 2659 cohort   -> tag all<N>
#   COV=rate    sbatch slurm/run_ablation.sbatch   # remi+propofol RATE, full cohort      -> tag all<N>_covrate
#   COV=pressor sbatch slurm/run_ablation.sbatch   # phenylephrine, 115-case subset       -> tag cases115_covpressor
#
# Optional multi-GPU sharding of one covariate (mirrors submit_duke_sweep.sh): pass an
# array + N_SHARDS, then merge afterwards. Not needed for a single A100 — see notes/CLUSTER.md.
set -euo pipefail

COV="${COV:-ce}"
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: submit this with 'sbatch slurm/run_ablation.sbatch' (not 'bash')." >&2; exit 1
fi
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
# tolerate submission from slurm/ (repo root is its parent)
[ -f "${PROJECT_ROOT}/requirements_gpu.txt" ] || [ ! -f "${PROJECT_ROOT}/../requirements_gpu.txt" ] || PROJECT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DEFAULT_SQSH="${HOME}/containers/tirex2.sqsh"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$( [ -f "$DEFAULT_SQSH" ] && echo "$DEFAULT_SQSH" || echo "nvcr.io#nvidia/pytorch:25.09-py3" )}"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs"
export SLURM_STEP_LAUNCH_TIMEOUT=600

# Optional sharding coordinates (from `sbatch --array` + N_SHARDS/SHARD_OFFSET env).
N_SHARDS="${N_SHARDS:-1}"
if [ "${N_SHARDS}" -gt 1 ]; then
  SHARD_IDX=$(( ${SLURM_ARRAY_TASK_ID:-0} + ${SHARD_OFFSET:-0} ))
  SHARD_ARGS="--n-shards ${N_SHARDS} --shard-idx ${SHARD_IDX}"
else
  SHARD_ARGS=""
fi

# Covariate -> phase3 arguments.
case "${COV}" in
  ce)      RUN_ARGS="--config datasets/vitaldb/configs/data.yaml --cov ce --all" ;;
  rate)    RUN_ARGS="--config datasets/vitaldb/configs/data.yaml --cov rate --all" ;;
  ce_remi) RUN_ARGS="--config datasets/vitaldb/configs/data.yaml --cov ce_remi --all" ;;
  pressor) RUN_ARGS="--config datasets/vitaldb/configs/data_pressor.yaml --cov pressor --cases-file datasets/vitaldb/pressor_cases_phen.txt" ;;
  *) echo "unknown COV=${COV}" >&2; exit 1 ;;
esac

echo "Project root : ${PROJECT_ROOT}"; echo "Container : ${CONTAINER_IMAGE}"
echo "COV=${COV}  RUN_ARGS=${RUN_ARGS}  SHARD_ARGS=${SHARD_ARGS}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT='${PROJECT_ROOT}'
# MUST be sourced (not 'bash setup_env.sh') so its exports — HF_HOME, HF_HUB_OFFLINE,
# TORCH_EXTENSIONS_DIR — reach the python process below. A subshell would lose them.
source '${PROJECT_ROOT}/slurm/setup_env.sh'
cd '${PROJECT_ROOT}'
echo '[job] phase3_ablation.py ${RUN_ARGS} ${SHARD_ARGS} --device cuda --batch-size 512'
PYTHONPATH=scripts:datasets/vitaldb python scripts/phase3_ablation.py ${RUN_ARGS} ${SHARD_ARGS} --device cuda --batch-size 512
echo '[job] Done.'
"
