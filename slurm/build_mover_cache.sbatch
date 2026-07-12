#!/bin/bash
#SBATCH -J mover-cache
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# ONE-TIME (CPU): stream the SIS tables and build the MOVER per-case cache + clinical_data.csv +
# cohort_manifest.csv. High mem (96G) because the 180-200 MB observations/vitals CSVs are
# accumulated per-PID in memory. After this, run TiRex/baselines/zeroshot with the MOVER config.
#
#   sbatch slurm/build_mover_cache.sbatch
set -euo pipefail
export NVIDIA_VISIBLE_DEVICES=void            # CPU node: skip the enroot NVIDIA hook
export NVIDIA_DRIVER_CAPABILITIES=void

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: submit with 'sbatch slurm/build_mover_cache.sbatch'." >&2; exit 1
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
cd '${PROJECT_ROOT}'
echo '[job] building MOVER cache ...'
PYTHONPATH=datasets/mover python datasets/mover/build_mover_cache.py \
    --config datasets/mover/configs/data.yaml
echo '[job] Done.'
"
