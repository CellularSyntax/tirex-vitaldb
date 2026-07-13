#!/bin/bash
#SBATCH -J tirex2-clineval
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p cpu
#SBATCH -q cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# CPU job (no GPU, no model): regenerate the TiRex-2 VitalDB clinical-translation JSON.
# The committed results/clinical_eval_all2873.json is STALE -- it was written before the
# lead_curve / alarm_tradeoff arrays were added to clinical_eval.py, so TiRex-2 is missing
# from Fig 5 panels a1 (detection yield) and a2 (alarm burden). The three baseline files
# already carry those arrays; only TiRex-2's needs rebuilding. Rebuilds from the full
# per-case cache (present on the cluster), which the numpy/scikit-learn stack inside the
# container needs -- login nodes lack both.
#
# Submit from project root:  sbatch slurm/clinical_eval_tirex.sbatch
# Then pull results/clinical_eval_all2873.json back to the Mac's results/ folder.
set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: submit this with 'sbatch slurm/clinical_eval_tirex.sbatch' (not 'bash')." >&2; exit 1
fi
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
DEFAULT_SQSH="${HOME}/containers/tirex2.sqsh"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$( [ -f "$DEFAULT_SQSH" ] && echo "$DEFAULT_SQSH" || echo "nvcr.io#nvidia/pytorch:25.09-py3" )}"
mkdir -p "${PROJECT_ROOT}/logs"
export SLURM_STEP_LAUNCH_TIMEOUT=600
# CPU-only job: skip enroot's NVIDIA hook (CPU nodes lack nvidia-container-cli).
export NVIDIA_VISIBLE_DEVICES=void
export NVIDIA_DRIVER_CAPABILITIES=void
echo "Project root : ${PROJECT_ROOT}"; echo "Container : ${CONTAINER_IMAGE}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
cd '${PROJECT_ROOT}'
echo '[cline] python:' \$(python -V)
echo '[cline] regenerating clinical_eval_all2873.json (TiRex-2 VitalDB) ...'
PYTHONPATH=scripts:datasets/vitaldb python scripts/clinical_eval.py all2873
echo '[cline] done. Pull results/clinical_eval_all2873.json to the Mac.'
"
