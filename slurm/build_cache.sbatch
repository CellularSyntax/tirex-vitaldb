#!/bin/bash
#SBATCH -J tirex2-buildcache
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p cpu
#SBATCH -q cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# CPU job: build the loader caches from vital_files/ (must be downloaded first —
# see slurm/download_vitalfiles.sh). Reads .vital, no GPU/model. Produces:
#   * results/cohort_manifest.csv + outputs/cache/*.npz   (anesthetic cohort, via build_cohort.py)
#   * outputs/cache_pressor/*.npz                          (phenylephrine subset, data_pressor.yaml)
# This is the long pole (I/O over ~2700 .vital); run it once, GPU jobs then reuse the caches.
#
# Submit from project root:  sbatch slurm/build_cache.sbatch
set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: submit this with 'sbatch slurm/build_cache.sbatch' (not 'bash')." >&2; exit 1
fi
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
# tolerate submission from slurm/ (repo root is its parent)
[ -f "${PROJECT_ROOT}/requirements_gpu.txt" ] || [ ! -f "${PROJECT_ROOT}/../requirements_gpu.txt" ] || PROJECT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DEFAULT_SQSH="${HOME}/containers/tirex2.sqsh"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$( [ -f "$DEFAULT_SQSH" ] && echo "$DEFAULT_SQSH" || echo "nvcr.io#nvidia/pytorch:25.09-py3" )}"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs"
export SLURM_STEP_LAUNCH_TIMEOUT=600
# CPU-partition nodes lack nvidia-container-cli, so enroot's NVIDIA hook (98-nvidia.sh)
# would fail on the image's baked NVIDIA_VISIBLE_DEVICES=all. This is a CPU-only job
# (reads .vital, no model), so tell enroot to skip GPU setup entirely.
export NVIDIA_VISIBLE_DEVICES=void
export NVIDIA_DRIVER_CAPABILITIES=void
echo "Project root : ${PROJECT_ROOT}"; echo "Container : ${CONTAINER_IMAGE}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT='${PROJECT_ROOT}'
# source (not 'bash') so setup_env.sh's exports persist into this shell. Harmless for the
# cache build (no HF), but keeps both jobs consistent.
source '${PROJECT_ROOT}/slurm/setup_env.sh'
cd '${PROJECT_ROOT}'

echo '[cache] Building anesthetic cohort cache + manifest (build_cohort.py) ...'
PYTHONPATH=scripts:datasets/vitaldb python datasets/vitaldb/build_cohort.py --workers ${SLURM_CPUS_PER_TASK}

echo '[cache] Building phenylephrine (pressor) cache from data_pressor.yaml ...'
PYTHONPATH=scripts:datasets/vitaldb python -c \"
import vitaldb_loader as L
cfg=L.load_config('datasets/vitaldb/configs/data_pressor.yaml'); clin=L._clinical_index(cfg['clinical_csv'])
ids=[l.strip() for l in open('datasets/vitaldb/pressor_cases_phen.txt') if l.strip()]
ok=sum(L.load_case(c,cfg,clin) is not None for c in ids)
print(f'[cache] pressor cache: {ok}/{len(ids)} cases built')
\"
echo '[cache] Done. Caches: outputs/cache/ (anesthetic) + outputs/cache_pressor/ (phenylephrine).'
"
