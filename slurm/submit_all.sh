#!/bin/bash
# Submit the full-cohort TiRex-2 run as dependent Slurm jobs (mirrors jaxfibers/submit_all.sh).
# Prereqs (do these once, manually):
#   1) vital_files/ downloaded          ->  bash slurm/download_vitalfiles.sh
#   2) container built                  ->  export HF_TOKEN=hf_xxx; bash slurm/build_container.sh
# Then:  export HF_TOKEN=hf_xxx; bash slurm/submit_all.sh
#
# Job graph:  build_cache (CPU) --afterok--> { ce, rate, pressor } GPU jobs (run in parallel).
# GPU work is light (zero-shot inference); one A100 per covariate is plenty — 3 of your 6 GPU slots.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || { echo "cannot cd to repo root" >&2; exit 1; }  # sbatch paths are relative to here
mkdir -p logs
: "${HF_TOKEN:?export HF_TOKEN=hf_xxx first (gated NX-AI/TiRex-2 weights)}"

echo "[submit] 1/4  build_cache (CPU) ..."
CACHE_JID=$(sbatch --parsable slurm/build_cache.sbatch)
echo "         job ${CACHE_JID}"

echo "[submit] 2/4  ce  full cohort (a100, afterok:${CACHE_JID}) ..."
CE_JID=$(COV=ce sbatch --parsable --dependency=afterok:${CACHE_JID} \
         -J tirex2-ce -q a100 --gres=gpu:a100:1 slurm/run_ablation.sbatch)
echo "         job ${CE_JID}"

echo "[submit] 3/4  rate full cohort (a100, afterok:${CACHE_JID}) ..."
RATE_JID=$(COV=rate sbatch --parsable --dependency=afterok:${CACHE_JID} \
           -J tirex2-rate -q a100 --gres=gpu:a100:1 slurm/run_ablation.sbatch)
echo "         job ${RATE_JID}"

echo "[submit] 4/4  pressor subset (a100, afterok:${CACHE_JID}) ..."
PRES_JID=$(COV=pressor sbatch --parsable --dependency=afterok:${CACHE_JID} \
           -J tirex2-pressor -q a100 --gres=gpu:a100:1 slurm/run_ablation.sbatch)
echo "         job ${PRES_JID}"

echo ""
echo "Submitted: cache=${CACHE_JID} -> ce=${CE_JID}, rate=${RATE_JID}, pressor=${PRES_JID}"
echo "Monitor:   squeue -u \$(whoami)    |    tail -f logs/tirex2-*-<jobid>.out"
echo "Outputs:   results/ablation_primary_all<N>{,_covrate}.json, ..._covpressor, + outputs/figs/dashboard_*.png"
