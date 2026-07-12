#!/usr/bin/env bash
# Run on the CLUSTER (which has the MOVER cache + manifest) to produce the small JSON aggregates
# that can't be computed on the Mac, then pull them down so the figures build locally:
#   - results/mover_cohort_flow.json      -> the MOVER side of the Fig 1b two-cohort curation funnel
#   - results/hypo_metrics_*mover*.json   -> MOVER operating points (a MOVER clinical supplement)
#   - results/clinical_eval_*mover*.json  -> MOVER lead-time / severity gradient
# (The MOVER subgroup forest is intentionally skipped: subgroup_forest.py is tied to VitalDB's
#  demographic schema — ASA/Department — which MOVER's SIS tables do not carry.)
#
# Usage (inside the project root, in the container / a python env with numpy):
#   bash scripts/mover_supp_analyses.sh
# Then from the Mac:
#   rsync -av <cluster>:~/tirex-2/tirex-vitaldb/results/ ./results/ \
#         --include='*mover*' --include='mover_cohort_flow.json' --exclude='*'
#   bash scripts/rebuild_local.sh    # Fig 1b now shows both cohorts
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="scripts:datasets/mover${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-$(command -v python3 || command -v python)}   # cluster login nodes usually have python3, not python
CFG=datasets/mover/configs/data.yaml
export CE_CONFIG="$CFG"                                   # cohort-agnostic clinical_eval -> MOVER loader
export HE_CLINICAL="datasets/mover/clinical_data.csv"     # cohort-agnostic hypo_eval demographics

echo "==> 1/2  MOVER cohort funnel (from the manifest — cheap, no multi-GB rescan)"
$PY - <<'PYEOF'
import csv, json, os
man = list(csv.DictReader(open("datasets/mover/cohort_manifest.csv")))
inc = sum(int(r["include"]) for r in man)
os.makedirs("results", exist_ok=True)
json.dump({"n_candidates": len(man), "included_N": inc},
          open("results/mover_cohort_flow.json", "w"), indent=2)
print(f"    wrote results/mover_cohort_flow.json  (candidates={len(man)}, included={inc})")
PYEOF

echo "==> 2/2  MOVER operating points + lead-time/severity (TiRex + trained + best foil)"
for tag in mover_art \
           baseline-tft_mover_art_covmover_rate \
           baseline-patchtst_mover_art_covmover_rate \
           baseline-chronos_mover_art; do
  [ -e "results/ablation_windows_${tag}.csv" ] || { echo "    [skip] $tag (no windows)"; continue; }
  $PY scripts/hypo_eval.py     "$tag"   >/dev/null 2>&1 || echo "    [warn] hypo_eval $tag failed"
  $PY scripts/clinical_eval.py "$tag"   >/dev/null 2>&1 || echo "    [warn] clinical_eval $tag failed"
  echo "    done $tag"
done
echo "==> MOVER supplement aggregates ready. Pull results/*mover* + results/mover_cohort_flow.json to the Mac."
