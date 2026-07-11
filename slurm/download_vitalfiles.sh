#!/bin/bash
# Re-download the VitalDB raw .vital files we need, into vital_files/ (zero-padded 0000.vital,
# matching the loader's str(caseid).zfill(4)). Downloads the union of:
#   * every caseid in results/cohort_manifest.csv   (reproduces the anesthetic cohort scan)
#   * results/pressor_cases_phen.txt                 (phenylephrine subset)
# Skips files already present; parallel; retries. Run on a login node or the `storage` partition
# (external access). ~5.4k cases; sizeable — ensure disk space.
#
#   bash slurm/download_vitalfiles.sh                 # default: manifest ∪ pressor list
#   PARALLEL=16 bash slurm/download_vitalfiles.sh     # more concurrent downloads
#
# NOTE: verify the VitalDB endpoint below once (open-dataset raw vitals). If it changed, adjust URL.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DEST="${PROJECT_ROOT}/vital_files"
URL_BASE="${URL_BASE:-https://api.vitaldb.net}"
PARALLEL="${PARALLEL:-8}"
mkdir -p "${DEST}"

# clinical_data.csv (loader dependency) — fetched from VitalDB, not committed. VERIFY endpoint once.
if [ ! -s "${PROJECT_ROOT}/clinical_data.csv" ]; then
  echo "[dl] fetching clinical_data.csv from ${URL_BASE}/cases ..."
  curl -fsSL --retry 4 -o "${PROJECT_ROOT}/clinical_data.csv" "${URL_BASE}/cases" \
    || echo "[dl] WARNING: clinical_data.csv fetch failed — provide it manually (loader needs it)."
fi

# collect bare caseids (manifest col1 minus header, drop leading zeros) ∪ pressor list
{ tail -n +2 "${PROJECT_ROOT}/results/cohort_manifest.csv" | cut -d, -f1
  cat "${PROJECT_ROOT}/results/pressor_cases_phen.txt" 2>/dev/null || true
} | sed 's/^0*//; s/^$//' | grep -E '^[0-9]+$' | sort -un > /tmp/tirex_caseids.txt
N=$(wc -l < /tmp/tirex_caseids.txt)
echo "[dl] ${N} unique caseids -> ${DEST}  (parallel=${PARALLEL}, base=${URL_BASE})"

fetch() {  # $1 = bare caseid
  local id="$1"; local pad; pad=$(printf '%04d' "$id"); local out="${DEST}/${pad}.vital"
  [ -s "${out}" ] && return 0
  curl -fsSL --retry 4 --retry-delay 2 -o "${out}.part" "${URL_BASE}/${id}.vital" \
    && mv "${out}.part" "${out}" || { rm -f "${out}.part"; echo "[dl] FAILED ${id}" >&2; }
}
export -f fetch; export DEST URL_BASE

xargs -a /tmp/tirex_caseids.txt -P "${PARALLEL}" -I{} bash -c 'fetch "$@"' _ {}

GOT=$(find "${DEST}" -name '*.vital' -size +0c | wc -l)
echo "[dl] done: ${GOT} .vital files present in ${DEST}"
