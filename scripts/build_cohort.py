"""Phase 3.1 — build the cohort manifest + run the infusion-fidelity gate.

Parallel scan over LOCAL .vital files (iCloud-evicted skipped). For each file: header-check required
tracks; if present, load (5 s grid, masked) and compute target/covariate coverage, duration, and
infusion activity. Applies inclusion criteria and writes:
  - results/cohort_manifest.csv   (one row per scanned candidate + include flag + exclude reason)
  - results/cohort_flow.json      (case-flow counts at each exclusion step -> [N])
  - results/infusion_fidelity.json (the §3.1 gate summary)

Incremental CSV write so partial results survive an early stop. Resumable via --skip-scanned.

  python scripts/build_cohort.py [--limit N] [--workers 8]
"""
from __future__ import annotations
import argparse, csv, glob, json, os
from multiprocessing import Pool
import numpy as np

import vitaldb_loader as L

# ---- inclusion thresholds (record in DATA_NOTES) ----
TARGET = "Solar8000/ART_MBP"
REMI_CE = "Orchestra/RFTN20_CE"
REMI_RATE = "Orchestra/RFTN20_RATE"
PPF_CE = "Orchestra/PPF20_CE"
MIN_DUR_MIN = 60.0          # room for rolling origins (30 ctx + 10 horizon + strides)
MIN_TARGET_COV = 0.6        # fraction finite MAP in window
MIN_REMI_COV = 0.5          # fraction finite remifentanil CE
REMI_ACTIVE_MIN_CE = 0.5    # ng/mL: max CE must exceed -> an actual infusion happened

_CFG = None
_CLIN = None


def _init(cfg, clin):
    global _CFG, _CLIN
    _CFG, _CLIN = cfg, clin


def _scan_one(path: str):
    caseid = os.path.splitext(os.path.basename(path))[0]
    row = {"caseid": caseid, "include": 0, "exclude_reason": ""}
    if not L._is_local(path):
        row["exclude_reason"] = "evicted"; return row
    # cheap header check first
    try:
        import vitaldb
        present = set(vitaldb.vital_trks(path))
    except Exception:
        row["exclude_reason"] = "unreadable_header"; return row
    row["has_ART_MBP"] = int(TARGET in present)
    row["has_RFTN_CE"] = int(REMI_CE in present)
    row["has_RFTN_RATE"] = int(REMI_RATE in present)
    row["has_PPF_CE"] = int(PPF_CE in present)
    if TARGET not in present:
        row["exclude_reason"] = "no_ART_MBP"; return row
    if REMI_CE not in present and REMI_RATE not in present:
        row["exclude_reason"] = "no_remifentanil"; return row
    # load (masked, 5s grid, anesthesia window)
    rec = L.load_case(caseid, _CFG, _CLIN, use_cache=True)
    if rec is None:
        row["exclude_reason"] = "load_failed"; return row
    dt = rec["interval_s"]; n = rec["n"]
    dur = n * dt / 60.0
    tgt_cov = rec["target_coverage"]
    ce = rec["future_cov"].get(REMI_CE)
    ce_cov = float(np.isfinite(ce).mean()) if ce is not None else 0.0
    ce_max = float(np.nanmax(ce)) if ce is not None and np.isfinite(ce).any() else 0.0
    ce_range = (float(np.nanmax(ce) - np.nanmin(ce))
                if ce is not None and np.isfinite(ce).any() else 0.0)
    ppf = rec["future_cov"].get(PPF_CE)
    ppf_max = float(np.nanmax(ppf)) if ppf is not None and np.isfinite(ppf).any() else 0.0
    m = rec["meta"]
    row.update({
        "n": n, "dur_min": round(dur, 1), "target_coverage": round(tgt_cov, 3),
        "remi_ce_coverage": round(ce_cov, 3), "remi_ce_max": round(ce_max, 2),
        "remi_ce_range": round(ce_range, 2), "ppf_ce_max": round(ppf_max, 2),
        "age": m.get("age"), "sex": m.get("sex"), "department": m.get("department"),
        "optype": m.get("optype"), "ane_type": m.get("ane_type"), "asa": m.get("asa"),
    })
    # inclusion cascade (record first failing reason)
    if dur < MIN_DUR_MIN:
        row["exclude_reason"] = "short_duration"; return row
    if tgt_cov < MIN_TARGET_COV:
        row["exclude_reason"] = "low_target_coverage"; return row
    if ce_cov < MIN_REMI_COV:
        row["exclude_reason"] = "low_remi_coverage"; return row
    if ce_max < REMI_ACTIVE_MIN_CE:
        row["exclude_reason"] = "remi_inactive"; return row
    row["include"] = 1
    return row


COLUMNS = ["caseid", "include", "exclude_reason", "has_ART_MBP", "has_RFTN_CE", "has_RFTN_RATE",
           "has_PPF_CE", "n", "dur_min", "target_coverage", "remi_ce_coverage", "remi_ce_max",
           "remi_ce_range", "ppf_ce_max", "age", "sex", "department", "optype", "ane_type", "asa"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = L.load_config(args.config)
    clin = L._clinical_index(cfg["clinical_csv"])
    files = sorted(glob.glob(os.path.join(cfg["vital_dir"], "*.vital")))
    local = [f for f in files if L._is_local(f)]
    if args.limit:
        local = local[:args.limit]
    print(f"{len(files)} files, {len(local)} local, scanning with {args.workers} workers", flush=True)

    os.makedirs("results", exist_ok=True)
    out_csv = "results/cohort_manifest.csv"
    rows = []
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        with Pool(args.workers, initializer=_init, initargs=(cfg, clin)) as pool:
            for i, row in enumerate(pool.imap_unordered(_scan_one, local, chunksize=8)):
                rows.append(row)
                w.writerow(row); fh.flush()
                if (i + 1) % 200 == 0:
                    inc = sum(r["include"] for r in rows)
                    print(f"  {i+1}/{len(local)} scanned, {inc} included so far", flush=True)

    # case-flow (order matters -> first exclusion reason wins in _scan_one)
    from collections import Counter
    reasons = Counter(r["exclude_reason"] for r in rows if not r["include"])
    N = sum(r["include"] for r in rows)
    flow = {
        "n_local_scanned": len(rows),
        "excluded": dict(reasons),
        "included_N": N,
        "thresholds": {"MIN_DUR_MIN": MIN_DUR_MIN, "MIN_TARGET_COV": MIN_TARGET_COV,
                       "MIN_REMI_COV": MIN_REMI_COV, "REMI_ACTIVE_MIN_CE": REMI_ACTIVE_MIN_CE,
                       "target": TARGET, "covariate": REMI_CE},
    }
    json.dump(flow, open("results/cohort_flow.json", "w"), indent=2)

    # infusion-fidelity gate (over INCLUDED cases)
    inc_rows = [r for r in rows if r["include"]]
    def arr(k): return np.array([r[k] for r in inc_rows if r.get(k) is not None], float)
    fidelity = {
        "n_included": N,
        "remi_ce_coverage": _summ(arr("remi_ce_coverage")),
        "remi_ce_max_ngml": _summ(arr("remi_ce_max")),
        "remi_ce_range_ngml": _summ(arr("remi_ce_range")),
        "target_coverage": _summ(arr("target_coverage")),
        "dur_min": _summ(arr("dur_min")),
        "also_have_RFTN_RATE_pct": round(100 * np.mean(arr("has_RFTN_RATE")), 1) if N else None,
        "also_have_PPF_CE_pct": round(100 * np.mean(arr("has_PPF_CE")), 1) if N else None,
        "gate_pass": bool(N > 0 and np.median(arr("remi_ce_coverage")) >= MIN_REMI_COV),
    }
    json.dump(fidelity, open("results/infusion_fidelity.json", "w"), indent=2)

    print("\n=== CASE FLOW ===")
    print(json.dumps(flow, indent=2))
    print("\n=== INFUSION-FIDELITY GATE ===")
    print(json.dumps(fidelity, indent=2))
    print(f"\n>>> [N] candidate cohort = {N}")


def _summ(a):
    if len(a) == 0:
        return None
    return {"median": round(float(np.median(a)), 2), "iqr": [round(float(np.percentile(a, 25)), 2),
            round(float(np.percentile(a, 75)), 2)], "min": round(float(a.min()), 2),
            "max": round(float(a.max()), 2), "n": int(len(a))}


if __name__ == "__main__":
    main()
