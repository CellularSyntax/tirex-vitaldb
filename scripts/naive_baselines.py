"""Naive reference baselines on IDENTICAL windows: persistence and MAP-trend/logistic.

Reviewer ask (unanimous): because MAP is strongly autocorrelated over 1-7 min, the high
short-horizon AUROC may partly reflect autocorrelation, not model skill. This script adds a
naive floor on the exact same windows/metrics/cohorts as the foundation models and trained
baselines, so the reader can see the skill above autocorrelation.

It does NOT re-run any forecaster. It reads the already-written per-window rows
(results/ablation_windows_<tag>.csv) for the ground-truth (caseid, t0, h_min, hypo_event) and
reconstructs, at each window origin t0, two label-free risk scores from the case cache context:

  (1) persistence  : risk = sigmoid over the last observed MAP relative to the 65 mmHg
                     threshold (lower last-MAP -> higher risk). A monotone transform of
                     last-MAP, so its AUROC equals that of the pure "last observed MAP <65"
                     ranker -- the exact naive alarm the paper levels at HPI.
  (2) map_trend    : a 2-feature logistic (last MAP, recent slope over the last 5 min) fit on a
                     disjoint 20% subject-level dev split (same split rule as hypo_eval.py) and
                     scored out-of-sample on the test split, per horizon.

Both scored with the project's own AUROC/AUPRC and a case-clustered bootstrap CI, so the
numbers are directly comparable to the existing tables. Persistence needs no fitting and is
reported on the full window set as well as the test split; the logistic is reported on the
test split only.

Run on the CLUSTER (holds the full per-case cache), from the project root:
    # VitalDB
    PYTHONPATH=scripts:datasets/vitaldb python scripts/naive_baselines.py all2873 \
        --cache datasets/vitaldb/cache --clinical datasets/vitaldb/data/clinical_data.csv
    # MOVER
    PYTHONPATH=scripts:datasets/mover python scripts/naive_baselines.py mover_art \
        --cache datasets/mover/cache --clinical datasets/mover/clinical_data.csv

Writes results/naive_metrics_<tag>.json (pull to the Mac; drops into the reviewer supplement).
Optionally also writes results/ablation_windows_naive-persist_<tag>.csv so downstream table code
can pick it up as another model row.
"""
from __future__ import annotations
import argparse, csv, glob, json, os, sys
import numpy as np

DT_DEFAULT = 15.0                 # window t0 index step (s); overridden per-cohort below
SLOPE_WIN_MIN = 5.0
HYPO_THR = 65.0
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ---------- metrics (match scripts/hypo_eval.py exactly) ----------
def _pairs(y, s):
    m = np.isfinite(s); return np.asarray(y, float)[m], np.asarray(s, float)[m]


def auroc(y, s):
    y, s = _pairs(y, s); P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0:
        return np.nan
    order = np.argsort(-s, kind="mergesort"); ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    return float(_trapz(np.r_[0, tp / P], np.r_[0, fp / N]))


def auprc(y, s):
    y, s = _pairs(y, s); P = y.sum()
    if P == 0:
        return np.nan
    order = np.argsort(-s, kind="mergesort"); ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / (tp + fp); rec = tp / P
    return float(_trapz(np.r_[prec[0], prec], np.r_[0, rec]))


def clustered_boot_ci(cid, y, s, fn, n_boot=2000, seed=0):
    cid = np.asarray(cid); uc = np.unique(cid); by = {c: np.where(cid == c)[0] for c in uc}
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        pick = rng.choice(uc, len(uc), replace=True)
        idx = np.concatenate([by[c] for c in pick]); v = fn(y[idx], s[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return [np.nan, np.nan]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


# ---------- data ----------
def load_rows(tag):
    files = sorted(glob.glob(f"results/ablation_windows_{tag}_sh*of*.csv")) or \
            sorted(glob.glob(f"results/ablation_windows_{tag}.csv"))
    if not files:
        sys.exit(f"no windows CSVs for tag={tag}")
    rows = []
    for f in files:
        rows += list(csv.DictReader(open(f)))
    return rows


def norm_caseid(c):
    """VitalDB ids are integers (strip zero padding); MOVER ids are hex hashes (verbatim)."""
    s = str(c).strip()
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def caseid_to_subject(clinical):
    m = {}
    for r in csv.DictReader(open(clinical, encoding="utf-8-sig")):
        try:
            m[norm_caseid(r["caseid"])] = str(r.get("subjectid", r["caseid"]))
        except KeyError:
            continue
    return m


def dev_subjects(caseids, c2s, dev_frac=0.2, seed=0):
    subs = sorted({c2s.get(str(c), str(c)) for c in caseids})
    rng = np.random.default_rng(seed); rng.shuffle(subs)
    ndev = max(1, int(round(len(subs) * dev_frac)))
    return set(subs[:ndev])


def load_case_target(cache_dir, cid):
    for name in (f"{int(cid):04d}.npz", f"{int(cid)}.npz", f"{cid}.npz"):
        p = os.path.join(cache_dir, name)
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            return z["time_min"].astype(float), z["target"].astype(float)
    return None


def naive_feats(cache_dir, cid, t0, dt):
    """Return (last_MAP, slope_mmHg_per_min) at window origin t0 (index units of dt seconds)."""
    r = load_case_target(cache_dir, cid)
    if r is None:
        return np.nan, np.nan
    tm, mp = r
    origin_min = t0 * dt / 60.0
    pre = tm <= origin_min + 1e-6
    fin = np.isfinite(mp)
    idx = np.where(pre & fin)[0]
    if len(idx) == 0:
        return np.nan, np.nan
    last = mp[idx[-1]]
    w = idx[tm[idx] >= tm[idx[-1]] - SLOPE_WIN_MIN]
    slope = float(np.polyfit(tm[w], mp[w], 1)[0]) if len(w) >= 2 else 0.0
    return float(last), slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--dt", type=float, default=None, help="window index step in s (default: infer 15 vitaldb / 60 mover)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--write-windows", action="store_true",
                    help="also write results/ablation_windows_naive-persist_<tag>.csv")
    args = ap.parse_args()

    dt = args.dt if args.dt else (60.0 if "mover" in args.tag else 15.0)
    rows = load_rows(args.tag)
    c2s = caseid_to_subject(args.clinical)
    horizons = sorted({int(r["h_min"]) for r in rows})
    allcases = [r["caseid"] for r in rows]
    dev = dev_subjects(allcases, c2s, seed=0)

    # cache last-MAP + slope once per (caseid, t0) — horizon-independent
    origins = sorted({(r["caseid"], int(r["t0"])) for r in rows})
    feat = {}
    miss = 0
    for cid, t0 in origins:
        last, slope = naive_feats(args.cache, cid, t0, dt)
        feat[(cid, t0)] = (last, slope)
        if not np.isfinite(last):
            miss += 1
    print(f"tag={args.tag} dt={dt}s origins={len(origins)} cache-miss={miss} "
          f"({100*miss/max(len(origins),1):.1f}%) horizons={horizons} dev_subj={len(dev)}", flush=True)
    if miss > 0.05 * len(origins):
        print("  [warn] >5% of origins missing from cache — coverage incomplete on this machine.", flush=True)

    persist_rows = []
    out = {"tag": args.tag, "dt_s": dt, "n_origins": len(origins), "cache_miss": miss,
           "horizons_min": horizons, "per_horizon": {}}

    for h in horizons:
        hr = [r for r in rows if int(r["h_min"]) == h]
        y = np.array([float(r["hypo_event"]) for r in hr])
        cid = np.array([r["caseid"] for r in hr])
        last = np.array([feat[(r["caseid"], int(r["t0"]))][0] for r in hr])
        slope = np.array([feat[(r["caseid"], int(r["t0"]))][1] for r in hr])
        is_dev = np.array([c2s.get(str(c), str(c)) in dev for c in cid])
        ok = np.isfinite(last)

        # (1) persistence: lower last-MAP -> higher risk. Monotone => AUROC of the last-MAP ranker.
        risk_persist = -last                       # rank-equivalent to P(MAP<65) from last value
        yt, rt, ct = y[~is_dev & ok], risk_persist[~is_dev & ok], cid[~is_dev & ok]
        rec = {"n_test": int((~is_dev & ok).sum()), "prevalence": round(float(y[ok].mean()), 4)}
        rec["persistence"] = {
            "auroc": round(auroc(yt, rt), 4), "auroc_CI95": clustered_boot_ci(ct, yt, rt, auroc, args.n_boot),
            "auprc": round(auprc(yt, rt), 4), "auprc_CI95": clustered_boot_ci(ct, yt, rt, auprc, args.n_boot),
            "auroc_fullset": round(auroc(y[ok], risk_persist[ok]), 4),
        }

        # (2) logistic on (last MAP, slope), fit on dev, scored on test
        dtr = is_dev & ok
        try:
            from sklearn.linear_model import LogisticRegression
            Xd = np.column_stack([last[dtr], slope[dtr]]); yd = y[dtr]
            if len(np.unique(yd)) == 2 and dtr.sum() > 20:
                clf = LogisticRegression(max_iter=1000).fit(Xd, yd)
                Xt = np.column_stack([last[~is_dev & ok], slope[~is_dev & ok]])
                st = clf.predict_proba(Xt)[:, 1]
                rec["map_trend_logit"] = {
                    "auroc": round(auroc(yt, st), 4), "auroc_CI95": clustered_boot_ci(ct, yt, st, auroc, args.n_boot),
                    "auprc": round(auprc(yt, st), 4), "auprc_CI95": clustered_boot_ci(ct, yt, st, auprc, args.n_boot),
                    "coef_lastMAP": round(float(clf.coef_[0][0]), 4), "coef_slope": round(float(clf.coef_[0][1]), 4),
                }
        except ImportError:
            rec["map_trend_logit"] = {"error": "sklearn not available"}

        out["per_horizon"][f"{h}min"] = rec

        if args.write_windows:
            for r in hr:
                last_i, _ = feat[(r["caseid"], int(r["t0"]))]
                persist_rows.append({"caseid": r["caseid"], "t0": r["t0"], "h_min": h,
                                     "stratum": r.get("stratum", ""), "hypo_event": r["hypo_event"],
                                     "risk_M1": (-last_i if np.isfinite(last_i) else ""),
                                     "risk_M0": (-last_i if np.isfinite(last_i) else "")})

    os.makedirs("results", exist_ok=True)
    json.dump(out, open(f"results/naive_metrics_{args.tag}.json", "w"), indent=2)
    print(f"wrote results/naive_metrics_{args.tag}.json")
    if args.write_windows and persist_rows:
        fn = f"results/ablation_windows_naive-persist_{args.tag}.csv"
        with open(fn, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(persist_rows[0].keys())); w.writeheader(); w.writerows(persist_rows)
        print(f"wrote {fn}")

    # concise console summary
    for h in horizons:
        r = out["per_horizon"][f"{h}min"]
        pl = r["persistence"]; lg = r.get("map_trend_logit", {})
        print(f"  h={h:2d}  prev={r['prevalence']:.3f}  persist AUROC={pl['auroc']:.3f} {pl['auroc_CI95']}"
              f"  logit AUROC={lg.get('auroc')}")


if __name__ == "__main__":
    main()
