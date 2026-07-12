"""Matched head-to-head: zero-shot TiRex-2 vs the trained supervised baseline on the
IDENTICAL held-out test subjects (canonical subject split), plus the published foils.

Both models are scored with the same metric code (hypo_eval.auroc etc.) on the same test
subjects, so the comparison is apples-to-apples — the objection the reviewers will raise
about the cross-paper foil numbers no longer applies to this row.

Run: PYTHONPATH=scripts:datasets/vitaldb python scripts/baselines/compare.py \
        --tirex all2873 --baseline baseline-tft_all2873
"""
from __future__ import annotations
import argparse, csv, json
import numpy as np
import hypo_eval as HE
from baselines.splits import subject_split

# Foils (external VitalDB), hypotension AUROC — from notes/RELATED_WORK.md
KAPRAL = {5: 0.903, 7: 0.867}
ZHU = {5: 0.904, 10: 0.892, 15: 0.882}


def load_rows(tag):
    import glob
    files = sorted(glob.glob(f"results/ablation_windows_{tag}_sh*of*.csv")) or [f"results/ablation_windows_{tag}.csv"]
    rows = []
    for f in files:
        rows.extend(list(csv.DictReader(open(f))))
    return rows


def auroc_ci(rows, h, c2s, test_subjects, risk_col="risk_M1", n_boot=1000):
    hr = [r for r in rows if int(r["h_min"]) == h and r.get(risk_col) not in ("", "nan", None)
          and c2s.get(str(r["caseid"]), str(r["caseid"])) in test_subjects]
    if not hr:
        return None
    y = np.array([float(r["hypo_event"]) for r in hr])
    s = np.array([float(r[risk_col]) for r in hr])
    cid = np.array([str(r["caseid"]) for r in hr])
    if y.sum() == 0 or y.sum() == len(y):
        return None
    au = HE.auroc(y, s)
    lo, hi = HE.clustered_boot_ci(cid, y, s, HE.auroc, n_boot=n_boot)
    return dict(auroc=round(au, 4), ci=[round(lo, 4), round(hi, 4)], n=len(y), n_events=int(y.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tirex", default="all2873")
    ap.add_argument("--baseline", default="baseline-tft_all2873")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    tirex = load_rows(args.tirex); base = load_rows(args.baseline)
    c2s = HE.caseid_to_subject()
    # canonical split over the FULL TiRex cohort (all splits present) -> identical test subjects
    cohort = sorted({str(r["caseid"]) for r in tirex})
    split = subject_split(cohort, c2s, seed=args.seed)
    test_subjects = {c2s.get(c, c) for c in cohort if split[c] == "test"}
    horizons = sorted({int(r["h_min"]) for r in base})   # baseline defines the horizons present
    print(f"matched test subjects: {len(test_subjects)}  horizons: {horizons}\n", flush=True)

    out = {"tirex_tag": args.tirex, "baseline_tag": args.baseline, "n_test_subjects": len(test_subjects),
           "per_horizon": {}}
    hdr = f"{'h(min)':>6} | {'TiRex M1':>20} | {'TFT M1 (trained)':>20} | {'TFT M0':>14} | {'Kapral':>7} | {'Zhu':>7}"
    print(hdr); print("-" * len(hdr))
    for h in horizons:
        t1 = auroc_ci(tirex, h, c2s, test_subjects, "risk_M1", args.n_boot)
        b1 = auroc_ci(base, h, c2s, test_subjects, "risk_M1", args.n_boot)
        b0 = auroc_ci(base, h, c2s, test_subjects, "risk_M0", args.n_boot)
        def fmt(d): return "—" if d is None else f"{d['auroc']:.3f} [{d['ci'][0]:.3f},{d['ci'][1]:.3f}]"
        def fmt0(d): return "—" if d is None else f"{d['auroc']:.3f}"
        kap = f"{KAPRAL[h]:.3f}" if h in KAPRAL else "—"
        zhu = f"{ZHU[h]:.3f}" if h in ZHU else "—"
        print(f"{h:>6} | {fmt(t1):>20} | {fmt(b1):>20} | {fmt0(b0):>14} | {kap:>7} | {zhu:>7}")
        out["per_horizon"][h] = {"tirex_M1": t1, "tft_M1": b1, "tft_M0": b0,
                                 "kapral_ext": KAPRAL.get(h), "zhu_ext": ZHU.get(h)}
    path = f"results/matched_comparison_{args.baseline}.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
