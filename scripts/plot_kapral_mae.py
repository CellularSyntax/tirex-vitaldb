"""Headline continuous-forecast figure: our zero-shot MAE vs Kapral et al. 2024 (digitized Fig 2),
across forecast horizons up to 7 min (Kapral's anchor; no external reference beyond 7).

Convention note (critical): OUR mae_M1 per horizon h is the MEAN abs error over steps 0..h
(cumulative-to-h; see phase3 `pinball`). Kapral's Fig 2 is INSTANTANEOUS error at each distance. To
compare apples-to-apples we convert Kapral's curve to the same cumulative-to-h average. (Instantaneous
per-horizon MAE is a next-run enhancement.)

Two panels: (left) continuous overlay of ours vs Kapral internal/external with bands;
(right) grouped points+error bars at 1/3/5/7 min. Reads results/kapral_mae_curves.csv + our windows CSVs.

Run:  PYTHONPATH=scripts <venv>/bin/python scripts/plot_kapral_mae.py n300_s1
Writes outputs/figs/mae_vs_kapral_<tag>.png
"""
import csv, glob, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OVERLAY_H = [1, 3, 5, 7]     # Kapral only forecasts to 7 min


def load_kapral():
    D = defaultdict(list)
    for r in csv.DictReader(open("results/kapral_mae_curves.csv")):
        D[(r["dataset"], r["curve"])].append((float(r["forecast_min"]), float(r["mae_mmHg"])))
    curves = {}
    for k, pts in D.items():
        pts = sorted(pts); curves[k] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    return curves


def cum(curves, ds, curve, t):
    """Kapral cumulative-to-t: mean of the instantaneous curve over [0, t]."""
    xs, ys = curves[(ds, curve)]
    g = np.linspace(0, t, max(2, int(t * 12)))
    return float(np.interp(g, xs, ys).mean())


def our_mae(tag):
    files = sorted(glob.glob(f"results/ablation_windows_{tag}_sh*of*.csv")) or \
            sorted(glob.glob(f"results/ablation_windows_{tag}.csv"))
    per_h = defaultdict(list)   # h_min -> list of (caseid, mae_M1)
    for f in files:
        for r in csv.DictReader(open(f)):
            v = r.get("mae_M1", "")
            if v not in ("", "nan"):
                per_h[int(r["h_min"])].append((r["caseid"], float(v)))
    out = {}
    for h, pairs in per_h.items():
        cids = np.array([c for c, _ in pairs]); vals = np.array([v for _, v in pairs])
        uc = np.unique(cids); by = {c: np.where(cids == c)[0] for c in uc}
        rng = np.random.default_rng(0); boot = []
        for _ in range(1000):
            pick = rng.choice(uc, len(uc), replace=True)
            boot.append(vals[np.concatenate([by[c] for c in pick])].mean())
        out[h] = (float(vals.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return out


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "n300_s1"
    K = load_kapral(); ours = our_mae(tag)
    tt = np.linspace(0.25, 7, 60)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- left: continuous overlay (cumulative convention) ----
    a = ax[0]
    for ds, c, lbl in (("internal", "C0", "Kapral internal (VitalDB-trained holdout)"),
                       ("external", "C3", "Kapral external (VitalDB)")):
        mean = [cum(K, ds, "mean", t) for t in tt]
        lo = [cum(K, ds, "lower", t) for t in tt]; hi = [cum(K, ds, "upper", t) for t in tt]
        a.plot(tt, mean, color=c, lw=1.8, label=lbl)
        a.fill_between(tt, lo, hi, color=c, alpha=0.12)
    hs = [h for h in OVERLAY_H if h in ours]
    m = [ours[h][0] for h in hs]; lo = [ours[h][1] for h in hs]; hi = [ours[h][2] for h in hs]
    a.errorbar(hs, m, yerr=[np.array(m) - np.array(lo), np.array(hi) - np.array(m)],
               fmt="-o", color="C1", lw=2, capsize=3, ms=7, label="Ours: TiRex-2 zero-shot M1", zorder=6)
    a.set_xlabel("forecast horizon (min)"); a.set_ylabel("MAE (mmHg), averaged over 0→horizon")
    a.set_xlim(0, 7.3); a.set_ylim(0, None); a.set_title("Cumulative MAE vs forecast horizon")
    a.legend(fontsize=8, loc="upper left")

    # ---- right: grouped points + error bars at 1/3/5/7 ----
    a = ax[1]
    groups = [("Kapral internal", "C0", lambda h: (cum(K, "internal", "mean", h),
              cum(K, "internal", "lower", h), cum(K, "internal", "upper", h))),
              ("Kapral external", "C3", lambda h: (cum(K, "external", "mean", h),
              cum(K, "external", "lower", h), cum(K, "external", "upper", h))),
              ("Ours (M1)", "C1", lambda h: ours.get(h))]
    x = np.arange(len(OVERLAY_H)); w = 0.24
    for i, (lbl, c, fn) in enumerate(groups):
        xs = x + (i - 1) * w
        mm = [fn(h)[0] for h in OVERLAY_H]; ll = [fn(h)[1] for h in OVERLAY_H]; uu = [fn(h)[2] for h in OVERLAY_H]
        a.errorbar(xs, mm, yerr=[np.array(mm) - np.array(ll), np.array(uu) - np.array(mm)],
                   fmt="o", color=c, capsize=3, ms=7, label=lbl)
    a.set_xticks(x); a.set_xticklabels([f"{h} min" for h in OVERLAY_H])
    a.set_ylabel("MAE (mmHg), 0→horizon"); a.set_ylim(0, None)
    a.set_title("Per-horizon comparison"); a.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"Zero-shot MAE vs Kapral'24 — cumulative-to-horizon convention — {tag}",
                 fontweight="bold")
    fig.text(0.5, 0.005, "Kapral Fig 2 digitized; converted to cumulative-to-horizon to match our metric. "
             "Kapral forecasts ≤7 min. Bands = digitized silhouette / case-clustered 95% CI (ours).",
             ha="center", fontsize=7, color="dimgrey")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"outputs/figs/mae_vs_kapral_{tag}.png", dpi=120); plt.close(fig)

    print(f"{'h':>3} {'ours M1 [CI]':>22} {'Kapral int (cum)':>17} {'Kapral ext (cum)':>17}")
    for h in OVERLAY_H:
        o = ours.get(h)
        print(f"{h:>3} {o[0]:6.2f} [{o[1]:.2f}, {o[2]:.2f}]      "
              f"{cum(K,'internal','mean',h):15.2f}  {cum(K,'external','mean',h):15.2f}")
    print(f"\nwrote outputs/figs/mae_vs_kapral_{tag}.png")


if __name__ == "__main__":
    main()
