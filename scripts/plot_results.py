"""Plotting for the Phase 3 ablation — progress-tracking figures.

- plot_dashboard(summary): per-horizon [X%] (covariate effect) per variant w/ CIs, CRPS curves,
  MAE curves, skill vs persistence. Refreshed as the run progresses.
- plot_examples(examples): grid of MAP forecast panels, WITH vs WITHOUT covariate vs ground truth.

Standalone:  python scripts/plot_results.py results/ablation_primary_<tag>.json
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANT_COLORS = {"rate": "C0", "ce": "C1", "ce_both": "C2"}

# SOTA reference points (TRAINED foils; numbers from notes/RELATED_WORK.md — single source of truth).
# Context only: different cohorts (foils excluded thoracic/cardiac/vascular) + trained on 73k/320k,
# so these are references, NOT a head-to-head. Kapral = continuous-MAP MAE; horizon 7 min.
FOILS_MAE = {
    "Kapral'24 ext (VitalDB), 7min": 7.0,
    "Kapral'24 internal, 7min": 4.0,
}
# hypotension AUROC references (trained foils; numbers from notes/RELATED_WORK.md). (horizon_min, auroc, label)
FOILS_AUROC = [
    (5, 0.909, "Kapral'24 TFT 5min (int)"),
    (5, 0.903, "Kapral'24 TFT 5min (ext)"),
    (7, 0.867, "Kapral'24 TFT 7min (ext)"),
    (15, 0.882, "Zhu'26 15min (int)"),
    (10, 0.892, "Zhu'26 10min (int)"),
    (5, 0.904, "Zhu'26 5min (int)"),
]


def plot_hypotension(summary: dict, out_png: str):
    """Impending-hypotension (MAP<65) [Z]: AUROC + AUPRC vs horizon (M1 vs M0) with foil references."""
    ph = summary["per_horizon"]
    hs = sorted(int(k.replace("min", "")) for k in ph if ph[k].get("hypo"))
    if not hs:
        return
    def hg(h, key):
        return ph[f"{h}min"]["hypo"].get(key)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    # AUROC vs horizon with CI + foils
    a = ax[0]
    a.plot(hs, [hg(h, "auroc_M1") for h in hs], "-o", color="C1", label="TiRex M1 (with drug)")
    ci = [hg(h, "auroc_M1_CI95") for h in hs]
    a.fill_between(hs, [c[0] for c in ci], [c[1] for c in ci], color="C1", alpha=0.15)
    a.plot(hs, [hg(h, "auroc_M0") for h in hs], "-o", color="C3", label="TiRex M0 (no drug)", ms=4)
    for hh, val, lbl in FOILS_AUROC:
        a.scatter([hh], [val], marker="*", s=90, color="dimgrey", zorder=5)
        a.annotate(lbl, (hh, val), fontsize=5.5, color="dimgrey", ha="left", va="bottom")
    a.set_xlabel("horizon (min)"); a.set_ylabel("AUROC"); a.set_title("Hypotension (MAP<65) AUROC vs SOTA"); a.legend(fontsize=7)
    # AUPRC vs horizon (+ prevalence baseline)
    a = ax[1]
    a.plot(hs, [hg(h, "auprc_M1") for h in hs], "-o", color="C1", label="M1 AUPRC")
    a.plot(hs, [hg(h, "auprc_M0") for h in hs], "-o", color="C3", label="M0 AUPRC", ms=4)
    a.plot(hs, [hg(h, "prevalence") for h in hs], "--", color="grey", label="prevalence (chance)")
    a.set_xlabel("horizon (min)"); a.set_ylabel("AUPRC"); a.set_title("Hypotension AUPRC vs horizon"); a.legend(fontsize=7)
    # covariate effect on AUPRC (M1-M0) with CI
    a = ax[2]
    d = [hg(h, "delta_auprc_M1_minus_M0") for h in hs]
    ci = [hg(h, "delta_auprc_CI95") for h in hs]
    a.plot(hs, d, "-o", color="C0"); a.fill_between(hs, [c[0] for c in ci], [c[1] for c in ci], color="C0", alpha=0.15)
    a.axhline(0, ls="--", c="grey", lw=0.8)
    a.set_xlabel("horizon (min)"); a.set_ylabel("Δ AUPRC (M1 − M0)")
    a.set_title("Covariate effect on hypotension detection")
    ev = [hg(hs[0], "n_events"), hg(hs[0], "n")]
    fig.suptitle(f"Impending hypotension [Z] — {summary.get('cases_done','?')}/{summary.get('n_cases','?')} cases "
                 f"({ev[0]}/{ev[1]} events @ {hs[0]}min)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out_png, dpi=110); plt.close(fig)


def plot_dashboard(summary: dict, out_png: str):
    ph = summary["per_horizon"]
    hs = sorted(int(k.replace("min", "")) for k in ph)
    def g(h, stratum, key):
        return ph[f"{h}min"].get(stratum, {}).get(key)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

    # (0,0) X% covariate effect vs horizon: with-past arm vs target-only arm (all windows), 95% CI
    a = ax[0, 0]
    for key, col, lbl in [("X_pct_withpast", "C1", "with past covs (M1 vs M0)"),
                          ("X_pct_targetonly", "C0", "target-only (M1_to vs M0_to)")]:
        x = [g(h, "all", key) for h in hs]
        ci = [g(h, "all", key + "_CI95") for h in hs]
        if any(v is None for v in x):
            continue
        a.plot(hs, x, "-o", color=col, label=lbl)
        a.fill_between(hs, [c[0] for c in ci], [c[1] for c in ci], color=col, alpha=0.15)
    a.axhline(0, ls="--", c="grey", lw=0.8)
    a.set_xlabel("horizon (min)"); a.set_ylabel("[X%] CRPS reduction (with vs without drug)")
    a.set_title("Covariate effect [X%] — with-past vs target-only, 95% CI"); a.legend(fontsize=8)

    # (0,1) X% (with-past) stratified: transition vs steady windows — the key insight
    a = ax[0, 1]
    for stratum, col in [("transition", "C3"), ("steady", "C2"), ("all", "k")]:
        x = [g(h, stratum, "X_pct_withpast") for h in hs]
        ci = [g(h, stratum, "X_pct_withpast_CI95") for h in hs]
        if any(v is None for v in x):
            continue
        a.plot(hs, x, "-o", color=col, label=f"{stratum} (n_win {g(hs[-1], stratum, 'n_windows')})", ms=4)
        a.fill_between(hs, [c[0] for c in ci], [c[1] for c in ci], color=col, alpha=0.12)
    a.axhline(0, ls="--", c="grey", lw=0.8)
    a.set_xlabel("horizon (min)"); a.set_ylabel("[X%] CRPS reduction")
    a.set_title("Covariate effect by window type (drug changes in horizon?)"); a.legend(fontsize=8)

    # (1,0) CRPS vs horizon: all conditions (all windows)
    a = ax[1, 0]
    for key, col, lbl in [("crps_M1", "C1", "M1 (past+drug)"), ("crps_M0", "C3", "M0 (past)"),
                          ("crps_M1_to", "C0", "M1_to (drug only)"), ("crps_M0_to", "C4", "M0_to (target only)"),
                          ("crps_persistence", "grey", "persistence")]:
        y = [g(h, "all", key) for h in hs]
        a.plot(hs, y, "-o" if key != "crps_persistence" else "--s", color=col, label=lbl, ms=4)
    a.set_xlabel("horizon (min)"); a.set_ylabel("CRPS (mmHg)")
    a.set_title("Forecast error (CRPS) vs horizon — all conditions"); a.legend(fontsize=7)

    # (1,1) MAE vs horizon with SOTA reference lines
    a = ax[1, 1]
    y = [g(h, "all", "mae_M1") for h in hs]
    a.plot(hs, y, "-o", color="C1", label="TiRex M1 (median MAE)")
    a.plot(hs, [g(h, "all", "mae_M0") for h in hs], "-o", color="C3", label="TiRex M0", ms=4)
    for i, (lbl, val) in enumerate(FOILS_MAE.items()):
        a.axhline(val, ls=":", lw=1.2, color="dimgrey")
        a.text(hs[-1], val, f" {lbl}", fontsize=6.5, va="bottom" if i == 0 else "top", ha="right", color="dimgrey")
    a.set_xlabel("horizon (min)"); a.set_ylabel("median MAE (mmHg)")
    a.set_title("Median MAE vs horizon — vs SOTA (Kapral'24 trained)"); a.legend(fontsize=7, loc="upper left")

    nc = summary.get("n_cases"); nw = summary.get("n_windows"); done = summary.get("cases_done", nc)
    fig.suptitle(f"TiRex-2 MAP ablation @ {summary.get('dt_s','?')}s — {done}/{nc} cases, {nw} windows "
                 f"(tag {summary.get('tag','')})", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=110); plt.close(fig)


def plot_examples(ex: dict, out_png: str, dt: float = 5.0):
    """ex: dict of arrays saved by the ablation. Keys: caseid,t0,context,truth,q_ce,q_M0,cov (each stacked)."""
    n = len(ex["caseid"])
    k = min(n, 12)
    idx = np.linspace(0, n - 1, k).round().astype(int)
    ncol = 3; nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 2.8 * nrow), squeeze=False)
    for j, i in enumerate(idx):
        a = axes[j // ncol][j % ncol]
        ctx = ex["context"][i]; truth = ex["truth"][i]
        qce = ex["q_ce"][i]; q0 = ex["q_M0"][i]
        L = len(ctx); H = truth.shape[0]
        tc = np.arange(-L, 0) * dt / 60; th = np.arange(H) * dt / 60
        a.plot(tc, ctx, color="k", lw=0.7)
        a.plot(th, truth, color="green", lw=1.3, label="truth")
        a.plot(th, qce[4], color="C1", lw=1.2, label="with cov")
        a.fill_between(th, qce[0], qce[8], color="C1", alpha=0.15)
        a.plot(th, q0[4], color="C3", lw=1.1, ls="--", label="no cov")
        a.axvline(0, ls="--", c="grey", lw=0.6); a.axhline(65, ls=":", c="grey", lw=0.6)
        mae_c = np.nanmean(np.abs(qce[4] - truth)); mae_0 = np.nanmean(np.abs(q0[4] - truth))
        a.set_title(f"case {ex['caseid'][i]} @ {ex['t0'][i]*dt/60:.0f}min  "
                    f"MAE {mae_c:.1f} vs {mae_0:.1f}", fontsize=8)
        a.tick_params(labelsize=7)
        if j == 0:
            a.legend(fontsize=7, loc="best")
    for j in range(k, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Example MAP forecasts: with-covariate (orange) vs without (red) vs truth (green)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=110); plt.close(fig)


if __name__ == "__main__":
    import sys
    s = json.load(open(sys.argv[1]))
    out = sys.argv[1].replace("ablation_primary_", "dashboard_").replace(".json", ".png")
    plot_dashboard(s, out)
    print("wrote", out)
