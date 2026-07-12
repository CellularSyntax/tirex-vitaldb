"""Publication figures + tables for the TiRex-2 / VitalDB paper (Nature Medicine style).

Classification-first framing:
  Fig 1  Study design & cohort
  Fig 2  Forecast accuracy & the value of the known future drug covariate
  Fig 3  Impending-hypotension prediction: zero-shot TiRex-2 vs supervised SOTA  (headline)
  Fig 4  Clinical translation & robustness
  Tables 1-3  cohort characteristics | accuracy | classification-vs-foils

Reads the finished full-cohort outputs already in results/ (tag=all2873 primary; the
covrate/pressor arms for the covariate-representation panel). Styling lives in paper_style.

Run:  PYTHONPATH=scripts:datasets/vitaldb python scripts/paper_figures.py [tag]
"""
from __future__ import annotations
import json, csv, os, sys
import numpy as np
import matplotlib.pyplot as plt
import paper_style as S
import hypo_eval as H   # roc_points, pr_points, calibration, auroc, load_rows, split_subjects, caseid_to_subject

TAG = sys.argv[1] if len(sys.argv) > 1 else "all2873"
RATE_TAG, PRESSOR_TAG = "all2873_covrate", "cases115_covpressor"
DT_S = 15.0
MAIN_H = [1, 3, 5, 7]            # horizons in main figures (<=7 min); supplement adds 10, 15


# ── data loaders ──────────────────────────────────────────────────────────────
def load_primary(tag):
    return json.load(open(f"results/ablation_primary_{tag}.json"))

def load_hypo(tag):
    return json.load(open(f"results/hypo_metrics_{tag}.json"))

def load_clinical(tag):
    return json.load(open(f"results/clinical_eval_{tag}.json"))

def load_subgroup(tag, h=5):
    return json.load(open(f"results/subgroup_forest_{tag}_h{h}.json"))

def strat(primary, h, s):
    return primary["per_horizon"][S.hkey(h)][s]

def _test_scores(rows, c2s, dev, h, risk_col="risk_M1", ev_col="hypo_event"):
    """(y, s) on the held-out test subjects for horizon h — matches hypo_eval's reported AUROC."""
    y, s = [], []
    for r in rows:
        if int(r["h_min"]) != h or r[risk_col] in ("", "nan"):
            continue
        if c2s.get(str(r["caseid"]), str(r["caseid"])) in dev:
            continue  # keep test only
        y.append(float(r[ev_col])); s.append(float(r[risk_col]))
    return np.array(y), np.array(s)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — study design & cohort
# ══════════════════════════════════════════════════════════════════════════════
def figure1(tag):
    prim = load_primary(tag); hyp = load_hypo(tag)
    flow = json.load(open("results/cohort_flow.json"))
    ex = np.load(f"outputs/figs/examples_{tag}.npz", allow_pickle=True)

    fig = plt.figure(figsize=(S.W2, S.W2 * 0.78))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=1.05, wspace=0.42)
    ax_sch = fig.add_subplot(gs[0, :2])     # a schematic (wide)
    ax_flow = fig.add_subplot(gs[0, 2])     # b cohort flow
    ax_ex = [fig.add_subplot(gs[1, i]) for i in range(3)]  # c examples

    # a — task schematic (illustrative)
    _schematic(ax_sch)
    S.panel_letter(ax_sch, "a", dx=-0.06)

    # b — cohort flow (funnel) + window/prevalence annotation
    _cohort_flow(ax_flow, flow, prim, hyp)
    S.panel_letter(ax_flow, "b", dx=-0.10)

    # c — three representative forecasts (steady / transition / hypotensive onset)
    picks = _pick_examples(ex)
    titles = ["Steady", "Transition", "Hypotensive onset"]
    for ax, idx, tt in zip(ax_ex, picks, titles):
        _example_panel(ax, ex, idx, tt)
    S.panel_letter(ax_ex[0], "c")
    ax_ex[0].legend(loc="lower left", fontsize=5.5, ncol=1)

    fig.suptitle("Zero-shot intraoperative MAP forecasting with TiRex-2 — study design",
                 y=0.99, fontsize=9, fontweight="bold")
    S.save_fig(fig, "Fig1_design_cohort")


def _schematic(ax):
    t = np.linspace(-30, 15, 300)
    rng = np.random.default_rng(3)
    base = 78 + 6*np.sin(t/6) - 0.25*t
    map_ = base + rng.normal(0, 1.2, t.size)
    ctx = t <= 0
    ax.plot(t[ctx], map_[ctx], color=S.C["ink"], lw=1.3)
    # median forecast + band on horizon
    fut = t > 0
    med = base[fut]
    ax.plot(t[fut], med, color=S.C["M1"], lw=1.6, label="TiRex-2 median")
    ax.fill_between(t[fut], med-6, med+6, color=S.C["M1_light"], alpha=0.5, lw=0, label="10–90% interval")
    ax.axvspan(-30, 0, color="#F2F2F2", zorder=0)
    ax.axvline(0, color="#888", lw=0.8, ls=":")
    ax.axhline(65, color=S.C["event"], lw=0.8, ls="--")
    ax.text(-29, 96, "context (30 min)", fontsize=6, color="#555")
    ax.text(1, 96, "forecast horizon (→15 min)", fontsize=6, color="#555")
    ax.text(13.5, 66.5, "MAP 65", fontsize=5.5, color=S.C["event"], ha="right")
    # covariate strip
    inf = 0.5*(1+np.tanh((t+2)/4))
    ax2 = ax.inset_axes([0, -0.40, 1, 0.24])
    ax2.plot(t, inf, color=S.C["M0"], lw=1.2)
    ax2.fill_between(t[fut], inf[fut], 0, color=S.C["M0_light"], alpha=0.6, lw=0)
    ax2.axvspan(-30, 0, color="#F2F2F2", zorder=0); ax2.axvline(0, color="#888", lw=0.8, ls=":")
    ax2.set_yticks([]); ax2.set_xlabel("time (min)")
    ax2.set_ylabel("drug\ninfusion", fontsize=6)
    ax2.text(1, 0.82, "known future covariate (M1)", fontsize=5.5, color=S.C["M0"])
    ax2.spines["left"].set_visible(False)
    ax.set_xlim(-30, 15); ax.set_ylim(55, 100); ax.set_xticks([])
    ax.set_ylabel("MAP (mmHg)"); ax.set_title("Forecasting task", loc="left")
    ax.legend(loc="lower left", fontsize=5.5)


def _cohort_flow(ax, flow, prim, hyp):
    ax.axis("off")
    n0 = flow["n_local_scanned"]; nN = flow["included_N"]
    exc = flow["excluded"]
    nw = prim["n_windows"]
    ntr = strat(prim, 7, "transition")["n_windows"]; nst = strat(prim, 7, "steady")["n_windows"]
    steps = [
        (f"VitalDB cases scanned\nn = {n0:,}", "#E8EEF2"),
        (f"Anesthetic cohort\n(remi + propofol)\nn = {nN:,}", S.C["M1_light"]),
        (f"Forecast windows\nn = {nw:,}", "#EAD9BD"),
    ]
    y = 0.92
    for i, (txt, col) in enumerate(steps):
        ax.add_patch(plt.Rectangle((0.08, y-0.16), 0.84, 0.15, transform=ax.transAxes,
                     facecolor=col, edgecolor="#888", lw=0.6, zorder=2))
        ax.text(0.5, y-0.085, txt, transform=ax.transAxes, ha="center", va="center", fontsize=6, zorder=3)
        if i < len(steps)-1:
            ax.annotate("", xy=(0.5, y-0.30), xytext=(0.5, y-0.17), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="#666", lw=0.9))
        y -= 0.30
    top_exc = sorted(exc.items(), key=lambda kv: -kv[1])[:3]
    ax.text(0.5, y+0.04, "excluded: " + "; ".join(f"{k.replace('_',' ')} {v:,}" for k, v in top_exc),
            transform=ax.transAxes, ha="center", va="top", fontsize=5, color="#666")
    ax.text(0.5, y-0.03, f"transition {ntr:,} · steady {nst:,} windows", transform=ax.transAxes,
            ha="center", va="top", fontsize=5.5, color="#333")
    prev = ", ".join(f"{h}m {hyp['per_horizon'][S.hkey(h)]['prevalence']*100:.0f}%" for h in [1,5,15])
    ax.text(0.5, y-0.11, f"hypotension prevalence: {prev}", transform=ax.transAxes,
            ha="center", va="top", fontsize=5.5, color=S.C["event"])
    ax.set_title("Cohort & windows", loc="center")


def _pick_examples(ex):
    """Choose a steady, a transition and a hypotensive-onset example from the npz.
    Filters artifact traces (context+horizon min < 60 mmHg) so panels look clean."""
    truth = ex["truth"]; ctx = ex["context"]; n = truth.shape[0]
    fin = lambda a: a[np.isfinite(a)]
    st = []
    for i in range(n):
        t = fin(truth[i]); c = fin(ctx[i])
        both = np.concatenate([c, t]) if (c.size and t.size) else (t if t.size else c)
        if t.size < 5 or both.size < 5:
            st.append(dict(i=i, rng=1e9, mn=0.0, below=0.0, cmin=0.0)); continue
        st.append(dict(i=i, rng=float(t.max()-t.min()), mn=float(t.min()),
                       below=float(np.mean(t < 65)), cmin=float(both.min())))
    hypo = max(st, key=lambda s: s["below"])                     # clearest onset below 65
    phys = [s for s in st if s["cmin"] >= 60 and s["mn"] >= 70]  # no artifacts, stays normotensive
    steady = min(phys or st, key=lambda s: s["rng"])             # flattest
    cand = [s for s in st if s["cmin"] >= 60 and s["mn"] >= 66 and s["i"] not in (steady["i"], hypo["i"])]
    cand.sort(key=lambda s: s["rng"])
    trans = cand[len(cand)//2] if cand else st[min(1, n-1)]      # mid-range drift
    return [steady["i"], trans["i"], hypo["i"]]


def _example_panel(ax, ex, i, title):
    dt_min = DT_S/60.0
    ctx = ex["context"][i]; truth = ex["truth"][i]; q = ex["q_ce"][i]
    tc = np.arange(-len(ctx), 0)*dt_min
    th = (np.arange(len(truth))+1)*dt_min
    ax.plot(tc, ctx, color=S.C["ink"], lw=1.0)
    ax.plot(th, truth, color=S.C["ink"], lw=1.3, label="observed")
    ax.plot(th, q[S.Q_MED], color=S.C["M1"], lw=1.3, label="M1 median")
    ax.fill_between(th, q[S.Q_LO], q[S.Q_HI], color=S.C["M1_light"], alpha=0.55, lw=0, label="10–90%")
    ax.axvline(0, color="#888", lw=0.7, ls=":")
    ax.axhline(65, color=S.C["event"], lw=0.7, ls="--")
    ax.set_title(title, loc="center", fontsize=7)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("MAP (mmHg)")
    ax.set_xlim(tc[0], th[-1])


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — accuracy & value of the drug covariate
# ══════════════════════════════════════════════════════════════════════════════
def figure2(tag):
    prim = load_primary(tag)
    fig, axs = plt.subplot_mosaic([["a", "b"], ["c", "d"]], figsize=(S.W2, S.W2*0.72),
                                  gridspec_kw=dict(hspace=0.55, wspace=0.35))

    # a — instantaneous MAE (mmHg) vs horizon, M1 vs M0
    hs = MAIN_H
    mae1 = [strat(prim, h, "all")["mae_M1"] for h in hs]
    mae0 = [strat(prim, h, "all")["mae_M0"] for h in hs]
    a = axs["a"]
    a.plot(hs, mae1, "-o", color=S.C["M1"], label="M1 (+ drug covariate)")
    a.plot(hs, mae0, "-s", color=S.C["M0"], label="M0 (target only)")
    y7 = strat(prim, 7, "all")["Y_pct_vs_persistence"]
    a.text(0.03, 0.06, f"vs persistence: −{y7:.0f}% CRPS", transform=a.transAxes, fontsize=6, color="#555")
    S.finish(a, "forecast horizon (min)", "MAE (mmHg)", "Forecast accuracy")
    a.set_xticks(hs); a.legend(loc="upper left"); S.panel_letter(a, "a")

    # b — covariate benefit X% vs horizon, by stratum
    b = axs["b"]
    for s_name, col, mk in [("all", S.C["M1"], "o"), ("transition", S.C["transition"], "^"),
                            ("steady", S.C["steady"], "v")]:
        xs = [strat(prim, h, s_name)["X_pct_withpast"] for h in hs]
        lo = [strat(prim, h, s_name)["X_pct_withpast_CI95"][0] for h in hs]
        hi = [strat(prim, h, s_name)["X_pct_withpast_CI95"][1] for h in hs]
        b.plot(hs, xs, "-", color=col, marker=mk, label=s_name)
        b.fill_between(hs, lo, hi, color=col, alpha=0.15, lw=0)
    b.axhline(0, color="#999", lw=0.7, ls="--")
    S.finish(b, "forecast horizon (min)", "CRPS reduction M0→M1 (%)", "Value of the drug covariate")
    b.set_xticks(hs); b.legend(loc="upper right"); S.panel_letter(b, "b")

    # c — covariate representation: CE vs RATE vs pressor (transition, 7 min) forest
    c = axs["c"]
    arms = [("CE (effect-site)", tag, S.C["M1"]),
            ("RATE (infusion)", RATE_TAG, S.C["rate"]),
            ("Phenylephrine", PRESSOR_TAG, S.C["pressor"])]
    ypos = list(range(len(arms)))[::-1]
    for yp, (lab, t, col) in zip(ypos, arms):
        p = load_primary(t); blk = strat(p, 7, "transition")
        x = blk["X_pct_withpast"]; ci = blk["X_pct_withpast_CI95"]
        c.errorbar(x, yp, xerr=[[x-ci[0]], [ci[1]-x]], fmt="o", color=col, capsize=2.5, lw=1.2)
        c.text(ci[1]+0.15, yp, f"{x:+.2f}% [{ci[0]:+.2f}, {ci[1]:+.2f}]", va="center", fontsize=5.8)
    c.axvline(0, color="#999", lw=0.7, ls="--")
    c.set_yticks(ypos); c.set_yticklabels([a[0] for a in arms])
    c.set_xlabel("CRPS reduction in transition windows @7 min (%)")
    c.set_title("Which covariate helps?", loc="center")
    c.set_xlim(-4, 4); c.set_ylim(-0.6, len(arms)-0.4); S.panel_letter(c, "c")

    # d — instantaneous MAE vs Kapral (external / internal)
    d = axs["d"]
    _kapral_panel(d, tag)
    S.panel_letter(d, "d")

    fig.suptitle("Forecast accuracy and the value of the known future drug trajectory",
                 y=0.99, fontsize=9, fontweight="bold")
    S.save_fig(fig, "Fig2_accuracy_covariate")


def _kapral_panel(ax, tag):
    # our instantaneous endpoint MAE from windows
    rows, _ = H.load_rows(tag)
    hs = MAIN_H
    our = []
    for h in hs:
        v = [float(r["mae_inst_M1"]) for r in rows if int(r["h_min"]) == h and r.get("mae_inst_M1") not in ("", "nan", None)]
        our.append(np.mean(v) if v else np.nan)
    # Kapral digitized curves (instantaneous)
    K = {}
    for r in csv.DictReader(open("results/kapral_mae_curves.csv")):
        K.setdefault((r["dataset"], r["curve"]), []).append((float(r["forecast_min"]), float(r["mae_mmHg"])))
    def curve(ds, c):
        pts = sorted(K.get((ds, c), []));
        return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    for ds, col, lab in [("internal", S.C["kapral"], "Kapral internal"),
                         ("external", "#B07FD0", "Kapral external")]:
        xm, ym = curve(ds, "mean")
        if xm.size:
            ax.plot(xm, ym, "-", color=col, lw=1.1, label=lab)
            xl, yl = curve(ds, "lower"); xu, yu = curve(ds, "upper")
            if xl.size and xu.size:
                yl_i = np.interp(xm, xl, yl); yu_i = np.interp(xm, xu, yu)
                ax.fill_between(xm, yl_i, yu_i, color=col, alpha=0.12, lw=0)
    ax.plot(hs, our, "-o", color=S.C["M1"], label="TiRex-2 (ours, M1)")
    ax.set_xlim(0, 7.4); ax.set_ylim(0, None)
    S.finish(ax, "forecast distance (min)", "instantaneous MAE (mmHg)", "Accuracy vs Kapral et al. (TFT)")
    ax.legend(loc="upper left")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — impending-hypotension prediction vs supervised SOTA  (headline)
# ══════════════════════════════════════════════════════════════════════════════
def figure3(tag):
    hyp = load_hypo(tag); clin = load_clinical(tag)
    rows, _ = H.load_rows(tag); c2s = H.caseid_to_subject()
    dev = H.split_subjects([r["caseid"] for r in rows], c2s, seed=0)

    fig = plt.figure(figsize=(S.W2, S.W2*0.66))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.42)
    ax_roc = fig.add_subplot(gs[0, 0])
    ax_auc = fig.add_subplot(gs[0, 1])
    ax_cal = fig.add_subplot(gs[0, 2])
    ax_pr  = fig.add_subplot(gs[1, 0])
    ax_dca = fig.add_subplot(gs[1, 1])
    ax_txt = fig.add_subplot(gs[1, 2]); ax_txt.axis("off")

    # a — ROC at 5 and 7 min with spec>=0.90 operating points
    for h, col, ls in [(5, S.C["M1"], "-"), (7, S.C["transition"], "-")]:
        y, s = _test_scores(rows, c2s, dev, h)
        fpr, tpr, _ = H.roc_points(y, s)
        au = hyp["per_horizon"][S.hkey(h)]["M1"]["auroc"]
        ax_roc.plot(fpr, tpr, ls, color=col, lw=1.4, label=f"{h} min (AUROC {au:.3f})")
        op = hyp["per_horizon"][S.hkey(h)]["M1"]["operating_points"]["spec90"]
        ax_roc.plot(1-op["spec"], op["sens"], "o", color=col, ms=5, mec="white", mew=0.6, zorder=5)
    ax_roc.plot([0, 1], [0, 1], color="#BBB", lw=0.7, ls=":")
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.005)
    S.finish(ax_roc, "1 − specificity", "sensitivity", "ROC (M1)", ygrid=False)
    ax_roc.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02)); S.panel_letter(ax_roc, "a")
    ax_roc.text(0.30, 0.55, "● spec ≥ 0.90\n  operating point", fontsize=5.5, color="#555", transform=ax_roc.transAxes)

    # b — AUROC vs horizon, M1 vs M0, with foils overlaid (THE panel)
    hs = S.horizons_sorted(hyp["per_horizon"])
    def series(model):
        a = [hyp["per_horizon"][S.hkey(h)][model]["auroc"] for h in hs]
        lo = [hyp["per_horizon"][S.hkey(h)][model]["auroc_CI95"][0] for h in hs]
        hi = [hyp["per_horizon"][S.hkey(h)][model]["auroc_CI95"][1] for h in hs]
        return a, lo, hi
    a1, l1, h1 = series("M1"); a0, l0, h0 = series("M0")
    ax_auc.fill_between(hs, l1, h1, color=S.C["M1"], alpha=0.15, lw=0)
    ax_auc.plot(hs, a1, "-o", color=S.C["M1"], label="TiRex-2 M1 (ours)")
    ax_auc.plot(hs, a0, "--s", color=S.C["M0"], ms=3, label="TiRex-2 M0 (ours)")
    for h, (ki, ke) in S.KAPRAL_AUROC.items():
        ax_auc.plot(h, ke, "D", color=S.C["kapral"], ms=5, mec="white", mew=0.5, zorder=6)
    ax_auc.plot([], [], "D", color=S.C["kapral"], label="Kapral (TFT, ext.)")
    for h, z in S.ZHU_AUROC.items():
        ax_auc.plot(h, z, "s", color=S.C["zhu"], ms=5, mec="white", mew=0.5, zorder=6)
    ax_auc.plot([], [], "s", color=S.C["zhu"], label="Zhu (Transformer, ext.)")
    ax_auc.axvspan(0.5, 7, color="#F4F7F8", zorder=0)
    S.finish(ax_auc, "forecast horizon (min)", "hypotension AUROC", "Zero-shot vs supervised SOTA")
    ax_auc.set_xticks(hs); ax_auc.set_ylim(0.80, 1.0)
    ax_auc.legend(loc="lower left", fontsize=5.6); S.panel_letter(ax_auc, "b")

    # c — calibration at 5 min (M1)
    y5, s5 = _test_scores(rows, c2s, dev, 5)
    mean_pred, obs_freq, _, _ = H.calibration(y5, s5, n_bins=10)
    ece = hyp["per_horizon"]["5min"]["M1"]["ece"]
    ax_cal.plot([0, 1], [0, 1], color="#BBB", lw=0.7, ls=":")
    ax_cal.plot(mean_pred, obs_freq, "-o", color=S.C["M1"], ms=3)
    ax_cal.text(0.05, 0.9, f"ECE = {ece:.3f}", transform=ax_cal.transAxes, fontsize=6)
    ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)
    S.finish(ax_cal, "predicted risk", "observed frequency", "Calibration @5 min", ygrid=False)
    S.panel_letter(ax_cal, "c")

    # d — AUPRC vs horizon with prevalence baseline
    ap = [hyp["per_horizon"][S.hkey(h)]["M1"]["auprc"] for h in hs]
    apl = [hyp["per_horizon"][S.hkey(h)]["M1"]["auprc_CI95"][0] for h in hs]
    aph = [hyp["per_horizon"][S.hkey(h)]["M1"]["auprc_CI95"][1] for h in hs]
    prev = [hyp["per_horizon"][S.hkey(h)]["prevalence"] for h in hs]
    ax_pr.fill_between(hs, apl, aph, color=S.C["M1"], alpha=0.15, lw=0)
    ax_pr.plot(hs, ap, "-o", color=S.C["M1"], label="AUPRC (M1)")
    ax_pr.plot(hs, prev, "--", color=S.C["persist"], label="prevalence (chance)")
    S.finish(ax_pr, "forecast horizon (min)", "AUPRC", "Precision–recall")
    ax_pr.set_xticks(hs); ax_pr.set_ylim(0, 1); ax_pr.legend(loc="upper right"); S.panel_letter(ax_pr, "d")

    # e — decision curve @5 min
    dc = clin["C_decision_curve"]["5"]
    pt = np.array(dc["pt"]); nb = np.array(dc["nb_model"]); nball = np.array(dc["nb_treat_all"])
    ax_dca.plot(pt, nb, color=S.C["M1"], lw=1.4, label="TiRex-2 M1")
    ax_dca.plot(pt, nball, color=S.C["persist"], lw=1.0, label="treat all")
    ax_dca.axhline(0, color="#999", lw=0.8, label="treat none")
    ax_dca.set_ylim(-0.02, max(0.02, np.nanmax(nb)*1.15)); ax_dca.set_xlim(pt.min(), pt.max())
    S.finish(ax_dca, "threshold probability", "net benefit", "Decision curve @5 min")
    ax_dca.legend(loc="upper right"); S.panel_letter(ax_dca, "e")

    # f (text) — headline summary numbers
    m5 = hyp["per_horizon"]["5min"]["M1"]; op = m5["operating_points"]["spec90"]
    lines = [
        "Headline (zero-shot, held-out test)",
        "",
        f"AUROC @5 min:  {m5['auroc']:.3f}  [{m5['auroc_CI95'][0]:.3f}, {m5['auroc_CI95'][1]:.3f}]",
        f"   vs Kapral ext. 0.903 · Zhu 0.904",
        f"AUROC @7 min:  {hyp['per_horizon']['7min']['M1']['auroc']:.3f}",
        f"   vs Kapral ext. 0.867",
        "",
        f"At spec ≥ 0.90 (5 min):",
        f"   sens {op['sens']:.2f} · PPV {op['ppv']:.2f} · NPV {op['npv']:.2f} · F1 {op['f1']:.2f}",
        f"pAUROC(spec≥0.9): {m5['pauroc_spec90']:.3f}  ·  ECE {m5['ece']:.3f}",
    ]
    ax_txt.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=6.2, family="sans-serif")
    S.panel_letter(ax_txt, "f", dx=0.0)

    fig.suptitle("Impending-hypotension prediction: zero-shot TiRex-2 matches or exceeds task-trained models",
                 y=0.99, fontsize=9, fontweight="bold")
    S.save_fig(fig, "Fig3_hypotension_vs_sota")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — clinical translation & robustness
# ══════════════════════════════════════════════════════════════════════════════
def figure4(tag):
    clin = load_clinical(tag); hyp = load_hypo(tag); sg = load_subgroup(tag, 5)
    fig, axs = plt.subplot_mosaic([["a", "c"], ["b", "c"], ["d", "c"]],
                                  figsize=(S.W2, S.W2*0.78),
                                  gridspec_kw=dict(hspace=0.7, wspace=0.35, width_ratios=[1, 1]))

    # a — lead time (early warning)
    A = clin["A_early_warning"]
    a = axs["a"]
    med = A["lead_time_min_median"]; iqr = A["lead_time_min_IQR"]
    # pct_detected_* are already in percent (0-100)
    a.barh([0], [A["pct_detected_ge2min_ahead"]], color=S.C["M1_light"], height=0.5, label="≥2 min ahead")
    a.barh([1], [A["pct_detected_ge5min_ahead"]], color=S.C["M1"], height=0.5, label="≥5 min ahead")
    a.set_yticks([0, 1]); a.set_yticklabels(["detected\n≥2 min", "detected\n≥5 min"])
    a.set_xlim(0, 100); a.set_xlabel("% of events flagged in advance")
    a.set_title(f"Early warning — median lead {med:.1f} min (IQR {iqr[0]:.1f}–{iqr[1]:.1f})", loc="center", fontsize=6.8)
    for yv, key in [(0, "pct_detected_ge2min_ahead"), (1, "pct_detected_ge5min_ahead")]:
        a.text(min(A[key]+1.5, 92), yv, f"{A[key]:.0f}%", va="center", fontsize=6)
    S.panel_letter(a, "a")

    # b — severity gradient (AUROC by threshold/duration vs horizon)
    b = axs["b"]
    sev = clin["B_severity"]
    styles = {"MAP<65 (≥1min)": (S.C["M1"], "-", "o"), "MAP<55 (≥1min)": (S.C["transition"], "-", "^"),
              "MAP<50 (≥1min)": ("#08313A", "-", "s"), "MAP<65 (≥5min, sustained)": (S.C["M0"], "--", "D")}
    for name, (col, ls, mk) in styles.items():
        d = sev.get(name, {}); hs = sorted(int(k) for k in d)
        au = [d[str(h)]["auroc"] for h in hs]
        b.plot(hs, au, ls, color=col, marker=mk, ms=3, label=name.replace(" (", "\n("))
    S.finish(b, "forecast horizon (min)", "AUROC", "Severity gradient")
    b.set_ylim(0.75, 1.0); b.legend(loc="lower left", fontsize=5.2); S.panel_letter(b, "b")

    # d — operating points vs horizon at spec>=0.90
    d = axs["d"]
    hs = S.horizons_sorted(hyp["per_horizon"])
    for metric, col, mk in [("sens", S.C["M1"], "o"), ("ppv", S.C["M0"], "s"),
                            ("npv", S.C["transition"], "^"), ("f1", S.C["zhu"], "D")]:
        vals = [hyp["per_horizon"][S.hkey(h)]["M1"]["operating_points"]["spec90"][metric] for h in hs]
        d.plot(hs, vals, "-", color=col, marker=mk, ms=3, label=metric.upper())
    S.finish(d, "forecast horizon (min)", "value at spec ≥ 0.90", "Operating characteristics")
    d.set_xticks(hs); d.set_ylim(0, 1); d.legend(loc="center left", fontsize=5.5, ncol=2); S.panel_letter(d, "d")

    # c — subgroup forest (tall panel)
    _forest(axs["c"], sg)
    S.panel_letter(axs["c"], "c", dx=-0.02)

    fig.suptitle("Clinical translation and robustness of the hypotension early-warning signal",
                 y=0.995, fontsize=9, fontweight="bold")
    S.save_fig(fig, "Fig4_clinical_robustness")


def _forest(ax, sg):
    subs = sg["subgroups"]; overall = sg["overall"]
    rows = []
    last_var = None
    for s in subs:
        if s["var"] != last_var:
            rows.append(("header", s["var"], None, None, None)); last_var = s["var"]
        rows.append(("row", s["level"], s["auroc"], s["ci"], s.get("n_cases")))
    rows = rows[::-1]
    y = 0; yticks = []; ylabels = []
    for kind, lab, au, ci, n in rows:
        if kind == "header":
            ax.text(-0.02, y, lab, fontsize=6.3, fontweight="bold", va="center",
                    transform=ax.get_yaxis_transform())
            yticks.append(y); ylabels.append("")
        else:
            ax.errorbar(au, y, xerr=[[au-ci[0]], [ci[1]-au]], fmt="o", color=S.C["M1"],
                        ms=3.2, capsize=1.8, lw=1.0)
            ax.text(1.005, y, f"{au:.3f}", va="center", fontsize=5.6, transform=ax.get_yaxis_transform())
            yticks.append(y); ylabels.append(f"  {lab} (n={n})")
        y += 1
    ax.axvline(overall["auroc"], color=S.C["persist"], lw=0.9, ls="--")
    ax.text(overall["auroc"], -1.15, f"overall {overall['auroc']:.3f}", fontsize=5.6,
            color="#555", ha="center", va="top")
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=5.8)
    ax.set_ylim(-1.4, y-0.4); ax.set_xlim(0.85, 1.0)
    ax.set_xlabel("hypotension AUROC @5 min"); ax.set_title("Subgroup robustness", loc="center")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)


# ══════════════════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════════════════
def _write_table(name, header, rows, caption):
    os.makedirs(S.TAB_DIR, exist_ok=True)
    # markdown
    with open(f"{S.TAB_DIR}/{name}.md", "w") as f:
        f.write(f"**{caption}**\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"]*len(header)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
    # csv
    with open(f"{S.TAB_DIR}/{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    # latex (booktabs)
    with open(f"{S.TAB_DIR}/{name}.tex", "w") as f:
        f.write("\\begin{table}[t]\\centering\\footnotesize\n")
        f.write("\\caption{" + caption + "}\n")
        f.write("\\begin{tabular}{" + "l"*len(header) + "}\n\\toprule\n")
        f.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(str(x) for x in r).replace("%", "\\%").replace("±", "$\\pm$") + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  wrote {S.TAB_DIR}/{name}.{{md,csv,tex}}", flush=True)


def table1_cohort(tag):
    rows = [r for r in csv.DictReader(open("datasets/vitaldb/cohort_manifest.csv")) if r["include"] in ("1","True","true")]
    hyp = load_hypo(tag)
    def num(col):
        v = [float(r[col]) for r in rows if r.get(col) not in ("", None) and r[col].replace(".","",1).lstrip("-").isdigit()]
        return np.array(v)
    def age_parse(x):
        try: return float(x)
        except: return 90.0 if ">" in str(x) else np.nan
    ages = np.array([age_parse(r["age"]) for r in rows]); ages = ages[~np.isnan(ages)]
    dur = num("dur_min")
    n = len(rows)
    males = sum(1 for r in rows if r["sex"].upper().startswith("M"))
    asa = {}
    for r in rows: asa[r["asa"]] = asa.get(r["asa"], 0)+1
    dept = {}
    for r in rows: dept[r["department"]] = dept.get(r["department"], 0)+1
    top_dept = sorted(dept.items(), key=lambda kv:-kv[1])[:3]
    H_rows = [
        ("Cases, n", f"{n:,}"),
        ("Age, y — median (IQR)", f"{np.median(ages):.0f} ({np.percentile(ages,25):.0f}–{np.percentile(ages,75):.0f})"),
        ("Male sex, n (%)", f"{males:,} ({males/n*100:.0f}%)"),
        ("Case duration, min — median (IQR)", f"{np.median(dur):.0f} ({np.percentile(dur,25):.0f}–{np.percentile(dur,75):.0f})"),
        ("ASA I–II, n (%)", f"{asa.get('1',0)+asa.get('2',0):,} ({(asa.get('1',0)+asa.get('2',0))/n*100:.0f}%)"),
        ("Top departments", "; ".join(f"{k} {v}" for k,v in top_dept)),
        ("Forecast windows, n", f"{load_primary(tag)['n_windows']:,}"),
    ]
    for h in [1,5,10,15]:
        p = hyp["per_horizon"][S.hkey(h)]
        H_rows.append((f"Hypotension prevalence @{h} min, %", f"{p['prevalence']*100:.1f}"))
    _write_table("Table1_cohort", ["Characteristic", "Value"], H_rows,
                 f"Table 1. Cohort characteristics (anesthetic cohort, tag={tag}).")


def table2_accuracy(tag):
    p = load_primary(tag); hs = S.horizons_sorted(p["per_horizon"])
    header = ["Horizon (min)", "MAE M1", "MAE M0", "CRPS M1", "CRPS M0", "CRPS persist.",
              "X% covariate [95% CI]", "Y% vs persist. [95% CI]"]
    rows = []
    for h in hs:
        a = strat(p, h, "all")
        rows.append([h, f"{a['mae_M1']:.2f}", f"{a['mae_M0']:.2f}", f"{a['crps_M1']:.3f}",
                     f"{a['crps_M0']:.3f}", f"{a['crps_persistence']:.3f}",
                     f"{a['X_pct_withpast']:+.2f} [{a['X_pct_withpast_CI95'][0]:+.2f}, {a['X_pct_withpast_CI95'][1]:+.2f}]",
                     f"{a['Y_pct_vs_persistence']:.1f} [{a['Y_pct_CI95'][0]:.1f}, {a['Y_pct_CI95'][1]:.1f}]"])
    _write_table("Table2_accuracy", header, rows,
                 f"Table 2. Forecast accuracy and covariate value, all windows (tag={tag}).")


def table3_classification(tag):
    hyp = load_hypo(tag); hs = S.horizons_sorted(hyp["per_horizon"])
    header = ["Horizon (min)", "AUROC M1 [95% CI]", "AUROC M0", "AUPRC M1", "pAUROC(sp≥.9)", "ECE",
              "Sens/PPV/NPV/F1 @sp≥.9", "Kapral ext.", "Zhu ext."]
    rows = []
    for h in hs:
        m1 = hyp["per_horizon"][S.hkey(h)]["M1"]; m0 = hyp["per_horizon"][S.hkey(h)]["M0"]
        op = m1["operating_points"]["spec90"]
        kap = f"{S.KAPRAL_AUROC[h][1]:.3f}" if h in S.KAPRAL_AUROC else "—"
        zhu = f"{S.ZHU_AUROC[h]:.3f}" if h in S.ZHU_AUROC else "—"
        rows.append([h, f"{m1['auroc']:.3f} [{m1['auroc_CI95'][0]:.3f}, {m1['auroc_CI95'][1]:.3f}]",
                     f"{m0['auroc']:.3f}", f"{m1['auprc']:.3f}", f"{m1['pauroc_spec90']:.3f}",
                     f"{m1['ece']:.3f}", f"{op['sens']:.2f}/{op['ppv']:.2f}/{op['npv']:.2f}/{op['f1']:.2f}",
                     kap, zhu])
    _write_table("Table3_classification", header, rows,
                 f"Table 3. Impending-hypotension classification vs supervised foils (tag={tag}). "
                 "Ours = zero-shot; foils = task-trained (external VitalDB).")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(S.FIG_DIR, exist_ok=True); os.makedirs(S.TAB_DIR, exist_ok=True)
    print(f"[paper] tag={TAG}  font={S.SANS}", flush=True)
    print("[paper] Figure 1 ..."); figure1(TAG)
    print("[paper] Figure 2 ..."); figure2(TAG)
    print("[paper] Figure 3 ..."); figure3(TAG)
    print("[paper] Figure 4 ..."); figure4(TAG)
    print("[paper] Tables ..."); table1_cohort(TAG); table2_accuracy(TAG); table3_classification(TAG)
    print("[paper] Done. Figures in outputs/figs/paper/ ; tables in results/tables/", flush=True)


if __name__ == "__main__":
    main()
