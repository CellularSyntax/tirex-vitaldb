"""Subgroup / generalizability forest plot for the TiRex-2 hypotension task (§3.9 TRIPOD-AI).
Standalone, post-hoc: joins the per-window rows (phase3_ablation.py output) to clinical/demographic
variables in datasets/vitaldb/cohort_manifest.csv and plots per-subgroup hypotension AUROC (M1) at a chosen
horizon, each with a case-clustered bootstrap CI, against the overall estimate. Descriptive
heterogeneity analysis over the whole cohort (not the dev/test split used for thresholding).

Run:  PYTHONPATH=scripts <venv>/bin/python scripts/subgroup_forest.py n300_s1 [horizon_min=5]
Writes results/subgroup_forest_<tag>_h<h>.json + outputs/figs/subgroup_forest_<tag>_h<h>.png
"""
import csv, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hypo_eval import auroc, clustered_boot_ci, load_rows


def parse_age(a):
    if a in ("", "nan", None):
        return np.nan
    return 90.0 if str(a).startswith(">") else float(a)


def manifest_meta():
    """caseid -> dict of subgroup fields."""
    m = {}
    for r in csv.DictReader(open("datasets/vitaldb/cohort_manifest.csv")):
        if r.get("include") != "1":
            continue
        m[str(int(r["caseid"]))] = r
    return m


def subgroup_defs(meta):
    """Return ordered [(variable, level_label, predicate(caseid)->bool)] with only non-empty levels."""
    age = {c: parse_age(v["age"]) for c, v in meta.items()}
    dur = {c: float(v["dur_min"]) for c, v in meta.items() if v.get("dur_min") not in ("", "nan")}
    dt1, dt2 = (np.nanpercentile(list(dur.values()), [33, 67]) if dur else (np.nan, np.nan))

    def asa_grp(c):
        a = meta[c].get("asa", "")
        return "I–II" if a in ("1", "2") else ("III+" if a in ("3", "4", "5", "6") else None)

    def dept_grp(c):
        d = meta[c].get("department", "")
        return d if d in ("General surgery", "Thoracic surgery") else ("Other" if d else None)

    defs = []
    defs += [("Sex", lbl, (lambda c, s=s: meta[c].get("sex") == s)) for lbl, s in [("Male", "M"), ("Female", "F")]]
    defs += [("Age", "< 50",  lambda c: age.get(c, np.nan) < 50),
             ("Age", "50–64", lambda c: 50 <= age.get(c, np.nan) < 65),
             ("Age", "65–79", lambda c: 65 <= age.get(c, np.nan) < 80),
             ("Age", "≥ 80",  lambda c: age.get(c, np.nan) >= 80)]
    defs += [("ASA", "I–II", lambda c: asa_grp(c) == "I–II"),
             ("ASA", "III+", lambda c: asa_grp(c) == "III+")]
    defs += [("Department", "General surgery", lambda c: dept_grp(c) == "General surgery"),
             ("Department", "Thoracic surgery", lambda c: dept_grp(c) == "Thoracic surgery"),
             ("Department", "Other", lambda c: dept_grp(c) == "Other")]
    if dur:
        defs += [("Case duration", "short (T1)", lambda c: dur.get(c, np.inf) <= dt1),
                 ("Case duration", "mid (T2)", lambda c: dt1 < dur.get(c, np.inf) <= dt2),
                 ("Case duration", "long (T3)", lambda c: dur.get(c, -1) > dt2)]
    return defs


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "n300_s1"
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_boot = int(sys.argv[3]) if len(sys.argv) > 3 else 800
    rows, files = load_rows(tag)
    meta = manifest_meta()
    hr = [r for r in rows if int(r["h_min"]) == H and r["risk_M1"] not in ("", "nan")
          and str(r["caseid"]) in meta]
    cid = np.array([str(r["caseid"]) for r in hr])
    y = np.array([float(r["hypo_event"]) for r in hr])
    s = np.array([float(r["risk_M1"]) for r in hr])
    overall = auroc(y, s); overall_ci = clustered_boot_ci(cid, y, s, auroc, n_boot)
    print(f"tag={tag} h={H}min  {len(files)} shards  overall M1 AUROC={overall:.3f} CI{overall_ci} "
          f"(N={len(np.unique(cid))} cases, {int(y.sum())} events / {len(y)} windows)", flush=True)

    results = []
    for var, lbl, pred in subgroup_defs(meta):
        mask = np.array([pred(c) for c in cid])
        yg, sg, cg = y[mask], s[mask], cid[mask]
        ncase = len(np.unique(cg)); ev = int(yg.sum())
        if ncase < 5 or ev < 5 or (yg == 0).sum() < 5:      # too small / degenerate -> skip metric
            results.append({"var": var, "level": lbl, "n_cases": ncase, "n_events": ev,
                            "n_windows": int(mask.sum()), "auroc": None, "ci": [None, None]})
            continue
        a = auroc(yg, sg); ci = clustered_boot_ci(cg, yg, sg, auroc, n_boot)
        results.append({"var": var, "level": lbl, "n_cases": ncase, "n_events": ev,
                        "n_windows": int(mask.sum()), "auroc": round(a, 4), "ci": ci})
        print(f"  {var:14s} {lbl:16s} n={ncase:4d} ev={ev:4d}  AUROC={a:.3f} CI{ci}", flush=True)

    out = {"tag": tag, "horizon_min": H, "metric": "hypotension AUROC (M1)",
           "overall": {"auroc": round(overall, 4), "ci": overall_ci,
                       "n_cases": len(np.unique(cid)), "n_events": int(y.sum())},
           "subgroups": results}
    json.dump(out, open(f"results/subgroup_forest_{tag}_h{H}.json", "w"), indent=1)
    forest(out, tag, H)


def forest(out, tag, H):
    res = out["subgroups"]
    # layout: a header row per variable, then its levels
    yrows, ylabels, is_header = [], [], []
    y = 0.0; last_var = None
    for r in res:
        if r["var"] != last_var:
            y -= 0.7; yrows.append(y); ylabels.append(r["var"]); is_header.append(True); last_var = r["var"]
        y -= 1.0; yrows.append(y); ylabels.append("   " + r["level"]); is_header.append(False)
    rowmap = [i for i, h in enumerate(is_header) if not h]

    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(yrows) + 1.6))
    ov = out["overall"]
    ax.axvspan(ov["ci"][0], ov["ci"][1], color="C0", alpha=0.08)
    ax.axvline(ov["auroc"], color="C0", ls="--", lw=1, label=f"overall {ov['auroc']:.3f}")
    for r, ri in zip(res, rowmap):
        yy = yrows[ri]
        if r["auroc"] is None:
            ax.text(0.505, yy, "n/s (too few events)", va="center", fontsize=7, color="grey")
            continue
        lo, hi = r["ci"]
        ax.plot([lo, hi], [yy, yy], "-", color="C1", lw=1.4)
        ax.plot(r["auroc"], yy, "s", color="C1", ms=7)
        ax.text(1.005, yy, f"{r['auroc']:.3f} [{lo:.3f}, {hi:.3f}]   n={r['n_cases']}, ev={r['n_events']}",
                va="center", fontsize=7.5, transform=ax.get_yaxis_transform())
    for ri, lbl, hdr in zip(range(len(yrows)), ylabels, is_header):
        ax.text(-0.01, yrows[ri], lbl, va="center", ha="right", fontsize=8.5,
                fontweight="bold" if hdr else "normal", transform=ax.get_yaxis_transform())
    ax.set_yticks([]); ax.set_ylim(min(yrows) - 0.8, 0.2)
    ax.set_xlim(0.5, 1.0); ax.set_xlabel(f"hypotension AUROC (M1) at {H} min")
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"Subgroup generalizability — hypotension detection @ {H} min — {tag}",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(); fig.savefig(f"outputs/figs/subgroup_forest_{tag}_h{H}.png", dpi=120,
                                    bbox_inches="tight"); plt.close(fig)
    print(f"wrote outputs/figs/subgroup_forest_{tag}_h{H}.png", flush=True)


if __name__ == "__main__":
    main()
