"""Merge sharded ablation window CSVs -> aggregate + dashboard. Run anytime (interim or final).

  python scripts/merge_dashboard.py n300_s1
"""
from __future__ import annotations
import csv, glob, json, sys
import numpy as np
import yaml

from phase3_ablation import aggregate, HORIZON_STEPS_MIN
from plot_results import plot_dashboard, plot_examples, plot_hypotension


def main():
    base = sys.argv[1]
    files = sorted(glob.glob(f"results/ablation_windows_{base}_sh*of*.csv"))
    assert files, f"no shard CSVs for {base}"
    rows, cases = [], set()
    for f in files:
        for r in csv.DictReader(open(f)):
            for k in list(r):
                if k in ("caseid", "stratum"):
                    continue
                if k == "h_min":
                    r[k] = int(round(float(r[k])))
                else:
                    r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
            rows.append(r); cases.add(r["caseid"])
    dt = float(yaml.safe_load(open("configs/data.yaml")).get("resample_to_s", 5.0))
    hsteps = [int(m * 60 / dt) for m in HORIZON_STEPS_MIN]
    n_boot = yaml.safe_load(open("configs/eval.yaml")).get("n_boot", 2000)
    rng = np.random.default_rng(0)
    summary = aggregate(rows, hsteps, dt, n_boot, rng,
                        {"tag": base, "n_cases": len(cases), "cases_done": len(cases), "dt_s": dt})
    json.dump(summary, open(f"results/ablation_primary_{base}.json", "w"), indent=2)
    plot_dashboard(summary, f"outputs/figs/dashboard_{base}.png")
    plot_hypotension(summary, f"outputs/figs/hypotension_{base}.png")

    # examples from shard-0
    exf = sorted(glob.glob(f"outputs/figs/examples_{base}_sh0of*.npz"))
    if exf:
        d = np.load(exf[0], allow_pickle=True)
        plot_examples({k: d[k] for k in d.files}, f"outputs/figs/examples_{base}.png", dt)

    a = summary["per_horizon"]
    print(f"merged {len(files)} shards, {len(cases)} cases, {summary['n_windows']} windows")
    for h in ["7min", "15min"]:
        s = a.get(h, {})
        print(f"  {h}: X%withpast(all)={s.get('all',{}).get('X_pct_withpast')} "
              f"CI{s.get('all',{}).get('X_pct_withpast_CI95')} | "
              f"transition={s.get('transition',{}).get('X_pct_withpast')} "
              f"CI{s.get('transition',{}).get('X_pct_withpast_CI95')} | "
              f"MAE_M1={s.get('all',{}).get('mae_M1')}")
    print(f"-> outputs/figs/dashboard_{base}.png")


if __name__ == "__main__":
    main()
