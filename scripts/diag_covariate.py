"""Covariate diagnostic — does TiRex-2 actually respond to the future infusion covariate?

For a window, forecast MAP under different FUTURE covariate trajectories (context part unchanged):
  real / flat(frozen at origin) / counterfactual +big / counterfactual -big.
If the forecast median moves across scenarios -> the model uses the covariate (mechanism works, and a
small [X%] just means the real covariate isn't very predictive in-context). If it never moves -> the
covariate is being ignored (bug/limitation).

  python scripts/diag_covariate.py 2521 --origins 144 20 90
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import vitaldb_loader as L

COV = "Orchestra/RFTN20_CE"
PAST = ["Solar8000/HR", "Solar8000/PLETH_SPO2", "Solar8000/CVP"]
MED = 4


def scenarios(cov_ctxh, Lc, H):
    """Return dict of future-covariate variants; context part [0:Lc] identical, horizon part varied."""
    base = cov_ctxh.copy()
    at_origin = base[Lc - 1]
    out = {}
    out["real"] = base.copy()
    flat = base.copy(); flat[Lc:] = at_origin; out["flat (frozen)"] = flat
    up = base.copy(); up[Lc:] = at_origin + np.linspace(0, 4, H).astype(np.float32); out["counterfactual +4"] = up
    dn = base.copy(); dn[Lc:] = np.maximum(at_origin - np.linspace(0, 4, H), 0).astype(np.float32); out["counterfactual -4"] = dn
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("caseid")
    ap.add_argument("--origins", type=float, nargs="+", default=[144])
    ap.add_argument("--context-min", type=float, default=30)
    ap.add_argument("--horizon-min", type=float, default=10)
    ap.add_argument("--resample-to-s", type=float, default=None, help="downsample grid (e.g. 15)")
    args = ap.parse_args()

    cfg = L.load_config("datasets/vitaldb/configs/data.yaml")
    if args.resample_to_s:
        cfg["resample_to_s"] = args.resample_to_s
    clin = L._clinical_index(cfg["clinical_csv"])
    rec = L.load_case(args.caseid, cfg, clin)
    dt = rec["interval_s"]; Lc = int(args.context_min * 60 / dt); H = int(args.horizon_min * 60 / dt)
    tgt = rec["target"]; cov = rec["future_cov"][COV]

    from tirex2 import TimeseriesType, load_model
    model = load_model("NX-AI/TiRex-2", device="cpu")

    fig, axes = plt.subplots(len(args.origins), 2, figsize=(13, 3.4 * len(args.origins)), squeeze=False)
    for r, om in enumerate(args.origins):
        t0 = int(om * 60 / dt)
        if t0 < Lc or t0 + H > len(tgt):
            print(f"skip origin {om}min: needs context {args.context_min}min before and horizon after", flush=True)
            continue
        target = torch.from_numpy(tgt[t0 - Lc:t0].astype(np.float32)).unsqueeze(0)
        past = torch.from_numpy(np.stack([rec["past_cov"][p][t0 - Lc:t0] for p in PAST]).astype(np.float32))
        cov_ctxh = cov[t0 - Lc:t0 + H].astype(np.float32)
        truth = tgt[t0:t0 + H]
        scen = scenarios(cov_ctxh, Lc, H)
        items = [TimeseriesType(target=target, past_covariates=past,
                                future_covariates=torch.from_numpy(s).unsqueeze(0)) for s in scen.values()]
        # add no-cov
        items.append(TimeseriesType(target=target, past_covariates=past, future_covariates=None))
        outs = model.forecast(items, prediction_length=H, output_type="numpy")
        meds = {name: np.asarray(o)[0][MED] for name, o in zip(list(scen) + ["no cov"], outs)}

        th = np.arange(H) * dt / 60
        tc = np.arange(-Lc, 0) * dt / 60
        a = axes[r][0]
        a.plot(tc, tgt[t0 - Lc:t0], color="k", lw=0.7)
        a.plot(th, truth, color="green", lw=1.4, label="truth")
        for name, m in meds.items():
            a.plot(th, m, lw=1.2, ls="--" if "cov" in name and name != "no cov" else "-", label=name)
        a.axvline(0, ls="--", c="grey"); a.set_title(f"case {rec['caseid']} @ {om:.0f}min — MAP under covariate scenarios", fontsize=9)
        a.set_ylabel("MAP"); a.legend(fontsize=6.5, ncol=2)
        # covariate panel
        a2 = axes[r][1]
        for name, s in scen.items():
            a2.plot(np.arange(-Lc, H) * dt / 60, s, lw=0.9, label=name)
        a2.axvline(0, ls="--", c="grey"); a2.set_title("RFTN20_CE scenarios", fontsize=9)
        a2.legend(fontsize=6.5)
        # sensitivity: spread of end-of-horizon median across real/flat/up/down
        endvals = {k: meds[k][-1] for k in scen}
        spread = max(endvals.values()) - min(endvals.values())
        print(f"origin {om:.0f}min: end-horizon MAP median by scenario "
              f"{ {k: round(float(v),1) for k,v in endvals.items()} }  -> spread {spread:.1f} mmHg", flush=True)

    fig.tight_layout(); out = f"outputs/diag_covariate_{rec['caseid']}.png"; fig.savefig(out, dpi=115)
    print("saved", out)


if __name__ == "__main__":
    main()
