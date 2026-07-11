"""Phase 2 — single-case proof of concept.

Forecast MAP at a rolling origin, WITH vs WITHOUT the drug future-covariate, on one real VitalDB case.
Confirms: shapes/horizon, the covariate visibly changes the forecast, quantile calibration on one case.

  python scripts/phase2_single_case.py 2521 --context-min 30 --horizon-min 10 [--origin-min 145]

If --origin-min is omitted, picks the origin whose horizon contains the largest change in the primary
drug covariate (a guaranteed "real intervention"), subject to enough finite target history+truth.
Saves forecasts+metrics to outputs/phase2/.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

from vitaldb_loader import load_config, _clinical_index, load_case  # noqa: E402

QUANTILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
MED = 4  # index of 0.5


def pinball(y, q_pred, q_levels):
    """Mean pinball loss over finite y. q_pred: (n_q, H); y: (H,)."""
    m = np.isfinite(y)
    if m.sum() == 0:
        return np.nan
    y, q_pred = y[m], q_pred[:, m]
    losses = []
    for i, a in enumerate(q_levels):
        d = y - q_pred[i]
        losses.append(np.mean(np.maximum(a * d, (a - 1) * d)))
    return float(np.mean(losses))  # avg pinball ~ CRPS proxy over the quantile grid


def coverage(y, q_pred, lo=0, hi=8):
    """Empirical coverage of the [q_levels[lo], q_levels[hi]] interval (default 0.1..0.9 -> nominal 0.8)."""
    m = np.isfinite(y)
    if m.sum() == 0:
        return np.nan
    return float(np.mean((y[m] >= q_pred[lo, m]) & (y[m] <= q_pred[hi, m])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("caseid")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--context-min", type=float, default=30.0)
    ap.add_argument("--horizon-min", type=float, default=10.0)
    ap.add_argument("--origin-min", type=float, default=None)
    ap.add_argument("--primary-cov", default="Orchestra/RFTN20_CE",
                    help="drug covariate used to auto-pick the intervention origin")
    args = ap.parse_args()

    cfg = load_config(args.config)
    clinical = _clinical_index(cfg["clinical_csv"])
    rec = load_case(args.caseid, cfg, clinical)
    assert rec is not None, "case unreadable"
    dt = rec["interval_s"]
    L = int(args.context_min * 60 / dt)
    H = int(args.horizon_min * 60 / dt)
    assert H <= 320, f"H={H} exceeds model.future_len=320"

    target = rec["target"].astype(np.float32)
    fut = rec["future_cov"]
    pst = rec["past_cov"]
    N = len(target)

    # choose origin
    if args.origin_min is not None:
        t0 = int(args.origin_min * 60 / dt)
    else:
        cov = fut[args.primary_cov]
        best, best_t0 = -1, None
        for t0 in range(L, N - H):
            seg = cov[t0:t0 + H]
            ctx_ok = np.isfinite(target[t0 - L:t0]).mean() > 0.5
            tru_ok = np.isfinite(target[t0:t0 + H]).mean() > 0.5
            if not (ctx_ok and tru_ok) or not np.isfinite(seg).all():
                continue
            change = np.nanmax(seg) - np.nanmin(seg)
            if change > best:
                best, best_t0 = change, t0
        t0 = best_t0
        assert t0 is not None, "no valid origin with an intervention found"
        print(f"auto origin at {t0*dt/60:.1f} min (primary-cov change {best:.2f} over horizon)")

    ctx = slice(t0 - L, t0)
    horizon = slice(t0, t0 + H)
    truth = target[horizon]

    # build TimeseriesType inputs
    tgt_ctx = torch.from_numpy(target[ctx]).unsqueeze(0)                      # (1, L)
    # future covariates span [t0-L, t0+H) -> (n_fut, L+H); known over the horizon
    fut_names = list(fut)
    fut_ctxh = np.stack([fut[n][t0 - L:t0 + H] for n in fut_names]).astype(np.float32)
    fut_cov_t = torch.from_numpy(fut_ctxh)
    # past covariates over context only -> (n_past, L)
    pst_names = list(pst)
    pst_ctx = np.stack([pst[n][ctx] for n in pst_names]).astype(np.float32)
    pst_cov_t = torch.from_numpy(pst_ctx)

    from tirex2 import TimeseriesType, load_model
    model = load_model("NX-AI/TiRex-2", device="cpu")

    ts_with = TimeseriesType(target=tgt_ctx, past_covariates=pst_cov_t, future_covariates=fut_cov_t)
    ts_without = TimeseriesType(target=tgt_ctx, past_covariates=pst_cov_t, future_covariates=None)
    out = model.forecast([ts_with, ts_without], prediction_length=H, output_type="numpy")
    q_with = np.asarray(out[0])[0]     # (9, H)
    q_without = np.asarray(out[1])[0]  # (9, H)

    metrics = {
        "caseid": rec["caseid"], "origin_min": round(t0 * dt / 60, 1),
        "context_min": args.context_min, "horizon_min": args.horizon_min, "H_steps": H,
        "future_covariates": fut_names, "past_covariates": pst_names,
        "with_cov": {
            "mae": float(np.nanmean(np.abs(q_with[MED] - truth))),
            "rmse": float(np.sqrt(np.nanmean((q_with[MED] - truth) ** 2))),
            "pinball_crps": pinball(truth, q_with, QUANTILES),
            "cov80": coverage(truth, q_with),
        },
        "without_cov": {
            "mae": float(np.nanmean(np.abs(q_without[MED] - truth))),
            "rmse": float(np.sqrt(np.nanmean((q_without[MED] - truth) ** 2))),
            "pinball_crps": pinball(truth, q_without, QUANTILES),
            "cov80": coverage(truth, q_without),
        },
    }
    mw, mn = metrics["with_cov"], metrics["without_cov"]
    metrics["covariate_effect"] = {
        "mae_reduction_pct": round(100 * (mn["mae"] - mw["mae"]) / mn["mae"], 1) if mn["mae"] else None,
        "crps_reduction_pct": round(100 * (mn["pinball_crps"] - mw["pinball_crps"]) / mn["pinball_crps"], 1) if mn["pinball_crps"] else None,
    }

    os.makedirs("outputs/phase2", exist_ok=True)
    tag = f"{rec['caseid']}_o{int(metrics['origin_min'])}_h{int(args.horizon_min)}"
    with open(f"outputs/phase2/metrics_{tag}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(f"outputs/phase2/forecast_{tag}.npz",
                        truth=truth, q_with=q_with, q_without=q_without,
                        context=target[ctx], quantiles=QUANTILES)
    print(json.dumps(metrics, indent=2))

    # plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tctx = np.arange(-L, 0) * dt / 60
    th = np.arange(0, H) * dt / 60
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[3, 1])
    ax[0].plot(tctx, target[ctx], color="k", lw=0.9, label="MAP context")
    ax[0].plot(th, truth, color="green", lw=1.4, label="MAP truth")
    ax[0].plot(th, q_with[MED], color="C0", lw=1.4, label=f"median WITH cov (MAE {mw['mae']:.1f})")
    ax[0].fill_between(th, q_with[0], q_with[8], color="C0", alpha=0.15)
    ax[0].plot(th, q_without[MED], color="C3", lw=1.4, ls="--", label=f"median WITHOUT cov (MAE {mn['mae']:.1f})")
    ax[0].fill_between(th, q_without[0], q_without[8], color="C3", alpha=0.10)
    ax[0].axvline(0, ls="--", c="grey"); ax[0].axhline(65, ls=":", c="k", lw=0.8)
    ax[0].set_ylabel("MAP (mmHg)"); ax[0].legend(fontsize=8, loc="best")
    ax[0].set_title(f"Case {rec['caseid']} @ {metrics['origin_min']} min — MAP forecast, drug-covariate ablation")
    for n in fut_names:
        ax[1].plot(np.arange(-L, H) * dt / 60, fut[n][t0 - L:t0 + H], lw=0.8, label=n.split("/")[-1])
    ax[1].axvline(0, ls="--", c="grey"); ax[1].legend(fontsize=7, ncol=2); ax[1].set_xlabel("time from origin (min)")
    ax[1].set_ylabel("drug cov")
    fig.tight_layout(); out_png = f"outputs/phase2/forecast_{tag}.png"; fig.savefig(out_png, dpi=120)
    print("saved", out_png)


if __name__ == "__main__":
    main()
