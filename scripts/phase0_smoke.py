"""Phase 0 smoke test for TiRex-2.

Runs all weight-dependent Phase 0 checks in one shot:
  1. Load model on cpu (time it), read quantile levels + output shape.
  2. Try mps; record whether it works / falls back / errors.
  3. Forecast a noisy sine; confirm shape (n_targets, n_quantiles, H).
  4. Controlled covariate test: target = shifted copy of a future-known covariate.
     Compare forecast WITH vs WITHOUT the covariate against the (known) truth.
  5. NaN-in-context test: inject NaNs, confirm no crash.

Requires HF auth for the gated NX-AI/TiRex-2 weights (HF_TOKEN or `huggingface-cli login`).
Saves a plot to outputs/phase0_smoke.png and prints a summary.
"""
from __future__ import annotations
import os, time, sys
import numpy as np
import torch

from tirex2 import TimeseriesType, load_model

RNG = np.random.default_rng(0)
torch.manual_seed(0)


def load(device):
    t0 = time.perf_counter()
    m = load_model("NX-AI/TiRex-2", device=device)
    return m, time.perf_counter() - t0


def fc(model, ts, H):
    t0 = time.perf_counter()
    out = model.forecast([ts], prediction_length=H, output_type="numpy")
    return out, time.perf_counter() - t0


def median(forecast_item, med_idx):
    """forecast_item has shape (n_targets, n_quantiles, H); return target-0 median (H,)."""
    return np.asarray(forecast_item)[0, med_idx]


def main():
    summary = {}

    # --- 1. load on cpu ---
    model, load_s = load("cpu")
    summary["cpu_load_s"] = round(load_s, 2)
    q = model.quantiles.detach().cpu().numpy()
    summary["quantiles"] = [round(float(x), 3) for x in q.ravel()]
    summary["max_prediction_length"] = int(model.future_len)
    med_idx = summary["quantiles"].index(0.5)  # = 4

    # --- 3. noisy sine smoke test ---
    T, H = 256, 64
    t = np.arange(T + H)
    clean = np.sin(t / 8.0)
    y = (clean + 0.1 * RNG.standard_normal(T + H)).astype(np.float32)
    ts = TimeseriesType(
        target=torch.from_numpy(y[:T]).unsqueeze(0),
        past_covariates=None, future_covariates=None,
    )
    out, inf_s = fc(model, ts, H)
    summary["cpu_infer_s"] = round(inf_s, 3)
    summary["sine_output_shape"] = list(np.asarray(out[0]).shape)
    sine_med = median(out[0], med_idx)

    # --- 4. controlled covariate test ---
    # A random-walk covariate fully known over [0, T+H]; target = covariate shifted by `lag`,
    # so the future covariate deterministically reveals the target's future.
    lag = 8
    cov = np.cumsum(0.2 * RNG.standard_normal(T + H + lag)).astype(np.float32)
    tgt_full = cov[lag:lag + T + H]                 # target = cov shifted left by lag
    cov_full = cov[:T + H]                           # covariate aligned to target window [0, T+H]
    truth_future = tgt_full[T:T + H]

    ts_with = TimeseriesType(
        target=torch.from_numpy(tgt_full[:T]).unsqueeze(0),
        past_covariates=None,
        future_covariates=torch.from_numpy(cov_full).unsqueeze(0),  # [1, T+H] -> known
    )
    ts_without = TimeseriesType(
        target=torch.from_numpy(tgt_full[:T]).unsqueeze(0),
        past_covariates=None, future_covariates=None,
    )
    ow, _ = fc(model, ts_with, H)
    on, _ = fc(model, ts_without, H)
    med_w = median(ow[0], med_idx)
    med_n = median(on[0], med_idx)
    mae_with = float(np.mean(np.abs(med_w - truth_future)))
    mae_without = float(np.mean(np.abs(med_n - truth_future)))
    summary["covariate_test"] = {
        "mae_with_cov": round(mae_with, 4),
        "mae_without_cov": round(mae_without, 4),
        "improvement_ratio": round(mae_without / max(mae_with, 1e-9), 2),
        "covariate_helps": mae_with < mae_without,
    }

    # --- 5. NaN-in-context ---
    y_nan = y[:T].copy()
    y_nan[50:60] = np.nan
    ts_nan = TimeseriesType(target=torch.from_numpy(y_nan).unsqueeze(0),
                            past_covariates=None, future_covariates=None)
    try:
        onan, _ = fc(model, ts_nan, H)
        med = median(onan[0], med_idx)
        summary["nan_test"] = {"ok": True, "any_nan_in_output": bool(np.isnan(med).any())}
    except Exception as e:  # noqa: BLE001
        summary["nan_test"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}

    # --- plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 7))
        ax[0].plot(range(T), y[:T], color="k", lw=0.8, label="context")
        ax[0].plot(range(T, T + H), clean[T:], color="green", lw=1.2, label="clean truth")
        ax[0].plot(range(T, T + H), sine_med, color="C0", label="forecast median")
        ax[0].axvline(T, ls="--", c="grey"); ax[0].set_title("Noisy sine smoke test"); ax[0].legend(loc="upper left", fontsize=8)
        ax[1].plot(range(T), tgt_full[:T], color="k", lw=0.8, label="target context")
        ax[1].plot(range(T, T + H), truth_future, color="green", lw=1.2, label="target truth")
        ax[1].plot(range(T, T + H), med_w, color="C0", label=f"median WITH cov (MAE {mae_with:.3f})")
        ax[1].plot(range(T, T + H), med_n, color="C3", label=f"median WITHOUT cov (MAE {mae_without:.3f})")
        ax[1].plot(range(T + H), cov_full, color="C1", lw=0.6, alpha=0.6, label="known covariate")
        ax[1].axvline(T, ls="--", c="grey"); ax[1].set_title("Controlled covariate test (target = covariate shifted)"); ax[1].legend(loc="upper left", fontsize=8)
        fig.tight_layout(); fig.savefig("outputs/phase0_smoke.png", dpi=110)
        summary["plot"] = "outputs/phase0_smoke.png"
    except Exception as e:  # noqa: BLE001
        summary["plot"] = f"plot failed: {e}"

    import json
    def save():
        with open("outputs/phase0_smoke_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Save CPU results NOW, before the (possibly-hanging) MPS probe.
    summary["mps"] = {"attempted": False, "note": "set TRY_MPS=1 to probe; run under `timeout` (may hang)"}
    save()

    # --- 2. mps attempt (opt-in; may hang -> run this script under a shell `timeout`) ---
    if os.environ.get("TRY_MPS") == "1" and torch.backends.mps.is_available():
        try:
            m_mps, mps_load_s = load("mps")
            _out, mps_inf_s = fc(m_mps, ts, H)
            summary["mps"] = {"attempted": True, "ok": True,
                              "load_s": round(mps_load_s, 2), "infer_s": round(mps_inf_s, 3)}
        except Exception as e:  # noqa: BLE001
            summary["mps"] = {"attempted": True, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
        save()

    print("\n===== PHASE 0 SMOKE SUMMARY =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
