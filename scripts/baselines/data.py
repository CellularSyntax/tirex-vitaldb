"""Build matched forecasting windows for the supervised baselines.

Uses the SAME window construction as the TiRex-2 evaluation (phase3_ablation.make_windows),
the SAME cache, and the SAME transition/steady tagging — so the supervised model sees
identical inputs on identical windows. Emits plain numpy arrays (not TiRex TimeseriesType):

  past   [Lc, 1+P]      MAP + past covariates (HR, SpO2, CVP)
  future [Lc+H, F]      known future drug-infusion covariates over context+horizon
  truth  [H]            ground-truth MAP over the horizon (may contain NaN)
  meta   caseid, t0, stratum

Normalisation statistics are computed on the TRAIN split only and applied to all splits.
"""
from __future__ import annotations
import numpy as np
import phase3_ablation as P
import vitaldb_loader as L


def _finite_fill(a):
    """Forward/backward fill NaNs, then set any remaining to 0 (channel absent)."""
    a = np.asarray(a, dtype=np.float64).copy()
    n = len(a)
    idx = np.where(np.isfinite(a))[0]
    if idx.size == 0:
        return np.zeros(n)
    # forward fill
    last = a[idx[0]]
    for i in range(n):
        if np.isfinite(a[i]):
            last = a[i]
        else:
            a[i] = last
    # back fill leading
    first = a[idx[0]]
    a[:idx[0]] = first
    return a


def build_windows(cases, cfg, clin, Lc, H, stride, warmup, max_origins, dt, min_run,
                  quiet=True):
    """Return a list of window dicts across the given cases (identical origins to TiRex)."""
    past_names = [x for x in P.PAST]
    fut_names = list(P.FUTURE_COV)
    out = []
    for caseid in cases:
        rec = L.load_case(caseid, cfg, clin)
        if rec is None:
            continue
        tgt = rec["target"]; primary = rec["future_cov"].get(P.PRIMARY_COV)
        for t0 in P.make_windows(rec, Lc, H, stride, warmup, max_origins):
            # past block: MAP + past covariates over the context
            past = [tgt[t0 - Lc:t0]]
            for x in past_names:
                v = rec["past_cov"].get(x)
                past.append(v[t0 - Lc:t0] if v is not None else np.full(Lc, np.nan))
            past = np.stack([_finite_fill(c) for c in past], axis=1)          # [Lc, 1+P]
            # future block: known drug covariates over context+horizon
            fut = np.stack([_finite_fill(rec["future_cov"][c][t0 - Lc:t0 + H]) for c in fut_names], axis=1)  # [Lc+H, F]
            truth = np.asarray(tgt[t0:t0 + H], dtype=np.float64)              # [H] (may hold NaN)
            seg = primary[t0:t0 + H]
            stratum = "transition" if (np.nanmax(seg) - np.nanmin(seg)) > P.TRANSITION_THR else "steady"
            out.append(dict(caseid=caseid, t0=int(t0), stratum=stratum,
                            past=past.astype(np.float32), future=fut.astype(np.float32),
                            truth=truth.astype(np.float32),
                            t_event_65=P.time_to_hypo(truth, min_run, dt)))
        if not quiet:
            print(f"  [win] case {caseid}: cumulative {len(out)} windows", flush=True)
    return out, past_names, fut_names


def fit_norm(windows):
    """Per-channel mean/std from TRAIN windows (past, future, and target)."""
    past = np.concatenate([w["past"] for w in windows], axis=0)               # [N*Lc, 1+P]
    fut = np.concatenate([w["future"] for w in windows], axis=0)              # [N*(Lc+H), F]
    tgt = np.concatenate([w["truth"][np.isfinite(w["truth"])] for w in windows])
    def ms(a):
        m = np.nanmean(a, axis=0); s = np.nanstd(a, axis=0); s[s < 1e-6] = 1.0
        return m.astype(np.float32), s.astype(np.float32)
    pm, ps = ms(past); fm, fs = ms(fut)
    tm, ts = float(np.mean(tgt)), float(np.std(tgt) + 1e-6)
    return dict(past_mean=pm, past_std=ps, fut_mean=fm, fut_std=fs, tgt_mean=tm, tgt_std=ts)


def to_tensors(windows, norm, use_future=True):
    """Stack + normalise a list of windows into model-ready float32 arrays.

    use_future=False zeroes the future covariate channels (the M0 / no-drug-covariate arm)."""
    P_ = np.stack([w["past"] for w in windows])                              # [N, Lc, 1+P]
    F_ = np.stack([w["future"] for w in windows])                            # [N, Lc+H, F]
    Y_ = np.stack([w["truth"] for w in windows])                             # [N, H]
    P_ = (P_ - norm["past_mean"]) / norm["past_std"]
    F_ = (F_ - norm["fut_mean"]) / norm["fut_std"]
    if not use_future:
        F_ = np.zeros_like(F_)
    Yn = (Y_ - norm["tgt_mean"]) / norm["tgt_std"]
    mask = np.isfinite(Y_).astype(np.float32)
    Yn = np.nan_to_num(Yn, nan=0.0)
    return P_.astype(np.float32), F_.astype(np.float32), Yn.astype(np.float32), mask, Y_.astype(np.float32)
