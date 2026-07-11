"""Clean, cached VitalDB loader for the TiRex-2 hemodynamic-response study (Phase 1).

Per case, yields aligned, artifact-masked, uniformly-resampled series:
  - target            : MAP (Solar8000/ART_MBP), physiologically masked
  - future_cov[name]  : drug infusion channels (known over the whole case)
  - past_cov[name]    : HR / SpO2 / CVP, masked
plus metadata + coverage stats. Robust to iCloud-evicted files (returns None on I/O error).

Design notes (see notes/DATA_NOTES.md):
  - Numerics are ~0.5 Hz; resample to `interval_s` (default 5 s) -> ~uniform grid.
  - Drug channels are step functions / smooth CE; resampled by last-value hold (no smooth interp).
  - Artifacts (art-line zeroing, fast-flush) -> NaN via plausibility ranges; TiRex masks NaN natively.
  - Windowing for rolling-origin evaluation is done downstream (Phase 2/3), not here.

CLI:  python scripts/vitaldb_loader.py 2521 [--no-cache] [--plot]
"""
from __future__ import annotations
import argparse, csv, os
from typing import Optional
import numpy as np
import yaml


def load_config(path: str = "configs/data.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _clinical_index(clinical_csv: str) -> dict[str, dict]:
    return {r["caseid"]: r for r in csv.DictReader(open(clinical_csv, encoding="utf-8-sig"))}


def _is_local(path: str) -> bool:
    try:
        return os.stat(path).st_blocks > 0
    except OSError:
        return False


def _mask_range(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = x.astype(np.float32).copy()
    bad = ~np.isfinite(x) | (x < lo) | (x > hi)
    x[bad] = np.nan
    return x


def _despike(x: np.ndarray, window: int = 7, thr: float = 25.0) -> np.ndarray:
    """NaN out points that deviate from the local (rolling) median by > thr (fast-flush artifacts).
    Kapral used plausibility masking + forward-fill (no smoothing); this is a stricter plausibility
    check that removes arterial-line flush/transient spikes the range mask misses. Applied at load."""
    x = x.astype(np.float32).copy()
    n = len(x)
    if n < window:
        return x
    from numpy.lib.stride_tricks import sliding_window_view
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    with np.errstate(all="ignore"):
        med = np.nanmedian(sliding_window_view(xp, window), axis=1)
    bad = np.isfinite(x) & np.isfinite(med) & (np.abs(x - med) > thr)
    x[bad] = np.nan
    return x


def _block_agg(x: np.ndarray, k: int, how: str) -> np.ndarray:
    n = (len(x) // k) * k
    xb = x[:n].reshape(-1, k)
    with np.errstate(all="ignore"):
        if how == "median":
            return np.nanmedian(xb, axis=1)
        if how == "mean":
            return np.nanmean(xb, axis=1)
        if how == "last":
            return xb[:, -1]
    raise ValueError(how)


def _finalize(rec: dict, cfg: dict) -> dict:
    """Load-time transforms on both fresh and cached records (no cache rebuild needed):
    despike the target, then optionally downsample to a coarser grid (resample_to_s)."""
    if rec is None:
        return rec
    dsp = cfg.get("despike")
    if dsp and dsp.get("enabled", False):
        rec["target"] = _despike(rec["target"], int(dsp.get("window", 7)), float(dsp.get("threshold_mmHg", 25)))

    rs = cfg.get("resample_to_s")
    if rs and float(rs) > rec["interval_s"]:
        k = int(round(float(rs) / rec["interval_s"]))
        if k > 1:
            rec["target"] = _block_agg(rec["target"], k, "median")     # denoise + robust
            rec["future_cov"] = {n: _block_agg(v, k, "last") for n, v in rec["future_cov"].items()}  # step-hold
            rec["past_cov"] = {n: _block_agg(v, k, "median") for n, v in rec["past_cov"].items()}
            rec["time_min"] = rec["time_min"][::k][:len(rec["target"])]
            rec["interval_s"] = rec["interval_s"] * k
            rec["n"] = len(rec["target"])
    rec["target_coverage"] = float(np.isfinite(rec["target"]).mean())
    return rec


def load_case(caseid: str, cfg: dict, clinical: Optional[dict] = None,
              use_cache: bool = True) -> Optional[dict]:
    """Return an aligned per-case record, or None if the file is unreadable/insufficient."""
    caseid = str(caseid).zfill(4)
    cache_path = os.path.join(cfg["cache_dir"], f"{caseid}.npz")
    if use_cache and os.path.exists(cache_path):
        return _finalize(_from_npz(cache_path), cfg)

    path = os.path.join(cfg["vital_dir"], f"{caseid}.vital")
    if not _is_local(path):
        return None  # iCloud-evicted; skip per Max's decision to work local-only

    tgt = cfg["tracks"]["target"]
    fut = cfg["tracks"]["future_cov"]
    pst = cfg["tracks"]["past_cov"]
    want = [tgt] + list(fut) + list(pst)

    try:
        import vitaldb
        vf = vitaldb.VitalFile(path, track_names=want)
        present = set(vf.get_track_names())
        arr = vf.to_numpy(want, interval=float(cfg["interval_s"]))  # (T, n)
    except Exception:  # noqa: BLE001  (iCloud timeout / corrupt / missing track)
        return None

    T = arr.shape[0]
    dt = float(cfg["interval_s"])
    time_s = np.arange(T) * dt

    # anesthesia-window trim
    clin = (clinical or {}).get(str(int(caseid)), None)
    win = (0, T)
    if cfg.get("trim_to_anesthesia") and clin and clin.get("anestart") and clin.get("aneend"):
        a0 = max(0.0, float(clin["anestart"]))
        a1 = float(clin["aneend"])
        i0, i1 = int(a0 / dt), min(T, int(a1 / dt))
        if i1 - i0 > 10:
            win = (i0, i1)
    s, e = win

    ranges = cfg["plausible_range"]
    col = {name: arr[s:e, i] for i, name in enumerate(want)}

    def get_masked(name):
        if name not in present:
            return np.full(e - s, np.nan, np.float32)
        if name in ranges:
            return _mask_range(col[name], *ranges[name])
        return col[name].astype(np.float32)

    def get_drug(name):
        if name not in present:
            return np.full(e - s, np.nan, np.float32)
        x = col[name].astype(np.float32).copy()
        x[x < cfg.get("drug_min", 0.0)] = cfg.get("drug_min", 0.0)  # clip pump negatives
        return x

    target = get_masked(tgt)
    future_cov = {name: get_drug(name) for name in fut}
    past_cov = {name: get_masked(name) for name in pst}

    cov = float(np.isfinite(target).mean())
    rec = {
        "caseid": caseid,
        "interval_s": dt,
        "time_min": (time_s[s:e] - time_s[s]) / 60.0,
        "target_name": tgt,
        "target": target,
        "future_cov": future_cov,
        "past_cov": past_cov,
        "target_coverage": cov,
        "n": e - s,
        "meta": {k: (clin.get(k) if clin else None)
                 for k in ("age", "sex", "department", "optype", "ane_type", "asa")},
    }
    if use_cache:
        _to_npz(cache_path, rec)   # cache stores range-masked (raw); despike applied at load
    return _finalize(rec, cfg)


def usable(rec: Optional[dict], cfg: dict) -> bool:
    return rec is not None and rec["target_coverage"] >= cfg.get("min_target_coverage", 0.6)


def _to_npz(path: str, rec: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat = {
        "caseid": rec["caseid"], "interval_s": rec["interval_s"],
        "time_min": rec["time_min"], "target": rec["target"],
        "target_name": rec["target_name"], "target_coverage": rec["target_coverage"],
        "future_names": np.array(list(rec["future_cov"])),
        "future_cov": np.stack(list(rec["future_cov"].values())) if rec["future_cov"] else np.zeros((0, rec["n"])),
        "past_names": np.array(list(rec["past_cov"])),
        "past_cov": np.stack(list(rec["past_cov"].values())) if rec["past_cov"] else np.zeros((0, rec["n"])),
        "meta_keys": np.array(list(rec["meta"])),
        "meta_vals": np.array([str(v) for v in rec["meta"].values()]),
    }
    np.savez_compressed(path, **flat)


def _from_npz(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    return {
        "caseid": str(d["caseid"]), "interval_s": float(d["interval_s"]),
        "time_min": d["time_min"], "target": d["target"], "target_name": str(d["target_name"]),
        "target_coverage": float(d["target_coverage"]), "n": len(d["target"]),
        "future_cov": {n: d["future_cov"][i] for i, n in enumerate(d["future_names"])},
        "past_cov": {n: d["past_cov"][i] for i, n in enumerate(d["past_names"])},
        "meta": {k: v for k, v in zip(d["meta_keys"], d["meta_vals"])},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("caseid")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    clinical = _clinical_index(cfg["clinical_csv"])
    rec = load_case(args.caseid, cfg, clinical, use_cache=not args.no_cache)
    if rec is None:
        print(f"case {args.caseid}: unreadable / evicted / missing tracks"); return

    print(f"case {rec['caseid']}: n={rec['n']} @ {rec['interval_s']}s = {rec['n']*rec['interval_s']/60:.0f} min")
    print(f"  target {rec['target_name']}: coverage {100*rec['target_coverage']:.0f}%  "
          f"median {np.nanmedian(rec['target']):.0f} mmHg  usable={usable(rec, cfg)}")
    for name, x in rec["future_cov"].items():
        print(f"  future_cov {name:22s}: nonNaN {100*np.isfinite(x).mean():3.0f}%  max {np.nanmax(x):.2f}")
    for name, x in rec["past_cov"].items():
        print(f"  past_cov   {name:22s}: nonNaN {100*np.isfinite(x).mean():3.0f}%  median {np.nanmedian(x):.1f}")
    print(f"  meta: {rec['meta']}")

    if args.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        tm = rec["time_min"]
        fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
        ax[0].plot(tm, rec["target"], color="C3", lw=0.8); ax[0].axhline(65, ls=":", c="k", lw=0.8)
        ax[0].set_ylabel(rec["target_name"]); ax[0].set_ylim(0, 160)
        for name, x in rec["future_cov"].items():
            ax[1].plot(tm, x, lw=0.7, label=name.split("/")[-1])
        ax[1].legend(fontsize=7); ax[1].set_xlabel("time (min)"); ax[1].set_ylabel("drug covariate")
        fig.suptitle(f"case {rec['caseid']} (loader output)"); fig.tight_layout()
        out = f"outputs/case{rec['caseid']}_loader.png"; fig.savefig(out, dpi=110)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
