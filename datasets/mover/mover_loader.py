"""MOVER (SIS) loader — same API as vitaldb_loader, reading the per-PID cache built by
build_mover_cache.py. Records use VitalDB CANONICAL channel names, so phase3 / baselines /
zeroshot / figures run unchanged. Only load-time transforms (despike, optional downsample)
are applied here; range-masking + gridding + rate-derivation happen once at cache build.

CLI:  PYTHONPATH=datasets/mover python datasets/mover/mover_loader.py <PID> --config datasets/mover/configs/data.yaml
"""
from __future__ import annotations
import argparse, csv, os
from typing import Optional
import numpy as np
import yaml


def load_config(path: str = "datasets/mover/configs/data.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _clinical_index(clinical_csv: str) -> dict:
    return {r["caseid"]: r for r in csv.DictReader(open(clinical_csv, encoding="utf-8-sig"))}


def _despike(x, window=5, thr=30.0):
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.asarray(x, np.float32).copy(); n = len(x)
    if n < window:
        return x
    pad = window // 2; xp = np.pad(x, pad, mode="edge")
    with np.errstate(all="ignore"):
        med = np.nanmedian(sliding_window_view(xp, window), axis=1)
    bad = np.isfinite(x) & np.isfinite(med) & (np.abs(x - med) > thr)
    x[bad] = np.nan
    return x


def _block_agg(x, k, how):
    n = (len(x) // k) * k; xb = x[:n].reshape(-1, k)
    with np.errstate(all="ignore"):
        return {"median": np.nanmedian, "mean": np.nanmean}[how](xb, axis=1) if how != "last" else xb[:, -1]


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


def _finalize(rec: dict, cfg: dict) -> dict:
    if rec is None:
        return rec
    dsp = cfg.get("despike")
    if dsp and dsp.get("enabled", False):
        rec["target"] = _despike(rec["target"], int(dsp.get("window", 5)), float(dsp.get("threshold_mmHg", 30)))
    rs = cfg.get("resample_to_s")
    if rs and float(rs) > rec["interval_s"]:
        k = int(round(float(rs) / rec["interval_s"]))
        if k > 1:
            rec["target"] = _block_agg(rec["target"], k, "median")
            rec["future_cov"] = {n: _block_agg(v, k, "last") for n, v in rec["future_cov"].items()}
            rec["past_cov"] = {n: _block_agg(v, k, "median") for n, v in rec["past_cov"].items()}
            rec["time_min"] = rec["time_min"][::k][:len(rec["target"])]
            rec["interval_s"] *= k; rec["n"] = len(rec["target"])
    rec["target_coverage"] = float(np.isfinite(rec["target"]).mean())
    return rec


def load_case(caseid: str, cfg: dict, clinical: Optional[dict] = None, use_cache: bool = True) -> Optional[dict]:
    """Return the aligned per-case record from cache, or None if not cached (skipped at build)."""
    path = os.path.join(cfg["cache_dir"], f"{caseid}.npz")
    if not os.path.exists(path):
        return None
    return _finalize(_from_npz(path), cfg)


def usable(rec, cfg):
    return rec is not None and rec["target_coverage"] >= cfg.get("min_target_coverage", 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("caseid")
    ap.add_argument("--config", default="datasets/mover/configs/data.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rec = load_case(args.caseid, cfg, _clinical_index(cfg["clinical_csv"]))
    if rec is None:
        print(f"PID {args.caseid}: not in cache"); return
    print(f"PID {rec['caseid']}: n={rec['n']} @ {rec['interval_s']}s = {rec['n']*rec['interval_s']/60:.0f} min")
    print(f"  target {rec['target_name']}: coverage {100*rec['target_coverage']:.0f}%  median {np.nanmedian(rec['target']):.0f} mmHg")
    for name, x in rec["future_cov"].items():
        nz = np.mean(np.asarray(x) > 0) * 100
        print(f"  future {name:24s}: {nz:3.0f}% on, max {np.nanmax(x):.1f}")
    for name, x in rec["past_cov"].items():
        print(f"  past   {name:24s}: nonNaN {100*np.isfinite(x).mean():3.0f}%  median {np.nanmedian(x):.1f}")
    print(f"  meta: {rec['meta']}")


if __name__ == "__main__":
    main()
