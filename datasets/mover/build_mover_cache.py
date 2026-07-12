"""Build the MOVER (SIS) per-case cache for the TiRex-2 pipeline (one-time preprocessing).

Streams the raw SIS tables once, resamples each case onto a uniform 60 s grid over its OR window,
derives per-minute infusion RATE trajectories (Dose / segment-duration, piecewise-constant), and
writes one .npz per PID in the VitalDB cache format — but keyed by VitalDB CANONICAL channel names
(HRe->Solar8000/HR, Propofol drip->Orchestra/PPF20_RATE, ...) so the whole downstream pipeline runs
unchanged. Also writes clinical_data.csv (demographics + OR times) and cohort_manifest.csv.

Run (cluster, CPU): PYTHONPATH=datasets/mover python datasets/mover/build_mover_cache.py \
                        --config datasets/mover/configs/data.yaml
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from datetime import datetime
import numpy as np
import yaml

csv.field_size_limit(1 << 24)
NULLS = {"", "\\N", "NA", "NaN", "null", "None"}


def parse_dt(s):
    """Parse either ISO 'YYYY-MM-DD HH:MM:SS' or SIS 'M/D/YY H:MM' -> datetime, else None."""
    if s is None or s.strip() in NULLS:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%y %H:%M", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def fnum(s):
    if s is None or s.strip() in NULLS:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _idx(cols, name):
    return cols.index(name) if name in cols else None


def _reader(path):
    f = open(path, newline="", encoding="utf-8", errors="replace")
    r = csv.reader(f)
    header = next(r)
    return f, r, header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="datasets/mover/configs/data.yaml")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap number of cases written")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    M = cfg["mover"]; sis = cfg["sis_dir"]; dt = float(cfg["interval_s"])
    ranges = cfg["plausible_range"]
    drug_map = M["drug_map"]; vitals_map = M["vitals_map"]; obs_map = M["obs_map"]
    target_src = M["target_source"]                          # 'MAP_ART' or 'nMAP'
    fut_channels = sorted(set(drug_map.values()))            # canonical future_cov channels
    past_channels = ["Solar8000/HR", "Solar8000/PLETH_SPO2", "Solar8000/CVP"]
    os.makedirs(cfg["cache_dir"], exist_ok=True)

    # ── 1. patient_information: demographics + OR window ───────────────────────────────
    info = {}
    f, r, h = _reader(f"{sis}/patient_information.csv")
    ci = {c: _idx(h, c) for c in ("PID", "Age", "Ht", "Wt", "Gender", "Procedure", "OR_start", "OR_end")}
    for row in r:
        pid = row[ci["PID"]]
        o0, o1 = parse_dt(row[ci["OR_start"]]), parse_dt(row[ci["OR_end"]])
        if not pid or o0 is None or o1 is None or (o1 - o0).total_seconds() < 600:
            continue
        info[pid] = dict(age=row[ci["Age"]], sex=row[ci["Gender"]], height=row[ci["Ht"]],
                         weight=row[ci["Wt"]], procedure=row[ci["Procedure"]], t0=o0, t1=o1,
                         wt=fnum(row[ci["Wt"]]))
    f.close()
    print(f"[mover] {len(info)} cases with valid OR window", flush=True)

    # ── 2. observations: invasive MAP_ART + CVPm ; vitals: HRe/SP02/nMAP ───────────────
    # accumulate per-PID lists of (t_seconds_from_or_start, value) per canonical channel
    acc = {}   # pid -> {canonical_channel: [(sec, val), ...], 'target': [...]}

    def add(pid, chan, t, v):
        if pid not in info or np.isnan(v):
            return
        sec = (t - info[pid]["t0"]).total_seconds()
        acc.setdefault(pid, {}).setdefault(chan, []).append((sec, v))

    # observations (invasive) — target MAP_ART lives here (or MAP_ART used as target_src)
    f, r, h = _reader(f"{sis}/patient_observations.csv")
    oi = {"PID": _idx(h, "PID"), "t": _idx(h, "Obs_time")}
    oi.update({col: _idx(h, col) for col in obs_map})
    for row in r:
        pid = row[oi["PID"]]; t = parse_dt(row[oi["t"]])
        if pid not in info or t is None:
            continue
        for col, chan in obs_map.items():
            if oi[col] is None:
                continue
            v = fnum(row[oi[col]])
            chan_ = "target" if col == target_src else chan
            add(pid, chan_, t, v)
    f.close(); print(f"[mover] observations scanned ({len(acc)} PIDs so far)", flush=True)

    # vitals (HR/SpO2/NIBP) — nMAP is target if target_source == nMAP
    f, r, h = _reader(f"{sis}/patient_vitals.csv")
    vi = {"PID": _idx(h, "PID"), "t": _idx(h, "Obs_time")}
    vi.update({col: _idx(h, col) for col in list(vitals_map) + ["nMAP"]})
    for row in r:
        pid = row[vi["PID"]]; t = parse_dt(row[vi["t"]])
        if pid not in info or t is None:
            continue
        for col, chan in vitals_map.items():
            if vi[col] is not None:
                add(pid, chan, t, fnum(row[vi[col]]))
        if target_src == "nMAP" and vi["nMAP"] is not None:
            add(pid, "target", t, fnum(row[vi["nMAP"]]))
    f.close(); print(f"[mover] vitals scanned", flush=True)

    # ── 3. medication: infusion segments -> per-minute rate ────────────────────────────
    seg = {}   # pid -> [(start_sec, end_sec, rate, canonical_channel), ...]
    has_infusion = set()
    f, r, h = _reader(f"{sis}/patient_medication.csv")
    mi = {c: _idx(h, c) for c in ("PID", "Start_time", "End_time", "Dose", "Drug_name")}
    for row in r:
        pid = row[mi["PID"]]; drug = (row[mi["Drug_name"]] or "").strip()
        if pid not in info or drug not in drug_map:
            continue
        s, e = parse_dt(row[mi["Start_time"]]), parse_dt(row[mi["End_time"]])
        dose = fnum(row[mi["Dose"]])
        if s is None or e is None or np.isnan(dose):      # boluses (no End) don't form a rate
            continue
        dur_min = (e - s).total_seconds() / 60.0
        if dur_min <= 0:
            continue
        rate = dose / dur_min                              # per-minute rate (mcg/min etc.)
        s0 = (s - info[pid]["t0"]).total_seconds(); e0 = (e - info[pid]["t0"]).total_seconds()
        seg.setdefault(pid, []).append((s0, e0, rate, drug_map[drug]))
        has_infusion.add(pid)
    f.close(); print(f"[mover] medication scanned ({len(has_infusion)} PIDs with a drip)", flush=True)

    # ── 4. per-PID: build 60 s grid, resample, mask, write npz ─────────────────────────
    def grid_mean(pairs, n):
        """bin (sec,val) pairs onto the 60 s grid -> mean per bin, NaN where empty."""
        s = np.zeros(n); c = np.zeros(n)
        for sec, v in pairs:
            k = int(round(sec / dt))
            if 0 <= k < n and np.isfinite(v):
                s[k] += v; c[k] += 1
        out = np.full(n, np.nan); m = c > 0; out[m] = s[m] / c[m]
        return out

    def mask_range(x, lo, hi):
        x = x.astype(np.float32)
        x[~np.isfinite(x) | (x < lo) | (x > hi)] = np.nan
        return x

    manifest = []; written = 0
    min_n = int(M["min_map_minutes"] * 60 / dt)
    for pid, meta in info.items():
        n = int(round((meta["t1"] - meta["t0"]).total_seconds() / dt))
        if n < min_n:
            manifest.append((pid, 0)); continue
        A = acc.get(pid, {})
        target = mask_range(grid_mean(A.get("target", []), n), *ranges[cfg["tracks"]["target"]])
        past = {}
        for chan in past_channels:
            past[chan] = mask_range(grid_mean(A.get(chan, []), n), *ranges[chan])
        # future (drug) channels: 0 = pump off (a KNOWN value), rate over each segment
        future = {chan: np.zeros(n, np.float32) for chan in fut_channels}
        for s0, e0, rate, chan in seg.get(pid, []):
            k0 = max(0, int(round(s0 / dt))); k1 = min(n, int(round(e0 / dt)))
            if k1 > k0:
                future[chan][k0:k1] = rate
        cov = float(np.isfinite(target).mean())
        infusion_ok = (pid in has_infusion) if M.get("require_infusion", True) else True
        include = int(cov >= cfg.get("min_target_coverage", 0.5) and n >= min_n and infusion_ok)
        manifest.append((pid, include))
        if not include:
            continue
        np.savez_compressed(
            f"{cfg['cache_dir']}/{pid}.npz",
            caseid=pid, interval_s=dt, time_min=(np.arange(n) * dt / 60.0).astype(np.float32),
            target=target.astype(np.float32), target_name=cfg["tracks"]["target"], target_coverage=cov,
            future_names=np.array(fut_channels), future_cov=np.stack([future[c] for c in fut_channels]),
            past_names=np.array(past_channels), past_cov=np.stack([past[c] for c in past_channels]),
            meta_keys=np.array(["age", "sex", "procedure", "weight"]),
            meta_vals=np.array([str(meta["age"]), str(meta["sex"]), str(meta["procedure"]), str(meta["weight"])]))
        written += 1
        if args.limit and written >= args.limit:
            break
        if written % 200 == 0:
            print(f"[mover] wrote {written} caches ...", flush=True)

    # ── 5. clinical_data.csv + cohort_manifest.csv ─────────────────────────────────────
    with open(cfg["clinical_csv"], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["caseid", "subjectid", "age", "sex", "height", "weight", "procedure",
                    "anestart", "aneend"])
        for pid, meta in info.items():
            dur = (meta["t1"] - meta["t0"]).total_seconds()
            w.writerow([pid, pid, meta["age"], meta["sex"], meta["height"], meta["weight"],
                        meta["procedure"], 0, int(dur)])
    with open(cfg["cohort_manifest"], "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["caseid", "include"])
        w.writerows(manifest)
    n_incl = sum(i for _, i in manifest)
    print(f"[mover] DONE: {written} caches written; {n_incl} cases included / {len(manifest)} candidates.", flush=True)
    print(f"[mover] wrote {cfg['clinical_csv']} and {cfg['cohort_manifest']}", flush=True)
    # cohort funnel for Fig 1b (pulled down to build the two-cohort curation panel locally)
    os.makedirs("results", exist_ok=True)
    json.dump({"n_or_window": len(info), "n_with_map": len(acc), "n_with_infusion": len(has_infusion),
               "included_N": n_incl, "n_cached": written,
               "thresholds": {"min_map_minutes": M["min_map_minutes"],
                              "min_target_coverage": cfg.get("min_target_coverage", 0.5),
                              "require_infusion": M.get("require_infusion", True)}},
              open("results/mover_cohort_flow.json", "w"), indent=2)
    print("[mover] wrote results/mover_cohort_flow.json", flush=True)


if __name__ == "__main__":
    main()
