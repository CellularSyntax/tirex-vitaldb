"""Generate a *curated* 3-example set for Fig 1c (steady / transition / hypotensive onset).

The phase3 run only dumps the first 12 windows of its first chunk, which rarely contains a
clean hypotensive-onset trace. Here we scan many cohort cases, pool candidate windows by
category from ground truth (no model), forecast M1 for the pool on CPU, then pick the
best-covered, best-tracked window in each category. Saves outputs/figs/examples_curated_<tag>.npz
with keys context/truth/q_ce/caseid/t0 (rows ordered: steady, transition, hypotensive).

Run:  PYTHONPATH=scripts:datasets/vitaldb /Users/admin/DATA/tirex2/venv/bin/python scripts/make_examples.py
"""
import csv, sys
import numpy as np, torch, yaml
import phase3_ablation as P
import vitaldb_loader as L

TAG = sys.argv[1] if len(sys.argv) > 1 else "all2873"
DEVICE = "cpu"
MAX_SCAN = int(sys.argv[2]) if len(sys.argv) > 2 else 240   # cases to scan
POOL_CAP = 45                                               # per-category candidate cap

ev = yaml.safe_load(open("configs/eval.yaml"))
cfg = L.load_config("datasets/vitaldb/configs/data.yaml"); clin = L._clinical_index(cfg["clinical_csv"])

# candidate cases: spread across the included cohort
man = [r for r in csv.DictReader(open("datasets/vitaldb/cohort_manifest.csv")) if r["include"] in ("1", "True", "true")]
ids = [r["caseid"] for r in man]
step = max(1, len(ids) // MAX_SCAN)
scan = ids[::step][:MAX_SCAN]

probe = next((r for r in (L.load_case(c, cfg, clin) for c in scan) if r is not None), None)
dt = probe["interval_s"]
Lc = int(ev["context_min"] * 60 / dt); H = max(int(m*60/dt) for m in P.HORIZON_STEPS_MIN)
stride = int(ev["origin_stride_min"] * 60 / dt); warmup = int(ev["warmup_min"] * 60 / dt)
min_run = max(1, int(ev.get("hypotension", {}).get("min_sustain_min", 1) * 60 / dt))
print(f"dt={dt} Lc={Lc} H={H} scanning {len(scan)} cases", flush=True)

pools = {"steady": [], "trans": [], "hypo": []}
recs = {}
for caseid in scan:
    rec = L.load_case(caseid, cfg, clin)
    if rec is None:
        continue
    for t0 in P.make_windows(rec, Lc, H, stride, warmup, max_origins=6):
        ctx = rec["target"][t0-Lc:t0]; truth = rec["target"][t0:t0+H]
        fc = ctx[np.isfinite(ctx)]; ft = truth[np.isfinite(truth)]
        if ft.size < 0.8*H or fc.size < 0.8*Lc:
            continue
        both = np.r_[fc, ft]
        if both.min() < 52 or both.max() > 165:          # drop artifacts
            continue
        last = fc[-1]; rng = ft.max()-ft.min(); tmin = ft.min()
        onset = P.time_to_hypo(truth, min_run, dt)
        if last >= 72 and np.isfinite(onset) and 1.5 <= onset <= 12 and len(pools["hypo"]) < POOL_CAP:
            pools["hypo"].append((caseid, t0))
        elif tmin >= 74 and rng <= 8 and len(pools["steady"]) < POOL_CAP:
            pools["steady"].append((caseid, t0))
        elif tmin >= 68 and 14 <= rng <= 45 and len(pools["trans"]) < POOL_CAP:
            pools["trans"].append((caseid, t0))
    recs[caseid] = rec
    if all(len(v) >= POOL_CAP for v in pools.values()):
        break
print({k: len(v) for k, v in pools.items()}, flush=True)

# forecast M1 for the whole pool
from tirex2 import load_model
model = load_model("NX-AI/TiRex-2", device=DEVICE)
allw = [(cat, cid, t0) for cat, lst in pools.items() for (cid, t0) in lst]
items = [P.build_ts(recs[cid], t0, Lc, H, use_past=True, use_future=True) for _, cid, t0 in allw]
print(f"forecasting {len(items)} candidate windows on {DEVICE} ...", flush=True)
fc = P.batched_forecast(model, items, H, bs=64)

# score each: coverage of 10-90 band + median tracking error
scored = {"steady": [], "trans": [], "hypo": []}
for (cat, cid, t0), q in zip(allw, fc):
    q = np.asarray(q)[0]                 # (9, H)
    truth = recs[cid]["target"][t0:t0+H]
    m = np.isfinite(truth)
    cov = float(np.mean((truth[m] >= q[0][m]) & (truth[m] <= q[8][m])))
    medmae = float(np.mean(np.abs(q[P.MED][m] - truth[m])))
    band_dips = float(np.nanmin(q[0])) < 65        # lower band anticipates hypotension
    scored[cat].append(dict(cid=cid, t0=t0, q=q, cov=cov, medmae=medmae, dips=band_dips))

def best(cat, key):
    return sorted(scored[cat], key=key)[0]
pick = {
    "steady": best("steady", lambda s: (-s["cov"], s["medmae"])),
    "trans":  best("trans",  lambda s: (s["medmae"], -s["cov"])),
    # hypo: want good coverage AND the band to anticipate the dip, then low median error
    "hypo":   best("hypo",   lambda s: (0 if (s["cov"] >= 0.75 and s["dips"]) else 1, s["medmae"])),
}
order = ["steady", "trans", "hypo"]
out = {k: [] for k in ("caseid", "t0", "context", "truth", "q_ce")}
for cat in order:
    s = pick[cat]; cid, t0 = s["cid"], s["t0"]
    out["caseid"].append(cid); out["t0"].append(t0)
    out["context"].append(recs[cid]["target"][t0-Lc:t0])
    out["truth"].append(recs[cid]["target"][t0:t0+H])
    out["q_ce"].append(s["q"])
    print(f"  {cat:7s} case={cid} t0={t0} cov={s['cov']:.2f} medMAE={s['medmae']:.1f} dips={s['dips']}", flush=True)
for k in out:
    out[k] = np.array(out[k], dtype=object if k == "caseid" else float)
np.savez_compressed(f"outputs/figs/examples_curated_{TAG}.npz", **out)
print(f"wrote outputs/figs/examples_curated_{TAG}.npz", flush=True)
