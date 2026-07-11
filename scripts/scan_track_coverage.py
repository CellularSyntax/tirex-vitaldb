"""Header-only track-presence scan across VitalDB .vital files.

Determines which cases have a MAP target + a drug-infusion covariate channel, to drive cohort
selection (§3.1) and the covariate choice (§3.2). Robust to iCloud-evicted placeholders (block-count
check + per-file try/except). Header-only read (~0.9 s/local file).

Usage:
  python scripts/scan_track_coverage.py --n 300          # random sample
  python scripts/scan_track_coverage.py --all            # full scan (LONG: ~0.9s x n_local files)
  python scripts/scan_track_coverage.py --all --out results/track_coverage.csv
"""
from __future__ import annotations
import argparse, glob, os, random, time
from collections import Counter
import vitaldb

DRUGS = ["PPF20_RATE", "PPF20_CE", "RFTN20_RATE", "RFTN20_CE", "RFTN50_RATE", "NEPI_RATE",
         "PHEN_RATE", "EPI_RATE", "VASO_RATE", "DOPA_RATE", "DOBU_RATE", "NTG_RATE",
         "DEX2_RATE", "DEX4_RATE"]


def is_local(path: str) -> bool:
    """True if the file is materialized on disk (not an iCloud placeholder)."""
    try:
        return os.stat(path).st_blocks > 0
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="vital_files")
    ap.add_argument("--n", type=int, default=300, help="random sample size (ignored with --all)")
    ap.add_argument("--all", action="store_true", help="scan every local file")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None, help="optional CSV of per-case track presence")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.vital")))
    local = [f for f in files if is_local(f)]
    print(f"{len(files)} files, {len(local)} materialized locally, {len(files) - len(local)} evicted")

    if args.all:
        sample = local
    else:
        random.seed(args.seed)
        sample = random.sample(local, min(args.n, len(local)))

    has = Counter()
    ok = err = 0
    rows = []
    t0 = time.time()
    for f in sample:
        try:
            tset = set(vitaldb.vital_trks(f))
            ok += 1
        except Exception:  # noqa: BLE001  (iCloud timeout / corrupt header)
            err += 1
            continue
        caseid = os.path.splitext(os.path.basename(f))[0]
        hm = "Solar8000/ART_MBP" in tset
        if hm:
            has["ART_MBP"] += 1
        if "Solar8000/NIBP_MBP" in tset:
            has["NIBP_MBP"] += 1
        if "SNUADC/ART" in tset:
            has["ART_wave"] += 1
        present = {}
        for d in DRUGS:
            p = ("Orchestra/" + d) in tset
            present[d] = p
            if p:
                has[d] += 1
            if hm and p:
                has["MBP+" + d] += 1
        if hm and any(present[d] for d in DRUGS):
            has["MBP+any_infusion"] += 1
        if args.out:
            rows.append((caseid, int(hm), *[int(present[d]) for d in DRUGS]))

    n = max(ok, 1)
    dt = time.time() - t0
    print(f"scanned {ok} ({err} errors) in {dt:.0f}s ({dt / n * 1000:.0f} ms/file)\n")
    print(f"ART_MBP {has['ART_MBP']} ({100*has['ART_MBP']//n}%)  "
          f"NIBP_MBP {has['NIBP_MBP']} ({100*has['NIBP_MBP']//n}%)  "
          f"ART_wave {has['ART_wave']} ({100*has['ART_wave']//n}%)")
    print(f"ART_MBP + any infusion: {has['MBP+any_infusion']} ({100*has['MBP+any_infusion']//n}%)\n")
    print(f"{'drug':14s}{'present':>9s}{'+ART_MBP':>10s}")
    for d in DRUGS:
        print(f"{d:14s}{has[d]:9d}{has['MBP+'+d]:10d}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["caseid", "ART_MBP", *DRUGS])
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
