"""Prevalence of vasoactive INFUSION tracks across local .vital files (header-only, parallel).
Answers: how many VitalDB cases actually record vasopressin/norepi/phenyleph/etc. as a continuous
infusion trajectory (not bolus), and how many co-occur with an arterial line (ART_MBP)."""
import os, glob
from collections import Counter
from multiprocessing import Pool
import vitaldb

TRK = ["Orchestra/VASO_RATE", "Orchestra/NEPI_RATE", "Orchestra/PHEN_RATE", "Orchestra/EPI_RATE",
       "Orchestra/DOPA_RATE", "Orchestra/DOBU_RATE", "Orchestra/NTG_RATE", "Orchestra/RFTN20_CE"]


def scan(f):
    try:
        ts = set(vitaldb.vital_trks(f))
    except Exception:
        return None
    return (ts & set(TRK), "Solar8000/ART_MBP" in ts)


def main():
    files = [f for f in sorted(glob.glob("vital_files/*.vital")) if os.stat(f).st_blocks > 0]
    print(f"scanning {len(files)} local files...", flush=True)
    present, with_art = Counter(), Counter()
    n = nart = 0
    with Pool(8) as p:
        for r in p.imap_unordered(scan, files, chunksize=16):
            if r is None:
                continue
            n += 1
            trks, hasart = r
            if hasart:
                nart += 1
            for t in trks:
                present[t] += 1
                if hasart:
                    with_art[t] += 1
    print(f"SCANNED {n} readable; {nart} have ART_MBP\n", flush=True)
    for t in TRK:
        print(f"{t:22s} present {present[t]:5d} ({100*present[t]/max(n,1):.1f}%)  "
              f"+ART_MBP {with_art[t]:5d} ({100*with_art[t]/max(nart,1):.1f}% of art-line)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
