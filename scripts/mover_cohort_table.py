"""MOVER cohort-characteristics for the two-cohort Table 1 (second column).

Mirrors scripts/paper_figures.py::table1_cohort but for the MOVER external cohort, whose
clinical_data.csv schema (caseid, subjectid, age, sex, height, weight, procedure, anestart,
aneend) carries age/sex/height/weight/duration/procedure but NOT ASA or department. BMI is
derived from height (cm) + weight (kg). The denominator is the set of cases that contributed
>=1 forecast window (results/ablation_windows_mover_art.csv), matching the VitalDB convention.

Run on the CLUSTER (which holds the MOVER cache, manifest and clinical_data.csv), from the
project root, in a python env with numpy:
    PYTHONPATH=scripts:datasets/mover python scripts/mover_cohort_table.py

Writes results/mover_cohort_characteristics.json  (pull to the Mac; the manuscript Table 1
second column is filled from it). Prints a human-readable summary.
"""
from __future__ import annotations
import csv, glob, json, os, sys
import numpy as np

WINDOWS = "results/ablation_windows_mover_art.csv"
CLINICAL = "datasets/mover/clinical_data.csv"
PRIMARY = "results/ablation_primary_mover_art.json"          # for n_windows + hypo prevalence
HORIZONS_PREV = [1, 5, 10, 15]


def fnum(r, k):
    try:
        return float(r[k])
    except (ValueError, TypeError, KeyError):
        return np.nan


def norm_caseid(c):
    """Normalise a caseid for cross-file joins. VitalDB ids are integers (strip zero
    padding via int); MOVER ids are hex hash strings (keep verbatim)."""
    s = str(c).strip()
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def windows_caseids(path):
    cids = set()
    for r in csv.DictReader(open(path)):
        cids.add(norm_caseid(r["caseid"]))
    return cids


def hkey(h, per):
    """Match the horizon-key spelling used in the primary JSON (e.g. '5min')."""
    for cand in (f"{h}min", str(h), f"{h}.0min"):
        if cand in per:
            return cand
    return f"{h}min"


def main():
    if not os.path.exists(WINDOWS):
        sys.exit(f"missing {WINDOWS} — run the MOVER ablation first")
    if not os.path.exists(CLINICAL):
        sys.exit(f"missing {CLINICAL} — build the MOVER cache first")

    keep = windows_caseids(WINDOWS)
    cd = {norm_caseid(r["caseid"]): r
          for r in csv.DictReader(open(CLINICAL, encoding="utf-8-sig"))}
    rows = [cd[c] for c in keep if c in cd]
    n = len(rows)
    if n == 0:
        sys.exit("no MOVER cases matched between windows and clinical_data.csv")

    ages = np.array([fnum(r, "age") for r in rows]); ages = ages[~np.isnan(ages)]
    ht = np.array([fnum(r, "height") for r in rows])          # cm
    wt = np.array([fnum(r, "weight") for r in rows])          # kg
    with np.errstate(divide="ignore", invalid="ignore"):
        bmi = wt / (ht / 100.0) ** 2
    bmi = bmi[np.isfinite(bmi) & (bmi > 5) & (bmi < 100)]
    dur = np.array([(fnum(r, "aneend") - fnum(r, "anestart")) / 60.0 for r in rows])
    dur = dur[np.isfinite(dur) & (dur > 0)]
    males = sum(1 for r in rows if str(r.get("sex", "")).strip().upper().startswith("M"))
    n_sex_known = sum(1 for r in rows if str(r.get("sex", "")).strip() not in ("", "nan", "None"))

    def iqr0(a):
        return None if len(a) == 0 else [round(float(np.median(a)), 1),
                                         round(float(np.percentile(a, 25)), 1),
                                         round(float(np.percentile(a, 75)), 1)]

    out = {
        "cohort": "MOVER",
        "denominator": "cases contributing >=1 forecast window",
        "n_cases": n,
        "age_median_iqr": iqr0(ages),
        "male_n": males, "male_pct": round(males / n * 100, 1) if n else None,
        "n_sex_known": n_sex_known,
        "bmi_median_iqr": iqr0(bmi), "n_bmi_known": int(len(bmi)),
        "anesthesia_min_median_iqr": iqr0(dur),
        "asa": None,          # MOVER SIS tables do not carry ASA
        "department": None,   # nor department
    }

    # forecast windows + hypotension prevalence from the primary JSON, if present
    if os.path.exists(PRIMARY):
        p = json.load(open(PRIMARY))
        out["n_windows"] = p.get("n_windows")
        per = p.get("per_horizon", {})
        prev = {}
        for h in HORIZONS_PREV:
            k = hkey(h, per)
            node = per.get(k, {})
            hy = node.get("hypo") or node.get("all", {})
            if "prevalence" in hy:
                prev[f"{h}min"] = round(hy["prevalence"] * 100, 1)
        out["hypo_prevalence_pct"] = prev

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/mover_cohort_characteristics.json", "w"), indent=2)

    print("MOVER cohort characteristics (n=%d cases contributing >=1 window)" % n)
    print("  age median (IQR)   :", out["age_median_iqr"])
    print("  male n (%%)         : %d (%.1f%%) of %d with sex known" % (males, out["male_pct"], n_sex_known))
    print("  BMI median (IQR)   : %s  (n=%d with height+weight)" % (out["bmi_median_iqr"], len(bmi)))
    print("  anaesthesia dur min:", out["anesthesia_min_median_iqr"])
    print("  ASA / department   : not recorded in MOVER SIS")
    print("  windows / prevalence:", out.get("n_windows"), out.get("hypo_prevalence_pct"))
    print("wrote results/mover_cohort_characteristics.json")


if __name__ == "__main__":
    main()
