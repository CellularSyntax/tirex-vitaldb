"""Definitive vasoactive-infusion prevalence across the ENTIRE VitalDB dataset via
vitaldb.find_cases (authoritative track-availability list, not our local download), then
cross-referenced against our study cohort. Supersedes scan_vaso.py, whose st_blocks>0 filter
under-counted while vital_files were iCloud-evicted placeholders.

Run:  PYTHONPATH=scripts <venv>/bin/python scripts/scan_vaso.py2   (network required for find_cases)
"""
import csv, json
import vitaldb

PRESSORS = {
    "Orchestra/VASO_RATE": "vasopressin",
    "Orchestra/NEPI_RATE": "norepinephrine",
    "Orchestra/PHEN_RATE": "phenylephrine",
    "Orchestra/EPI_RATE":  "epinephrine",
    "Orchestra/DOPA_RATE": "dopamine",
    "Orchestra/DOBU_RATE": "dobutamine",
}
ART = "Solar8000/ART_MBP"


def main():
    # our study cohort (included cases) + full "usable target" set from the manifest
    included = set()
    with open("datasets/vitaldb/cohort_manifest.csv") as fh:
        for r in csv.DictReader(fh):
            if r.get("include") == "1":
                included.add(str(int(r["caseid"])))
    print(f"cohort: {len(included)} included cases\n", flush=True)

    # authoritative full-dataset counts (find_cases returns caseids that have ALL listed tracks)
    art_cases = set(str(c) for c in vitaldb.find_cases([ART]))
    print(f"full dataset: {len(art_cases)} cases with an arterial line ({ART})\n", flush=True)

    rows = []
    for trk, drug in PRESSORS.items():
        cases = set(str(c) for c in vitaldb.find_cases([trk]))
        with_art = cases & art_cases
        in_cohort = cases & included
        rows.append({
            "track": trk, "drug": drug,
            "dataset_total": len(cases),
            "dataset_with_art": len(with_art),
            "in_our_cohort": len(in_cohort),
        })
        print(f"{drug:15s} {trk:22s}  dataset={len(cases):5d}  +art={len(with_art):5d}  "
              f"in_cohort={len(in_cohort):5d}", flush=True)

    with open("datasets/vitaldb/vaso_prevalence.json", "w") as fh:
        json.dump({"cohort_n": len(included), "art_dataset_n": len(art_cases), "pressors": rows},
                  fh, indent=1)
    print("\nwrote datasets/vitaldb/vaso_prevalence.json\nDONE", flush=True)


if __name__ == "__main__":
    main()
