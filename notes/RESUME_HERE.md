# RESUME HERE (paused 2026-07-11 to fix iCloud sync)

## Why we paused
iCloud Desktop sync repeatedly evicted files to placeholders (→ `.vital` read timeouts, an evicted
`.venv`, disk pressure). Max is deactivating Desktop/Documents/Downloads sync and re-pulling the
`Postdoc/2026` folder to a non-synced location so work continues cleanly.

## Environment facts for the iCloud fix
- **Safe, already OFF iCloud (do not touch):** venv `/Users/admin/DATA/tirex2/venv`, loader cache
  `/Users/admin/DATA/tirex2/cache` (2710 npz). These use ABSOLUTE paths in `configs/data.yaml`, so they
  keep working even if the project folder moves.
- **Project folder is portable:** `vital_dir: vital_files` is relative → the 88 GB `vital_files/` travels
  with the project; `cache_dir`/venv are absolute to `/Users/admin/DATA`. So you can MOVE the whole
  project off Desktop and just run from the new location with the same venv. Nothing to reconfigure.
- ⚠️ **Disk caveat:** turning off "Desktop & Documents in iCloud" with "Keep a Copy" tries to download
  ALL evicted files (88 GB vital_files + package files). Free disk was ~19–35 GB — ensure enough space
  or the download may fail/stall. Moving the folder to an external/local drive is an alternative.
- ⚠️ **Preserve today's edits:** everything from this session is on the LOCAL Desktop copy. When
  toggling sync, choose to KEEP LOCAL so a stale cloud version doesn't overwrite `notes/`, `scripts/`,
  `configs/`, `results/cohort_manifest.csv`.

## How to run (after resume)
```
cd <project>                       # wherever it now lives
PY=/Users/admin/DATA/tirex2/venv/bin/python
PYTHONPATH=scripts $PY scripts/phase3_ablation.py --n-cases 300 --seed 1 --n-shards 5 --shard-idx <i>   # 5 shards i=0..4
PYTHONPATH=scripts $PY scripts/merge_dashboard.py n300_s1                                                # merge -> dashboard + hypotension figs
```
(HF_TOKEN already logged in via `~/.cache/huggingface/token`. device=cpu on Mac.)

## State of the work
**DONE & validated:**
- Phase 0 (API), Phase 1 (loader, cohort **[N]=2659**, infusion-fidelity gate PASS), Phase 2 PoC.
- 15 s resample + despike; rolling-origin harness; **covariate ablation runs** (interim 25-case: effect
  ~0 overall, **+3–4 % in transition windows**; TiRex zero-shot MAE ≈ Kapral external).
- Sharded CPU runner (`--n-shards`) + `merge_dashboard.py`. HPC/CUDA port guide `notes/HPC_PORTING.md`.

**IMPLEMENTED but NOT yet validated (smoke test got killed at pause):**
- **Hypotension task [Z]** (MAP<65) added to `phase3_ablation.py`: per-window event + risk from forecast
  quantiles, AUROC/AUPRC per horizon (M1 vs M0), case-clustered bootstrap; `plot_hypotension` +
  FOILS_AUROC (Kapral 0.909/0.903, Zhu 0.904). **First step on resume: smoke-test it**
  (`--n-cases 10 --seed 7 --max-origins 12`) before the big run.

**PENDING / open threads:**
- **Vasopressor availability — RESOLVED 2026-07-11** (definitive, `scripts/scan_vaso2.py` via
  `vitaldb.find_cases` over the FULL dataset; counts in `results/vaso_prevalence.json`). The old "~0"
  was correct for VASOPRESSIN only (**1 case in all of VitalDB**) but WRONG as a blanket "no pressors":
  in-cohort counts — phenylephrine **97**, norepinephrine **47**, dopamine 20, epi 4, dobu 1;
  **any pressor = 157**, phen-or-nepi = **139**. (The old `scan_vaso.py` also under-counted because its
  `st_blocks>0` filter skipped the then-iCloud-evicted files.)
  → TWO follow-ups, deferred by Max ("note for later"): (a) **fix RESULTS.md limitation** — "pressor
  channels near-absent" is false (true only for vasopressin); (b) optional **pressor-covariate arm**
  (phenylephrine ±norepi as the future-known covariate) on the ~139-case subset — a more direct/faster
  BP effect than anesthetic CE; run like the primary ablation. Kapral Fig 3d: vasopressin is the most
  important DRUG (~0.32), remi only ~0.13 vs MAP ~0.79 — see RELATED_WORK.
- **etCO2 covariate enrichment** (needs a cache rebuild adding `Solar8000/ETCO2` to past_cov).
- **Full 2659-case run on HPC** (Linux+CUDA, `--device cuda`).
- Rolling-origin density: on HPC can drop `max_origins` cap + finer `origin_stride_min` (≈ Kapral's
  near-continuous eval).

## Ledger status (`notes/PAPER_TARGET.md`)
[N]=2659 ✓, [H]=15 ✓. [X%], [Y%], [Z] await the full sharded/HPC run.
