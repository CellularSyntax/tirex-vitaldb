# VitalDB data notes (verified)

Living record of verified facts + hygiene decisions for the VitalDB data. `[UNVERIFIED]` = still to
confirm by reading actual `.vital` files.

## What's on disk (verified)

- `vital_files/NNNN.vital` — **6388 files, 88 GB total**, one per caseid (`0001.vital` … `6388.vital`,
  zero-padded). This is the full VitalDB open dataset. Filename ↔ `caseid` (strip leading zeros).
- `.vital` = gzip-compressed VitalDB binary (~13–14 MB gz, up to ~77 MB uncompressed each). Read with
  the `vitaldb` Python package (`VitalFile`), not pandas.
- `clinical_data.csv` — 6388 rows, per-case metadata (demographics, surgery, lines, intraop drug
  totals). Schema dictionary in `clinical_parameters.csv`. `lab_data.csv`/`lab_parameters.csv` = labs.
- `track_names.csv` — the **master track dictionary** (196 tracks). Individual `.vital` files contain a
  *subset*; per-case presence must be read from each file's header.

## Key clinical_data.csv fields for cohort selection (§3.1)

- `caseid`, `subjectid` (patient — use for patient-level splits, not caseid).
- **`casestart` is anonymized to 0** → `caseend` (sec) *is* the recording duration. `casedur` (min) also
  in the dictionary. anestart/aneend/opstart/opend give phase boundaries (sec from casestart).
- `aline1`, `aline2` — arterial-line site (proxy for invasive ABP availability).
- `ane_type` — General / Spinal / Sedationalgesia.
- Intraop drug **totals** (not time series): `intraop_phe` (phenylephrine mcg), `intraop_epi`
  (epinephrine mcg), `intraop_eph` (ephedrine mg), `intraop_ppf`, `intraop_ftn`, etc. **Caveat:** these
  are cumulative doses / boluses, *not* the continuous infusion-pump trajectory we condition on.

## Cohort scoping (from clinical_data.csv, upper bounds)

- Arterial line present (`aline1` set): **3463 / 6388 (54%)**.
- Duration: median **165 min** (IQR 103–251), 6109 cases ≥60 min, 4262 ≥120 min.
- Bolus pressor recorded: phenylephrine 844, epinephrine 89, ephedrine 3211 cases.
- **Upper-bound cohort** (art line + any bolus pressor + ≥60 min): **~2108 cases**. Mostly General
  surgery (1502), Thoracic (448). Real `[N]` will be smaller after requiring a MAP track + a present
  Orchestra infusion channel.

## Drug / covariate tracks (from track_names.csv dictionary)

All Orchestra channels are numeric (`N`) step functions. Per drug: `*_RATE` (mL/hr), `*_VOL` (mL),
and for TCI drugs also `*_CE` (effect-site), `*_CP` (plasma), `*_CT` (target).

- **Continuous future-known covariate candidates** (what we condition on):
  - Anesthetics (TCI, lower MAP): `Orchestra/PPF20_RATE` + `PPF20_CE` (propofol);
    `Orchestra/RFTN20_RATE`/`RFTN50_RATE` + `_CE` (remifentanil). Likely the most reliably present.
  - Vasopressor infusions (raise MAP — cleanest stimulus): `Orchestra/PHEN_RATE` (phenylephrine),
    `NEPI_RATE` (norepinephrine), `EPI_RATE`, `VASO_RATE` (vasopressin), `DOPA_RATE`, `DOBU_RATE`.
  - Vasodilators (lower MAP): `NTG_RATE`, `NPS_RATE`.
- **§3.2 decision (pending):** test both `_RATE` and `_CE` as the covariate; keep the stronger.

## Target / vitals tracks (dictionary)

- MAP target candidates: `Solar8000/ART_MBP` (numeric, from art line) — primary; `Solar8000/NIBP_MBP`
  (non-invasive) fallback. ABP waveform `SNUADC/ART` (W/500 Hz) for the harder waveform variant later.
- Past covariates: `Solar8000/HR` (or `Solar8000/PLETH_HR`), `Solar8000/PLETH_SPO2`, `Solar8000/CVP`
  (numeric) / `SNUADC/CVP` (wave). Depth: `BIS/BIS`.
- Sampling: numerics `N` (~1/2 s in VitalDB, [UNVERIFIED] exact rate per file); waves `W/500` (ART, ECG,
  PLETH, CVP), plus some W/62.5, W/180, W/128.

## ⚠️ Environment gotcha: data is on iCloud Desktop + disk is tight

- The project lives under `~/Desktop` which is **iCloud-managed** (`com.apple.CloudDocs` full-sync).
- Of 6388 `.vital` files: **~5409 (84%) materialized locally (~94 GB)**, **~979 (15%) evicted iCloud
  placeholders (~8 GB apparent)** that download-on-access. Reading an evicted file triggers a download
  and **can time out** (`TimeoutError: [Errno 60]` seen on `0100.vital`).
- **Disk free ~19 GB and shrinking** as iCloud syncs down (was ~48 GB at session start). Full dataset
  ~102 GB apparent → **cannot materialize everything**.
- **Consequences / rules:**
  - Loader + any scan must be **robust to per-file I/O timeouts** (try/except, skip-or-retry, log
    failures). Never assume a random `.vital` is instantly readable.
  - Detect local-vs-evicted cheaply via `os.stat(f).st_blocks > 0` (block count; no download). Prefer
    building the cohort from **already-materialized** files (5409 is far more than we need).
  - `vitaldb.vital_trks(path)` / `VitalFile(..., header_only=True)` read the header only (~0.5–0.8 s on
    a local file) — use for track-presence scans; full load is ~3.5 s/file.
  - **DECISION (Max, 2026-07-11):** work from local files as-is — no data move. Build the cohort from
    already-materialized files only; skip/lazy-handle evicted placeholders; keep the loader robust to
    iCloud timeouts.

## Covariate coverage scan (300 random LOCAL files, header-only) — KEY FINDING

`vitaldb.vital_trks`, 297 readable / 300 (3 iCloud timeouts, ~1%). ~0.9 s/file.

| Track | Present | + ART_MBP |
|---|---|---|
| Solar8000/ART_MBP (invasive MAP) | 64% | — |
| Solar8000/NIBP_MBP (non-invasive) | 89% | — |
| SNUADC/ART (ABP wave) | 64% | — |
| ART_MBP + **any** infusion channel | **59%** | — |
| Orchestra/**RFTN20** RATE & CE (remifentanil) | 247/297 (83%) | **175 (59%)** |
| Orchestra/**PPF20** RATE & CE (propofol) | 163 (55%) | 100 (34%) |
| RFTN50 / NEPI / PHEN / EPI / DOPA / NTG / VASO / DOBU / DEX | **≤5 each (~0–2%)** | ≤5 |

**Decision for §3.2 — the drug covariate is the ANESTHETIC infusion (remifentanil ± propofol), NOT
vasopressors.** Continuous vasopressor *infusion* channels are essentially absent (pressors here are
given as boluses → recorded only as `intraop_phe/epi/eph` totals, not as pump tracks). So:
- Primary covariate = **`Orchestra/RFTN20_RATE` / `RFTN20_CE`** (best coverage), secondary = propofol.
- Both `_RATE` and `_CE` available for these → test both per §3.2, keep stronger.
- **Framing implication (flag to Max):** the intervention→MAP story is **anesthetic-induced blood-
  pressure change** (propofol/remi lower MAP & blunt responses), not pressor-driven BP rise. This is
  clinically coherent and *aligns tightly with the hypotension early-warning secondary task* (§3.5).
  Abstract wording "known drug-infusion trajectory" still holds.
- **Cohort size projection:** ~59% of ~5409 local files have ART_MBP + remifentanil ≈ **~3200 candidate
  cases** before quality filtering → `[N]` will be comfortably in the hundreds+.
- Non-invasive `NIBP_MBP` (89%) is intermittent (every few min) — keep invasive `ART_MBP` as target.

## Cohort inclusion thresholds (§3.1) — `scripts/build_cohort.py`

Applied over LOCAL files only (evicted skipped). Cascade (first failing reason recorded):
- **Target present:** `Solar8000/ART_MBP` (invasive MAP).
- **Remifentanil present:** `Orchestra/RFTN20_CE` and/or `RFTN20_RATE`.
- **Duration** ≥ 60 min (room for rolling origins: 30 ctx + 10 horizon + strides).
- **Target coverage** ≥ 0.60 (fraction finite MAP in anesthesia window, 5 s grid, masked [20,220]).
- **Remi CE coverage** ≥ 0.50 (fraction finite `RFTN20_CE`).
- **Remi active:** max `RFTN20_CE` > 0.5 ng/mL (an actual infusion occurred — not flat zero).
Outputs: `results/cohort_manifest.csv`, `results/cohort_flow.json`, `results/infusion_fidelity.json`.

## COHORT RESULT (§3.1) + infusion-fidelity GATE — status: ✅ PASS (2026-07-11)

Full scan of 5406 local files (`scripts/build_cohort.py`). **`[N]` = 2659 included.**

Case flow (excluded): no ART_MBP 2142 · no remifentanil 515 · unreadable header 39 · low remi
coverage 20 · low target coverage 17 · short duration 12 · remi inactive 2. → **included 2659**.
(Outputs: `results/cohort_manifest.csv`, `cohort_flow.json`, `infusion_fidelity.json`.)

**GATE = PASS** (`gate_pass: true`): remifentanil `RFTN20_CE` coverage median **0.96** (IQR 0.90–0.99,
min 0.50); CE max median **5.0 ng/mL** (IQR 4–6) → real infusions with dynamic range; target MAP
coverage median 0.93; duration median 223 min (IQR 165–299). 100% of included cases also have
`RFTN20_RATE`; 62% also have `PPF20_CE`. → Continuous infusion trajectory exists at adequate fidelity;
§3.1 can be signed off.

**Cohort characterization:** age median 61 (8–89); 1523 M / 1136 F; ASA 2 dominant (1668), ASA 1 (548),
ASA 3 (365); anesthesia General (2658/2659). Departments: General surgery 1653, **Thoracic surgery 837
(31%)**, Gynecology 97, Urology 72.

⚠️ **Decision point for Max (do NOT auto-change per framing guardrail):** our cohort is **31% thoracic
surgery**, which *both* foils (Kapral, Zhu) deliberately **excluded** (cardiac/thoracic/vascular have
atypical hemodynamics / one-lung ventilation; remifentanil-heavy). Keeping them is a cohort
*differentiator* but reduces comparability and adds hemodynamic heterogeneity. Options: (a) keep all
2659 (broadest), (b) add an exclusion for thoracic/cardiac/vascular for comparability with foils. This
is a principled scientific choice for Max — flagged, not changed. Per framing rules we do not tune the
threshold to match/differ from foils.
- **DECISION (Max, 2026-07-11): keep both.** Primary ablation on the full 2659; report a general-
  surgery-only subgroup as a **sensitivity analysis**. Most defensible; add subgroup column in results.

## Infusion-fidelity GATE — (superseded by COHORT RESULT above; PASS)

We use VitalDB as **primary** and **select for** infusion cases (both foils Kapral 2024 / Zhu 2026 used
it only as an external set) — so before Phase 3 we must explicitly verify the continuous infusion
trajectory exists at adequate fidelity in the chosen tracks, for the selected cohort:
- [ ] Sampling density / non-NaN coverage of `RFTN20_RATE` & `_CE` (± propofol) over evaluation windows.
- [ ] Plausible step dynamics (rate changes look like pump steps; CE looks like smooth PK integration).
- [ ] Fraction of cohort windows with a *usable* future covariate over `[0, T+H]`.
- **Caveat (Kapral):** VitalDB **lacks bolus-drug information** → our covariate must be the *infusion-
  pump* channel (rate/CE), not boluses. Consistent with the §3.2 anesthetic-infusion choice above.
- **Outcome:** _record here when run; do not sign off §3.1 until this passes._ See `PROJECT_PLAN.md` §3.1.

## Standing limitation: realized vs planned infusion (cross-ref)

The "future-known" infusion we feed is what the clinician **actually did** (retrospective) — an
optimistic upper bound vs deployment; the mirror image of Kapral's inability to anticipate future
interventions. Full statement in `RESULTS.md` (Limitations); ties to the abstract's "gap between
observed and truly planned infusions" line — do not duplicate wording.

## Signal preprocessing decisions (align w/ Kapral; verified against their Methods)

Kapral'24 preprocessing (from PDF): plausibility masking + **forward-fill** imputation + per-variate
**standardization (mean 0, std 1)** + drugs normalized by average dose. **No moving-average/smoothing.**
Our alignment:
- **Standardization:** TiRex normalizes **per-variate internally** (scaler `nanmean/std` over time, per
  channel — verified in source). So drug channels (0–400 mL/hr) and MAP (60–120) are NOT confused; a
  covariate flat-in-context but ramping-in-future gets its ramp *amplified* after normalization.
- **Missing/artifact:** we mask to **NaN** (TiRex handles natively) instead of forward-fill — cleaner.
- **Despike (added):** rolling-median (7@15s) despike NaNs MAP points >25 mmHg from local median —
  removes arterial-line fast-flush transients the range mask misses. Applied at load (config `despike`),
  no cache rebuild. (Removed ~9 spikes on case 2521, max 204→177.)
- **Resolution = 15 s** (config `resample_to_s: 15`, downsample cached 5 s at load): matches Kapral's
  15 s; **validated** to raise the model's covariate sensitivity at transitions (counterfactual spread
  0.8→2.7 mmHg at emergence). Cache stays 5 s; downsampled on load (target=median, drug=step-hold,
  vitals=median).

## Covariate mechanism — is TiRex using the drug? (diagnostic, `scripts/diag_covariate.py`)

Counterfactual test (feed same window real / frozen / ±4 ng/mL remi-CE trajectories): forecast median
moves only **0.2–0.8 mmHg (5 s)** / up to **2.7 mmHg at transitions (15 s)**. So: **mechanism is wired
correctly** (Phase 0 controlled test gave 13× when target≈covariate), but **zero-shot the model assigns
low weight to the drug** because the in-context drug↔MAP coupling is weak/noisy vs MAP's own dynamics —
unlike the near-deterministic synthetic demos (demo target ≈ covariate + tiny noise, by construction).
→ Honest finding; effect should concentrate in transition windows (hence §3.4 stratification).

**Covariate feasibility note (Kapral used 52 features incl. vasopressors):** in VitalDB only remi +
propofol exist as continuous *infusion* trajectories (future-known). Vasopressors/pressors are **boluses
recorded as untimed totals** → no trajectory to feed. So the drugs likeliest to cause sharp MAP swings
(pressor boluses) are exactly the ones we cannot provide as a future covariate — a stated limitation.
etCO2 enrichment (past covariate) planned as a follow-up cache rebuild.

## Still TODO (Phase 1)

- [ ] Read a sample `.vital` header: confirm per-case track list + exact sampling rates + timestamps.
- [ ] Header-only scan across cases: co-occurrence of (ART_MBP) × (each Orchestra `_RATE`/`_CE`) → pick
      the covariate drug(s) with best coverage. This determines the real cohort + `[N]`.
- [ ] Confirm filename↔caseid mapping and any missing files.
- [ ] Artifact handling for ART line (zeroing/flush/damping); numeric↔wave alignment; drug step
      resampling (no smooth interpolation).
