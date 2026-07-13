# MOVER Dataset — Schema Reconnaissance Report

**Purpose:** enable porting an intraoperative mean-arterial-pressure (MAP) forecasting pipeline (VitalDB-style: MAP target + past vitals + known-future drug-infusion covariates) onto the UCI **MOVER** dataset (Medical Informatics Operating Room Vitals and Events Repository).

**Scope:** reconnaissance only — **no data was modified**. All figures were obtained by streaming / sampling (Python `csv` module row-by-row, `head`, `wc -l`); no table was loaded whole into memory.

**Extraction root:** `D:\DATA\mover`

**Bottom line up front:**
- **Two independent record systems are present — SIS and EPIC — with *no* linking crosswalk between them** (~0 ID overlap). Pick one as the modelling cohort.
- **SIS (`EMR\`) is the cleaner, forecast-ready source**: wide tables, HR/SpO₂ at **1-minute** cadence, NIBP MAP at **3–5 min**, invasive arterial MAP (`MAP_ART`) at **1-minute in ~3,800 cases**, and drug **infusions recorded as timed segments** from which a per-minute rate trajectory is directly reconstructable.
- **EPIC is larger (~64k cases)** but long-format, messier, and its MAR spans the whole admission; usable but needs more wrangling.

---

## 1. Layout & README

### 1.1 Extraction root and tree (`find -maxdepth 2`)

```
D:\DATA\mover
├── README.txt                         (README archive)          469 B
├── ADDITIONAL-INFO-EPIC_MRN_PAT_ID.txt                          204 B
├── EPIC_MRN_PAT_ID.csv                (ID crosswalk)            3.3 MB
├── waveform_decode.py                 (waveform decoder stub)   2.4 KB
├── all_size_listing.txt / all_md5sum_listing.txt
│
├── EMR\                        ← extracted from  sis_emr.tar          (SIS anesthesia record)
│   ├── patient_a_line.csv
│   ├── patient_information.csv
│   ├── patient_input_output.csv
│   ├── patient_labs.csv
│   ├── patient_medication.csv
│   ├── patient_observations.csv
│   ├── patient_procedure_events.csv
│   ├── patient_ventilator.csv
│   └── patient_vitals.csv
│
├── EPIC_EMR\EMR\               ← extracted from  EPIC_EMR.tar.gz      (Epic clinical EMR)
│   ├── patient_coding.csv
│   ├── patient_history.csv
│   ├── patient_information.csv
│   ├── patient_labs.csv
│   ├── patient_lda.csv
│   ├── patient_medications.csv
│   ├── patient_post_op_complications.csv
│   ├── patient_procedure events.csv    (note the space in the filename)
│   └── patient_visit.csv
│
├── flowsheets_cleaned\        ← extracted from  Epic_flowsheets_cleaned.tar.gz  (Epic vitals, long, 19 parts)
│   └── flowsheet_part1.csv … flowsheet_part19.csv
│
└── srv\disk00\EPIC_flowsheets\ ← extracted from EPIC_patient_measurments.tar   (raw per-case Epic flowsheets)
    └── 1_2018-4633_flowsheets_20211006.csv     ⚠ ONLY 1 FILE PRESENT (see caveat)
```

> ⚠ **`EPIC_patient_measurments.tar` (360 GB uncompressed) is only *partially* extracted** — a single per-case file (`1_2018-4633_flowsheets_20211006.csv`, 144 MB, 406 LOG_IDs / 129 patients) is on disk. The complete raw per-case flowsheets are **not** available; use `flowsheets_cleaned\` (which is complete) for EPIC vitals. The waveform archives (`sis_wave`, `epic_wave_*`) were **not** extracted and are out of scope for MAP forecasting.

### 1.2 README contents (there is *no* column-level data dictionary)

The `README.txt` is minimal — reproduced in full:

```
MOVER NOTES:
1. The waveforms are all alphanumeric encoded and can be decoded using the provided Python script waveform_decode.py
2. For the EPIC dataset, the EMR patient measurements files were compressed separately because of the size of the files.

UPDATE v2 (2024-05):
Waveform data has been adjusted and replaced by these files:
  sis_wave_v2.tar.gz, epic_wave_3_v2.tar.gz, epic_wave_2_v2.tar.gz, epic_wave_1_v2.tar.gz
Reason: some of the wave gains were off
```

`ADDITIONAL-INFO-EPIC_MRN_PAT_ID.txt` (2023-09-19): *"A new file has been added which is a cross listing of patient ID numbers. There was a mismatch of IDs in the data files. This should help match the records. File: EPIC_MRN_PAT_ID.csv."*

**There is no table/column dictionary, no unit spec, and no timestamp-format spec in the README.** Everything below was inferred directly from the data. `waveform_decode.py` only documents waveform gain/offset (`GE_ART` gain 0.25, `INVP1` gain 0.01, base-64 SmallInt decoding) — not relevant to the tabular vitals used here.

### 1.3 Table inventory (all files are **CSV**; dtypes are read as text — see §per-table notes)

| Source (archive) | File | Size | Rows (data) | Notes |
|---|---|---|---|---|
| **SIS** (`EMR\`) | `patient_information.csv` | 3.3 MB | 19,114 | one row per case; demographics + case timing |
| SIS | `patient_vitals.csv` | 191 MB | 3,595,591 | **wide** 1-min vitals (HR, NIBP, SpO₂) |
| SIS | `patient_observations.csv` | 171 MB | 2,700,509 | **wide** invasive/derived (art MAP, CVP, SVV, CO, temp…) |
| SIS | `patient_medication.csv` | 19.2 MB | 373,852 | boluses **and** infusion segments |
| SIS | `patient_ventilator.csv` | 72.6 MB | 1,048,574 | vent + **volatile agent** Fi/Et |
| SIS | `patient_labs.csv` | 1.4 MB | 14,733 | wide ABG panel |
| SIS | `patient_input_output.csv` | 5.8 MB | 100,993 | fluids in/out |
| SIS | `patient_procedure_events.csv` | 2.1 MB | 40,801 | intraop event markers |
| SIS | `patient_a_line.csv` | 178 KB | 2,989 | arterial-line placements |
| **EPIC** (`EPIC_EMR\EMR\`) | `patient_information.csv` | 17.8 MB | 65,728 | demographics, ASA, case timing |
| EPIC | `patient_medications.csv` | **7.18 GB** | **27,961,524** | **MAR** — infusion rates + admin events |
| EPIC | `patient_labs.csv` | 3.32 GB | 29,079,344 | long-format labs |
| EPIC | `patient_coding.csv` | 220 MB | 2,033,948 | billing/diagnosis codes |
| EPIC | `patient_history.csv` | 51.5 MB | 970,741 | past diagnoses |
| EPIC | `patient_procedure events.csv` | 40.4 MB | 640,223 | intraop event markers |
| EPIC | `patient_visit.csv` | 15.7 MB | 219,257 | encounter diagnoses |
| EPIC | `patient_lda.csv` | 123.6 MB | 465,801 | Lines/Drains/Airways (incl. art lines) |
| EPIC | `patient_post_op_complications.csv` | 20 MB | 203,945 | post-op complications |
| **EPIC flowsheets** (`flowsheets_cleaned\`) | `flowsheet_part1.csv` | 48.4 GB | ≈ n/a | **long** vitals; part1 dominates |
| EPIC flowsheets | `flowsheet_part2.csv` | 4.5 GB | — | |
| EPIC flowsheets | `flowsheet_part3.csv` | 848 MB | 8,691,534 | (fully scanned) |
| EPIC flowsheets | `flowsheet_part4…19.csv` | ~5.5 GB each | — | 16 parts |
| | **flowsheets total** | **≈ 152 GB** | **≈ 1.5 billion (est.)** | est. from part3 density × total bytes — *not exhaustively counted* |
| **EPIC raw meas.** (`srv\…\`) | `1_2018-4633_flowsheets_20211006.csv` | 144 MB | 433,631 | only file extracted (partial) |
| crosswalk | `EPIC_MRN_PAT_ID.csv` | 3.3 MB | 65,728 | LOG_ID ↔ PAT_ID ↔ MRN |

**Common encodings:** null is stored as the literal token `\N` **and/or** empty string (both appear). Some SIS tables quote every field (`patient_a_line`, `patient_input_output`, `patient_procedure_events`, `patient_labs`), others are unquoted. All IDs are 16-character hex hashes.

---

## 2. Keys & joins

### 2.1 SIS (`EMR\`)
- **Single identifier: `PID`** (16-hex). Present in **every** SIS table.
- `PID` = **one surgical case / anesthesia record**. In `patient_information.csv` there are **19,114 rows, 19,114 distinct PIDs, zero duplicates**, each with exactly one `Procedure` and one set of OR times.
- **SIS has NO patient-level identifier.** You cannot tell whether two PIDs are the same physical patient. → The **only** leak-free split granularity is `PID` (case-level); repeat-patient leakage is undetectable in SIS.
- **Unique key for "one surgical case" (SIS): `PID`.**

### 2.2 EPIC (`EPIC_EMR\`, flowsheets, raw)
- **`MRN`** = patient. **`LOG_ID`** = OR case / surgical log. **`PAT_ID`** = Epic patient ID (**1:1 with MRN**).
- Crosswalk `EPIC_MRN_PAT_ID.csv` (`LOG_ID, PAT_ID, MRN`) links them: 65,728 rows → **64,353 distinct LOG_ID, 39,683 distinct PAT_ID = 39,683 distinct MRN**.
- In `patient_information.csv`: **64,354 distinct LOG_ID, 39,685 distinct MRN**. **12,591 MRNs have >1 surgery; max = 52 surgeries for one patient.** → **Yes, one patient can have many surgeries.**
- **Unique key for "one surgical case" (EPIC): `LOG_ID`** (dedupe; 1,374 LOG_IDs appear on >1 info row). **Leak-free split must be at `MRN` (patient) level.**

### 2.3 Which ID columns each table carries

| Table | ID columns |
|---|---|
| SIS – *all 9 tables* | `PID` only |
| EPIC `patient_information` | `LOG_ID`, `MRN` |
| EPIC `patient_medications` | `LOG_ID`, `MRN` |
| EPIC `patient_labs` | `LOG_ID`, `MRN` |
| EPIC `patient_lda` | `LOG_ID`, `MRN` |
| EPIC `patient_post_op_complications` | `LOG_ID`, `MRN` |
| EPIC `patient_procedure events` | `LOG_ID`, `MRN` |
| EPIC `patient_visit` | `LOG_ID`, `mrn` |
| EPIC `patient_coding` | `MRN` only (+ `SOURCE_KEY`) |
| EPIC `patient_history` | `mrn` only |
| EPIC `flowsheets_cleaned` | `LOG_ID`, `MRN` |
| EPIC raw measurements | `OR_CASE_ID`, `LOG_ID`, `PAT_ID`, `MRN`, `HSP_ACCOUNT_ID`, `OR_LINK_CSN`, `PAT_ENC_CSN_ID` |

### 2.4 ⚠ SIS ↔ EPIC do **not** join
Intersecting the 19,114 SIS `PID`s against the EPIC ID universe: **1** hit vs LOG_ID, **1** vs PAT_ID, **0** vs MRN — i.e. essentially zero (coincidental hash collisions). **SIS and EPIC are separate ID spaces with no provided crosswalk.** Treat them as **two independent cohorts**; do not attempt to merge a patient across the two systems.

---

## 3. Intraoperative vitals (forecast target + past covariates)

There are two vital sources. **SIS is wide and clean; EPIC is long.**

### 3.1 SIS — `patient_vitals.csv` (wide, non-invasive + HR + SpO₂)
Columns: `PID, Obs_time, HRe, HRp, nSBP, nMAP, nDBP, SP02`. One row per `PID` per minute; `Obs_time` is ISO `YYYY-MM-DD HH:MM:SS`. 3,595,591 rows over **19,023 PIDs**.

| Signal | Column | Unit | Storage | Median per-case cadence | Value range (p1 / p50 / p99) | Cases with data |
|---|---|---|---|---|---|---|
| **NIBP MAP** | `nMAP` | mmHg | wide | **~300 s (3–5 min; p25 180 / p75 300)** | 50 / **76** / 126 | 18,935 |
| NIBP systolic | `nSBP` | mmHg | wide | ~300 s | 71 / 108 / 186 | 18,950 |
| **Heart rate (ECG)** | `HRe` | bpm | wide | **60 s** | 46 / **75** / 126 | 19,011 |
| Heart rate (pleth) | `HRp` | bpm | wide | 60 s | 48 / 75 / 126 | 19,000 |
| **SpO₂** | `SP02` | % | wide | **60 s** | 91 / **100** / 100 | 19,021 |

(`nDBP` also present.) Units confirmed mmHg / bpm / % by the value distributions.

### 3.2 SIS — `patient_observations.csv` (wide, invasive arterial + CVP + derived)
Columns: `PID, Obs_time, SVV, PAPs, PAPm, PAPd, LAPm, CO, Cer_ox_r, Cer_ox_l, Temp, Temp2, TOF, SBP_FEM, MAP_FEM, DBP_FEM, ICPm, SBP_ART, MAP_ART, DBP_ART, CVPm`. `Obs_time` ISO seconds. 2,700,509 rows over 17,976 PIDs.

| Signal | Column | Unit | Median per-case cadence | Value range (p1 / p50 / p99) | Cases with data |
|---|---|---|---|---|---|
| **Invasive arterial MAP** | `MAP_ART` | mmHg | **60 s** | (–30) / **76** / 186 | **3,800** |
| Invasive arterial SBP | `SBP_ART` | mmHg | 60 s | (–29) / 109 / 190 | 3,800 |
| **Central venous pressure** | `CVPm` | mmHg | 60 s | (–40) / **10** / 303 | 735 |
| Femoral arterial MAP | `MAP_FEM` | mmHg | 60 s | –24 / 65 / 276 | 24 (rare) |
| Temperature | `Temp` | °C | 60 s | 34 / 36 / 38 | 16,888 |
| Cardiac output | `CO` | L/min | irregular | 1 / 3 / 7 | 109 |

> ⚠ The invasive columns (`MAP_ART`, `SBP_ART`, `DBP_ART`, `CVPm`, `MAP_FEM`) contain **artifacts/sentinels** — negative values and spikes (observed min ≈ –319, max ≈ 349). Physiologic range-filtering is required (e.g. keep MAP ∈ [20, 250], CVP ∈ [–5, 40]).

### 3.3 Invasive vs non-invasive MAP availability (SIS)
- **Invasive arterial MAP (`MAP_ART`): 3,800 cases** (1-min).
- **NIBP MAP (`nMAP`): 18,935 cases** (3–5 min).
- Both: **3,719** · arterial-only: 81 · NIBP-only: 15,216.

→ The **high-quality forecasting target** (dense, beat-derived, 1-min) exists for ~3,800 SIS cases; the remaining ~15k cases have only 3–5 min NIBP.

### 3.4 EPIC — long-format flowsheets (`flowsheets_cleaned\` and raw)
Cleaned schema: `"",LOG_ID, MRN, FLO_NAME, FLO_DISPLAY_NAME, RECORD_TYPE, RECORDED_TIME, MEAS_VALUE, UNITS`. **Long** (one row per measurement per timestamp). The vital is identified by the **`FLO_DISPLAY_NAME`** string; `MEAS_VALUE` is the value; `RECORDED_TIME` is ISO seconds.

**EPIC records the SAME vital under two different documentation streams — you must pick the anesthesia (dense) names:**

| Vital | Anesthesia intraop name (dense) | Cadence | Coarse nursing name (avoid) | Cadence | Unit |
|---|---|---|---|---|---|
| Arterial MAP | **`MAP-ART A-line`** | **60 s** | `Arterial Line MAP (ART)` | ~1800 s | mmHg |
| Arterial BP | `BP-ART A-line` | 60 s | `Arterial Line BP (ART)` | ~hourly | mmHg |
| NIBP MAP | **`NIBP - MAP`** | **~300 s** | `MAP (mmHg)` | ~3600 s | mmHg |
| Heart rate | **`Heart Rate`** | **60 s** | `Pulse` | ~1800 s | bpm |
| SpO₂ | **`SpO2`** | **60 s** | — | — | % |

Value ranges confirm units: `MAP-ART A-line` p50 ≈ 82, `MAP (mmHg)` p50 ≈ 89, `Arterial Line MAP (ART)` p50 ≈ 78 mmHg; `SpO2` p50 = 100 %. Also present: `Min/Max Systolic`, `Min/Max Diastolic`, `BP Location`, `NIBP Site`, `Cardiac Rhythm`, `Temp`, plus ventilator/gas rows (`ETCO2 (mmHg)`, `FiO2 (%)`, `SEV In`, `ETO2`, …).

> ⚠ Cadence figures here are from sampling `flowsheet_part1` (first 2.5 M rows) plus per-case computation on the one raw file — **not** a full-dataset per-case computation (150 GB). **CVP was not positively identified** in the EPIC flowsheet sample (`PAWP` present is an airway pressure, not central venous). Confirm CVP naming before relying on it in EPIC. *(`part3` happens to contain only Lines/Drains + cardiac-output rows — the parts are not cleanly split by measure type.)*

---

## 4. ⚠ CRITICAL — intraoperative drug infusions (known-future covariate)

**Answer up front — YES, a continuous per-minute infusion-rate trajectory can be built per surgery, in BOTH systems, but via different mechanisms:**

### 4.1 SIS — `patient_medication.csv` (segments → derive rate)
Columns: `PID, Start_time, End_time, Dose, Drug_name, Drug_units`. Timestamps `M/D/YY H:MM`. 373,852 rows; **6.3 % have `End_time` populated**.

- **Boluses:** `End_time = \N`, single `Start_time`, `Dose` in an **absolute** unit (`mg`/`mcg`/`mL`/`units`).
- **Infusions ("drips"):** **both `Start_time` and `End_time` populated**; each row is **one infusion segment**; `Dose` = **total amount delivered over that segment**; `Drug_units` are still **absolute amounts, NOT rates**. Consecutive segments for the same drug tile the timeline (a rate change = a new row).
- → **Derive rate = `Dose ÷ (End_time − Start_time)`**, lay piecewise-constant over each segment ⇒ a per-minute rate series. (Optionally ÷ weight from `patient_information.Wt` for mcg/kg/min.)

**Continuous IV anaesthetic / vasoactive infusions available in SIS (segment count = rows with `End_time`):**

| Drug (`Drug_name`, verbatim) | Units | Infusion segments | Analogue to VitalDB channel |
|---|---|---|---|
| `Propofol  drip` (two spaces) | mcg | 5,789 | propofol |
| `Remifentanyl-d` | mcg | 1,994 | remifentanil |
| `Phenylephrine` | mcg | 950 | phenylephrine |
| `Norepinephrine` | mcg | 424 | norepinephrine |
| `Dobutamine` | mcg | 260 | inotrope |
| `Nitroglycerin` | mcg | 258 | vasodilator |
| `Vasopressin` | units | 197 | vasopressor |
| `Dexmedetomidine` | mcg | 164 | sedative |
| `Nicardipine` | mg | 163 | vasodilator |
| `Epinephrine` | mcg | 139 | inotrope/pressor |

Plus fluid/blood infusions (Plasmalyte, Lactated ringers, Albumin, PRBC, FFP) and long-run antibiotics.

**Volatile agent (SIS):** the inhaled anaesthetic is a *separate* time series in `patient_ventilator.csv` — columns `Agent` (name), `Agent_Fi` (inspired %), `Agent_Et` (end-tidal %) — 1.05 M rows. This is the natural covariate for volatile-based cases (analogous to an effect-site channel).

**Top ~30 SIS drugs by row frequency (drug | count | unit):**
```
Fentanyl 55686 mcg | Propofol 32347 mg | Phenylephrine 29990 mcg | Plasmalyte 20781 mL
Ephedrine 18363 mg | Vecuronium 16106 mg | Midazolam 15552 mg | Ondansetron 14211 mg
Lidocaine 13943 mg | Lactated ringers 12463 mL | Glycopyrrolate 11191 mg | Dexamethasone 10841 mg
Rocuronium 10740 mg | Hydromorphone 10063 mg | Cefazolin 10018 mg | Neostigmine 8715 mg
Succinylcholine 6858 mg | Sodium chloride 0.9% 6276 mL | Propofol  drip 5921 mcg | Albumin human 5% 4752 mL
Acetaminophen 3846 mg | Calcium chloride 3802 mg | Cisatracurium 3333 mg | Vasopressin 3327 units
Packed red blood cells 3178 units | Esmolol 2404 mg | Ketamine 2359 mg | Remifentanyl-d 2035 mcg
Vancomycin 1760 mg | Morphine 1756 mg
```
(`Fentanyl`, `Propofol`, `Phenylephrine`, `Ephedrine` etc. appear mostly as **boluses** — absolute mg/mcg with no end time; the drip form of propofol/remifentanil is the separate `Propofol  drip` / `Remifentanyl-d`.)

### 4.2 EPIC — `patient_medications.csv` (MAR, explicit rates)
Columns: `ENC_TYPE_C, ENC_TYPE_NM, LOG_ID, MRN, ORDERING_DATE, ORDER_CLASS_NM, MEDICATION_ID, DISPLAY_NAME, MEDICATION_NM, START_DATE, END_DATE, ORDER_STATUS_NM, RECORD_TYPE, MAR_ACTION_NM, MED_ACTION_TIME, ADMIN_SIG, DOSE_UNIT_NM, MED_ROUTE_NM`. ~28 M rows.

This is a **classic infusion MAR — rates with rate-change timestamps, no derivation needed**:
- **`MAR_ACTION_NM`** ∈ {`Given`, `New Bag`, `Rate Change`, `Rate Verify`, `Stopped`, `Restarted`, `Bolus From Bag`, `Patch Applied/Removed`, `Held`/`MAR Hold`/`MAR Unhold`, …}.
- **`ADMIN_SIG`** = the numeric **rate (or dose)** value; **`MED_ACTION_TIME`** = event timestamp; **`DOSE_UNIT_NM`** = the unit.
- **Rate units are explicit** — sample counts (first 1.5 M rows): `mcg/min` 50,622 · `mcg/kg/min` 39,523 · `mcg/hr` 25,316 · `mg/hr` 20,125 · `Units/hr` 18,658 · `mL/hr` 18,489 · `mcg/kg/hr` 16,763 · `Units/min` 9,826 · `mg/min` 2,152 · `mg/kg/hr` 2,625 (plus discrete `mg`, `mL`, `Units`, tablets, drops for boluses/PO).
- Infusion products seen: `propofol (DIPRIVAN) infusion`, `fentaNYL … continuous infusion`, `phenylephrine dilution (NEO-SYNEPHRINE)`, `norepinephrine (LEVOPHED) 4 mg/250 mL`, `nitroGLYcerin 50 mg/250 mL infusion`, `DOBUTamine premix infusion`, insulin infusions, etc.

**Worked MAR excerpt (a nitroglycerin infusion → rate step-function):**
```
DISPLAY_NAME=nitroGLYcerin 50 mg/250 mL infusion  MAR_ACTION=Rate Change  time=2021-02-05 08:21  ADMIN_SIG=30  unit=mcg/min
                                                   Rate Change  09:06  40 mcg/min
                                                   Rate Verify  15:00  40 mcg/min
                                                   Rate Change  2021-02-06 14:10  5 mcg/min
propofol (DIPRIVAN) infusion   Rate Verify  30  mcg/kg/min
DOBUTamine premix infusion     Rate Change  5→3  mcg/kg/min
```

> ⚠ **EPIC MAR spans the whole hospital admission (incl. ICU) over multiple days** — the excerpt above is an ICU course with hourly `Rate Verify`. For intraop covariates you **must window `MED_ACTION_TIME` to `IN_OR_DTTM … OUT_OR_DTTM`**. Also, intraop anaesthesia-pump infusions in Epic are sometimes documented in the anaesthesia *flowsheet* rather than the MAR — completeness of intraop infusion capture in the MAR should be verified per-case before relying on EPIC over SIS.

### 4.3 Plain statement
- **SIS: YES.** Reconstruct a per-minute rate trajectory as `Dose/(End−Start)` (piecewise-constant) for **Propofol drip, Remifentanyl-d, Phenylephrine, Norepinephrine, Dobutamine, Nitroglycerin, Vasopressin, Dexmedetomidine, Nicardipine, Epinephrine** (and fluids). Volatile agent via `patient_ventilator.Agent_Fi/Agent_Et`.
- **EPIC: YES.** Read the rate directly from the MAR (`ADMIN_SIG` + `DOSE_UNIT_NM` rate units + `MED_ACTION_TIME`) as a step-function, **after** windowing to the OR interval.

---

## 5. Case timing (analysis window)

| System | Table | Columns | Format | Notes |
|---|---|---|---|---|
| **SIS** | `EMR\patient_information.csv` | `OR_start`, `OR_end`, `Surgery_start`, `Surgery_end` | `M/D/YY H:MM` (minute res, e.g. `2/15/18 12:13`) | absolute local datetimes, **no seconds, no timezone** |
| SIS | `EMR\patient_procedure_events.csv` | `Event_time`, `Event_name` | ISO seconds | intraop milestones/events |
| **EPIC** | `EPIC_EMR\EMR\patient_information.csv` | **`IN_OR_DTTM`, `OUT_OR_DTTM`, `AN_START_DATETIME`, `AN_STOP_DATETIME`** (+ `SURGERY_DATE`, `HOSP_ADMSN_TIME`, `HOSP_DISCH_TIME`) | `M/D/YY H:MM` | anaesthesia + OR in/out; ~89–90 % populated |
| EPIC | raw measurements file | same four columns embedded per row | ISO seconds | per-case window carried inline |
| EPIC | `patient_procedure events.csv` | `EVENT_TIME`, `EVENT_DISPLAY_NAME` | — | intraop milestones |

All times appear to be **absolute local wall-clock datetimes** (not offsets). **Timezone is undocumented.** Caution: within SIS the `patient_information` timing columns are minute-resolution `M/D/YY H:MM`, whereas the SIS vitals/observations use ISO `YYYY-MM-DD HH:MM:SS` — parse each format explicitly. Use `AN_START…AN_STOP` (EPIC) / `OR_start…OR_end` (SIS) as the modelling window.

---

## 6. Demographics / metadata

### 6.1 EPIC — `EPIC_EMR\EMR\patient_information.csv`
| Field | Column | Encoding | Coverage |
|---|---|---|---|
| **Age** | `BIRTH_DATE` | ⚠ **actually the AGE in years** (integer, e.g. `47`, `81`) — **not a date** | 100 % |
| **Sex** | `SEX` | `Male` (34,911) / `Female` (30,816) / `Unknown` (1) | 100 % |
| **ASA** | `ASA_RATING` (text) + `ASA_RATING_C` (code) | 1.0 Healthy · 2.0 Mild · 3.0 Severe · 4.0 Incapacitating · 5.0 Moribund · 6.0 Brain Dead | 89 % |
| Height | `HEIGHT` | ⚠ **feet/inches string**, e.g. `5' 6` | 80 % |
| Weight | `WEIGHT` | ⚠ **ounces**, e.g. `2832.47` (÷ 35.274 → ≈ 80.3 kg) | 96 % |
| Procedure | `PRIMARY_PROCEDURE_NM` | free text | 100 % |
| Anaesthesia type | `PRIMARY_ANES_TYPE_NM` | text | ~100 % |
| Class / service | `PATIENT_CLASS_GROUP`, `PATIENT_CLASS_NM` | text | 100 % |
| **BMI** | — | **not stored** → derive from parsed `HEIGHT` (ft/in → m) and `WEIGHT` (oz → kg) | — |

Also: `DISCH_DISP`, `LOS`, `ICU_ADMIN_FLAG`, `HOSP_ADMSN_TIME`, `HOSP_DISCH_TIME`. Diagnoses/comorbidities live in `patient_history.csv` (`dx_name`, `diagnosis_code`), `patient_coding.csv`, `patient_visit.csv`; post-op outcomes in `patient_post_op_complications.csv`.

### 6.2 SIS — `EMR\patient_information.csv` (self-contained, cleaner)
`Age` (years, int) · `Ht` (**cm**, e.g. `183`) · `Wt` (**kg**, e.g. `113`) · `Gender` (`M`/`F`) · `Procedure` (free text). **BMI derivable directly** from Ht/Wt (proper units). **No ASA class in SIS.**

---

## 7. One fully-worked example

**PID `f912e36b1d623f95` (SIS)** — *Craniotomy for resection of brain tumor*. Male, 60 y, 183 cm, 113 kg.

**ID key:** `PID = f912e36b1d623f95` (SIS case; no EPIC/MRN link exists).
**Timing:** `OR_start 2/15/18 12:13 → OR_end 20:30`; `Surgery_start 13:49 → Surgery_end 20:00`. Arterial line placed `2018-02-15 12:36` (`patient_a_line.csv`).

**Invasive arterial MAP (`MAP_ART`, from `patient_observations.csv`) — 491 samples, 1-min, span 12:55 → 20:25 (450 min):**
```
first 10  (time, MAP_ART mmHg | SBP_ART | DBP_ART)
  2018-02-15 12:55:00  82  109  62
  2018-02-15 12:56:00  78  105  61
  2018-02-15 12:57:00  78  104  62
  2018-02-15 12:58:00  75  103  61
  2018-02-15 12:59:00 101  100  61
  2018-02-15 13:00:00  76  101  62
  2018-02-15 13:01:00  73   96  59
  2018-02-15 13:02:00  85  114  72
  2018-02-15 13:03:00  89  120  75
  2018-02-15 13:04:00  90  119  73
last 5
  2018-02-15 20:21:00 135  180  108
  2018-02-15 20:22:00 141  209  120
  2018-02-15 20:23:00 160  223  128
  2018-02-15 20:24:00 145  192  108
  2018-02-15 20:25:00 140  183  104
```
**MAP samples = 491; duration ≈ 450 min.** (No `CVPm` for this case.)

**HR / SpO₂ availability (`patient_vitals.csv`):** `HRe` = 494 samples, `SP02` = 482 samples, `nMAP` (NIBP) = 44 samples; vitals span `12:13 → 20:27`. → dense 1-min HR & SpO₂ alongside the 1-min arterial MAP.

**Infusions for this case (`patient_medication.csv`) with derived rates (÷113 kg):**
```
drug              dose(mcg)   start → end            duration   → avg rate
Propofol  drip     254250    12:40 → 12:56 (16 min)  → 15,891 mcg/min ≈ 141 mcg/kg/min
Propofol  drip     169500    12:57 → 13:15 (18 min)  →  9,417 mcg/min ≈  83 mcg/kg/min
Propofol  drip     168088    13:15 → 13:32 (17 min)  →  9,888 mcg/min ≈  88 mcg/kg/min
Propofol  drip     345827    13:33 → 13:56 (23 min)  → 15,036 mcg/min ≈ 133 mcg/kg/min
Propofol  drip     277980    13:56 → 14:21 (25 min)  → 11,119 mcg/min ≈  98 mcg/kg/min
Propofol  drip    2690070    14:22 → 19:22 (300 min) →  8,967 mcg/min ≈  79 mcg/kg/min
Remifentanyl-d       678    12:40 → 13:15  (+7 more segments to 19:57)
Phenylephrine       1676.17 17:12 → 20:09 (177 min)  →   9.5 mcg/min  ≈ 0.084 mcg/kg/min
(boluses: Midazolam, Propofol induction 150+50 mg, Rocuronium, Lidocaine, Fentanyl 150 mcg,
 Ephedrine ×6, Calcium chloride ×3, Dexamethasone, Ondansetron, Mannitol, Vancomycin, Cefepime …)
```
The derived propofol rates (79–141 mcg/kg/min) are physiologically sensible TIVA maintenance values — **confirming the segment→rate reconstruction works.** This case has a 1-min arterial MAP target **and** overlapping propofol + remifentanil + phenylephrine rate trajectories — exactly the VitalDB-style setup.

---

## 8. Counts for cohort sizing

| Quantity | SIS | EPIC |
|---|---|---|
| Distinct **patients** | **unknown** (no patient ID; ≤ 19,114) | **39,685** (`MRN`) |
| Distinct **surgeries/cases** | **19,114** (`PID`) | **64,354** (`LOG_ID`) |
| Cases with **≥30 min MAP** (any) | **18,798** | not computed (150 GB) |
| — of which NIBP-MAP ≥30 min | 18,572 | — |
| — of which **arterial MAP ≥30 min** | **3,294** | — |
| Cases ≥30 min MAP **AND** continuous anaesthetic/vasoactive infusion | **6,190** | not computed |
| — arterial ≥30 min **AND** infusion | **1,866** | — |
| — arterial ≥30 min **AND** vasopressor/inotrope infusion | **827** | — |
| Cases with any continuous-infusion (whitelist) | 6,311 | — |

**How counted (SIS, exact):** streamed each of `patient_vitals` (`nMAP`) and `patient_observations` (`MAP_ART`) row-by-row, tracked first/last non-null timestamp per `PID`, kept PIDs with span ≥ 1800 s. "Infusion" = PID appearing in `patient_medication` with `End_time` populated and `Drug_name` in the continuous-infusion whitelist {Propofol drip, Remifentanyl-d, Phenylephrine, Norepinephrine, Epinephrine, Dexmedetomidine, Dobutamine, Vasopressin, Nitroglycerin, Nicardipine, Esmolol}; "vasopressor/inotrope" = the pressor subset.

**EPIC ≥30-min-MAP & infusion counts were not computed** — it would require a windowed per-case scan of the 152 GB flowsheets + 28 M-row MAR. Given the EPIC intraop anaesthesia template runs at 1-min, a large fraction of the 64,354 EPIC cases are expected to qualify, but this needs a dedicated pass to quantify.

**Recommended primary cohort:** SIS. For the *dense invasive* MAP-forecasting task with drug covariates, the ready set is **≈ 3,294 cases (arterial MAP ≥30 min)**, of which **≈ 1,866 have a concurrent continuous infusion** (827 with a pressor). For a larger NIBP-based task, **≈ 18,572 cases** with 3–5 min MAP are available.

---

## Open questions / caveats

1. **No SIS↔EPIC linkage.** ~0 ID overlap and no crosswalk. Choose one system; you cannot pool a patient across both. *(Confirmed by direct set intersection.)*
2. **No formal data dictionary.** README is 3 lines; all column meanings/units above were inferred from the data. Units (esp. EPIC WEIGHT=ounces, HEIGHT=ft/in, BIRTH_DATE=age) should be validated against Epic documentation before use.
3. **`EPIC_patient_measurments.tar` is only partially extracted** (1 per-case file present of a 360 GB archive). Use `flowsheets_cleaned\` (complete) for EPIC vitals; the raw per-case files are otherwise unavailable on disk.
4. **SIS has no patient-level ID** → repeat-patient leakage is undetectable; split at `PID` (case). *(Whether the same patient recurs under multiple PIDs is unknowable from the data.)*
5. **Timezone undocumented** (assumed local wall-clock). Mixed timestamp formats within SIS (`M/D/YY H:MM` in `patient_information`/`patient_medication` vs ISO seconds in vitals/observations) — parse explicitly.
6. **SIS invasive channels contain artifacts** (negative sentinels ≈ –319, spikes to ~349 in `MAP_ART`/`SBP_ART`/`CVPm`). Physiologic filtering mandatory.
7. **SIS infusion rate is *derived* (segment average = Dose/duration), not an instantaneous pump rate**; boluses have no end time. `Propofol  drip` has two spaces in the name; remifentanil is spelled `Remifentanyl-d`.
8. **EPIC MAR spans the full admission** (ICU/ward included, multi-day). Must window to `IN_OR…OUT_OR`. Intraop-pump completeness in the MAR vs the anaesthesia flowsheet is **unverified** — spot-check before preferring EPIC.
9. **EPIC has two flowsheet streams** for each vital (dense anaesthesia names vs coarse nursing names). Select the anaesthesia names (`MAP-ART A-line`, `Heart Rate`, `SpO2`, `NIBP - MAP`) for 1-min data; do **not** use `MAP (mmHg)`/`Pulse`/`Arterial Line MAP (ART)` (hourly).
10. **CVP in EPIC flowsheets not confirmed** (only SIS `CVPm` verified). Search other `FLO_DISPLAY_NAME`s in unscanned parts if EPIC CVP is needed.
11. **EPIC cadence & flowsheet row count are from sampling**, not a full per-case pass (part1 first 2.5 M rows + one raw file). Flowsheet total ≈ 1.5 B rows is an estimate.
12. **EPIC missingness:** ASA / anaesthesia timing ~10–11 % missing, HEIGHT ~20 % missing.
