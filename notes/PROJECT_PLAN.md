# Project plan — TiRex-2 as a zero-shot surrogate for intraoperative hemodynamic response

North-star plan of record. Paper framing & abstract: `notes/PAPER_TARGET.md`. Verified API facts:
`notes/API_NOTES.md`. Data facts: `notes/DATA_NOTES.md`.

## Core question
Given a patient's recent hemodynamic history + the known planned drug infusion, can TiRex-2 forecast
the hemodynamic response **zero-shot**, and does conditioning on the drug covariate improve the
forecast? Primary result = the **covariate ablation** (with vs without the drug covariate).

## Hard constraints
- **Zero-shot only.** No fine-tuning / training of TiRex-2 anywhere. Streaming, fine-tuning, and
  classification/regression heads are Pro-gated — do not use.
- Framing is monitoring/management (not control). Closed-loop = discussion-only implication.
- Verify every API detail and every VitalDB track name against the actual package / files.

## Working style
Small before large. Write as you learn (`API_NOTES.md`, `DATA_NOTES.md`). Reproducible: venv pinned in
`requirements.txt`, seeds, YAML configs, forecasts+metrics saved to disk. Don't fabricate. Ask before
large downloads / long runs.

---

## Phase 0 — Stand up TiRex-2 & characterize it  →  `notes/API_NOTES.md`
Isolated env; load gated weights (HF auth); device (cpu, try mps); forecast smoke test; characterize
`TimeseriesType`/`forecast` shapes & outputs; validate covariate mechanism with a controlled test;
NaN handling. **Status: near-complete — see API_NOTES.md.**

## Phase 1 — Ingest VitalDB  →  `notes/DATA_NOTES.md` + cached loader
Inspect format (`.vital` via `vitaldb` pkg); enumerate tracks & rates; select cases (art line + TCI/
vasoactive infusion + duration); hygiene (art-line artifact masking, numeric/waveform alignment, drug
step-function resampling, units, monotonic timestamps); cached loader → per-case aligned
`target`/`past_covariates`/`future_covariates` + metadata + sanity plots.

## Phase 2 — Single-case proof of concept
Target = MAP; future covariate = drug rate/CE over horizon; past covariates = HR/SpO2/CVP. One
rolling-origin window with a real intervention; forecast with vs without covariate; confirm shapes,
horizon, that the covariate visibly changes the forecast; check quantile calibration on the one case.
**Do not start Phase 3 until Phases 0–2 are validated.**

---

## Phase 3 — The study (organized around producing each ledger number)
Zero-shot throughout. Config-driven. Save per-window forecasts, per-case metrics, aggregate tables,
plots. Each ledger placeholder → a named experiment → a named output file (see `PAPER_TARGET.md`).

### 3.1 Cohort selection → produces `[N]`
- **Inclusion:** arterial line present (reliable MAP); ≥1 relevant drug infusion with rate and/or CE;
  minimum case duration; adequate signal quality.
- **Exclusion:** target mostly missing/artifact; case too short; unusable infusion channel.
- **Output:** `results/cohort_manifest.csv` + case-flow diagram (n at each exclusion step). `[N]` =
  final count. Record all thresholds in `notes/DATA_NOTES.md`.
- **GATE (blocks §3.1 sign-off) — infusion-fidelity verification.** Unlike both foils (which used
  VitalDB only as an *external* set), we use it as *primary* and *select for* infusion cases. Before
  Phase 3, explicitly confirm the continuous infusion trajectory (rate and/or CE) exists at adequate
  fidelity in the chosen tracks for the selected cohort — sampling density, non-NaN coverage over the
  window, plausible step dynamics. Note Kapral's caveat that **VitalDB lacks bolus-drug information**
  (so our covariate is the *infusion-pump* channel, not boluses — consistent with the §3.2 anesthetic-
  infusion choice). Record the check + outcome in `notes/DATA_NOTES.md`; do not proceed until it passes.

### 3.2 Task definition & evaluation protocol → produces `[H]`
- Target: MAP (numeric). Past covariates: HR, SpO2, CVP (as available). Future-known covariate: drug
  trajectory. **Test both infusion rate and CE**; keep the stronger, report both.
- Context length & horizon set: finalize from Phase 0 API behavior + Phase 2 MAP dynamics. Default
  horizon set **{1, 3, 5, 10} min → `[H]` = 10**; confirm sensible behavior at the longest horizon
  before committing. (Note: model max prediction length = 320 steps — see API_NOTES.)
- Rolling-origin: fixed stride, multiple origins per case; when aggregating, **cluster by case** and
  account for window overlap so error bars aren't overstated. Protocol → `configs/eval.yaml`.

### 3.3 Conditions
- **M1** — TiRex-2 **with** drug future-covariate (primary).
- **M0** — TiRex-2 **without** covariate (ablation partner).
- **B1** — persistence / last-value. **B2** — seasonal-naive (if periodicity). **B3** — per-case
  autoregressive fit on that window's own history only (no cross-case learning — keeps zero-shot claim
  clean; state explicitly).
- Optional **B4** — mechanistic differentiable cardiovascular model (Julia; David Lung) → the optional
  abstract sentence. Optional **B5** — another TSFM (Chronos/MOMENT) for context.
- **Baseline motivation:** B1 (persistence) and B3 (per-case AR) are motivated by the Vistisen
  "does a model beat simply extrapolating the MAP trajectory?" debate (via `notes/RELATED_WORK.md`,
  Kapral ref 15) — they test whether TiRex-2 earns its keep over trajectory extrapolation.
- **Trained-model reference points (cite, do NOT run as baselines):** Kapral 2024 (TFT, continuous
  forecast + binary) and Zhu 2026 (Transformer, classification) are supervised foils. We report their
  published numbers *alongside* ours in the results/comparison table (`results/comparison_foils.md`),
  not as methods we execute. See `notes/RELATED_WORK.md` for all foil numbers.

### 3.4 Primary metrics & the ablation → produces `[X%]` and `[Y%]`
- Primary metric: **CRPS** (uses TiRex-2 quantiles) + **MAE/RMSE on the median**. Per-horizon curves +
  aggregate.
- `[X%]` = relative error reduction M1 vs M0 = `(err_M0 − err_M1)/err_M0 × 100` on the primary metric.
  Per-horizon + one clearly-disclosed headline aggregate (state which).
- `[Y%]` = relative error reduction M1 vs the **best** baseline (B1–B3).
- Report **quantile calibration** (coverage/reliability) to support the "probabilistic" claim.
- **Statistics:** paired across cases (paired test or case-clustered bootstrap); report CIs for `[X%]`
  and `[Y%]`. Point estimates without CIs are not acceptable.
- **Produce `[X%]` FIRST — it's the paper's spine. If it doesn't hold, STOP and tell Max before
  proceeding.**
- **Interpretation guardrail (foils):** frame `[X%]` as *"a zero-shot foundation model exploits the
  known future infusion trajectory"* — **NOT** as *"the drug covariate helps forecasting"* (the latter
  is Kapral's already-published finding, Fig 3a). Our claim is the conjunction of zero-shot + future-
  known conditioning, quantified as a paired per-horizon ablation with CIs. See `notes/RELATED_WORK.md`
  and §3.7. This guardrail constrains wording only; it puts no pressure on the computed value.

### 3.5 Secondary task: impending hypotension → produces `[Z]`
- Event: MAP < 65 mmHg sustained ≥ [define, e.g. 1 min] within horizon `[H]`.
- Predictor: derive risk from probabilistic forecast (e.g. forecast P(MAP<65 within horizon)).
- Evaluate event-level: **AUROC (= `[Z]`) and AUPRC** (report both — AUPRC matters under imbalance) +
  median lead time. Select thresholds on a **disjoint dev set of cases**; report on held-out cases;
  cluster by case.
- Run the with/without-covariate ablation here too (covariate effect on AUPRC).
- **Report `[Z]` beside the trained supervised foils** as reference points: Kapral 5-min TFT AUROC
  0.909 internal / 0.903 external, and Zhu 5-min AUC 0.904 (see `notes/RELATED_WORK.md`). The claim is
  *training-free approaches the supervised benchmark* — **NOT** novelty of the hypotension task itself
  (that task is established). The novelty add is the future-known drug covariate on a zero-shot model.

### 3.6 Optional: mechanistic comparison
- If in scope, compare M1 vs B4 on the primary metric → supports optional abstract sentence. If not
  run, **delete that sentence** — don't leave it unsupported.

### 3.7 Guardrails (non-negotiable)
- Zero-shot: no training/fine-tuning of TiRex-2 anywhere.
- Case/patient-level separation for any dev/test split; no leakage across rolling origins in aggregate
  stats.
- **Integrity:** the abstract is a hypothesis template, not a target. Report whatever the data show. If
  the covariate does not help (`[X%]` ≈ 0 or negative), that is the finding — record it and revise the
  abstract. Never tune the analysis to hit a desired number.

### 3.8 Deliverables
- `results/` tables named as in the ledger; `RESULTS.md` summarizing findings **with CIs**.
- Update the ledger in `notes/PAPER_TARGET.md` with each computed value + CI + source file.
- Flag in `RESULTS.md` any abstract number that can't be supported, with a proposed abstract edit.
- Foil comparison table: `results/comparison_foils.md` (Zhu / Kapral / Ours), numbers sourced from
  `notes/RELATED_WORK.md`; "Ours" cells stay as ledger placeholders until Phase 3 computes them.

### 3.9 Reporting standards (flag only — do not implement now)
- Both foils follow TRIPOD / TRIPOD-AI. For the clinical-facing framing of the **secondary task**
  (hypotension early warning, §3.5), check TRIPOD-AI applicability/checklist when writing up. Flagged
  here as a to-do; no action this phase.
