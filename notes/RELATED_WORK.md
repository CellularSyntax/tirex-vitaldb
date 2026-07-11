# Related work — the two trained foils

**Single source of truth for foil facts/numbers.** The intro, discussion, and comparison table draw
from here — do not re-type these numbers elsewhere (cross-reference this file). Numbers verified
against the PDFs in repo (`PIIS2589537024003766.pdf` = Kapral; `journal.pmed.1005024.pdf` = Zhu) on
2026-07-11 unless marked *[from brief, not independently re-confirmed]*.

Framing constraint (see `PAPER_TARGET.md` framing rules, `PROJECT_PLAN.md` §3.7): our contribution is
the **conjunction** of (zero-shot foundation model) + (future-known covariate conditioning), quantified
with CIs — **never** "the drug covariate helps forecasting," which Kapral already published.

---

## Foil A — Kapral et al., *eClinicalMedicine* 2024;75:102797 (THE close one)

Code: github.com/lorenzkap/MAP_TFT. **Same institution as us** (Medical University of Vienna / General
Hospital of Vienna).

- **What they did:** Trained a Temporal Fusion Transformer (TFT) *from scratch* on **73,009 patients**
  (non-cardiothoracic general anaesthesia, GH Vienna 2017–2020; demographics + vital signs +
  **medication** [52 features incl. propofol, vasopressors] + ventilation, sampled every 15 s).
  Forecasts continuous MAP **7 min ahead** (28 steps @ 15 s) from an 8-min input window, and derives a
  binary hypotension (MAP < 65) prediction at 1/3/5/7 min. Same overall shape as our study.
- **Headline numbers (verified):** continuous MAP **MAE 4 mmHg internal / 7 mmHg external**
  (0.376 / 0.622 SD). Binary hypotension **AUROC mean 0.933 internal / 0.919 external**. Internal test
  n = 8113; **external test = VitalDB, n = 5065.** Per-horizon *[from brief]*: 5-min AUROC 0.909
  internal / 0.903 external; 7-min 0.880 / 0.867. vs Alarm/HPI: TFT 5-min 0.909 vs HPI 0.926 AUROC,
  but higher specificity (0.960 vs 0.858).
- **Medication matters (already published — NOT our novelty):** Fig 3a shows forecasts degrade when
  propofol is omitted; they call intraoperative medication "the most important factor" behind
  performance. The directional claim *drug data improves the MAP forecast* is theirs.
- **Fig 3d feature importance (relative relevance) — KEY for our discussion (values approx. from bar
  chart):** `Meanbp (MAP) ≈ 0.79` (dominant), `Vasopressin perfusor ≈ 0.32`, Cisatracurium ≈ 0.16,
  Succinylcholine ≈ 0.15, Invasive BP ≈ 0.14, **etCO2 ≈ 0.13, Remifentanil perfusor ≈ 0.13**, Fentanyl
  ≈ 0.12, Noradrenaline perfusor ≈ 0.11, Phenylephrine ≈ 0.11. **Implications for us:** (1) even in a
  *trained* model the MAP trajectory carries ~80% of the signal and remifentanil only ~0.13 → our
  *small* covariate effect is consistent with their own importance ranking, not a defect. (2) The most
  informative drug (vasopressin ≈0.32) plus the pressors (noradrenaline/phenylephrine ≈0.11) are
  exactly the ones VitalDB does **not** record as continuous infusion trajectories (boluses/untimed) →
  a ceiling on any covariate's achievable value here — a stated limitation, and a motivation for the
  future-known framing on the drugs we *do* have. (3) etCO2 ≈0.13 (≈ remi) → supports our queued etCO2
  covariate enrichment.
- **Why it does NOT scoop us — their limitation is our opening (verified quotes):** their medication
  input is **past/observed only**. They modified the DeepMind TFT code because it "cannot be used when
  lacking future-known time points." Fig 1 caption, verbatim: *"As the administration of phenylephrine
  occurs after the prediction start, it cannot be taken into account for forecasting MAP."* They
  attribute forecast overestimation/miscalibration to the model *"not anticipating future"* medical
  interventions. **We feed the known future infusion trajectory over the horizon — precisely the gap
  they name.** Their propofol-omission check is a post-hoc interpretability sidebar on a trained model,
  not a paired, per-horizon, CI-bearing ablation.
- **Also useful:** they frame the Vistisen debate (their ref 15 — does a model beat simply
  extrapolating the MAP trajectory?). We cite this to motivate our persistence (B1) and per-case AR
  (B3) baselines. They also caveat that **VitalDB lacks bolus-drug information** (relevant to our §3.1
  infusion-fidelity gate).

## Foil B — Zhu et al., *PLOS Medicine* 2026;23(3):e1005024

Code: github.com/ShouqiangZhu/IOH_Transformer.

- **What they did:** Trained a Transformer on **319,699 surgical cases** (tertiary hospital, Nanjing,
  China, 2013–2023) for **binary IOH classification** (MAP < 65; *not* continuous forecasting) at
  **5/10/15-min** horizons, using **vital-sign time series only**. External validation on **VitalDB
  (n = 5,260)** *[from brief]* — note the abstract describes the external step as a real-time **alert
  simulation on 10 representative VitalDB cases**; treat n=5,260 as the external cohort and the 10-case
  sim as the deployment demo.
- **Headline numbers (verified):** internal AUCs **0.904 / 0.892 / 0.882** (5/10/15 min; recall ≥88.3%).
  Higher recall than XGBoost (internal 5-min recall 0.891 vs 0.737).
- **Why it does NOT scoop us — and why it's a clean intro foil:** they **deliberately exclude
  medication/drug data**, arguing vital signs *implicitly* capture drug effects. **Our covariate
  ablation is the direct empirical test of an assumption they assert but never test.** Also a
  classification task, not probabilistic forecasting.

---

## Our differentiators (anchor all framing on these)

1. **Zero-shot vs trained.** Both foils train site-specific models on large labeled cohorts
   (73k / 320k). We use TiRex-2 off the shelf, **no training**. They *are* the "large, site-specific
   labeled training set" our abstract positions against.
2. **Future-known covariate vs past-only.** The sharp edge. Kapral conditions on medication as observed
   history and explicitly cannot use post-origin interventions; we condition on the drug trajectory
   **over the forecast horizon**, quantified as a **paired, per-horizon, CI-bearing ablation**.
3. **Venue/contribution type.** Foils are clinical prediction-model papers (TRIPOD-AI); ours is a
   **methods contribution** (BSPC / JBHI). Keep framing methodological.

## Canonical novelty statement (anchor wording — adapt lightly, don't drift)

> A zero-shot foundation model, conditioned on the future-known infusion trajectory that trained
> past-covariate models such as Kapral's explicitly cannot exploit, with the covariate contribution
> quantified as a paired, per-horizon ablation with confidence intervals.
