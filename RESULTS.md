# RESULTS (stub)

Phase 3 findings will be summarized here **with CIs** once computed. Nothing below is a result yet;
the ledger in `notes/PAPER_TARGET.md` is authoritative for all numbers.

## Flagship run — design + interim (NOT final; ledger unfilled until the full run)

Setup (`scripts/phase3_ablation.py`, `results/ablation_primary_n300_s1.json`, dashboard
`outputs/figs/dashboard_n300_s1.png`): 15 s grid, horizons {1,3,5,7,10,15} min, drug covariate =
remi+propofol CE. Four conditions — **M1** (past covs + drug), **M0** (past covs), **M1_to** (drug only),
**M0_to** (target only) — + persistence. Case-clustered bootstrap. **Stratified by transition vs steady**
windows (does remi CE change >0.5 ng/mL in the horizon?). `[X%]_withpast`=M1 vs M0; `[X%]_targetonly`
=M1_to vs M0_to (tests Zhu's "vitals implicitly capture drugs").

Interim signals (small-n pilots, will be replaced by the 300-case run):
- **TiRex vs SOTA:** zero-shot median MAE ≈ 5.9 mmHg at 7 min — at/below Kapral'24 external VitalDB MAE
  (7.0) — and ~3× better CRPS than persistence. Strong "training-free approaches/beats supervised" line.
- **Covariate effect is small** (see counterfactual, DATA_NOTES): M1≈M0≈target-only in CRPS. Watching
  whether the transition-window stratum shows a larger, CI-positive `[X%]`. If `[X%]`≈0 even there,
  that is the honest finding and the abstract claim gets revised (§3.7).

## Limitations (standing scaffold — expand as work proceeds)

- **Realized vs planned infusion (optimistic upper bound).** In retrospective VitalDB the "future-known"
  infusion trajectory we feed is **what the clinician actually did**, recorded post hoc. This is an
  *optimistic upper bound* relative to deployment, where the future drug plan is not known exactly ahead
  of time. It is the **mirror image of Kapral's reactive-intervention miscalibration**: their trained
  model overestimates because it *cannot* anticipate future medication; we benefit because we are *given*
  it. State this plainly. (Cross-references the abstract's existing "gap between observed and truly
  planned infusions" line — do not duplicate that wording; point to it. See `notes/RELATED_WORK.md`,
  `notes/DATA_NOTES.md`.)
- **Retrospective, single-center-style primary data** (VitalDB, Seoul National University), used here as
  *primary* and selected for infusion cases — see the §3.1 infusion-fidelity gate (`PROJECT_PLAN.md`).
- **Covariate = anesthetic infusion, not vasopressors** (pressor infusion channels near-absent; VitalDB
  lacks bolus-drug info, per Kapral). Frames the intervention→response as anesthetic-induced BP change.
- _(add per-horizon calibration, cohort-generalizability, zero-shot-vs-tuned caveats as results land.)_
