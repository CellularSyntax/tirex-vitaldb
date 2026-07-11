# Paper target — north star

**Every experiment must map to a claim in the abstract below.** If the data don't support a claim,
we revise the claim, not the analysis (see integrity guardrail in `PROJECT_PLAN.md` §3.7).

## Venue & framing

- **Venue:** *Biomedical Signal Processing and Control* (primary); *IEEE JBHI* as reach.
- **Framing:** monitoring-forward; the covariate ablation (with vs without the drug covariate) is the
  scientific spine.
- **Framing rules for all work:**
  1. Lead with intraoperative **monitoring/management**, not control theory.
  2. The **ablation** (with vs without the drug covariate) is the central result.
  3. Closed-loop / surrogate language is a **forward-looking implication only** — confine to Discussion.
  4. **Position against two trained foils (Kapral 2024, Zhu 2026):** our claim is the **conjunction** of
     zero-shot + future-known conditioning, quantified with CIs — **never** "the drug covariate helps"
     (already published by Kapral). See `notes/RELATED_WORK.md`.

## Abstract (verbatim — keep bracketed placeholders until computed)

> **Abstract.** Anticipating how a patient's hemodynamics will respond to drug administration is
> central to intraoperative monitoring and management, yet models that link interventions to
> physiological signals are typically either bespoke and patient-calibrated or require large,
> site-specific labeled training sets, limiting portability. We investigate whether a general-purpose
> time-series foundation model can forecast intraoperative hemodynamic response without any
> task-specific training. Using TiRex-2, a zero-shot multivariate model that conditions on past and
> future-known covariates, we forecast mean arterial pressure from recent vital-sign history while
> conditioning on the known drug-infusion trajectory over the forecast horizon, on [N] surgical cases
> from VitalDB. Our central analysis is an ablation isolating the contribution of the intervention
> covariate — forecasting with versus without the drug-infusion input — evaluated with rolling-origin,
> probabilistic metrics over clinically relevant horizons up to [H] minutes. Conditioning on the
> infusion covariate reduced [median/CRPS] forecast error by [X%] and outperformed persistence and
> per-case autoregressive baselines by [Y%], entirely zero-shot. We further show the forecasts support
> a monitoring-relevant secondary task, early warning of impending hypotension (MAP < 65 mmHg), with
> an AUROC of [Z]. [Optionally: forecasts approached a mechanistic differentiable cardiovascular model
> without any physiological priors.] These results indicate that covariate-conditioned foundation
> models can provide training-free, portable forecasting of intervention response, with potential as
> an off-the-shelf surrogate for model-based and closed-loop hemodynamic control. We discuss
> limitations, including retrospective single-center data and the gap between observed and truly
> planned infusions.

## Abstract number ledger

Filled by the Phase 3 experiments. **Point estimates without CIs are not acceptable.**

| Placeholder | Meaning | Produced by | Value (fill when computed) | 95% CI | Source file |
|---|---|---|---|---|---|
| `[N]` | final cohort size | cohort selection (§3.1) | **2659** (eligible; may subsample for compute) | n/a (exact count) | `results/cohort_manifest.csv` |
| `[H]` | max forecast horizon (min) | eval config (§3.2) | **15** (set {1,3,5,7,10,15}; 7=Kapral, 15=Zhu anchor) | n/a (design choice) | `configs/eval.yaml` |
| `[X%]` | error reduction, covariate vs no-covariate | primary ablation (§3.4) | — | — | `results/ablation_primary.csv` |
| `[Y%]` | error reduction, model vs best baseline | baseline comparison (§3.4) | — | — | `results/baselines.csv` |
| `[Z]` | hypotension early-warning AUROC | secondary task (§3.5) | — | — | `results/hypotension.csv` |
| optional | vs mechanistic CV model | mechanistic comparison (§3.6) | — | — | `results/mechanistic.csv` |

`[median/CRPS]`: state which metric is the headline for `[X%]` once computed (default CRPS).

**Note on the `[X%]` row (interpretation only — value/CI/source unchanged):** frame `[X%]` as *a
zero-shot FM exploiting the known future infusion*, not *"the drug covariate helps"* (Kapral's
published finding). See `PROJECT_PLAN.md` §3.4 guardrail and `notes/RELATED_WORK.md`. This constrains
wording only — the computed value is reported exactly as the data show (§3.7).
