# Reproducing all results (VitalDB)

End-to-end recipe to regenerate every number, table, and figure in the paper from raw data.
Two environments: **cluster** (SLURM + GPU, for the heavy runs) and **local Mac** (CPU, fine for
the post-hoc figures/tables). All commands run from the repo root with:

```bash
export PYP="PYTHONPATH=scripts:datasets/vitaldb"      # shared pipeline finds the dataset loader
```

Tags used throughout: `all2873` = anesthetic cohort, remi+propofol **CE** covariate (primary);
`all2873_covrate` = same cohort, **RATE** covariate; `cases115_covpressor` = phenylephrine subset.
Baselines are tagged `baseline-<model>_<cohort>` (e.g. `baseline-tft_all2873`).

---

## 0. One-time setup (cluster)

```bash
export HF_TOKEN=hf_xxx                       # gated NX-AI/TiRex-2 weights (put in ~/.bashrc)
bash slurm/download_vitalfiles.sh            # .vital + clinical_data.csv -> datasets/vitaldb/data/
bash slurm/build_container.sh                # one-time Pyxis image (deps + weights + kernels baked)
sbatch slurm/build_cache.sbatch              # CPU: loader caches + cohort_manifest.csv (the long pole)
```
Produces: `datasets/vitaldb/cache/` (anesthetic), `datasets/vitaldb/cache_pressor/` (phenylephrine),
`datasets/vitaldb/cohort_manifest.csv`, `results/cohort_flow.json`.

---

## 1. TiRex-2 zero-shot runs (GPU) — the primary results + covariate ablation

```bash
COV=ce      sbatch slurm/run_ablation.sbatch     # -> results/ablation_{windows,primary}_all2873.{csv,json}
COV=rate    sbatch slurm/run_ablation.sbatch     # -> ..._all2873_covrate.*
COV=pressor sbatch slurm/run_ablation.sbatch     # -> ..._cases115_covpressor.*
# or all three with auto-dependency after the cache job:
bash slurm/submit_all.sh
```
Each writes per-window forecasts (`ablation_windows_*.csv`, phase3 schema) + a summary
(`ablation_primary_*.json`) with MAE/CRPS, the covariate effect **X%** (M0→M1) and **Y%** (vs
persistence), stratified all/transition/steady, with case-clustered CIs.

Zero-shot ⇒ no training. The full cohort runs on one A100 in ~1–2 h each.

---

## 2. Matched supervised baselines (GPU) — TFT & PatchTST

Trained on the **same windows / subject-splits / metrics** as TiRex (canonical 60/20/20 split),
so the comparison is apples-to-apples. Each job trains M1 (with drug covariate) and M0 (without).

```bash
# --- classification head-to-head (CE cohort); auto-runs compare.py ---
MODEL=tft       COV=ce sbatch slurm/train_baseline.sbatch   # -> baseline-tft_all2873.*  + matched_comparison_*.json
MODEL=patchtst  COV=ce sbatch slurm/train_baseline.sbatch   # -> baseline-patchtst_all2873.* + matched_comparison_*.json

# --- covariate-representation parity for Fig 2c (RATE + phenylephrine arms) ---
MODEL=tft COV=rate    sbatch slurm/train_baseline.sbatch     # -> baseline-tft_all2873_covrate.*
MODEL=tft COV=pressor MATCH=results/ablation_windows_cases115_covpressor.csv \
    CONFIG=datasets/vitaldb/configs/data_pressor.yaml \
    sbatch slurm/train_baseline.sbatch                       # -> baseline-tft_cases115_covpressor.*
```
Each writes `ablation_windows_<baseline-tag>.csv` (test split, phase3 schema),
`baseline_history_<tag>.json` (train/val loss curves), `baseline_meta_<tag>.json`, and for COV=ce
`matched_comparison_<baseline-tag>.json` (TiRex vs baseline on identical test subjects + foils).

Add more architectures by registering them in `scripts/baselines/models.py` (`MODELS` dict) and
submitting with `MODEL=<name>`.

---

## 3. Pull results to the Mac (for figures/tables)

`outputs/` and the big `ablation_windows_*.csv` are git-ignored, so copy them off the cluster:
```bash
scp -r <cluster>:~/tirex-2/tirex-vitaldb/results ./            # JSONs, matched comparisons, windows CSVs, histories
```
(The small manifest, foil tables, and kapral curves are already in the repo.)

---

## 4. Post-hoc analyses (CPU) — feeds Fig 3/4 & Tables 3–5

Run locally with a Python that has numpy/scipy/pandas/matplotlib (or inside the container on the
cluster via `slurm/make_figures.sbatch`). These read the windows CSVs:
```bash
$PYP python scripts/hypo_eval.py       all2873      # hypotension ROC/PR/calibration/operating-points/pAUROC
$PYP python scripts/clinical_eval.py   all2873      # lead time / severity gradient / decision curves
$PYP python scripts/subgroup_forest.py all2873 5    # subgroup AUROC forest @5 min
$PYP python scripts/plot_kapral_mae.py all2873      # MAE vs Kapral overlay (standalone)
# on the cluster instead: TAG=all2873 sbatch slurm/make_figures.sbatch
```

Matched comparison (re-scores TiRex + a baseline on the identical canonical test split):
```bash
$PYP python scripts/baselines/compare.py --tirex all2873 --baseline baseline-tft_all2873
$PYP python scripts/baselines/compare.py --tirex all2873 --baseline baseline-patchtst_all2873
```

---

## 5. Paper figures + tables (CPU)

One command builds everything:
```bash
$PYP python scripts/paper_figures.py all2873
```
Outputs (Nature-style, PDF + 600-dpi PNG in `outputs/figs/paper/`, tables in `results/tables/`):

| Artifact | Content |
|---|---|
| Fig 1 | study design, cohort funnel, example forecasts |
| Fig 2 | (a) forecast accuracy TiRex vs TFT; (b) covariate value by window type, TiRex+TFT; (c) CE/RATE/pressor + TFT overlay; (d) MAE vs Kapral |
| Fig 3 | hypotension: ROC, AUROC-vs-horizon, calibration, PR, decision curve, head-to-head bars — TiRex vs TFT vs foils |
| Fig 4 | lead time, severity gradient, subgroup forest, operating characteristics |
| Fig S | TFT M1/M0 training curves |
| Table 1 | cohort characteristics (n=2,708 windows-contributing) |
| Table 2 | forecast accuracy + covariate value (TiRex) |
| Table 3 | hypotension classification vs foils |
| Table 4 | matched classification AUROC (TiRex vs TFT vs foils) |
| Table 5 | matched forecasting CRPS/MAE (TiRex vs TFT) |

`paper_figures.py` degrades gracefully: panels/tables that need a baseline (Fig 2c TFT overlay,
Fig 3 TFT curves, Tables 4/5, Fig S) are drawn only if the corresponding baseline files are present.

---

## Dependencies between steps
```
setup(0) ── cache ──> TiRex runs(1) ──> post-hoc(4) ─┐
                          └────────────> baselines(2) ┴─> figures/tables(5)
```
Regenerating figures after any run is just step 5 (seconds). The heavy steps (1–2) are
deterministic given the seed, so reruns reproduce identical numbers.

## Adding a new dataset (e.g. MOVER)
Drop it in as `datasets/mover/` (loader + configs + cache), then steps 1–5 run unchanged with the
new cohort tag — the pipeline in `scripts/` is dataset-agnostic.
