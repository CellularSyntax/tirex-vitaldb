# tirex-vitaldb

Zero-shot forecasting of intraoperative mean arterial pressure (MAP) with **TiRex-2**, conditioned on
the known future drug-infusion trajectory, on the **VitalDB** open dataset. Primary result is a paired
covariate ablation (with vs without the drug covariate); secondary task is impending-hypotension
(MAP < 65 mmHg) early warning. No training / fine-tuning of TiRex-2 anywhere — zero-shot throughout.

See `notes/` for the full plan (`PROJECT_PLAN.md`), paper framing (`PAPER_TARGET.md`), data facts
(`DATA_NOTES.md`), the trained-model foils we compare against (`RELATED_WORK.md`: Kapral 2024, Zhu 2026),
and the cluster runbook (`CLUSTER.md`).

## Layout
Dataset-specific code/data live under `datasets/<name>/`; the forecasting + analysis pipeline is shared,
so a second dataset (e.g. MOVER) drops in as `datasets/mover/` reusing `scripts/`.
```
datasets/vitaldb/   loader, cohort builder, scan tools; configs/ (data.yaml, data_pressor.yaml);
                    cohort_manifest.csv + pressor_cases_phen.txt; data/ (raw, gitignored) + cache/
scripts/            shared pipeline: phase3 ablation (flagship) + post-hoc analyses
                    (hypo_eval, clinical_eval, subgroup_forest, plot_kapral_mae, merge_dashboard)
configs/eval.yaml   shared evaluation protocol
slurm/              cluster deployment (Pyxis container build, cache build, GPU ablation, submit_all)
notes/              plan, related work, HPC porting, cluster runbook, resume/handoff
results/            run outputs + curated (kapral digitized curves, foil comparison tables)
```
Run scripts with `PYTHONPATH=scripts:datasets/vitaldb` so the shared pipeline finds the dataset loader.

## Data (not included — fetch it)
The **VitalDB raw `.vital` files, `clinical_data.csv`, `lab_data.csv`, and the papers are not committed**
(see `.gitignore`). Only the small case manifest + pressor case list are included. VitalDB is an open
dataset (https://vitaldb.net); please cite it and follow its terms. On a cluster, re-fetch the raw data
(`.vital` + `clinical_data.csv`) with `slurm/download_vitalfiles.sh` into `datasets/vitaldb/data/`.

## Run locally (macOS, CPU)
```bash
PY=/path/to/venv/bin/python
PYTHONPATH=scripts:datasets/vitaldb $PY scripts/phase3_ablation.py --n-cases 300 --seed 1     # anesthetic (remi+propofol CE)
PYTHONPATH=scripts:datasets/vitaldb $PY scripts/hypo_eval.py n300_s1                          # hypotension ROC/PR/calibration
PYTHONPATH=scripts:datasets/vitaldb $PY scripts/clinical_eval.py n300_s1                      # lead time / severity / decision curve
```

## Run the full cohort on the cluster (SLURM + GPU)
See `notes/CLUSTER.md`. In short:
```bash
export HF_TOKEN=hf_xxx                 # gated NX-AI/TiRex-2 weights
bash slurm/download_vitalfiles.sh      # re-fetch .vital
bash slurm/build_container.sh          # one-time Pyxis image (deps + weights + kernels baked)
bash slurm/submit_all.sh               # cache (CPU) -> ce + rate + pressor (GPU), auto-dependency
```

Zero-shot foundation model; not a medical device; research use only.
