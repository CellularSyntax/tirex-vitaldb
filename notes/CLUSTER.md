# Running the full cohort on the MSC Slurm cluster

Mirrors the proven jaxfibers Pyxis/enroot container workflow. GPU work here is **zero-shot inference**
(no training) — one A100 handles the whole 2659 cohort on the sLSTM CUDA fast path, so we need ~1 CPU
job + 3 GPU jobs, not the full 4×a100+2×h100 allowance. See `notes/HPC_PORTING.md` for the CUDA notes.

## Files (in `slurm/`)
| file | what |
|---|---|
| `download_vitalfiles.sh` | re-download the needed `.vital` into `vital_files/` (manifest ∪ pressor list) |
| `build_container.sh` | one-time: build `$HOME/containers/tirex2.sqsh` (deps + gated weights + kernels baked) |
| `setup_env.sh` | per-job env (skips pip if baked; sets HF/kernel caches on the mounted FS) |
| `build_cache.sbatch` | CPU job: `build_cohort.py` (anesthetic cache+manifest) + phenylephrine cache |
| `run_ablation.sbatch` | GPU job for one covariate: `COV=ce|rate|pressor` |
| `submit_all.sh` | orchestrates cache → {ce, rate, pressor} with `--dependency=afterok` |
| `../requirements_gpu.txt` | pip deps installed into the container |

## One-time setup
```bash
cd <project on cluster>            # copy the repo over; keep datasets/vitaldb/cohort_manifest.csv + datasets/vitaldb/pressor_cases_phen.txt
export HF_TOKEN=hf_xxxxx           # gated NX-AI/TiRex-2 weights (put in ~/.bashrc)

bash slurm/download_vitalfiles.sh  # ~5.4k .vital -> vital_files/ (login or `storage` partition). VERIFY the
                                   #   VitalDB URL in the script once; ensure disk space.
bash slurm/build_container.sh      # ~10-15 min on one GPU; pins BASE_IMAGE (torch 2.8<=v<2.10 — see below)
```
**BASE_IMAGE / torch:** `build_container.sh` defaults to `nvcr.io#nvidia/pytorch:25.09-py3`. tirex-2 0.1.1
needs `2.8<=torch<2.10`; check the tag's torch (`25.06`≈2.8, `25.09`≈2.9) and override `BASE_IMAGE=` if needed.

## Run
```bash
export HF_TOKEN=hf_xxxxx
bash slurm/submit_all.sh           # cache (CPU) -> ce + rate + pressor (3× a100), auto-dependency
# or individually:
sbatch slurm/build_cache.sbatch
COV=ce      sbatch --dependency=afterok:<cacheJID> slurm/run_ablation.sbatch
COV=rate    sbatch --dependency=afterok:<cacheJID> slurm/run_ablation.sbatch
COV=pressor sbatch --dependency=afterok:<cacheJID> slurm/run_ablation.sbatch
```
Monitor: `squeue -u $(whoami)` · `tail -f logs/tirex2-*-<jobid>.out`.

Outputs (same as the Mac subset, full-cohort tags): `results/ablation_primary_all<N>.json` (ce),
`..._covrate.json`, `..._covpressor` (via `cases115_covpressor`), plus `outputs/figs/dashboard_*.png`.
The full run natively captures the instrumentation columns (`mae_inst_*`, `t_event_65`,
`hypo_event_55/50`, `risk_*_55/50`), so all post-hoc scripts (`hypo_eval`, `clinical_eval`,
`subgroup_forest`, `plot_kapral_mae`) run unchanged on the cluster outputs.

## Optional: shard one covariate across GPUs (faster, uses more of the 6 slots)
Not needed for a single A100, but the `run_ablation.sbatch` supports it (mirrors `submit_duke_sweep.sh`):
```bash
# e.g. split the ce full run across 4 a100 + 2 h100 (6 disjoint shards), then merge:
COV=ce N_SHARDS=6 SHARD_OFFSET=0 sbatch --array=0-3 -q a100 --gres=gpu:a100:1 slurm/run_ablation.sbatch
COV=ce N_SHARDS=6 SHARD_OFFSET=4 sbatch --array=0-1 -q h100 --gres=gpu:h100:1 slurm/run_ablation.sbatch
# after both finish (single container run, no GPU): python scripts/merge_dashboard.py all<N>
```

## Resource sizing (MSC limits: 6 GPU jobs, 12 CPU jobs; QOS 72h)
- **CPU cache build**: `-p cpu -q cpu`, 32 cpus, ~a few hours (I/O over ~2700 `.vital`). The long pole.
- **GPU runs**: `-p gpu -q a100 --gres=gpu:a100:1`, ~1-2 h each on one A100. 3 jobs = 3/6 slots.
- No need for h100/b200 here (inference-bound, not compute-bound). a100 is plenty; a30/a16 also fine but slower.
