# Porting to HPC (Linux + CUDA) for the full-cohort run

The Mac run is CPU-only (no MPS — `device="mps"` raises; sLSTM kernels are CUDA-only). On HPC you get
the custom CUDA kernels → the full 2659-case run is fast and none of the macOS/iCloud issues apply.

## What to change (only these)

1. **Env:** Linux + CUDA (compute capability ≥ 8.0). `pip install "tirex-2[examples,gluonts]"` with a
   CUDA torch (`torch>=2.8,<2.10`). Custom kernels **compile on first `load_model`** — needs the CUDA
   toolkit + `nvcc`; optionally set `TORCH_CUDA_ARCH_LIST` (e.g. `"8.0;8.6;9.0"`). `HF_TOKEN` env var
   (gated weights). `requirements.txt` pins the Mac set — regenerate on HPC after install.
2. **Device:** pass `--device cuda` (already wired). On GPU use a larger `--batch-size` (e.g. 512–1024).
   No sharding needed — a single GPU process handles it; the sLSTM custom kernel is the fast path.
3. **Paths (`configs/data.yaml`):** set `vital_dir` (copy `vital_files/` over) and `cache_dir` to local
   HPC scratch. **iCloud-specific bits are irrelevant on Linux** (no eviction). `resample_to_s: 15`,
   `despike`, ranges — keep as-is (they're portable analysis choices).
4. **Run:** `PYTHONPATH=scripts python scripts/phase3_ablation.py --all --device cuda --batch-size 512`.
   (`--all` = full 2659 cohort. Drop `--no-plot`/sharding; single process refreshes the dashboard.)

## Reuse vs rebuild the cache

- The npz loader cache (`/Users/admin/DATA/tirex2/cache/*.npz`) and `results/cohort_manifest.csv` are
  plain, portable files — **copy them to HPC** and point `cache_dir` there to skip re-reading `.vital`.
- Or rebuild from scratch on HPC: `python scripts/build_cohort.py --workers N` (re-reads `.vital`;
  fast on a real filesystem). Produces the same manifest + caches.

## Scripts that use the GPU (need `--device cuda`)
- `scripts/phase3_ablation.py` (flagship; `--device` arg) — the main one.
- `scripts/diag_covariate.py`, `scripts/phase2_single_case.py` — hardcode `device="cpu"`; change to
  `"cuda"` if you rerun diagnostics on HPC (or add a `--device` arg mirroring the flagship).
- `build_cohort.py` / `vitaldb_loader.py` — **CPU only** (no model), unchanged.

## Sanity check on HPC before the full run
1. `python scripts/phase3_ablation.py --n-cases 20 --device cuda` → confirm it matches Mac numbers
   (zero-shot is deterministic given the same inputs; CRPS/MAE should agree within tiny fp differences).
2. Then `--all --device cuda`. Expect the covariate effect to be the same science (modest, concentrated
   in transition windows) — GPU only changes speed, not results.
