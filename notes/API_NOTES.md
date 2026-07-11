# TiRex-2 API notes (verified)

Living cheat-sheet for the *open* `tirex-2` package. Everything here is checked against the
actually-installed package unless marked **[UNVERIFIED]** (needs gated weights to test).

## Environment (verified)

- **venv location: `/Users/admin/DATA/tirex2/venv` (LOCAL, non-iCloud).** Invoke as
  `/Users/admin/DATA/tirex2/venv/bin/python`. Loader cache: `/Users/admin/DATA/tirex2/cache`.
- ⚠️⚠️ **iCloud gotcha (cost hours) — the venv MUST NOT live under Desktop/Documents/Downloads.** Those
  folders are all iCloud-synced. The original venv at `./.venv` (on Desktop) had ~4000 package files
  **evicted to iCloud placeholders**; importing `tirex2` → `torch._dynamo` → `sympy` hung at 0% CPU in
  `importlib get_data` reading an evicted file (waiting on an iCloud download that stalls). Symptom =
  process alive, ~0% CPU, stuck in module import. **Fix = venv + cache on local disk (`/Users/admin/DATA/...`).**
  Same root cause as the `.vital` I/O timeouts. Recreate: `python3 -m venv /Users/admin/DATA/tirex2/venv
  && .../bin/pip install -r requirements.txt vitaldb`.
- Original (deprecated, on iCloud): `./.venv`. Do not use — subject to random import hangs.
- Installed versions: **tirex-2 0.1.1**, torch **2.9.1**, numpy **2.1.3**, xlstm **2.0.5**, einops, flashrnn.
- `requires_python`: `>=3.11,<3.14`. Deps: `torch>=2.8,<2.10`, `numpy~=2.1.3`, `einops~=0.8.1`,
  `flashrnn>=1.0.4`, `xlstm~=2.0.3`, `huggingface-hub~=0.32.0`. `[examples]` adds matplotlib/plotly/jupyterlab.
- MPS: `torch.backends.mps.is_built() == True`, `is_available() == True`. **But** the sLSTM/mLSTM cell
  code types `device` as `Literal["cpu","cuda"]` (see `bi_xlstm.py`), so the recurrent kernels are
  cpu/cuda only — **mps is untested and may fall back or error**. Test empirically (see smoke test). Default to `device="cpu"` on this Mac.

## Access / auth (BLOCKER)

- Weights `NX-AI/TiRex-2` on HF are **gated**. Steps: (1) accept license on the HF model page while
  logged in, (2) provide a token with "Read access to public gated repos" via `HF_TOKEN` env var or
  `huggingface-cli login`. No token currently present on this machine (`~/.cache/huggingface/` has no `token`).
- Model size: ~38.4M params univariate + 44.1M multivariate ≈ 82.5M params → small download (~a few hundred MB, not flagged as large).

## Public API (verified by introspection + source)

Top-level `tirex2` exports: `load_model`, `TimeseriesType`, `ForecastModel`, `TiRex2`, `api_adapter`, `base`, `model`.

### `load_model`
```python
load_model(ckpt_path: str | Path = "NX-AI/TiRex-2", device: str = "cuda", *, hf_kwargs: dict | None = None) -> ForecastModel
```
- **Correction vs brief:** default `device="cuda"` — must pass `device="cpu"` explicitly on Mac.

### `TimeseriesType` (dataclass, `tirex2/model/types.py`)
```python
TimeseriesType(target: Tensor, past_covariates: Tensor | None, future_covariates: Tensor | None)
```
- `target`          : `[V_t, T]`   (multivariate targets allowed)
- `past_covariates` : `[V_p, T]`   or None
- `future_covariates`: `[V_f, >= T+H]` or None. **Extra future steps are ignored.**
- Derived props: `past_length = target.shape[-1]`;
  `future_length = future_covariates.shape[-1] - past_length` → **H is inferred** from future-covariate length.
- So a future covariate spans the *whole* window `[0 .. T+H]` (context then horizon concatenated),
  confirmed by `demo.Demo.to_timeseries_type`: `future_cov = concatenate([c.context, c.future])`.

### `ForecastModel.forecast` (`tirex2/api_adapter/forecast.py`)
```python
model.forecast(
    timeseries: list[TimeseriesType],
    prediction_length: int,
    output_type: Literal["torch","numpy","gluonts","fev"] = "torch",
    batch_size: int = 512,
    yield_per_batch: bool = False,
    **predict_kwargs,
)
```
- **Corrections vs brief:** input is `timeseries=[ts]` (a list) + explicit `prediction_length`;
  default `output_type="torch"` (README example passes `"numpy"`).
- Other methods: `forecast_fev`, `forecast_gluon`.
- **Output shape (from README, [UNVERIFIED] locally):** `(n_targets, n_quantiles=9, prediction_length)`.
  i.e. quantile-major, **not** `[?, H, n_quantiles]` as the brief guessed. 9 quantile levels by default
  (levels TBD — read `model.quantiles` buffer once loaded).

## Covariate mechanism (verified in source)

- Bidirectional xLSTM stack with `split_method="reverse_known_only"` (`bi_xlstm.py`):
  - forward cell sees **all** variates; reverse cell sees only **known** covariates.
  - "known" = **non-NaN over the future window**. Past-only covariates (all-NaN future) are forward-only.
- In `tirex2.py` forward pass: `has_future = ~all(isnan(x[:, -future_len:]), dim=-1)`;
  `known_covariate_mask = has_future & ~target_mask`.
- **Takeaway for us:** a fully-observed drug schedule over `[0..T+H]` → treated as *known* → gets
  bidirectional conditioning. This is exactly the "scheduled intervention" path we want. Ablation
  "without covariate" = drop `future_covariates` (or NaN-out the future window → forward-only).

## Missing values (verified in source, behavior [UNVERIFIED])

- NaN-native: `x_mask = (~isnan(x))`, `x = nan_to_num(x, nan=self.nan_mask_value)`, mask concatenated to
  input embedding. NaN-safe scaler. So NaNs in context are expected and masked, not fatal. Confirm empirically.

## Plotting (`tirex2/plotting.py`)

- Public: `plot_forecast`, `plot_multivariate`, `plot_covariate` (+ color constants). **[correction to brief:
  it's `tirex2.plotting`, and `plot_multivariate` exists here — good.]**

## Demo (`tirex2.demo`)

- Dataclasses `Demo(title, description, target_context, target_future, covariates)` and
  `Covariate(label, context, future, kind="flag"|"cont")`. `Demo.to_timeseries_type(include_covariates=True)`
  shows the canonical build: past-cov = covariates with `future is None`; future-cov = `concat(context, future)`.
- Use this as the template for building VitalDB `TimeseriesType`s.

## Phase 0 empirical results (VERIFIED — `scripts/phase0_smoke.py`, `outputs/phase0_smoke_summary.json`)

Ran on this Mac, `device="cpu"`, checkpoint cached locally. **Phase 0 complete.**

- **Load + inference (CPU):** load **0.9 s**, single-window inference **~1.2 s** (T=256, H=64). Fast —
  the model is tiny; no GPU needed for our scale.
- **`model.quantiles` = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]** (9 levels; median = index 4).
  **`model.future_len` = 320** = max prediction length (steps). Our horizon set must fit within 320 steps.
- **Output shape confirmed: `(n_targets, 9, H)`** — e.g. sine test returned `(1, 9, 64)`. Extract
  median = `forecast_item[0, 4]`.
- **Covariate mechanism WORKS (the paper's spine):** controlled test where target = a fully-observed
  future covariate shifted by a lag → forecast **MAE 0.114 WITH** covariate vs **1.49 WITHOUT**
  (**~13× error reduction**). Confirms TiRex-2 genuinely conditions on a non-NaN future covariate and
  treats it as "known". Plot: `outputs/phase0_smoke.png`.
- **NaN-in-context:** injecting NaNs into the target context does **not** crash and produces **no NaNs**
  in the output median. Native masking works as advertised.
- **MPS: DO NOT USE.** `device="mps"` **hangs** (froze ~4.5 min at 0% CPU inside Metal/MPSGraph; the
  sLSTM/mLSTM kernels are cpu/cuda-only). The smoke script gates MPS behind `TRY_MPS=1` + must be run
  under a shell `timeout`. **Default device = `cpu` for all work on this machine.**

### Gotcha (cost me a run): never `pip install` into `.venv` while a job is using it
Installing `vitaldb` into the venv *concurrently* with a running forecast deadlocked the running
process on the Python import lock (pip rewrote shared deps mid-import). Do env changes when no job is
running against the venv.
