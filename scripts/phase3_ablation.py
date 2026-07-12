"""Phase 3.4 — FLAGSHIP covariate ablation (rigorous characterization). Produces [X%].

Four conditions per window (drug covariate = remifentanil+propofol CE, known over horizon):
  M1     : target + past covs (HR/SpO2/CVP) + drug future cov   [primary]
  M0     : target + past covs, NO drug future cov               [ablation partner]
  M1_to  : target + drug future cov, NO past covs               [target-only arm]
  M0_to  : target only                                         [target-only baseline]
plus persistence (B1). [X%]_withpast = (CRPS_M0-CRPS_M1)/CRPS_M0 ; [X%]_targetonly = M1_to vs M0_to
(tests whether HR/SpO2/CVP already capture the drug, per Zhu). Stratified by transition vs steady
windows (does the drug CE change in the horizon?), since the covariate should matter only there.

Case-clustered bootstrap CIs. ZERO-SHOT. Chunked -> figures/results refresh as it runs.

  python scripts/phase3_ablation.py --n-cases 300 --seed 1
"""
from __future__ import annotations
import argparse, csv, json, os, time, importlib
import numpy as np
import torch


def get_loader(config_path):
    """Pick the dataset loader named in the config (`loader:` key), default vitaldb_loader.
    Lets the SAME pipeline run on VitalDB or MOVER — the loader just returns the per-case record."""
    import yaml
    raw = yaml.safe_load(open(config_path))
    return importlib.import_module(raw.get("loader", "vitaldb_loader"))

QLEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]); MED = 4
HORIZON_STEPS_MIN = [1, 3, 5, 7, 10, 15]     # steps computed from dt at runtime
FUTURE_COV = ["Orchestra/RFTN20_CE", "Orchestra/PPF20_CE"]   # known drug-infusion trajectory (model input)
# --cov presets: `future` = covariate FED TO THE MODEL; `primary`/`trans_thr` = the channel that anchors
# window validity + transition/steady tagging. For anesthetic variants the anchor stays on remi CE so
# rate-vs-CE compare on identical strata; the pressor arm anchors on phenylephrine (its own titration).
COV_PRESETS = {
    "ce":        {"future": ["Orchestra/RFTN20_CE", "Orchestra/PPF20_CE"], "primary": "Orchestra/RFTN20_CE", "trans_thr": 0.5},
    "rate":      {"future": ["Orchestra/RFTN20_RATE", "Orchestra/PPF20_RATE"], "primary": "Orchestra/RFTN20_CE", "trans_thr": 0.5},
    "ce_remi":   {"future": ["Orchestra/RFTN20_CE"], "primary": "Orchestra/RFTN20_CE", "trans_thr": 0.5},
    "rate_remi": {"future": ["Orchestra/RFTN20_RATE"], "primary": "Orchestra/RFTN20_CE", "trans_thr": 0.5},
    "pressor":   {"future": ["Orchestra/PHEN_RATE"], "primary": "Orchestra/PHEN_RATE", "trans_thr": 0.5},  # phenylephrine mL/hr
    # --- MOVER (SIS): infusion rates are DERIVED (Dose/duration), so the anchor is a rate channel,
    # not CE (MOVER has no effect-site model). trans_thr is in the derived rate unit (mcg/min) and is
    # a first guess — refine once the cache reveals the rate distributions (see build_mover_cache).
    "mover_rate":    {"future": ["Orchestra/RFTN20_RATE", "Orchestra/PPF20_RATE"], "primary": "Orchestra/PPF20_RATE", "trans_thr": 1000.0},
    "mover_pressor": {"future": ["Orchestra/PHEN_RATE"], "primary": "Orchestra/PHEN_RATE", "trans_thr": 5.0},
}
FUTURE_COV = COV_PRESETS["ce"]["future"]     # reassigned from --cov in main()
PRIMARY_COV = "Orchestra/RFTN20_CE"          # used for window validity + transition tagging
PAST = ["Solar8000/HR", "Solar8000/PLETH_SPO2", "Solar8000/CVP"]
TRANSITION_THR = 0.5                          # change of PRIMARY_COV over the horizon -> "transition"
COND_SPEC = {"M1": (True, True), "M0": (True, False), "M1_to": (False, True), "M0_to": (False, False)}
STRATA = ["all", "transition", "steady"]


def pinball(y, qp):
    m = np.isfinite(y)
    if m.sum() == 0:
        return np.nan, np.nan
    y, qp = y[m], qp[:, m]
    losses = [np.mean(np.maximum(a * (y - qp[i]), (a - 1) * (y - qp[i]))) for i, a in enumerate(QLEVELS)]
    return float(np.mean(losses)), float(np.mean(np.abs(qp[MED] - y)))


# ---- secondary task: impending hypotension (MAP < 65), §3.5 -> [Z] ----
HYPO_THR = 65.0
HYPO_THRS = [65.0, 55.0, 50.0]     # severity gradient (clinical_eval analysis B)
def hypo_event(truth_h, min_run, thr=HYPO_THR):
    """1 if MAP < thr sustained >= min_run samples within the horizon (from ground truth)."""
    below = np.isfinite(truth_h) & (truth_h < thr)
    run = mx = 0
    for b in below:
        run = run + 1 if b else 0
        mx = max(mx, run)
    return int(mx >= min_run)


def time_to_hypo(truth_h, min_run, dt, thr=HYPO_THR):
    """Minutes to onset of the first sustained (>=min_run) MAP<thr run within the horizon; NaN if none.
    Onset = start index of that run. Fine-grained lead time for clinical_eval analysis A."""
    below = np.isfinite(truth_h) & (truth_h < thr)
    run = 0
    for i, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= min_run:
            return float((i - min_run + 1) * dt / 60)
    return float("nan")


def hypo_risk(qp, thr=HYPO_THR):
    """Predicted P(MAP<thr at some point in horizon) from quantile forecast qp (9,h): max_t F_t(thr),
    F_t(thr) via inverse-CDF interpolation over the 9 quantile levels (values increasing per step)."""
    if qp.shape[1] == 0:
        return np.nan
    r = [float(np.interp(thr, qp[:, t], QLEVELS)) for t in range(qp.shape[1])]
    return max(r)


def _avg_ranks(s):
    order = np.argsort(s, kind="mergesort"); ss = s[order]; ranks = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j < len(s) and ss[j] == ss[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0; i = j
    return ranks


def auroc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float); m = np.isfinite(s)
    y, s = y[m], s[m]; P = y.sum(); N = len(y) - P
    if P == 0 or N == 0:
        return np.nan
    ranks = _avg_ranks(s)
    return float((ranks[y == 1].sum() - P * (P + 1) / 2) / (P * N))


def auprc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float); m = np.isfinite(s)
    y, s = y[m], s[m]
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-s, kind="mergesort"); y = y[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y); prec = tp / (tp + fp); rec = tp / y.sum()
    ap = 0.0; prev = 0.0
    for i in range(len(y)):
        ap += (rec[i] - prev) * prec[i]; prev = rec[i]
    return float(ap)


def make_windows(rec, Lc, H, stride, warmup, max_origins):
    tgt = rec["target"]; cov = rec["future_cov"].get(PRIMARY_COV)
    if cov is None:
        return []
    n = rec["n"]; origins = []
    for t0 in range(max(Lc, warmup), n - H, stride):
        if np.isfinite(tgt[t0 - Lc:t0]).mean() < 0.5 or np.isfinite(tgt[t0:t0 + H]).mean() < 0.5:
            continue
        if not np.isfinite(cov[t0 - Lc:t0 + H]).all():
            continue
        origins.append(t0)
    if max_origins and len(origins) > max_origins:
        idx = np.linspace(0, len(origins) - 1, max_origins).round().astype(int)
        origins = [origins[i] for i in idx]
    return origins


def build_ts(rec, t0, Lc, H, use_past, use_future):
    from tirex2 import TimeseriesType
    target = torch.from_numpy(rec["target"][t0 - Lc:t0].astype(np.float32)).unsqueeze(0)
    past_t = None
    if use_past:
        p = [rec["past_cov"][x][t0 - Lc:t0] for x in PAST if x in rec["past_cov"]]
        past_t = torch.from_numpy(np.stack(p).astype(np.float32)) if p else None
    fut_t = None
    if use_future:
        fc = np.stack([rec["future_cov"][c][t0 - Lc:t0 + H] for c in FUTURE_COV]).astype(np.float32)
        fut_t = torch.from_numpy(fc)
    return TimeseriesType(target=target, past_covariates=past_t, future_covariates=fut_t)


def batched_forecast(model, items, H, bs):
    out = []
    for i in range(0, len(items), bs):
        out.extend(model.forecast(items[i:i + bs], prediction_length=H, output_type="numpy"))
    return out


def boot_reduction(cm1, cm0, n_boot, rng):
    """relative CRPS reduction (%) of cm1 vs cm0, case-clustered bootstrap. arrays are per-case means."""
    if len(cm1) < 3:
        return None, [None, None]
    pt = 100 * (cm0.mean() - cm1.mean()) / cm0.mean()
    b = []
    k = len(cm1)
    for _ in range(n_boot):
        s = rng.integers(0, k, k)
        b.append(100 * (cm0[s].mean() - cm1[s].mean()) / cm0[s].mean())
    return round(float(pt), 2), [round(float(np.percentile(b, 2.5)), 2), round(float(np.percentile(b, 97.5)), 2)]


def aggregate(rows, hsteps, dt, n_boot, rng, meta):
    summary = {**meta, "n_windows": len({(r["caseid"], r["t0"]) for r in rows}), "per_horizon": {}}
    for h in hsteps:
        hm = round(h * dt / 60)
        hrows = [r for r in rows if r["h_min"] == hm]
        hres = {}
        for stratum in STRATA:
            sr = [r for r in hrows if (stratum == "all" or r["stratum"] == stratum) and np.isfinite(r["crps_M1"])]
            by_case = {}
            for r in sr:
                by_case.setdefault(r["caseid"], []).append(r)
            cids = list(by_case)
            def cmean(m):
                return np.array([np.nanmean([r[m] for r in by_case[c]]) for c in cids])
            if not cids:
                hres[stratum] = {"n_windows": 0, "n_cases": 0}; continue
            cM1, cM0 = cmean("crps_M1"), cmean("crps_M0")
            cM1t, cM0t = cmean("crps_M1_to"), cmean("crps_M0_to")
            cP = cmean("crps_persist")
            xw, xwci = boot_reduction(cM1, cM0, n_boot, rng)
            xt, xtci = boot_reduction(cM1t, cM0t, n_boot, rng)
            yv, yci = boot_reduction(cM1, cP, n_boot, rng)
            hres[stratum] = {
                "n_windows": len(sr), "n_cases": len(cids),
                "crps_M1": round(float(cM1.mean()), 4), "crps_M0": round(float(cM0.mean()), 4),
                "crps_M1_to": round(float(cM1t.mean()), 4), "crps_M0_to": round(float(cM0t.mean()), 4),
                "crps_persistence": round(float(cP.mean()), 4),
                "mae_M1": round(float(cmean("mae_M1").mean()), 3), "mae_M0": round(float(cmean("mae_M0").mean()), 3),
                "X_pct_withpast": xw, "X_pct_withpast_CI95": xwci,
                "X_pct_targetonly": xt, "X_pct_targetonly_CI95": xtci,
                "Y_pct_vs_persistence": yv, "Y_pct_CI95": yci,
            }
        # --- secondary task: impending hypotension [Z] (pooled over all windows this horizon) ---
        hh = [r for r in hrows if np.isfinite(r.get("risk_M1", np.nan))]
        if hh:
            y = np.array([r["hypo_event"] for r in hh]); s1 = np.array([r["risk_M1"] for r in hh])
            s0 = np.array([r["risk_M0"] for r in hh])
            by_c = {}
            for i, r in enumerate(hh):
                by_c.setdefault(r["caseid"], []).append(i)
            cids = list(by_c)
            b_a1, b_da = [], []
            for _ in range(min(n_boot, 1000)):
                samp = rng.integers(0, len(cids), len(cids))
                idx = np.concatenate([by_c[cids[k]] for k in samp])
                b_a1.append(auroc(y[idx], s1[idx]))
                b_da.append((auprc(y[idx], s1[idx]) or np.nan) - (auprc(y[idx], s0[idx]) or np.nan))
            p1, p0 = auprc(y, s1), auprc(y, s0)
            hres["hypo"] = {
                "n": len(y), "n_events": int(y.sum()), "prevalence": round(float(y.mean()), 3),
                "auroc_M1": round(auroc(y, s1), 3), "auroc_M0": round(auroc(y, s0), 3),
                "auroc_M1_CI95": [round(float(np.nanpercentile(b_a1, 2.5)), 3), round(float(np.nanpercentile(b_a1, 97.5)), 3)],
                "auprc_M1": round(p1, 3), "auprc_M0": round(p0, 3),
                "delta_auprc_M1_minus_M0": round(p1 - p0, 3),
                "delta_auprc_CI95": [round(float(np.nanpercentile(b_da, 2.5)), 3), round(float(np.nanpercentile(b_da, 97.5)), 3)],
            }
        summary["per_horizon"][f"{hm}min"] = hres
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="datasets/vitaldb/configs/data.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--n-cases", type=int, default=300)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-origins", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--no-plot", action="store_true", help="skip per-chunk figures (for sharded runs)")
    ap.add_argument("--device", default="cpu", help="cpu (Mac) or cuda (HPC). On CUDA use larger --batch-size.")
    ap.add_argument("--cov", default="ce", choices=list(COV_PRESETS),
                    help="covariate preset (future channel + stratification anchor).")
    ap.add_argument("--cases-file", default=None,
                    help="file with one caseid per line -> use these instead of the random cohort sample.")
    ap.add_argument("--quiet", action="store_true", help="suppress per-case / per-chunk progress lines.")
    args = ap.parse_args()

    import yaml
    from plot_results import plot_dashboard, plot_examples
    ev = yaml.safe_load(open(args.eval_config))
    L = get_loader(args.config)                          # vitaldb_loader or mover_loader (per config)
    cfg = L.load_config(args.config); clin = L._clinical_index(cfg["clinical_csv"])
    rec0 = None
    # infer dt after resample by loading one case
    rng = np.random.default_rng(args.seed)
    if args.cases_file:
        cases = [ln.strip() for ln in open(args.cases_file) if ln.strip()]
    else:
        manifest = cfg.get("cohort_manifest", "datasets/vitaldb/cohort_manifest.csv")   # dataset-specific
        included = [r["caseid"] for r in csv.DictReader(open(manifest)) if r["include"] in ("1", "True", "true")]
        cases = included if args.all else list(rng.choice(included, min(args.n_cases, len(included)), replace=False))
    global FUTURE_COV, PRIMARY_COV, TRANSITION_THR
    preset = COV_PRESETS[args.cov]
    FUTURE_COV = preset["future"]; PRIMARY_COV = preset["primary"]; TRANSITION_THR = preset["trans_thr"]
    cov_sfx = "" if args.cov == "ce" else f"_cov{args.cov}"
    default_tag = f"cases{len(cases)}" if args.cases_file else (f"all{len(cases)}" if args.all else f"n{len(cases)}_s{args.seed}")
    base_tag = args.tag or (default_tag + cov_sfx)
    if args.n_shards > 1:
        cases = cases[args.shard_idx::args.n_shards]        # disjoint shard
        tag = f"{base_tag}_sh{args.shard_idx}of{args.n_shards}"
        args.no_plot = True
    else:
        tag = base_tag

    # infer dt from the first case that actually loads (a partial cache / in-progress
    # vital_files download can leave leading cases unavailable -> load_case returns None).
    probe = next((r for r in (L.load_case(c, cfg, clin) for c in cases) if r is not None), None)
    if probe is None:
        raise SystemExit(f"No loadable cases among {len(cases)} selected (cache empty and "
                         f"vital_files missing?). Check cache_dir/vital_dir in {args.config}.")
    dt = probe["interval_s"]
    Lc = int(ev["context_min"] * 60 / dt)
    hsteps = [int(m * 60 / dt) for m in HORIZON_STEPS_MIN]
    H = max(hsteps)
    stride = int(ev["origin_stride_min"] * 60 / dt); warmup = int(ev["warmup_min"] * 60 / dt)
    min_run = max(1, int(ev.get("hypotension", {}).get("min_sustain_min", 1) * 60 / dt))  # hypotension sustain
    n_boot = ev.get("n_boot", 2000)
    assert H <= 320, f"H={H} steps > future_len 320"
    print(f"dt={dt}s L={Lc} H={H} horizons(min)={HORIZON_STEPS_MIN} stride={stride} cases={len(cases)}", flush=True)
    os.makedirs("results", exist_ok=True); os.makedirs("outputs/figs", exist_ok=True)

    from tirex2 import load_model
    print(f"[load] loading TiRex-2 on {args.device} (custom kernels compile on first call) ...", flush=True)
    model = load_model("NX-AI/TiRex-2", device=args.device)
    print(f"[load] model ready. cov={args.cov} future_cov={FUTURE_COV}", flush=True)

    t_start = time.time(); n_skip = 0
    n_chunks = (len(cases) + args.chunk - 1) // args.chunk
    all_rows = []; ex = None
    for start in range(0, len(cases), args.chunk):
        chunk = cases[start:start + args.chunk]; ci = start // args.chunk + 1
        keys, persist, trans, recs = [], [], [], {}
        items = {c: [] for c in COND_SPEC}
        for caseid in chunk:
            rec = L.load_case(caseid, cfg, clin)
            if rec is None:
                n_skip += 1
                if not args.quiet:
                    print(f"  [case {caseid}] skip — no data / covariate missing", flush=True)
                continue
            truth = rec["target"]; cov = rec["future_cov"][PRIMARY_COV]
            nt = ns = 0
            for t0 in make_windows(rec, Lc, H, stride, warmup, args.max_origins):
                keys.append((caseid, t0)); recs[(caseid, t0)] = rec
                persist.append((truth[t0 - Lc:t0], truth[t0:t0 + H]))
                seg = cov[t0:t0 + H]
                st = "transition" if (np.nanmax(seg) - np.nanmin(seg)) > TRANSITION_THR else "steady"
                trans.append(st); nt += st == "transition"; ns += st == "steady"
                for c, (up, uf) in COND_SPEC.items():
                    items[c].append(build_ts(rec, t0, Lc, H, up, uf))
            if not args.quiet:
                print(f"  [case {caseid}] {nt + ns} windows ({nt} transition, {ns} steady)", flush=True)
        if not keys:
            continue
        if not args.quiet:
            print(f"[chunk {ci}/{n_chunks}] forecasting {len(keys)} windows x{len(COND_SPEC)} conditions "
                  f"on {args.device} ...", flush=True)
        tf = time.time()
        fc = {c: batched_forecast(model, items[c], H, args.batch_size) for c in COND_SPEC}
        if not args.quiet:
            print(f"[chunk {ci}/{n_chunks}] forecast done in {time.time() - tf:.1f}s", flush=True)

        for wi, ((caseid, t0), (ctx, truth), st) in enumerate(zip(keys, persist, trans)):
            q = {c: np.asarray(fc[c][wi])[0] for c in COND_SPEC}
            fin = ctx[np.isfinite(ctx)]; plast = fin[-1] if len(fin) else np.nan
            # per-origin (horizon-independent): time to first sustained MAP<65 (fine-grained lead time)
            t_ev65 = time_to_hypo(truth, min_run, dt)
            for h in hsteps:
                tr = truth[:h]; hm = round(h * dt / 60)
                row = {"caseid": caseid, "t0": t0, "h_min": hm, "stratum": st, "t_event_65": t_ev65}
                for c in COND_SPEC:
                    cr, ma = pinball(tr, q[c][:, :h]); row[f"crps_{c}"] = cr; row[f"mae_{c}"] = ma
                    # instantaneous MAE at exactly h (median forecast at the horizon endpoint vs truth)
                    yl = tr[-1]; row[f"mae_inst_{c}"] = (float(abs(q[c][MED, h - 1] - yl))
                                                         if np.isfinite(yl) else np.nan)
                row["crps_persist"] = float(np.nanmean(np.abs(plast - tr))) if np.isfinite(tr).any() else np.nan
                # secondary task: impending hypotension — events + covariate-ablation risks at each severity
                for thr in HYPO_THRS:
                    tk = "" if thr == HYPO_THR else f"_{int(thr)}"
                    row[f"hypo_event{tk}"] = hypo_event(tr, min_run, thr)
                    row[f"risk_M1{tk}"] = hypo_risk(q["M1"][:, :h], thr)
                    row[f"risk_M0{tk}"] = hypo_risk(q["M0"][:, :h], thr)
                all_rows.append(row)

        if ex is None and (not args.no_plot or args.shard_idx == 0):
            sel = np.linspace(0, len(keys) - 1, min(12, len(keys))).round().astype(int)
            ex = {k: [] for k in ("caseid", "t0", "context", "truth", "q_ce", "q_M0")}
            for i in sel:
                cid, t0 = keys[i]
                ex["caseid"].append(cid); ex["t0"].append(t0)
                ex["context"].append(recs[(cid, t0)]["target"][t0 - Lc:t0])
                ex["truth"].append(persist[i][1])
                ex["q_ce"].append(np.asarray(fc["M1"][i])[0]); ex["q_M0"].append(np.asarray(fc["M0"][i])[0])
            for k in ex:
                ex[k] = np.array(ex[k], dtype=object if k == "caseid" else float)
            np.savez_compressed(f"outputs/figs/examples_{tag}.npz", **ex)
            if not args.no_plot:
                plot_examples(ex, f"outputs/figs/examples_{tag}.png", dt)

        done = min(start + args.chunk, len(cases))
        summary = aggregate(all_rows, hsteps, dt, n_boot, rng,
                            {"tag": tag, "n_cases": len(cases), "cases_done": done, "dt_s": dt})
        with open(f"results/ablation_windows_{tag}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
        if not args.no_plot:
            json.dump(summary, open(f"results/ablation_primary_{tag}.json", "w"), indent=2)
            plot_dashboard(summary, f"outputs/figs/dashboard_{tag}.png")
            from plot_results import plot_hypotension
            plot_hypotension(summary, f"outputs/figs/hypotension_{tag}.png")
        ph = summary["per_horizon"]
        a15 = ph.get("15min", {}).get("all", {}); a7 = ph.get("7min", {}).get("all", {})
        t15 = ph.get("15min", {}).get("transition", {})
        elapsed = time.time() - t_start; eta = elapsed / max(done, 1) * (len(cases) - done)
        print(f"[{done}/{len(cases)} cases | {summary['n_windows']} win | {n_skip} skipped | "
              f"{elapsed / 60:.1f}min elapsed, ~{eta / 60:.0f}min left]", flush=True)
        print(f"    7min : MAE_M1={a7.get('mae_M1')} CRPS_M1={a7.get('crps_M1')} "
              f"(vs persist {a7.get('crps_persistence')})", flush=True)
        print(f"    15min: X%withpast(all)={a15.get('X_pct_withpast')} CI{a15.get('X_pct_withpast_CI95')} | "
              f"transition={t15.get('X_pct_withpast')} CI{t15.get('X_pct_withpast_CI95')} | "
              f"targetonly={a15.get('X_pct_targetonly')}", flush=True)

    print(f"\n=== SHARD DONE tag={tag} cases={len(cases)} ({n_skip} skipped) "
          f"windows={summary['n_windows']} in {(time.time() - t_start) / 60:.1f}min ===", flush=True)


if __name__ == "__main__":
    main()
