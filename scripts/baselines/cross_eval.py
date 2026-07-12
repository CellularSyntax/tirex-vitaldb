"""Cross-dataset transfer: apply a trained baseline (checkpoint from train.py --all-train on
dataset A) to dataset B's windows, using A's normalization. Writes per-window forecasts in the
phase3 schema tagged e.g. xfer-tft_vitaldb60TOmover_art, so the existing post-hoc/figures score it.

M0 (covariate-free) is the default — it tests pure MAP-dynamics transfer and sidesteps the
covariate-unit mismatch between datasets (VitalDB pump mL/hr vs MOVER derived mcg/min). Input/output
shapes MUST match between source and target (same cadence + channel count) — harmonize both to 60 s.

Run: PYTHONPATH=scripts:datasets/vitaldb:datasets/mover python scripts/baselines/cross_eval.py \
        --ckpt results/baseline_ckpt_vitaldb60.pt --config datasets/mover/configs/data.yaml \
        --cov mover_rate --tag xfer-tft_vitaldb60TOmover_art
"""
from __future__ import annotations
import argparse, csv, json, os, time
import numpy as np
import torch
import yaml

import phase3_ablation as P
from baselines import data as D
from baselines.models import build_model
from baselines.train import predict, _window_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="source checkpoint from train.py --all-train")
    ap.add_argument("--config", required=True, help="TARGET dataset config")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--cov", required=True, choices=list(P.COV_PRESETS), help="TARGET covariate preset (for windowing)")
    ap.add_argument("--arm", default="M0", choices=["M0", "M1"], help="which trained arm to transfer (default M0)")
    ap.add_argument("--tag", required=True, help="output tag, e.g. xfer-tft_vitaldb60TOmover_art")
    ap.add_argument("--match", default=None, help="optional windows CSV to lock the target cohort")
    ap.add_argument("--max-origins", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ev = yaml.safe_load(open(args.eval_config))
    L = P.get_loader(args.config)
    cfg = L.load_config(args.config); clin = L._clinical_index(cfg["clinical_csv"])
    preset = P.COV_PRESETS[args.cov]
    P.FUTURE_COV = list(preset["future"]); P.PRIMARY_COV = preset["primary"]; P.TRANSITION_THR = preset["trans_thr"]

    if args.match:
        cases = sorted({r["caseid"] for r in csv.DictReader(open(args.match))})
    else:
        manifest = cfg.get("cohort_manifest", "datasets/vitaldb/cohort_manifest.csv")
        cases = [r["caseid"] for r in csv.DictReader(open(manifest)) if r["include"] in ("1", "True", "true")]

    probe = next((r for r in (L.load_case(c, cfg, clin) for c in cases) if r is not None), None)
    dt = probe["interval_s"]; Lc = int(ev["context_min"] * 60 / dt)
    hsteps = [int(m * 60 / dt) for m in P.HORIZON_STEPS_MIN]; H = max(hsteps)
    stride = int(ev["origin_stride_min"] * 60 / dt); warmup = int(ev["warmup_min"] * 60 / dt)
    min_run = max(1, int(ev.get("hypotension", {}).get("min_sustain_min", 1) * 60 / dt))

    # shapes MUST match the source model (harmonize cadence/channels or this transfer is invalid)
    mism = []
    if Lc != ck["Lc"]: mism.append(f"context {Lc}!={ck['Lc']}")
    if H != ck["H"]: mism.append(f"horizon {H}!={ck['H']}")
    if mism:
        raise SystemExit(f"[xfer] shape mismatch source vs target ({', '.join(mism)}). "
                         f"Harmonize both datasets (e.g. resample_to_s: 60) before cross-eval.")

    t0 = time.time()
    win, past_names, fut_names = D.build_windows(cases, cfg, clin, Lc, H, stride, warmup,
                                                 args.max_origins, dt, min_run, quiet=args.quiet)
    n_past, n_fut = 1 + len(past_names), len(fut_names)
    if (n_past, n_fut) != (ck["n_past"], ck["n_fut"]):
        raise SystemExit(f"[xfer] channel mismatch: target (n_past={n_past},n_fut={n_fut}) vs "
                         f"source ({ck['n_past']},{ck['n_fut']}). Covariate/past channels must align.")
    print(f"[xfer] {ck['model']} ({args.ckpt}) -> {args.tag}: {len(win)} target windows "
          f"({len(cases)} cases, arm {args.arm})", flush=True)

    model = build_model(ck["model"], ck["n_past"], ck["n_fut"], ck["H"], context_len=ck["Lc"], d=ck["d_model"]).to(dev)
    model.load_state_dict(ck["state"][args.arm])
    use_future = (args.arm == "M1")
    q = predict(model, D.to_tensors(win, ck["norm"], use_future), dev, ck["norm"], bs=512)   # [N,Q,H], source norm

    # M0 transfer: both schema columns use the same covariate-free forecast (risk_M1==risk_M0)
    os.makedirs("results", exist_ok=True); MED = P.MED
    cols = ["caseid", "t0", "h_min", "stratum", "t_event_65",
            "crps_M1", "mae_M1", "mae_inst_M1", "crps_M0", "mae_M0", "mae_inst_M0",
            "crps_M1_to", "mae_M1_to", "mae_inst_M1_to", "crps_M0_to", "mae_M0_to", "mae_inst_M0_to",
            "crps_persist", "hypo_event", "risk_M1", "risk_M0",
            "hypo_event_55", "risk_M1_55", "risk_M0_55", "hypo_event_50", "risk_M1_50", "risk_M0_50", "split"]
    path = f"results/ablation_windows_{args.tag}.csv"; n_rows = 0
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader()
        for wi, w in enumerate(win):
            for row in _window_rows(w, {"M1": q[wi], "M0": q[wi]}, hsteps, dt, min_run, MED, "xfer"):
                wr.writerow(row); n_rows += 1
    json.dump({"tag": args.tag, "source_ckpt": args.ckpt, "model": ck["model"], "arm": args.arm,
               "n_windows": len(win), "n_cases": len(cases), "transfer": True},
              open(f"results/baseline_meta_{args.tag}.json", "w"), indent=1)
    print(f"[xfer] wrote {path} ({n_rows} rows). Done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
