"""Train a matched supervised baseline (default: TFT) on the SAME windows/splits/metrics
as the zero-shot TiRex-2 evaluation, and write per-window quantile forecasts in the exact
phase3 schema (+ a `split` column) so the existing post-hoc pipeline compares them directly.

Trains two arms to mirror TiRex's ablation:
  M1 = with the known future drug-infusion covariate
  M0 = same architecture, future covariate zeroed (no drug information)

Run (cluster): PYTHONPATH=scripts:datasets/vitaldb python scripts/baselines/train.py \
                 --config datasets/vitaldb/configs/data.yaml --model tft --all --device cuda
"""
from __future__ import annotations
import argparse, csv, json, os, time
import numpy as np
import torch
import yaml

import phase3_ablation as P
import vitaldb_loader as L
from baselines import data as D
from baselines.models import build_model
from baselines.splits import subject_split

QL = torch.tensor(P.QLEVELS, dtype=torch.float32)


def pinball_loss(pred, y, mask, ql):
    y = y.unsqueeze(-1)                                   # [B,H,1]
    q = ql.view(1, 1, -1).to(pred.device)                # [1,1,Q]
    diff = y - pred
    loss = torch.maximum(q * diff, (q - 1) * diff).mean(-1)   # [B,H]
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def caseid_to_subject(clinical_csv):
    m = {}
    for r in csv.DictReader(open(clinical_csv, encoding="utf-8-sig")):
        m[str(r["caseid"])] = str(r.get("subjectid") or r["caseid"])
    return m


def train_arm(model, tr, va, args, dev):
    Ptr, Ftr, Ytr, Mtr, _ = tr
    Pva, Fva, Yva, Mva, _ = va
    to = lambda a: torch.from_numpy(a).to(dev)
    Ptr, Ftr, Ytr, Mtr = map(to, (Ptr, Ftr, Ytr, Mtr))
    Pva, Fva, Yva, Mva = map(to, (Pva, Fva, Yva, Mva))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    n = Ptr.shape[0]; bs = args.batch_size
    best = float("inf"); best_state = None; bad = 0; best_ep = 0
    history = []
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n, device=dev); tr_sum = 0.0; nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            pred = model(Ptr[idx], Ftr[idx])
            loss = pinball_loss(pred, Ytr[idx], Mtr[idx], QL)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_sum += loss.item(); nb += 1
        tr_loss = tr_sum / max(nb, 1)
        model.eval()
        with torch.no_grad():
            vloss = pinball_loss(model(Pva, Fva), Yva, Mva, QL).item()
        sched.step(vloss)
        history.append({"epoch": ep, "train_pinball": round(tr_loss, 5), "val_pinball": round(vloss, 5)})
        if vloss < best - 1e-5:
            best = vloss; best_ep = ep; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
        print(f"    epoch {ep:02d} train_pinball={tr_loss:.4f} val_pinball={vloss:.4f} best={best:.4f}{'  *' if bad == 0 else ''}", flush=True)
        if bad >= args.patience:
            print(f"    early stop at epoch {ep}", flush=True); break
    if best_state:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_ep, "best_val": round(best, 5), "curve": history}


@torch.no_grad()
def predict(model, X, dev, norm, bs=512):
    Px, Fx = torch.from_numpy(X[0]), torch.from_numpy(X[1])
    outs = []
    model.eval()
    for i in range(0, Px.shape[0], bs):
        q = model(Px[i:i + bs].to(dev), Fx[i:i + bs].to(dev)).cpu().numpy()   # [b,H,Q]
        outs.append(q)
    q = np.concatenate(outs, axis=0)
    q = np.sort(q, axis=-1)                                # enforce monotone quantiles
    q = q * norm["tgt_std"] + norm["tgt_mean"]             # de-normalise to mmHg
    return np.transpose(q, (0, 2, 1))                      # [N, Q, H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="datasets/vitaldb/configs/data.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--model", default="tft", help="architecture in baselines.models.MODELS")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n-cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--max-origins", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--match-tirex", default=None,
                    help="path to a TiRex ablation_windows_*.csv; locks the cohort to its exact "
                         "caseids so windows/splits are identical (the matched comparison).")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ev = yaml.safe_load(open(args.eval_config))
    cfg = L.load_config(args.config); clin = L._clinical_index(cfg["clinical_csv"])
    if args.match_tirex:
        cases = sorted({r["caseid"] for r in csv.DictReader(open(args.match_tirex))})
        stem = os.path.basename(args.match_tirex).replace("ablation_windows_", "").replace(".csv", "")
        tag = args.tag or f"baseline-{args.model}_{stem}"
    else:
        man = [r["caseid"] for r in csv.DictReader(open("datasets/vitaldb/cohort_manifest.csv"))
               if r["include"] in ("1", "True", "true")]
        rng = np.random.default_rng(args.seed)
        cases = man if args.all else list(rng.choice(man, min(args.n_cases, len(man)), replace=False))
        tag = args.tag or (f"baseline-{args.model}_" + ("all" if args.all else f"n{len(cases)}"))

    probe = next((r for r in (L.load_case(c, cfg, clin) for c in cases) if r is not None), None)
    dt = probe["interval_s"]; Lc = int(ev["context_min"] * 60 / dt)
    hsteps = [int(m * 60 / dt) for m in P.HORIZON_STEPS_MIN]; H = max(hsteps)
    stride = int(ev["origin_stride_min"] * 60 / dt); warmup = int(ev["warmup_min"] * 60 / dt)
    min_run = max(1, int(ev.get("hypotension", {}).get("min_sustain_min", 1) * 60 / dt))
    dev = torch.device(args.device)
    print(f"[base] model={args.model} dt={dt} Lc={Lc} H={H} cases={len(cases)} device={args.device} tag={tag}", flush=True)

    c2s = caseid_to_subject(cfg["clinical_csv"])
    split = subject_split(cases, c2s, seed=args.seed)     # canonical 60/20/20 subject split
    by = {s: [c for c in cases if split[c] == s] for s in ("train", "val", "test")}
    print(f"[base] subjects/cases split -> train {len(by['train'])}  val {len(by['val'])}  test {len(by['test'])}", flush=True)

    t0 = time.time()
    win = {}
    for s in ("train", "val", "test"):
        w, past_names, fut_names = D.build_windows(by[s], cfg, clin, Lc, H, stride, warmup,
                                                   args.max_origins, dt, min_run, quiet=args.quiet)
        win[s] = w
        print(f"[base] {s}: {len(w)} windows ({time.time()-t0:.0f}s)", flush=True)
    norm = D.fit_norm(win["train"])
    n_past, n_fut = 1 + len(past_names), len(fut_names)

    preds = {}; hist = {}
    for arm, use_future in [("M1", True), ("M0", False)]:
        print(f"[base] === training arm {arm} (use_future={use_future}) ===", flush=True)
        tr = D.to_tensors(win["train"], norm, use_future)
        va = D.to_tensors(win["val"], norm, use_future)
        te = D.to_tensors(win["test"], norm, use_future)
        torch.manual_seed(args.seed)
        model = build_model(args.model, n_past, n_fut, H, d=args.d_model).to(dev)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"[base] {arm}: {n_par/1e3:.0f}k params, {tr[0].shape[0]} train windows", flush=True)
        model, hist[arm] = train_arm(model, tr, va, args, dev)
        preds[arm] = predict(model, te, dev, norm, bs=max(256, args.batch_size))
    json.dump({"tag": tag, "model": args.model, "arms": hist},
              open(f"results/baseline_history_{tag}.json", "w"), indent=1)
    print(f"[base] wrote results/baseline_history_{tag}.json (train/val loss curves)", flush=True)

    # ---- write per-window rows in the phase3 schema (test split only) ----
    os.makedirs("results", exist_ok=True)
    MED = P.MED
    cols = ["caseid", "t0", "h_min", "stratum", "t_event_65",
            "crps_M1", "mae_M1", "mae_inst_M1", "crps_M0", "mae_M0", "mae_inst_M0",
            "crps_M1_to", "mae_M1_to", "mae_inst_M1_to", "crps_M0_to", "mae_M0_to", "mae_inst_M0_to",
            "crps_persist", "hypo_event", "risk_M1", "risk_M0",
            "hypo_event_55", "risk_M1_55", "risk_M0_55", "hypo_event_50", "risk_M1_50", "risk_M0_50", "split"]
    path = f"results/ablation_windows_{tag}.csv"
    n_rows = 0
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader()
        for wi, w in enumerate(win["test"]):
            truth = w["truth"]; q = {"M1": preds["M1"][wi], "M0": preds["M0"][wi]}   # each [Q,H]
            pastMAP = w["past"][:, 0]; fin = pastMAP[np.isfinite(pastMAP)]
            plast = fin[-1] if len(fin) else np.nan
            for h in hsteps:
                tr = truth[:h]; hm = round(h * dt / 60)
                row = {"caseid": w["caseid"], "t0": w["t0"], "h_min": hm, "stratum": w["stratum"],
                       "t_event_65": w["t_event_65"], "split": "test"}
                for c in ("M1", "M0"):
                    cr, ma = P.pinball(tr, q[c][:, :h]); row[f"crps_{c}"] = cr; row[f"mae_{c}"] = ma
                    yl = tr[-1]
                    row[f"mae_inst_{c}"] = float(abs(q[c][MED, h - 1] - yl)) if np.isfinite(yl) else np.nan
                    row[f"crps_{c}_to"] = cr; row[f"mae_{c}_to"] = ma      # baseline has no target-only variant
                    row[f"mae_inst_{c}_to"] = row[f"mae_inst_{c}"]
                row["crps_persist"] = float(np.nanmean(np.abs(plast - tr))) if np.isfinite(tr).any() else np.nan
                for thr in P.HYPO_THRS:
                    tk = "" if thr == P.HYPO_THR else f"_{int(thr)}"
                    row[f"hypo_event{tk}"] = P.hypo_event(tr, min_run, thr)
                    row[f"risk_M1{tk}"] = P.hypo_risk(q["M1"][:, :h], thr)
                    row[f"risk_M0{tk}"] = P.hypo_risk(q["M0"][:, :h], thr)
                wr.writerow(row); n_rows += 1
    json.dump({"tag": tag, "model": args.model, "n_test_windows": len(win["test"]), "n_rows": n_rows,
               "n_params_per_arm": n_par, "split_seed": args.seed,
               "cases": {s: len(by[s]) for s in by}}, open(f"results/baseline_meta_{tag}.json", "w"), indent=1)
    print(f"[base] wrote {path}  ({n_rows} rows, test split) + results/baseline_meta_{tag}.json", flush=True)
    print(f"[base] Done in {time.time()-t0:.0f}s. Compare with scripts/baselines/compare.py", flush=True)


if __name__ == "__main__":
    main()
