import numpy as np, json, warnings, csv
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
root="/Users/admin/Desktop/DATA/Uni/2026/Projects/tirex-2/results"
models=["tirex2","chronos","timesfm","moirai","tft","patchtst"]
def linear_cka(X,Y):
    X=X-X.mean(0); Y=Y-Y.mean(0)
    return np.linalg.norm(X.T@Y,'fro')**2/(np.linalg.norm(X.T@X,'fro')*np.linalg.norm(Y.T@Y,'fro'))
def probe_reg(X,y,g,ns=5):
    m=np.isfinite(y); X,y,g=X[m],y[m],g[m]; r=[]
    for tr,te in GroupKFold(ns).split(X,y,g):
        rg=make_pipeline(StandardScaler(),Ridge(alpha=10.0)); rg.fit(X[tr],y[tr])
        r.append(r2_score(y[te],rg.predict(X[te])))
    return float(np.mean(r)),float(np.std(r))
TRAJ={}; RSA={}
for tag in ["all2873","mover_art"]:
    T=dict(np.load(f"{root}/embeddings_targets_{tag}.npz",allow_pickle=True))
    E={m:dict(np.load(f"{root}/embeddings_{m}_{tag}.npz",allow_pickle=True)) for m in models}
    ref=E["tirex2"]
    g=np.array([str(c) for c in ref["caseid"]])
    targets={"MAP_now":T["last_map"],"MAP_fut5_mean":T["fut5_mean"],"MAP_fut5_min":T["fut5_min"],
             "MAP_futH_mean":T["futH_mean"],"MAP_slope5":T["slope5"],"MAP_slopeH":T["slopeH"]}
    TRAJ[tag]={}
    for m in models:
        X=E[m]["emb"].astype(np.float64); TRAJ[tag][m]={}
        for tname,y in targets.items():
            r2,sd=probe_reg(X,y,g); TRAJ[tag][m][tname]={"r2":round(r2,3),"sd":round(sd,3)}
    # RSA
    Xs={m:StandardScaler().fit_transform(E[m]["emb"].astype(np.float64)) for m in models}
    M=np.array([[linear_cka(Xs[a],Xs[b]) for b in models] for a in models])
    RSA[tag]={"models":models,"cka":M.round(3).tolist()}
    print(f"\n=== {tag} trajectory probe (R2) ===")
    print(f"{'model':9s} "+" ".join(f"{t:13s}" for t in targets))
    for m in models:
        print(f"{m:9s} "+" ".join(f"{TRAJ[tag][m][t]['r2']:+.3f}       " for t in targets))
json.dump({"trajectory":TRAJ},open(f"{root}/_expl_trajectory.json","w"),indent=1)
json.dump(RSA,open(f"{root}/_expl_rsa_both.json","w"),indent=1)
print("\nsaved trajectory + rsa (both cohorts)")
