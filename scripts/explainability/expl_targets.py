import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
root="/Users/admin/Desktop/DATA/Uni/2026/Projects/tirex-2/results"
models=["tirex2","chronos","timesfm","moirai","tft","patchtst"]
for tag in ["all2873","mover_art"]:
    T=dict(np.load(f"{root}/embeddings_targets_{tag}.npz",allow_pickle=True))
    E={m:dict(np.load(f"{root}/embeddings_{m}_{tag}.npz",allow_pickle=True)) for m in models}
    ref=E["tirex2"]
    # row-alignment of targets vs embeddings
    k_t=np.array([f"{c}@{t}" for c,t in zip(T["caseid"],T["t0"])])
    k_e=np.array([f"{c}@{t}" for c,t in zip(ref["caseid"],ref["t0"])])
    aligned=(k_t.shape==k_e.shape) and (k_t==k_e).all()
    lm=T["last_map"]
    print(f"\n===== {tag} ===== targets n={len(T['last_map'])} aligned={aligned}")
    print(f"  keys={list(T.keys())}")
    print(f"  last_map: mean={np.nanmean(lm):.1f} sd={np.nanstd(lm):.1f} finite={np.isfinite(lm).mean():.2f}")
    print("  --- 2-cluster split vs MAP level ---")
    for m in ["timesfm","tirex2","tft","patchtst","chronos","moirai"]:
        X=StandardScaler().fit_transform(E[m]["emb"].astype(np.float64))
        c=KMeans(2,random_state=0,n_init=10).fit_predict(X)
        mm=np.isfinite(lm)
        a=roc_auc_score(c[mm],lm[mm]); auc=max(a,1-a)
        print(f"  {m:9s} split~lastMAP AUC={auc:.3f} | MAP mean0={lm[(c==0)&mm].mean():.1f} mean1={lm[(c==1)&mm].mean():.1f}")
