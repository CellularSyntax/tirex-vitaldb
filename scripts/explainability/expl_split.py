import numpy as np, json, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami, roc_auc_score
root="/Users/admin/Desktop/DATA/Uni/2026/Projects/tirex-2/results"
models=["tirex2","chronos","timesfm","moirai","tft","patchtst"]
for tag in ["all2873","mover_art"]:
    print(f"\n========== {tag} ==========")
    E={m:dict(np.load(f"{root}/embeddings_{m}_{tag}.npz",allow_pickle=True)) for m in models}
    ref=E["tirex2"]
    strat=(ref["stratum"]=="transition").astype(int)
    hypo=ref["hypo_event_5"].astype(int)
    t0=ref["t0"].astype(float)                      # step index within case ~ time position
    caseid=ref["caseid"]
    tev=ref["t_event_65"].astype(float); has_event=np.isfinite(tev).astype(int)
    for m in models:
        X=StandardScaler().fit_transform(E[m]["emb"].astype(np.float64))
        c=KMeans(2,random_state=0,n_init=10).fit_predict(X)
        # gap: silhouette of the 2-split itself (is it really bimodal?)
        from sklearn.metrics import silhouette_score
        sil=silhouette_score(X,c)
        # association with each metadata var
        a_strat=ami(c,strat); a_hypo=ami(c,hypo); a_hasev=ami(c,has_event)
        # t0: does cluster predict time position?
        auc_t0=roc_auc_score(c, t0) if len(np.unique(c))>1 else np.nan
        auc_t0=max(auc_t0,1-auc_t0)
        # caseid: is the split just per-case? (mean windows/case within cluster vs across)
        # fraction of cases that fall ENTIRELY in one cluster (case-level split)
        import pandas as pd
        df=pd.DataFrame({"case":caseid,"c":c})
        case_pure=df.groupby("case")["c"].nunique()
        pure_frac=(case_pure==1).mean()
        # size balance
        bal=min(np.bincount(c))/len(c)
        print(f"{m:9s} split_sil={sil:.3f} bal={bal:.2f} | AMI strat={a_strat:.3f} hypo={a_hypo:.3f} hasEvent={a_hasev:.3f} | t0_AUC={auc_t0:.3f} | case_pure={pure_frac:.2f}")
