import numpy as np, csv, json, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, adjusted_mutual_info_score as ami, silhouette_score
import pandas as pd
root="/Users/admin/Desktop/DATA/Uni/2026/Projects/tirex-2/results"
models=["tirex2","chronos","timesfm","moirai","tft","patchtst"]
# VitalDB clinical join (department only meaningful there; MOVER lacks it)
clin={}
with open("/Users/admin/Desktop/DATA/Uni/2026/Projects/tirex-2/datasets/vitaldb/data/clinical_data.csv",encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): clin[str(int(r["caseid"]))]=r
rows=[]
for tag in ["all2873","mover_art"]:
    T=dict(np.load(f"{root}/embeddings_targets_{tag}.npz",allow_pickle=True))
    E={m:dict(np.load(f"{root}/embeddings_{m}_{tag}.npz",allow_pickle=True)) for m in models}
    ref=E["tirex2"]
    lm=T["last_map"]
    strat=(ref["stratum"]=="transition").astype(int)
    hypo=ref["hypo_event_5"].astype(int)
    if tag=="all2873":
        gid=np.array([str(int(c)) for c in ref["caseid"]])
        dept=np.array([clin.get(c,{}).get("department","") for c in gid])
    else:
        gid=np.array([str(c) for c in ref["caseid"]]); dept=None
    def AUC(c,v):
        m=np.isfinite(v); a=roc_auc_score(c[m],v[m]); return max(a,1-a)
    for m in models:
        X=StandardScaler().fit_transform(E[m]["emb"].astype(np.float64))
        c=KMeans(2,random_state=0,n_init=10).fit_predict(X)
        sil=silhouette_score(X,c)
        en=np.linalg.norm(E[m]["emb"].astype(np.float64),axis=1)
        dfc=pd.DataFrame({"g":gid,"c":c}); pure=(dfc.groupby("g")["c"].nunique()==1).mean()
        row=dict(cohort=tag,model=m,
                 split_silhouette=round(float(sil),3),
                 balance=round(float(min(np.bincount(c))/len(c)),3),
                 ami_stratum=round(float(ami(c,strat)),3),
                 ami_hypo5=round(float(ami(c,hypo)),3),
                 auc_MAPlevel=round(float(AUC(c,lm)),3),
                 auc_embL2norm=round(float(AUC(c,en)),3),
                 case_purity=round(float(pure),3))
        row["ami_department"]=round(float(ami(c,dept)),3) if dept is not None else None
        rows.append(row)
df=pd.DataFrame(rows)
df.to_csv(f"{root}/_expl_cluster_characterization.csv",index=False)
import pandas as pd
pd.set_option("display.width",200,"display.max_columns",20)
print(df.to_string(index=False))
