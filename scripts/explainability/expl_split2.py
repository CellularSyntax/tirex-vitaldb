import numpy as np, csv, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami, roc_auc_score
root="/Users/admin/Desktop/DATA/Uni/2026/Projects/tirex-2/results"
clin={}
with open("/Users/admin/Desktop/DATA/Uni/2026/Projects/tirex-2/datasets/vitaldb/data/clinical_data.csv",encoding="utf-8-sig") as f:
    for r in csv.DictReader(f): clin[str(int(r["caseid"]))]=r
def f2(x):
    try: return float(x)
    except: return np.nan
cat_cols=["sex","department","optype","ane_type","position","approach"]
num_cols=["age","bmi","height","weight","asa"]
E={m:dict(np.load(f"{root}/embeddings_{m}_all2873.npz",allow_pickle=True)) for m in ["tirex2","timesfm","tft","patchtst"]}
ref=E["tirex2"]; groups=np.array([str(int(c)) for c in ref["caseid"]])
gn=lambda name: np.array([f2(clin.get(c,{}).get(name,"")) for c in groups])
gc=lambda name: np.array([clin.get(c,{}).get(name,"") for c in groups])
for m in ["tft","patchtst","timesfm","tirex2"]:
    X=StandardScaler().fit_transform(E[m]["emb"].astype(np.float64))
    c=KMeans(2,random_state=0,n_init=10).fit_predict(X)
    print(f"\n--- {m} (sizes {np.bincount(c)}) ---")
    scores=[]
    for col in num_cols:
        v=gn(col); mm=np.isfinite(v)
        if mm.sum()<100: continue
        a=roc_auc_score(c[mm],v[mm]); auc=max(a,1-a)
        scores.append((auc,f"num {col:9s} AUC={auc:.3f} mean0={v[(c==0)&mm].mean():.1f} mean1={v[(c==1)&mm].mean():.1f}"))
    for col in cat_cols:
        v=gc(col); a=ami(c,v)
        # dominant category per cluster
        def top(cl):
            vals,cnt=np.unique(v[c==cl][v[c==cl]!=""],return_counts=True)
            return f"{vals[cnt.argmax()]}({cnt.max()/max(1,(c==cl).sum()):.0%})" if len(vals) else "na"
        scores.append((a,f"cat {col:11s} AMI={a:.3f}  c0:{top(0)} c1:{top(1)}"))
    for s,line in sorted(scores,reverse=True): print("  ",line)
