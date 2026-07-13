import numpy as np, json, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
root="/Users/admin/Desktop/DATA/Uni/2026/Projects/tirex-2/results"
models=["tirex2","chronos","timesfm","moirai","tft","patchtst"]
E={m:dict(np.load(f"{root}/embeddings_{m}_all2873.npz",allow_pickle=True)) for m in models}
ref=E["tirex2"]; groups=np.array([str(int(c)) for c in ref["caseid"]])
tev=ref["t_event_65"].astype(float)
def label(h): return (np.isfinite(tev)&(tev<=h)).astype(int)
def probe(X,y,g,ns=5):
    au=[]
    for tr,te in StratifiedGroupKFold(ns,shuffle=True,random_state=0).split(X,y,g):
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced"))
        clf.fit(X[tr],y[tr]); au.append(roc_auc_score(y[te],clf.predict_proba(X[te])[:,1]))
    return float(np.mean(au)),float(np.std(au))
hs=[1,2,3,5]
print("prevalence:",{h:round(float(label(h).mean()),3) for h in hs},"| n_pos:",{h:int(label(h).sum()) for h in hs},flush=True)
out={"horizons":hs,"prevalence":{h:float(label(h).mean()) for h in hs}}
for m in models:
    X=E[m]["emb"].astype(np.float64); out[m]={}
    row=[]
    for h in hs:
        y=label(h)
        if y.sum()<40: row.append(f"{h}min n/a"); continue
        au,sd=probe(X,y,groups); out[m][h]={"auroc":au,"sd":sd}
        row.append(f"{h}min={au:.3f}")
    print(f"{m:9s} "+" | ".join(row),flush=True)
json.dump(out,open(f"{root}/_expl_probe_short.json","w"),indent=1)
print("saved")
