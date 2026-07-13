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
tev=ref["t_event_65"].astype(float)   # minutes to first MAP<65 within window; nan if none
def label(hmin): return (np.isfinite(tev) & (tev<=hmin)).astype(int)
def probe(X,y,g,ns=5):
    au,ap=[],[]
    for tr,te in StratifiedGroupKFold(ns,shuffle=True,random_state=0).split(X,y,g):
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced"))
        clf.fit(X[tr],y[tr]); p=clf.predict_proba(X[te])[:,1]
        au.append(roc_auc_score(y[te],p)); ap.append(average_precision_score(y[te],p))
    return float(np.mean(au)),float(np.std(au)),float(np.mean(ap))
horizons=[5,10,15]
out={"horizons":horizons,"prevalence":{}}
for h in horizons: out["prevalence"][h]=float(label(h).mean())
print("prevalence:",{h:round(out['prevalence'][h],3) for h in horizons},flush=True)
for m in models:
    X=E[m]["emb"].astype(np.float64); out[m]={}
    row=[]
    for h in horizons:
        au,sd,ap=probe(X,label(h),groups); out[m][h]={"auroc":au,"sd":sd,"auprc":ap}
        row.append(f"{h}min AUROC={au:.3f}±{sd:.3f}(AUPRC={ap:.2f})")
    print(f"{m:9s} "+" | ".join(row),flush=True)
json.dump(out,open(f"{root}/_expl_probe_horizons.json","w"),indent=1)
print("saved")
