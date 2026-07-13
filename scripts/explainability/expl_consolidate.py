import numpy as np, json, warnings, csv as csvmod
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
import pandas as pd
root="/Users/admin/Desktop/DATA/Uni/2026/Projects/tirex-2/results"
models=["tirex2","chronos","timesfm","moirai","tft","patchtst"]
LABEL={"tirex2":"TiRex-2","chronos":"Chronos-Bolt","timesfm":"TimesFM-2.5","moirai":"Moirai-1.1-R",
       "tft":"TFT","patchtst":"PatchTST"}
TYPE={"tirex2":"zero-shot","chronos":"zero-shot","timesfm":"zero-shot","moirai":"zero-shot",
      "tft":"task-trained","patchtst":"task-trained"}
horizons=[1,2,3,5,10,15]

# ---- TABLE A: hypotension probe AUROC/AUPRC per model x horizon x cohort ----
def probe(X,y,g,ns=5):
    au,ap=[],[]
    for tr,te in StratifiedGroupKFold(ns,shuffle=True,random_state=0).split(X,y,g):
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced"))
        clf.fit(X[tr],y[tr]); p=clf.predict_proba(X[te])[:,1]
        au.append(roc_auc_score(y[te],p)); ap.append(average_precision_score(y[te],p))
    return np.mean(au),np.std(au),np.mean(ap)
rowsA=[]
for tag,cname in [("all2873","VitalDB"),("mover_art","MOVER")]:
    E={m:dict(np.load(f"{root}/embeddings_{m}_{tag}.npz",allow_pickle=True)) for m in models}
    ref=E["tirex2"]; g=np.array([str(c) for c in ref["caseid"]]); tev=ref["t_event_65"].astype(float)
    for m in models:
        X=E[m]["emb"].astype(np.float64)
        for h in horizons:
            y=(np.isfinite(tev)&(tev<=h)).astype(int)
            if y.sum()<40: continue
            au,sd,ap=probe(X,y,g)
            rowsA.append(dict(cohort=cname,model=LABEL[m],type=TYPE[m],horizon_min=h,
                              prevalence=round(float(y.mean()),3),
                              AUROC=round(au,3),AUROC_sd=round(sd,3),AUPRC=round(ap,3)))
pd.DataFrame(rowsA).to_csv(f"{root}/suppl_probe_hypotension.csv",index=False)
print("A hypotension probe:",len(rowsA),"rows")

# ---- TABLE B: trajectory R2 ----
TR=json.load(open(f"{root}/_expl_trajectory.json"))["trajectory"]
tnames=["MAP_now","MAP_fut5_mean","MAP_fut5_min","MAP_futH_mean","MAP_slope5","MAP_slopeH"]
rowsB=[]
for tag,cname in [("all2873","VitalDB"),("mover_art","MOVER")]:
    for m in models:
        r=dict(cohort=cname,model=LABEL[m],type=TYPE[m])
        for t in tnames: r[t+"_R2"]=TR[tag][m][t]["r2"]
        rowsB.append(r)
pd.DataFrame(rowsB).to_csv(f"{root}/suppl_probe_trajectory.csv",index=False)
print("B trajectory probe:",len(rowsB),"rows")

# ---- TABLE C: cluster characterization (already computed) ----
dfc=pd.read_csv(f"{root}/_expl_cluster_characterization.csv")
dfc["model"]=dfc["model"].map(LABEL); dfc["cohort"]=dfc["cohort"].map({"all2873":"VitalDB","mover_art":"MOVER"})
dfc.to_csv(f"{root}/suppl_cluster_characterization.csv",index=False)
print("C cluster char:",len(dfc),"rows")

# ---- TABLE D: RSA/CKA matrices ----
RSA=json.load(open(f"{root}/_expl_rsa_both.json"))
for tag,cname in [("all2873","VitalDB"),("mover_art","MOVER")]:
    M=np.array(RSA[tag]["cka"]); labs=[LABEL[m] for m in RSA[tag]["models"]]
    pd.DataFrame(M,index=labs,columns=labs).to_csv(f"{root}/suppl_rsa_cka_{cname}.csv")
print("D RSA matrices written (VitalDB, MOVER)")
print("\nall CSVs in results/")
