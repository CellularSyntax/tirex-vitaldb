import numpy as np, json, warnings, csv
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from sklearn.pipeline import make_pipeline

root = "/Users/admin/Desktop/DATA/Uni/2026/Projects/tirex-2/results"
models = ["tirex2","chronos","timesfm","moirai","tft","patchtst"]
E = {m: dict(np.load(f"{root}/embeddings_{m}_all2873.npz", allow_pickle=True)) for m in models}
ref = E["tirex2"]
# normalize caseid to int string for the join (emb is zero-padded '0004', clinical is '4')
groups = np.array([str(int(c)) for c in ref["caseid"]])
hypo = ref["hypo_event_5"].astype(int)
strat = (ref["stratum"]=="transition").astype(int)

clin={}
with open("/Users/admin/Desktop/DATA/Uni/2026/Projects/tirex-2/datasets/vitaldb/data/clinical_data.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        clin[str(int(r["caseid"]))] = r
def col(name):
    out=[]
    for c in groups:
        v = clin.get(c,{}).get(name,"")
        try: out.append(float(v))
        except: out.append(np.nan)
    return np.array(out)
age=col("age"); asa=col("asa"); bmi=col("bmi")
print(f"join: age known={np.isfinite(age).sum()}/{len(age)} asa={np.isfinite(asa).sum()} bmi={np.isfinite(bmi).sum()}", flush=True)

def probe_clf(X,y,groups,ns=5):
    aucs,aps=[],[]
    for tr,te in StratifiedGroupKFold(ns,shuffle=True,random_state=0).split(X,y,groups):
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced"))
        clf.fit(X[tr],y[tr]); p=clf.predict_proba(X[te])[:,1]
        aucs.append(roc_auc_score(y[te],p)); aps.append(average_precision_score(y[te],p))
    return float(np.mean(aucs)),float(np.std(aucs)),float(np.mean(aps))
def probe_reg(X,y,groups,ns=5):
    m=np.isfinite(y); X,y,g=X[m],y[m],groups[m]
    r2s=[]
    for tr,te in GroupKFold(ns).split(X,y,g):
        reg=make_pipeline(StandardScaler(),Ridge(alpha=10.0)); reg.fit(X[tr],y[tr])
        r2s.append(r2_score(y[te],reg.predict(X[te])))
    return float(np.mean(r2s)),float(np.std(r2s))

res={}
for m in models:
    X=E[m]["emb"].astype(np.float64)
    a=probe_clf(X,hypo,groups); s=probe_clf(X,strat,groups)
    ag=probe_reg(X,age,groups); bm=probe_reg(X,bmi,groups)
    res[m]={"hypo_auroc":a[0],"hypo_auroc_sd":a[1],"hypo_auprc":a[2],
            "stratum_auroc":s[0],"stratum_auroc_sd":s[1],"stratum_auprc":s[2],
            "age_r2":ag[0],"age_r2_sd":ag[1],"bmi_r2":bm[0],"bmi_r2_sd":bm[1]}
    print(f"{m:9s} hypo AUROC={a[0]:.3f}±{a[1]:.3f} AUPRC={a[2]:.3f} | stratum AUROC={s[0]:.3f} | age R2={ag[0]:+.3f} | bmi R2={bm[0]:+.3f}",flush=True)
res["_prevalence"]={"hypo":float(hypo.mean()),"stratum":float(strat.mean()),
                    "asa_known":int(np.isfinite(asa).sum())}
json.dump(res,open(f"{root}/_expl_probes.json","w"),indent=1)
print("saved")
