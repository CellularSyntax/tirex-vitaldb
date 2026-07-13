import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
os.makedirs("/tmp/numba_cache", exist_ok=True)
import numpy as np, json, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap

root = "/Users/admin/Desktop/DATA/Uni/2026/Projects/tirex-2/results"
models = ["tirex2","chronos","timesfm","moirai","tft","patchtst"]
E = {m: dict(np.load(f"{root}/embeddings_{m}_all2873.npz", allow_pickle=True)) for m in models}
strat = E["tirex2"]["stratum"]; hypo = E["tirex2"]["hypo_event_5"].astype(int)
strat_bin = (strat=="transition").astype(int)
umaps, clust = {}, {}
for m in models:
    X = StandardScaler().fit_transform(E[m]["emb"])
    umaps[m] = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=0).fit_transform(X)
    clust[m] = {"sil_stratum": float(silhouette_score(X, strat_bin)),
                "sil_hypo": float(silhouette_score(X, hypo))}
    print(f"{m:9s} silhouette stratum={clust[m]['sil_stratum']:+.4f} hypo={clust[m]['sil_hypo']:+.4f}", flush=True)
np.savez(f"{root}/_expl_umaps.npz", **{m: umaps[m] for m in models}, stratum=strat, hypo=hypo)
json.dump(clust, open(f"{root}/_expl_clustering.json","w"), indent=1)
print("saved")
