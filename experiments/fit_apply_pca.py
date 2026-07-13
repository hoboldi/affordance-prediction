"""Fit PCA-128 on the raw 768-d DINO features (L2-normalized, as stored) from a sample of objects,
then apply to all → vertex_dino_pca128.pt. Compares PCA vs the random projection used by the final model."""
import sys, glob, os
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.decomposition import PCA
DR = "/home/datasets/customDatasets/cmr2/reconstructions"
paths = sorted(glob.glob(f"{DR}/*/vertex_dino_raw768.pt"))
print(f"{len(paths)} objects with raw768 features", flush=True)
assert len(paths) > 0, "run extraction first"

# ---- fit PCA on a seeded sample of objects (subsample vertices to keep memory bounded) ----
rng = np.random.default_rng(0)
fit_idx = sorted(rng.choice(len(paths), min(40, len(paths)), replace=False))
rows = []
for i in fit_idx:
    f = torch.load(paths[i], map_location="cpu", weights_only=False)["features"].float().numpy()  # (V,768) L2-normed
    v = f.shape[0]; take = rng.choice(v, min(3000, v), replace=False)
    rows.append(f[take])
X = np.concatenate(rows, 0)
print(f"fitting PCA-128 on {X.shape} sampled vertices", flush=True)
pca = PCA(n_components=128, svd_solver="randomized", random_state=0).fit(X)
print(f"explained variance (128 comps): {pca.explained_variance_ratio_.sum():.3f}", flush=True)
torch.save({"components": torch.tensor(pca.components_), "mean": torch.tensor(pca.mean_)}, "experiments/dino_pca128.pt")

# ---- apply to all objects ----
for j, p in enumerate(paths):
    d = torch.load(p, map_location="cpu", weights_only=False)
    f = d["features"].float().numpy()
    z = pca.transform(f).astype(np.float16)                     # (V,128)
    out = os.path.join(os.path.dirname(p), "vertex_dino_pca128.pt")
    torch.save({"features": torch.from_numpy(z), "visible_in_any_view": d.get("visible_in_any_view")}, out)
    if (j + 1) % 50 == 0: print(f"  applied {j+1}/{len(paths)}", flush=True)
print("done → vertex_dino_pca128.pt written for all objects", flush=True)
