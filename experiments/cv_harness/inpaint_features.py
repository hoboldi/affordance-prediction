"""Tier-1 feature inpainting: fill INVISIBLE per-vertex CLIP + DINO features from the nearest VISIBLE
neighbor (3D cKDTree). Writes vertex_semantics_clean_filled.pt + vertex_dino_fine_filled.pt per recon dir
for the objects used by the n100 FT (train) + val. Originals are untouched."""
import os, sys, json, glob
import numpy as np, torch
from scipy.spatial import cKDTree

ROOT = "/home/datasets/customDatasets/cmr2"
CLIP_SRC, CLIP_DST = "vertex_semantics_clean.pt", "vertex_semantics_clean_filled.pt"
DINO_SRC, DINO_DST = "vertex_dino_fine.pt", "vertex_dino_fine_filled.pt"

objs = set(json.load(open("human_split_n100.json"))["train"]) | set(json.load(open("human_split.json"))["val"])
print(f"{len(objs)} objects to inpaint")

def feats_of(blob):
    return (blob["features"] if isinstance(blob, dict) and "features" in blob else blob)

done, skip, cov = 0, 0, []
for o in sorted(objs):
    rd = f"{ROOT}/reconstructions/{o}"
    cs, ds_ = f"{rd}/{CLIP_SRC}", f"{rd}/{DINO_SRC}"
    if not (os.path.exists(cs) and os.path.exists(ds_) and os.path.exists(f"{rd}/vertex_positions.pt")):
        skip += 1; continue
    pos = np.asarray(torch.load(f"{rd}/vertex_positions.pt", weights_only=False), dtype=np.float64)
    cb = torch.load(cs, weights_only=False)                       # dict: features (V,Dc) + visible_in_any_view (V,)
    clip = feats_of(cb).float().numpy()
    vis = cb.get("visible_in_any_view") if isinstance(cb, dict) else None
    if vis is None:
        vis = (np.linalg.norm(clip, axis=1) > 1e-6)
    vis = np.asarray(vis).astype(bool)
    db = torch.load(ds_, weights_only=False)
    dino = feats_of(db).float().numpy()
    V = len(pos)
    if not (len(clip) == len(dino) == len(vis) == V):
        skip += 1; continue
    cov.append(float(vis.mean()))
    covered = vis & (np.linalg.norm(clip, axis=1) > 1e-6)
    if covered.sum() == 0 or covered.all():
        # nothing to fill (or nothing visible) -> just copy through
        clip_f, dino_f = clip, dino
    else:
        tree = cKDTree(pos[covered])
        cov_idx = np.where(covered)[0]
        _, nn = tree.query(pos[~covered], k=1)              # nearest covered vertex for each uncovered
        src = cov_idx[nn]
        clip_f, dino_f = clip.copy(), dino.copy()
        clip_f[~covered] = clip[src]
        dino_f[~covered] = dino[src]
    out_c = {"features": torch.from_numpy(clip_f).float(), "visible_in_any_view": torch.from_numpy(vis)}
    torch.save(out_c, f"{rd}/{CLIP_DST}")
    torch.save({"features": torch.from_numpy(dino_f).float()}, f"{rd}/{DINO_DST}")
    done += 1
    if done % 25 == 0:
        print(f"  {done} done (mean visible frac so far {np.mean(cov):.3f})")
print(f"DONE: filled {done}, skipped {skip}, mean visible frac {np.mean(cov):.3f}")
