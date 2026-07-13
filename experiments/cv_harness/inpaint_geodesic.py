"""Tier-1b GEODESIC feature inpainting: fill invisible per-vertex CLIP+DINO from the nearest visible
neighbor ALONG THE MESH SURFACE (multi-source Dijkstra over mesh edges), not Euclidean 3D.
Fixes the thin-slab smearing (seat underside no longer grabs seat-top features). Writes
*_geofilled.pt per recon dir. Originals + the Euclidean *_filled.pt are untouched."""
import os, json
import numpy as np, torch, trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

ROOT = "/home/datasets/customDatasets/cmr2"
CLIP_SRC, CLIP_DST = "vertex_semantics_clean.pt", "vertex_semantics_clean_geofilled.pt"
DINO_SRC, DINO_DST = "vertex_dino_fine.pt", "vertex_dino_fine_geofilled.pt"

objs = sorted(set(json.load(open("human_split_n100.json"))["train"]) | set(json.load(open("human_split.json"))["val"]))
print(f"{len(objs)} objects")
feats_of = lambda b: (b["features"] if isinstance(b, dict) and "features" in b else b)

done = skip = euclid_obj = 0
geo_frac, euclid_fallback_verts = [], 0
for o in objs:
    rd = f"{ROOT}/reconstructions/{o}"
    cs, ds_, gp = f"{rd}/{CLIP_SRC}", f"{rd}/{DINO_SRC}", f"{rd}/mesh.glb"
    if not all(os.path.exists(p) for p in (cs, ds_, gp, f"{rd}/vertex_positions.pt")):
        skip += 1; continue
    pos = np.asarray(torch.load(f"{rd}/vertex_positions.pt", weights_only=False), dtype=np.float64)
    cb = torch.load(cs, weights_only=False); clip = feats_of(cb).float().numpy()
    vis = np.asarray(cb["visible_in_any_view"] if isinstance(cb, dict) and "visible_in_any_view" in cb
                     else np.linalg.norm(clip, axis=1) > 1e-6).astype(bool)
    dino = feats_of(torch.load(ds_, weights_only=False)).float().numpy()
    V = len(pos)
    covered = vis & (np.linalg.norm(clip, axis=1) > 1e-6)
    clip_f, dino_f = clip.copy(), dino.copy()
    inv = np.where(~covered)[0]
    if len(inv) == 0 or covered.sum() == 0 or not (len(clip) == len(dino) == len(vis) == V):
        if not (len(clip) == len(dino) == len(vis) == V):
            skip += 1; continue
    else:
        src_idx = np.where(covered)[0]
        use_geo = False
        try:
            m = trimesh.load(gp, process=False, force="mesh")
            faces = np.asarray(m.faces)
            if len(m.vertices) == V and len(faces) > 0:
                e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
                i, j = e[:, 0], e[:, 1]; w = np.linalg.norm(pos[i] - pos[j], axis=1)
                g = csr_matrix((np.concatenate([w, w]), (np.concatenate([i, j]), np.concatenate([j, i]))), shape=(V, V))
                _, _, sources = dijkstra(g, directed=False, indices=src_idx, min_only=True, return_predecessors=True)
                src_for_inv = sources[inv]; reach = src_for_inv >= 0
                clip_f[inv[reach]] = clip[src_for_inv[reach]]
                dino_f[inv[reach]] = dino[src_for_inv[reach]]
                geo_frac.append(float(reach.mean())); use_geo = True
                if (~reach).any():  # disconnected components -> euclidean fallback
                    bad = inv[~reach]; tree = cKDTree(pos[covered]); _, nn = tree.query(pos[bad], k=1)
                    clip_f[bad] = clip[src_idx[nn]]; dino_f[bad] = dino[src_idx[nn]]
                    euclid_fallback_verts += len(bad)
        except Exception as ex:
            print(f"  mesh fail {o}: {ex}")
        if not use_geo:  # whole-object euclidean fallback (mesh missing/mismatch)
            tree = cKDTree(pos[covered]); _, nn = tree.query(pos[inv], k=1)
            clip_f[inv] = clip[src_idx[nn]]; dino_f[inv] = dino[src_idx[nn]]
            euclid_obj += 1
    torch.save({"features": torch.from_numpy(clip_f).float(), "visible_in_any_view": torch.from_numpy(vis)}, f"{rd}/{CLIP_DST}")
    torch.save({"features": torch.from_numpy(dino_f).float()}, f"{rd}/{DINO_DST}")
    done += 1
    if done % 25 == 0:
        print(f"  {done} done")
print(f"DONE: {done} filled, {skip} skipped | objects via geodesic={done-euclid_obj}, euclid-fallback objects={euclid_obj}, "
      f"euclid-fallback verts(disconnected)={euclid_fallback_verts} | mean geo-reachable frac={np.mean(geo_frac):.4f}")
