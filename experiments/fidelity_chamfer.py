"""Reconstruction fidelity: Chamfer distance + F-score between SAM3D meshes and CO3D ground-truth
point clouds (on disk at cmr2/source). Normalize both to unit scale, ICP-align (orientation restarts),
report per-object + by-category. Restricted to CV objects. CPU."""
import sys, os, json, glob, argparse
sys.path.insert(0, "src")
import numpy as np, trimesh
from scipy.spatial import cKDTree
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
SRC = "/home/datasets/customDatasets/cmr2/source"
DR  = "/home/datasets/customDatasets/cmr2/reconstructions"
NPTS = 12000; TAU = 0.05
ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); args = ap.parse_args()

cv = set()
for k in range(5):
    cv |= set(json.load(open(f"{SCR}/cv_fold{k}.json"))["val"])
cv = sorted(cv)

def src_ply(stem):
    p = stem.split("__")
    return f"{SRC}/{p[0]}/{p[1]}/pointcloud.ply" if len(p) >= 2 else None

have = [s for s in cv if src_ply(s) and os.path.exists(src_ply(s)) and os.path.exists(f"{DR}/{s}/mesh.glb")]
print(f"CV objects: {len(cv)}  with CO3D pointcloud + mesh: {len(have)}", flush=True)
if args.limit:  # spread across categories for the test
    bycat = {}
    for s in have: bycat.setdefault(s.split("__")[0], []).append(s)
    have = [v[0] for v in bycat.values()][:args.limit]
    print(f"TEST mode: {len(have)} objects, one per category: {[h.split('__')[0] for h in have]}", flush=True)

def norm(P):
    P = P - P.mean(0); s = np.sqrt((P ** 2).sum(1)).mean(); return P / (s + 1e-9)
def cham_f(A, B, tau=TAU):
    dab, _ = cKDTree(B).query(A); dba, _ = cKDTree(A).query(B)
    p = (dba < tau).mean(); r = (dab < tau).mean(); f = 2 * p * r / (p + r + 1e-9)
    return float(dab.mean() + dba.mean()), float(f)
def octahedral_inits():  # 24 proper axis-aligned rotations to resolve orientation ambiguity
    import itertools
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i, p in enumerate(perm): R[i, p] = signs[i]
            if abs(np.linalg.det(R) - 1) < 1e-6:
                M = np.eye(4); M[:3, :3] = R; out.append(M)
    return out
INITS = octahedral_inits()

rows = []
for i, s in enumerate(have):
    try:
        gt = trimesh.load(src_ply(s), process=False)
        G = np.asarray(gt.vertices, np.float64)
        if len(G) > NPTS: G = G[np.random.default_rng(0).choice(len(G), NPTS, replace=False)]
        mesh = trimesh.load(f"{DR}/{s}/mesh.glb", process=False, force="mesh")
        M = np.asarray(mesh.sample(NPTS), np.float64)
        G = norm(G); M = norm(M)
        best = None
        for init in INITS:
            try:
                mat, tr, _ = trimesh.registration.icp(M, G, initial=init, max_iterations=30, threshold=1e-5)
            except Exception:
                continue
            c, f = cham_f(tr, G)
            if best is None or c < best[0]: best = (c, f)
        if best is None: continue
        rows.append((s.split("__")[0], s, best[0], best[1]))
        if args.limit: print(f"  {s.split('__')[0]:9s} chamfer={best[0]:.4f}  F@{TAU}={best[1]:.3f}", flush=True)
        elif (i + 1) % 25 == 0: print(f"  {i+1}/{len(have)}", flush=True)
    except Exception as e:
        print(f"  SKIP {s}: {e!r}", flush=True)

if rows:
    ch = np.array([r[2] for r in rows]); fs = np.array([r[3] for r in rows])
    print(f"\n=== Reconstruction fidelity (n={len(rows)}) ===")
    print(f"  mean Chamfer (normalized): {ch.mean():.4f} ± {ch.std():.4f}")
    print(f"  mean F-score @ {TAU}:       {fs.mean():.3f} ± {fs.std():.3f}")
    cats = sorted(set(r[0] for r in rows))
    print("  by category:")
    for c in cats:
        cc = [r[2] for r in rows if r[0] == c]; ff = [r[3] for r in rows if r[0] == c]
        print(f"    {c:10s} n={len(cc):3d}  chamfer={np.mean(cc):.4f}  F={np.mean(ff):.3f}")
