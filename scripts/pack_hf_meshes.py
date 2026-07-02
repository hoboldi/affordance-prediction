"""Build a meshes-ONLY export of the reconstruction dataset for human labeling on HuggingFace.

For each reconstruction: verify mesh.glb's vertex order == vertex_positions.pt (best rigid+scale fit
residual ~0 => index i<->i, so a label on glb vertex i maps 1:1 onto our training vertex i), then copy
mesh.glb + vertex_positions.pt into an HF-ready folder. Latents/features are deliberately EXCLUDED.
Writes alignment_manifest.json {object: {n_vertices, sha256(vertex_positions), residual, ok}} so HF<->local
alignment can be diffed in one command, and so order-NOT-preserved objects are flagged & excluded.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path("/home/datasets/customDatasets/cmr2")
SRC = ROOT / "reconstructions"
OUT = ROOT / "hf_meshes_only"
OUT.mkdir(exist_ok=True)
RESID_TOL = 0.02  # mean residual after best rigid+scale fit; > tol => order NOT preserved -> exclude


def load_vp(p: Path) -> np.ndarray:
    v = torch.load(p, weights_only=False)
    return np.asarray(v["vertices"] if isinstance(v, dict) else v, dtype=float)


def residual(vp: np.ndarray, gv: np.ndarray) -> float:
    """Mean per-vertex residual after the best rigid+scale fit assuming index i<->i."""
    if vp.shape != gv.shape:
        return 1e9
    A = vp - vp.mean(0)
    B = gv - gv.mean(0)
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    s = S.sum() / max((A ** 2).sum(), 1e-12)
    return float(np.sqrt(((B - s * (A @ R.T)) ** 2).sum(1)).mean())


def main() -> None:
    dirs = sorted(d for d in SRC.iterdir() if d.is_dir())
    manifest: dict[str, dict] = {}
    ok = fail = skip = 0
    for i, d in enumerate(dirs):
        glb, vpp = d / "mesh.glb", d / "vertex_positions.pt"
        if not (glb.exists() and vpp.exists()):
            skip += 1
            continue
        try:
            vp = load_vp(vpp)
            gv = np.asarray(trimesh.load(glb, force="mesh", process=False).vertices, dtype=float)
            res = residual(vp, gv)
        except Exception as e:  # noqa: BLE001
            fail += 1
            manifest[d.name] = {"ok": False, "error": str(e)}
            print(f"ERR {d.name}: {e}", flush=True)
            continue
        if res > RESID_TOL:
            fail += 1
            manifest[d.name] = {"ok": False, "residual": round(res, 6), "n_vertices": int(len(vp))}
            print(f"FAIL order {d.name} resid={res:.4f} (EXCLUDED)", flush=True)
            continue
        od = OUT / d.name
        od.mkdir(exist_ok=True)
        shutil.copy2(glb, od / "mesh.glb")
        shutil.copy2(vpp, od / "vertex_positions.pt")
        manifest[d.name] = {
            "ok": True,
            "n_vertices": int(len(vp)),
            "residual": round(res, 6),
            "sha256_vpos": hashlib.sha256(vpp.read_bytes()).hexdigest(),
        }
        ok += 1
        if i % 200 == 0:
            print(f"{i}/{len(dirs)} ok={ok} fail={fail} skip={skip}", flush=True)

    (OUT / "alignment_manifest.json").write_text(json.dumps(manifest, indent=0))
    print(f"DONE: {ok} packed, {fail} failed-order, {skip} missing-files (of {len(dirs)})", flush=True)


if __name__ == "__main__":
    main()
