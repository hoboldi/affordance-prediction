"""Rebuild per-vertex fine DINO with DIFFERENT multi-view fusion operators (ablation of the aggregation
of projected DINO latents on vertices). One render+DINO pass per object, fused 3 ways:
  mean  = facing-weighted mean  (== current vertex_dino_fine.pt; sanity baseline)
  max   = element-wise max over contributing views (peak-preserving)
  smax  = softmax(facing/tau) weighted mean (sharper view selection than linear clip)
Identical random-projection matrix (default_rng(0)) so ONLY the fusion differs. Writes
vertex_dino_fine_{mean,max,smax}.pt per recon dir. GPU."""
import os, sys, json, argparse, time
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, trimesh
from datasets.data_root_dataset import _parse_row, resolve_data_root
from datasets.mesh_loading import load_mesh
from projection.patch_vertex_mapping import render_uv_to_patch_index
from rendering.renderer import render_mesh_views
from utils.config import load_config
from generate_vertex_dino import DinoExtractor

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--tau", type=float, default=0.25)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--skip_existing", action="store_true")
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

cfg = load_config(); cfg.setdefault("rendering", {})["backend"] = "mesh"
clip_image_size = int(cfg.get("projection", {}).get("clip_image_size", 224))
data_root = resolve_data_root(cfg)
raw = [json.loads(l) for l in open(args.manifest) if l.strip()]
dirs = []
seen = set()
for r in raw:
    row = _parse_row(r, data_root=data_root)
    d = row.sam3d_reconstruction_dir
    if d is not None and str(d) not in seen:
        seen.add(str(d)); dirs.append(d)
if args.limit: dirs = dirs[:args.limit]

dev = args.device
print(f"device={dev} | {len(dirs)} objects | tau={args.tau}", flush=True)
ext = DinoExtractor(dev, image_size=448)          # dinov2-base, 32x32 fine patches
rng = np.random.default_rng(0)
proj = torch.from_numpy((rng.standard_normal((ext.dim, 128)) / np.sqrt(128)).astype(np.float32))

def norm_proj(feats):
    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-6)
    return (feats @ proj).half()

def fuse(V, idxs, patches, wmean, wsmax, mode):
    D = patches[0].shape[-1]
    if mode == "max":
        fm = torch.full((V, D), float("-inf"))
        for pidx, pat in zip(idxs, patches):
            vis = pidx >= 0
            if not vis.any(): continue
            vids = np.where(vis)[0]
            fm[vids] = torch.maximum(fm[vids], pat[pidx[vis]])
        fm[torch.isinf(fm)] = 0.0
        return fm
    W = wmean if mode == "mean" else wsmax
    fs = torch.zeros(V, D); cs = torch.zeros(V)
    for pidx, pat, w in zip(idxs, patches, W):
        vis = pidx >= 0
        if not vis.any(): continue
        vids = np.where(vis)[0]
        wv = torch.as_tensor(w[vids], dtype=torch.float32)
        fs[vids] += pat[pidx[vis]] * wv.unsqueeze(-1); cs[vids] += wv
    return fs / cs.clamp(min=1e-6).unsqueeze(-1)

n_ok = n_skip = n_fail = 0; t0 = time.time()
for i, d in enumerate(dirs):
    outs = {m: d / f"vertex_dino_fine_{m}.pt" for m in ["mean", "max", "smax"]}
    if args.skip_existing and all(p.is_file() for p in outs.values()):
        n_skip += 1; continue
    try:
        mesh = load_mesh(d / "mesh.glb", process=False)
        views = render_mesh_views(mesh, cfg)
        patches = ext.patches([v.rgb for v in views])
        V = views[0].vertex_uv.shape[0]
        idxs = [render_uv_to_patch_index(v.vertex_uv, v.vertex_visible, render_height=v.rgb.shape[0],
                render_width=v.rgb.shape[1], clip_image_size=clip_image_size, grid_h=p.grid_h, grid_w=p.grid_w)
                for v, p in zip(views, patches)]
        pats = [p.patches for p in patches]
        n = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False).vertex_normals
        n = np.asarray(n, np.float64); n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
        pos = np.asarray(mesh.vertices, np.float64)
        wmean, wsmax = [], []
        for v in views:
            dvec = np.asarray(v.camera_pose[:3, 3], np.float64)[None, :] - pos
            dvec = dvec / (np.linalg.norm(dvec, axis=1, keepdims=True) + 1e-8)
            facing = (n * dvec).sum(axis=1)
            wmean.append(np.clip(facing, 0.1, 1.0).astype(np.float32))
            wsmax.append(np.exp(np.clip(facing, 0.1, 1.0) / args.tau).astype(np.float32))
        vis_any = np.zeros(V, bool)
        for pidx in idxs: vis_any |= pidx >= 0
        vis_t = torch.from_numpy(vis_any)
        for m in ["mean", "max", "smax"]:
            feats = norm_proj(fuse(V, idxs, pats, wmean, wsmax, m))
            torch.save({"features": feats, "visible_in_any_view": vis_t}, outs[m])
        n_ok += 1
        if i < 3 or (i + 1) % 25 == 0:
            print(f"[{i+1}/{len(dirs)}] {d.name} V={V} vis={vis_any.mean():.2f} {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        n_fail += 1; print(f"FAIL {d.name}: {e!r}", flush=True)
print(f"DONE ok={n_ok} skip={n_skip} fail={n_fail} {time.time()-t0:.0f}s", flush=True)
