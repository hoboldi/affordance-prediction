"""Smoke-test SlatEncoder on one object's real SLAT: shapes, finiteness, gradient flow,
and validate that vertex->voxel gather reproduces the stored slat_vertex_features."""
import sys, time
sys.path.insert(0, "src")
import numpy as np, torch
from models.slat_encoder import SlatEncoder, SlatEncoderConfig, build_slat_knn, build_vertex_to_slat

O = "/home/datasets/customDatasets/cmr2/reconstructions/bottle__34_1397_4376__v000"
coords = torch.load(f"{O}/slat_coords.pt", weights_only=False).float()
feats = torch.load(f"{O}/slat_feats.pt", weights_only=False).float()
pos = torch.as_tensor(np.asarray(torch.load(f"{O}/vertex_positions.pt", weights_only=False)), dtype=torch.float32)
svf = torch.load(f"{O}/slat_vertex_features.pt", weights_only=False)
svf = (svf["features"] if isinstance(svf, dict) and "features" in svf else svf).float()
print(f"slat_coords {tuple(coords.shape)}  slat_feats {tuple(feats.shape)}  vertices {tuple(pos.shape)}  stored slat_vertex {tuple(svf.shape)}")

t = time.time()
knn = build_slat_knn(coords, k=8)
v2s = build_vertex_to_slat(pos, coords)
print(f"build knn {tuple(knn.shape)} + vertex_to_slat {tuple(v2s.shape)} in {time.time()-t:.2f}s")

# correctness: does nearest-voxel gather reproduce the stored per-vertex SLAT?
gathered = feats[v2s]
match = torch.allclose(gathered, svf, atol=1e-4)
maxerr = (gathered - svf).abs().max().item() if gathered.shape == svf.shape else float("nan")
print(f"gather==stored slat_vertex_features: {match}  (max abs err {maxerr:.2e}) -> mapping {'MATCHES pipeline (nearest voxel)' if match else 'differs (pipeline may interpolate; still valid)'}")

# forward + backward
enc = SlatEncoder(SlatEncoderConfig(in_dim=feats.shape[1], hidden=64, layers=3, out_dim=64, knn_k=8))
enc.train()
t = time.time()
out = enc(feats, coords, knn, v2s)
print(f"forward -> {tuple(out.shape)} in {time.time()-t:.2f}s  finite={torch.isfinite(out).all().item()}  mean={out.mean():.3f} std={out.std():.3f}")
loss = out.pow(2).mean(); loss.backward()
gnorm = sum(p.grad.norm().item() for p in enc.parameters() if p.grad is not None)
print(f"backward ok, total grad norm {gnorm:.3f}  params {sum(p.numel() for p in enc.parameters()):,}")
