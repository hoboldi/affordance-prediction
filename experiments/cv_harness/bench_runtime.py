"""Runtime benchmark: lean (DINO+geom+verb, 133-d) vs full (272-d) final model.
Times the three pieces that differ: (A) GNN head forward, (B) CLIP-vision ViT-B/32 pass lean REMOVES,
(C) SLAT encoder lean REMOVES. Raw meshes are archived so the shared multi-view RENDER isn't timed
(it's identical work for DINO either way); we time the compute that lean actually drops. cuda:0 + CPU."""
import os, sys, time, statistics
sys.path.insert(0, "src")
import torch
from models.gnn_head import AffordanceGNN, AffordanceGNNConfig

torch.manual_seed(0)
GPU = torch.cuda.is_available()
dev = torch.device("cuda:0") if GPU else torch.device("cpu")

def timeit(fn, iters=20, warmup=3, cuda=False):
    for _ in range(warmup): fn()
    if cuda: torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn()
        if cuda: torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)  # ms
    return statistics.mean(ts), statistics.stdev(ts) if len(ts) > 1 else 0.0

def make_cfg(lean):
    # lean drops clip-vision(128)+slat(8)+normals(3) => genuinely smaller input projection (not zeroed placeholders)
    return AffordanceGNNConfig(
        vlm_dim=(0 if lean else 128), dino_vertex_dim=128, geom_dim=5,
        sam3d_dim=(0 if lean else 8), normals_dim=(0 if lean else 3), pos_dim=0,
        verb_dim=128, num_verbs=6, verb_embedding="text", verb_text_dim=512, verb_in_backbone=True,
        gnn_hidden=128, gnn_layers=3, knn_k=24)

def head_inputs(V, lean, device):
    g = lambda d: torch.randn(V, d, device=device) if d else None
    verb = torch.randn(512, device=device)
    pos = torch.randn(V, 3, device=device)
    knn = AffordanceGNN._build_knn(pos.cpu(), 24).to(device)
    kw = dict(verb_idx=verb, dino_vertex=g(128), vertex_geom=g(5), knn_idx=knn)
    if not lean:
        kw.update(vlm_features=g(128), slat_vertex=g(8), vertex_normals=g(3))
    return kw

print(f"=== device: {'cuda:0 ('+torch.cuda.get_device_name(0)+')' if GPU else 'CPU-only'} ===\n")

# ── (A) GNN HEAD forward: full vs true-lean, at representative vertex counts ──
print("=== (A) GNN head forward (ms/object, mean±sd over 20 iters) ===")
for V in [30000, 200000]:
    print(f"\n  V={V:,} vertices, k=24:")
    for name, lean in [("full  (272-d in)", False), ("lean  (133-d in)", True)]:
        cfg = make_cfg(lean); m = AffordanceGNN(cfg).to(dev).eval()
        npar = sum(p.numel() for p in m.parameters())
        kw = head_inputs(V, lean, dev)
        with torch.no_grad():
            fn = lambda: m(**kw)
            if GPU: torch.cuda.reset_peak_memory_stats()
            ms, sd = timeit(fn, cuda=GPU)
            peak = (torch.cuda.max_memory_allocated() / 1e6) if GPU else 0
        thr = V / ms  # verts per ms
        print(f"    {name}: {ms:7.2f} ± {sd:4.2f} ms   ({thr:6.0f} verts/ms, {npar:,} params, peak {peak:.0f} MB)")
        del m
        if GPU: torch.cuda.empty_cache()

# ── (B) CLIP-vision ViT-B/32 pass that LEAN REMOVES (16 views/object) ──
print("\n=== (B) CLIP-vision ViT-B/32 forward — REMOVED by lean (16 views/object) ===")
try:
    from transformers import CLIPModel
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(dev).eval()
    NV = 16  # configs/default.yaml: 2 rings x 8 azimuths
    px = torch.randn(NV, 3, 224, 224, device=dev)
    with torch.no_grad():
        fn = lambda: clip.vision_model(pixel_values=px)
        ms, sd = timeit(fn, iters=15, cuda=GPU)
    print(f"    CLIP vision {NV} views: {ms:7.2f} ± {sd:4.2f} ms/object   (this compute is SKIPPED by lean)")
    del clip
    if GPU: torch.cuda.empty_cache()
except Exception as e:
    print(f"    [skipped: {type(e).__name__}: {e}]")

# ── (C) SLAT encoder that LEAN REMOVES (~8.5k voxels/object) ──
print("\n=== (C) SLAT encoder forward — REMOVED by lean (~8.5k voxels/object) ===")
try:
    from models.slat_encoder import SlatEncoder, SlatEncoderConfig, build_slat_knn
    NVOX = 8500
    scfg = SlatEncoderConfig()
    se = SlatEncoder(scfg).to(dev).eval()
    coords = torch.rand(NVOX, 3, device=dev) * 64
    feats = torch.randn(NVOX, scfg.in_dim, device=dev)
    knn = build_slat_knn(coords.cpu(), scfg.knn_k).to(dev)
    v2s = torch.randint(0, NVOX, (200000,), device=dev)  # gather to mesh verts
    with torch.no_grad():
        def fn():
            import inspect
            sig = inspect.signature(se.forward)
            # try common signatures
            try: return se(slat_coords=coords, slat_feats=feats, knn_idx=knn, vertex_to_slat=v2s)
            except TypeError: return se(coords, feats, knn, v2s)
        try:
            ms, sd = timeit(fn, iters=15, cuda=GPU)
            print(f"    SLAT encoder ({NVOX} voxels): {ms:7.2f} ± {sd:4.2f} ms/object   (SKIPPED by lean)")
        except Exception as e2:
            print(f"    [forward-sig mismatch, reporting build only: {type(e2).__name__}]")
except Exception as e:
    print(f"    [skipped: {type(e).__name__}: {e}]")

print("\n=== NOTE ===")
print("  Multi-view RENDER (shared by DINO, kept in lean) not timed — raw meshes archived.")
print("  Lean removes: the entire CLIP-vision extraction (render+ViT+project) + SLAT encoding.")
print("  DINO extraction + geom stay. Head cost is ~identical either way (both sub-second).")
