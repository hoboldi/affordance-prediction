"""End-to-end smoke test: build AffordanceMLPSlat, warm-start from base, forward+backward on one object."""
import sys, dataclasses, time
sys.path.insert(0, "src")
import numpy as np, torch
from models.mlp_head import mlp_head_config_from_model_cfg
from models.slat_encoder import SlatEncoderConfig, build_slat_knn, build_vertex_to_slat
from models.mlp_slat import AffordanceMLPSlat, load_base_into_slat_model
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

O = "/home/datasets/customDatasets/cmr2/reconstructions/bottle__34_1397_4376__v000"
def feat(p):
    d = torch.load(f"{O}/{p}", weights_only=False)
    return (d["features"] if isinstance(d, dict) and "features" in d else d)

base_ck = torch.load("outputs/ov_concat_finepatch/best.pt", map_location="cpu", weights_only=False)
base_cfg = mlp_head_config_from_model_cfg(base_ck["model_cfg"])
print("base per-vertex dims: vlm", base_cfg.vlm_dim, "dino", base_cfg.dino_vertex_dim, "sam3d", base_cfg.sam3d_dim, "normals", base_cfg.normals_dim)

SLAT_OUT = 64
slat_cfg = SlatEncoderConfig(in_dim=8, hidden=64, layers=3, out_dim=SLAT_OUT, knn_k=8)
mlp_cfg = dataclasses.replace(base_cfg, sam3d_dim=SLAT_OUT)      # encoder output replaces the flat 8-d SLAT
vte = base_ck["model"].get("verb_text")
model = AffordanceMLPSlat(mlp_cfg, slat_cfg, verb_text_embeddings=vte)
loaded, skipped = load_base_into_slat_model(model, "outputs/ov_concat_finepatch/best.pt")
n_enc = sum(p.numel() for p in model.slat_encoder.parameters())
print(f"warm-start: {loaded} params loaded from base, {skipped} skipped (reshaped input layers) | SlatEncoder adds {n_enc:,} params")

# real features
pos = torch.as_tensor(np.asarray(feat("vertex_positions.pt")), dtype=torch.float32)
slat_feats = feat("slat_feats.pt").float(); slat_coords = feat("slat_coords.pt").float()
knn = build_slat_knn(slat_coords, 8); v2s = build_vertex_to_slat(pos, slat_coords)
kw = dict(
    vlm_features=feat("vertex_semantics_clean.pt").float(),
    dino_vertex=feat("vertex_dino_fine.pt").float(),
    dino_cls=torch.load(f"{O}/dino_cls.pt", weights_only=False).float(),
    ss_dino_cls=torch.load(f"{O}/ss_dino_cls.pt", weights_only=False).float(),
    vertex_normals=torch.as_tensor(np.asarray(torch.load(f"{O}/vertex_normals.pt", weights_only=False)), dtype=torch.float32),
)
emb = VLMWrapper(VLMConfig(device="cpu")).encode_text(["grasp"])[0]

model.train()
t = time.time()
out = model(emb, slat_feats=slat_feats, slat_coords=slat_coords, slat_knn=knn, vertex_to_slat=v2s, **kw)
print(f"forward -> {tuple(out.shape)} in {time.time()-t:.2f}s  finite={torch.isfinite(out).all().item()}  logit mean={out.mean():.3f}")
loss = torch.nn.functional.binary_cross_entropy_with_logits(out, (out.detach() > 0).float())
loss.backward()
genc = sum(p.grad.norm().item() for p in model.slat_encoder.parameters() if p.grad is not None)
gmlp = sum(p.grad.norm().item() for p in model.mlp.parameters() if p.grad is not None)
print(f"backward ok: grad norm encoder={genc:.3f}  mlp={gmlp:.3f}  (both >0 => both branches train)")
