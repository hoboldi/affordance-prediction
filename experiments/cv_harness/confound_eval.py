"""CONFOUND resolution: score baseline-full-FT (no SLAT) vs SLAT-full-FT on fold-0 val, IDENTICAL settings.
baseline = outputs/basefull_fold0/last.pt (AffordanceMLP, --no_freeze_backbone).
SLAT     = outputs/cv_ft3_fold0.pt        (AffordanceMLPSlat, --full_ft).
Same fold-0 val objects, same 25k subsample, same seed, trained verbs only. CPU."""
import os, sys, json, dataclasses, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from models.slat_encoder import SlatEncoderConfig, build_slat_knn, build_vertex_to_slat
from models.mlp_slat import AffordanceMLPSlat
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

ROOT = "/home/datasets/customDatasets/cmr2"
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
SUB = 25000; rng = np.random.default_rng(0); f = lambda t: t.float() if t is not None else None

# baseline full-FT (plain MLP)
bck = torch.load("outputs/basefull_fold0/last.pt", map_location="cpu", weights_only=False)
bcfg = mlp_head_config_from_model_cfg(bck["model_cfg"])
base = AffordanceMLP(bcfg); base.load_state_dict(bck["model"]); base.eval()
# SLAT full-FT
slat_cfg = SlatEncoderConfig(in_dim=8, hidden=64, layers=3, out_dim=64, knn_k=8)
mlp_cfg = dataclasses.replace(bcfg, sam3d_dim=64)
sm = AffordanceMLPSlat(mlp_cfg, slat_cfg, verb_text_embeddings=bck["model"].get("verb_text"))
sm.load_state_dict(torch.load("outputs/cv_ft3_fold0.pt", map_location="cpu", weights_only=False)["model"]); sm.eval()

ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=bcfg.dino_filename)
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in TRAINED}
scache = {}
def sdata(o):
    if o not in scache:
        rd = f"{ROOT}/reconstructions/{o}"
        sf = torch.load(f"{rd}/slat_feats.pt", weights_only=False).float(); sc = torch.load(f"{rd}/slat_coords.pt", weights_only=False).float()
        pos = torch.as_tensor(np.asarray(torch.load(f"{rd}/vertex_positions.pt", weights_only=False)), dtype=torch.float32)
        scache[o] = (sf, sc, build_slat_knn(sc, 8), build_vertex_to_slat(pos, sc))
    return scache[o]

val = json.load(open(f"{SCR}/cv_fold0.json"))["val"]
byv = collections.defaultdict(lambda: {"b": [], "s": []})
for o in val:
    if o not in o2r: continue
    it = None
    for v in TRAINED:
        hf = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
        if not os.path.exists(hf): continue
        hb = (np.asarray(torch.load(hf, weights_only=False)).reshape(-1) >= 0.5).astype(int)
        if not (0 < hb.sum() < len(hb)): continue
        if it is None: it = ds[o2r[o]]
        V = len(hb); sel = rng.choice(V, min(SUB, V), replace=False); idx = torch.as_tensor(sel, dtype=torch.long)
        sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
        sf, sc, sk, v2s = sdata(o)
        with torch.no_grad():
            lb = base(emb[v], slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"), dino_cls=f(it.get("dino_cls")),
                      ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"), vertex_positions=None, dino_vertex=sl("dino_vertex_features"))
            ls = sm(emb[v], slat_feats=sf, slat_coords=sc, slat_knn=sk, vertex_to_slat=v2s[idx], vlm_features=sl("vertex_features"),
                    dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"), dino_vertex=sl("dino_vertex_features"))
        yb = hb[sel]
        byv[v]["b"].append(average_precision_score(yb, torch.sigmoid(lb).numpy()))
        byv[v]["s"].append(average_precision_score(yb, torch.sigmoid(ls).numpy()))

print("=== CONFOUND (fold-0 val, identical eval): baseline-full-FT vs SLAT-full-FT ===")
print(f"{'verb':9s} {'BASE-fullFT':>12s} {'SLAT-fullFT':>12s} {'d(SLAT-BASE)':>13s}")
mb, ms = [], []
for v in TRAINED:
    b, s = np.mean(byv[v]["b"]), np.mean(byv[v]["s"]); mb.append(b); ms.append(s)
    print(f"{v:9s} {b:12.3f} {s:12.3f} {s-b:+13.3f}")
print(f"{'TRAINED':9s} {np.mean(mb):12.3f} {np.mean(ms):12.3f} {np.mean(ms)-np.mean(mb):+13.3f}")
print("\nVERDICT: if d ~ 0 => SLAT encoder adds nothing on top of full-FT (full-FT itself is the driver).")
