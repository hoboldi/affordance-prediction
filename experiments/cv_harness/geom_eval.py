"""Does geometry help cross-category grasp? Score the held-out bottles with two LOCO models, identical settings:
  NO-GEOM  outputs/loco_bottle/last.pt       (full-FT, no geom)      -> the 0.36-grasp baseline
  GEOM     outputs/locogeom_bottle/last.pt   (full-FT + 5 geom ch)   -> passes vertex_geom
Both never saw a bottle. Per-verb AUPRC vs human, SUB=25000, seed 0. CPU."""
import os, sys, json, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

CAT = sys.argv[1] if len(sys.argv) > 1 else "bottle"
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
TRAINED = ["contain", "pour", "grasp"]   # script skips verbs a category lacks
SUB = 25000; rng = np.random.default_rng(0); f = lambda t: t.float() if t is not None else None

def load(p):
    c = torch.load(p, map_location="cpu", weights_only=False); m = AffordanceMLP(mlp_head_config_from_model_cfg(c["model_cfg"])); m.load_state_dict(c["model"]); m.eval()
    return m, int(getattr(mlp_head_config_from_model_cfg(c["model_cfg"]), "geom_dim", 0))
nogeom, _ = load(f"outputs/loco_{CAT}/last.pt")
geom, gdim = load(f"outputs/locogeom_{CAT}/last.pt")
print(f"[{CAT}] geom model geom_dim = {gdim} (should be 5)")

bcfg = mlp_head_config_from_model_cfg(torch.load(f"outputs/locogeom_{CAT}/last.pt", map_location="cpu", weights_only=False)["model_cfg"])
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=bcfg.dino_filename,
                     load_vertex_geom=True, geom_filename="vertex_geom.pt")
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in TRAINED}
cat_objs = [o for o in o2r if o.split("__")[0] == CAT]

byv = collections.defaultdict(lambda: {"ng": [], "g": []})
for o in cat_objs:
    it = None
    for v in TRAINED:
        hf = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
        if not os.path.exists(hf): continue
        hb = (np.asarray(torch.load(hf, weights_only=False)).reshape(-1) >= 0.5).astype(int)
        if not (0 < hb.sum() < len(hb)): continue
        if it is None: it = ds[o2r[o]]
        V = len(hb); sel = rng.choice(V, min(SUB, V), replace=False); idx = torch.as_tensor(sel, dtype=torch.long)
        sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
        kw = dict(vlm_features=sl("vertex_features"), dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")),
                  vertex_normals=sl("vertex_normals"), vertex_positions=None, dino_vertex=sl("dino_vertex_features"))
        with torch.no_grad():
            png = torch.sigmoid(nogeom(emb[v], slat_vertex=sl("slat_vertex_features"), **kw)).numpy()
            pg = torch.sigmoid(geom(emb[v], slat_vertex=sl("slat_vertex_features"), vertex_geom=sl("vertex_geom"), **kw)).numpy()
        byv[v]["ng"].append(average_precision_score(hb[sel], png))
        byv[v]["g"].append(average_precision_score(hb[sel], pg))

print(f"\n=== LOCO({CAT}) transfer: does geometry help? ({len(cat_objs)} {CAT}s) ===")
print(f"{'verb':9s} {'NO-GEOM':>8s} {'+GEOM':>8s} {'delta':>8s}")
for v in TRAINED:
    ng, g = np.mean(byv[v]["ng"]), np.mean(byv[v]["g"])
    print(f"{v:9s} {ng:8.3f} {g:8.3f} {g-ng:+8.3f}")
