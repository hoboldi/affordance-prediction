"""Aggregate 5-fold CV: each object scored by the fold model that held it OUT (last.pt). Per-verb means
(FT vs GEAL) over ALL labeled objects + bootstrap CI on the gap. Reuses cv_fold{k}.json + manifest.cv.jsonl.
CPU only."""
import os, sys, json, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

ROOT = "/home/datasets/customDatasets/cmr2"
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
K = 5
MAN = f"{SCR}/manifest.cv.jsonl"

cfg0 = mlp_head_config_from_model_cfg(torch.load("outputs/cv_fold0/last.pt", map_location="cpu", weights_only=False)["model_cfg"])
ds = DataRootDataset(manifest_path=MAN, load_vertex_labels_eager=False, load_vertex_semantics_eager=True,
                     load_vertex_dino=True, dino_filename=cfg0.dino_filename)
obj2row = {}
for i, r in enumerate(ds.rows):
    obj2row.setdefault(os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")), i)
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in TRAINED}

def predict(model, it, e):
    f = lambda t: t.float() if t is not None else None
    with torch.no_grad():
        lo = model(e, slat_vertex=f(it.get("slat_vertex_features")), vlm_features=f(it.get("vertex_features")),
                   dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")),
                   vertex_normals=f(it.get("vertex_normals")), vertex_positions=None,
                   dino_vertex=f(it.get("dino_vertex_features")), vertex_geom=None)
    return torch.sigmoid(lo).numpy()

ftmap, gmap = collections.defaultdict(dict), collections.defaultdict(dict)
for k in range(K):
    val_objs = json.load(open(f"{SCR}/cv_fold{k}.json"))["val"]
    ck = torch.load(f"outputs/cv_fold{k}/last.pt", map_location="cpu", weights_only=False)
    m = AffordanceMLP(mlp_head_config_from_model_cfg(ck["model_cfg"])); m.load_state_dict(ck["model"]); m.eval()
    for o in val_objs:
        if o not in obj2row:
            continue
        it = None
        for v in TRAINED:
            hf = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
            gf = f"{ROOT}/reconstructions/{o}/vertex_pseudolabels_{v}.pt"
            if not (os.path.exists(hf) and os.path.exists(gf)):
                continue
            hum = np.asarray(torch.load(hf, weights_only=False)).reshape(-1); hb = (hum >= 0.5).astype(int)
            geal = np.asarray(torch.load(gf, weights_only=False)).reshape(-1)
            if hb.sum() == 0:
                continue
            if it is None:
                it = ds[obj2row[o]]
            pf = predict(m, it, emb[v])
            if not (len(pf) == len(geal) == len(hb)):
                continue
            ftmap[o][v] = average_precision_score(hb, pf); gmap[o][v] = average_precision_score(hb, geal)
    print(f"fold {k}: scored {sum(1 for o in val_objs if o in ftmap)} held-out objects")

objs = np.array(sorted(ftmap.keys()))
def tmean(draw, M):
    pv = [np.mean([M[o][v] for o in draw if v in M[o]]) for v in TRAINED]
    return float(np.mean([x for x in pv if not np.isnan(x)]))
print(f"\n=== CROSS-VALIDATED per-verb (out-of-fold FT vs GEAL, vs human, n=all labeled) ===")
print(f"{'verb':9s} {'n':>4s} {'FT':>7s} {'GEAL':>7s} {'gap':>7s}")
for v in TRAINED:
    fts = [ftmap[o][v] for o in objs if v in ftmap[o]]; gs = [gmap[o][v] for o in objs if v in gmap[o]]
    print(f"{v:9s} {len(fts):4d} {np.mean(fts):7.3f} {np.mean(gs):7.3f} {np.mean(fts)-np.mean(gs):+7.3f}")
pt_ft, pt_g = tmean(objs, ftmap), tmean(objs, gmap)
rng = np.random.default_rng(0); B = 3000; d = np.empty(B)
for b in range(B):
    dr = rng.choice(objs, size=len(objs), replace=True); d[b] = tmean(dr, ftmap) - tmean(dr, gmap)
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"\nCV trained-mean: FT={pt_ft:.3f}  GEAL={pt_g:.3f}  gap=+{pt_ft-pt_g:.3f} over {len(objs)} objects")
print(f"bootstrap gap mean={d.mean():+.3f}  95% CI=[{lo:+.3f},{hi:+.3f}]  P(FT>GEAL)={(d>0).mean():.3f}")
print("(compare single-split: gap +0.020, CI [-0.041,+0.077], P=0.75)")
