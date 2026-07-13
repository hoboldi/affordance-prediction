"""Alt-work item 3 (aggregation): how best to combine the two full-FT models, out-of-fold over folds 0/1/2?
Models per fold: baseline-full-FT (basefull_fold{k}) + SLAT-full-FT (cv_ft3_fold{k}). Sweep aggregation rules:
avg-prob, max-prob, min-prob, per-verb-pick (choose the better single model per verb by trained-mean). CPU."""
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
b0 = torch.load("outputs/cv_fold0/last.pt", map_location="cpu", weights_only=False)
slat_cfg = SlatEncoderConfig(in_dim=8, hidden=64, layers=3, out_dim=64, knn_k=8)
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True,
                     dino_filename=mlp_head_config_from_model_cfg(b0["model_cfg"]).dino_filename)
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

rows = []  # (verb, obj, prob_base, prob_slat, y)  -- store per-(verb,obj) AUPRC under each rule
byv = collections.defaultdict(lambda: collections.defaultdict(list))
for k in [0, 1, 2]:
    bm = AffordanceMLP(mlp_head_config_from_model_cfg(torch.load(f"outputs/basefull_fold{k}/last.pt", map_location="cpu", weights_only=False)["model_cfg"]))
    bm.load_state_dict(torch.load(f"outputs/basefull_fold{k}/last.pt", map_location="cpu", weights_only=False)["model"]); bm.eval()
    scfg = dataclasses.replace(mlp_head_config_from_model_cfg(torch.load(f"outputs/basefull_fold{k}/last.pt", map_location="cpu", weights_only=False)["model_cfg"]), sam3d_dim=64)
    sm = AffordanceMLPSlat(scfg, slat_cfg, verb_text_embeddings=b0["model"].get("verb_text"))
    sm.load_state_dict(torch.load(f"outputs/cv_ft3_fold{k}.pt", map_location="cpu", weights_only=False)["model"]); sm.eval()
    for o in json.load(open(f"{SCR}/cv_fold{k}.json"))["val"]:
        if o not in o2r: continue
        it = None
        for v in TRAINED:
            hf = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
            if not os.path.exists(hf): continue
            hb = (np.asarray(torch.load(hf, weights_only=False)).reshape(-1) >= 0.5).astype(int)
            if not (0 < hb.sum() < len(hb)): continue
            if it is None: it = ds[o2r[o]]
            V = len(hb); sel = rng.choice(V, min(SUB, V), replace=False); idx = torch.as_tensor(sel, dtype=torch.long)
            sl = lambda key: (it.get(key).float()[idx] if it.get(key) is not None else None)
            sf, sc, sk, v2s = sdata(o)
            with torch.no_grad():
                pb = torch.sigmoid(bm(emb[v], slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"), dino_cls=f(it.get("dino_cls")),
                                      ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"), vertex_positions=None, dino_vertex=sl("dino_vertex_features"))).numpy()
                ps = torch.sigmoid(sm(emb[v], slat_feats=sf, slat_coords=sc, slat_knn=sk, vertex_to_slat=v2s[idx], vlm_features=sl("vertex_features"),
                                      dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"), dino_vertex=sl("dino_vertex_features"))).numpy()
            y = hb[sel]
            byv[v]["base"].append(average_precision_score(y, pb))
            byv[v]["slat"].append(average_precision_score(y, ps))
            byv[v]["avg"].append(average_precision_score(y, (pb + ps) / 2))
            byv[v]["max"].append(average_precision_score(y, np.maximum(pb, ps)))
            byv[v]["min"].append(average_precision_score(y, np.minimum(pb, ps)))
    print(f"fold {k} scored", flush=True)

def vm(rule): return np.mean([np.mean(byv[v][rule]) for v in TRAINED if byv[v][rule]])
print(f"\n=== full-FT ensemble aggregation (out-of-fold folds 0-2, trained-mean AUPRC) ===")
for rule in ["base", "slat", "avg", "max", "min"]:
    print(f"  {rule:6s} {vm(rule):.3f}")
# per-verb pick: best single model per verb
pick = np.mean([max(np.mean(byv[v]["base"]), np.mean(byv[v]["slat"])) for v in TRAINED if byv[v]["base"]])
print(f"  pick   {pick:.3f}  (per-verb best of base/slat)")
print("\nper-verb (base | slat | avg):")
for v in TRAINED:
    print(f"  {v:9s} {np.mean(byv[v]['base']):.3f} | {np.mean(byv[v]['slat']):.3f} | {np.mean(byv[v]['avg']):.3f}")
