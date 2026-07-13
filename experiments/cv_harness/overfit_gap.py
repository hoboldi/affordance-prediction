"""Train-vs-val gap for the full-FT SLAT model (0.823). For each fold k in {0,1,2}, load cv_ft3_fold{k}.pt
and score BOTH its own TRAIN objects (seen) and its VAL objects (held out) with identical settings, per-verb
AUPRC vs human. gap = train_mean - val_mean. Large gap => fitting the training objects; small => clean.
CPU only, no GPU."""
import os, sys, json, dataclasses, collections, time
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import mlp_head_config_from_model_cfg
from models.slat_encoder import SlatEncoderConfig, build_slat_knn, build_vertex_to_slat
from models.mlp_slat import AffordanceMLPSlat
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

ROOT = "/home/datasets/customDatasets/cmr2"
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
SUB = 15000; rng = np.random.default_rng(0); f = lambda t: t.float() if t is not None else None
torch.set_num_threads(8)

b0 = torch.load("outputs/cv_fold0/last.pt", map_location="cpu", weights_only=False)
bcfg = mlp_head_config_from_model_cfg(b0["model_cfg"])
slat_cfg = SlatEncoderConfig(in_dim=8, hidden=64, layers=3, out_dim=64, knn_k=8)
mlp_cfg = dataclasses.replace(bcfg, sam3d_dim=64); vte = b0["model"].get("verb_text")
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=bcfg.dino_filename)
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in TRAINED}
scache = {}
def sdata(o):
    if o not in scache:
        rd = f"{ROOT}/reconstructions/{o}"
        sf = torch.load(f"{rd}/slat_feats.pt", weights_only=False).float()
        sc = torch.load(f"{rd}/slat_coords.pt", weights_only=False).float()
        pos = torch.as_tensor(np.asarray(torch.load(f"{rd}/vertex_positions.pt", weights_only=False)), dtype=torch.float32)
        scache[o] = (sf, sc, build_slat_knn(sc, 8), build_vertex_to_slat(pos, sc))
    return scache[o]

def score(model, o, v):
    hf = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
    if not (o in o2r and os.path.exists(hf)): return None
    hb = np.asarray(torch.load(hf, weights_only=False)).reshape(-1)
    yb = (hb >= 0.5).astype(int)
    if not (0 < yb.sum() < len(yb)): return None
    it = ds[o2r[o]]; V = len(hb); sel = rng.choice(V, min(SUB, V), replace=False); idx = torch.as_tensor(sel, dtype=torch.long)
    sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
    sf, sc, sk, v2s = sdata(o)
    with torch.no_grad():
        p = torch.sigmoid(model(emb[v], slat_feats=sf, slat_coords=sc, slat_knn=sk, vertex_to_slat=v2s[idx],
                                vlm_features=sl("vertex_features"), dino_cls=f(it.get("dino_cls")),
                                ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"),
                                dino_vertex=sl("dino_vertex_features"))).numpy()
    return float(average_precision_score(yb[sel], p))

res = {"train": collections.defaultdict(list), "val": collections.defaultdict(list)}
t0 = time.time()
for k in [0, 1, 2]:
    sp = json.load(open(f"{SCR}/cv_fold{k}.json"))
    m = AffordanceMLPSlat(mlp_cfg, slat_cfg, verb_text_embeddings=vte)
    m.load_state_dict(torch.load(f"outputs/cv_ft3_fold{k}.pt", map_location="cpu", weights_only=False)["model"]); m.eval()
    for part in ["train", "val"]:
        objs = [o for o in sp[part] if o in o2r]
        for o in objs:
            for v in TRAINED:
                a = score(m, o, v)
                if a is not None: res[part][v].append(a)
        print(f"fold{k} {part}: {len(objs)} objs scored  ({time.time()-t0:.0f}s)", flush=True)

def vm(d): return np.mean([np.mean(d[v]) for v in TRAINED if d[v]])
print("\n=== full-FT train-vs-val gap (human GT, per-verb AUPRC) ===")
print(f"{'verb':9s} {'TRAIN':>7s} {'VAL':>7s} {'gap':>7s}")
for v in TRAINED:
    tr, va = res["train"][v], res["val"][v]
    if tr and va: print(f"{v:9s} {np.mean(tr):7.3f} {np.mean(va):7.3f} {np.mean(tr)-np.mean(va):+7.3f}  (n_tr={len(tr)},n_va={len(va)})")
tmean, vmean = vm(res["train"]), vm(res["val"])
print(f"\nTRAINED-MEAN  train={tmean:.3f}  val={vmean:.3f}  GAP={tmean-vmean:+.3f}")
print("(val here should ~match the 0.823 headline; GAP is the overfitting margin)")
json.dump({"train": {v: res["train"][v] for v in TRAINED}, "val": {v: res["val"][v] for v in TRAINED}},
          open("/tmp/overfit_gap.json", "w"))
print("=== DONE ===", flush=True)
