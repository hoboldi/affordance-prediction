"""Anchor eval + ensemble for the 5-fold SLAT CV. Each object scored out-of-fold by BOTH the baseline
(cv_fold{k}) and the SLAT model (cv_slat_cv_fold{k}). Reports per-verb: baseline, SLAT, ensemble(avg logits),
ensemble(per-verb pick). Bootstrap CI on the SLAT-vs-baseline trained-mean gap (fair, same objects). CPU."""
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
SUB = 25000; K = int(os.environ.get("KFOLDS", "5")); rng = np.random.default_rng(0)

b0 = torch.load("outputs/cv_fold0/last.pt", map_location="cpu", weights_only=False)
bcfg = mlp_head_config_from_model_cfg(b0["model_cfg"])
slat_cfg = SlatEncoderConfig(in_dim=8, hidden=64, layers=3, out_dim=64, knn_k=8)
mlp_cfg = dataclasses.replace(bcfg, sam3d_dim=64)
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=bcfg.dino_filename)
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in TRAINED}
f = lambda t: t.float() if t is not None else None
scache = {}
def sdata(o):
    if o not in scache:
        rd = f"{ROOT}/reconstructions/{o}"
        sf = torch.load(f"{rd}/slat_feats.pt", weights_only=False).float(); sc = torch.load(f"{rd}/slat_coords.pt", weights_only=False).float()
        pos = torch.as_tensor(np.asarray(torch.load(f"{rd}/vertex_positions.pt", weights_only=False)), dtype=torch.float32)
        scache[o] = (sf, sc, build_slat_knn(sc, 8), build_vertex_to_slat(pos, sc))
    return scache[o]

rows = []   # (verb, obj, ap_base, ap_slat, ap_ens)
for k in range(K):
    val = json.load(open(f"{SCR}/cv_fold{k}.json"))["val"]
    bm = AffordanceMLP(mlp_head_config_from_model_cfg(torch.load(f"outputs/cv_fold{k}/last.pt", map_location="cpu", weights_only=False)["model_cfg"]))
    bm.load_state_dict(torch.load(f"outputs/cv_fold{k}/last.pt", map_location="cpu", weights_only=False)["model"]); bm.eval()
    sm = AffordanceMLPSlat(mlp_cfg, slat_cfg, verb_text_embeddings=b0["model"].get("verb_text"))
    sm.load_state_dict(torch.load(f"outputs/cv_slat_cv_fold{k}.pt", map_location="cpu", weights_only=False)["model"]); sm.eval()
    for o in val:
        if o not in o2r: continue
        it = None
        for v in TRAINED:
            hf = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
            if not os.path.exists(hf): continue
            hb = (np.asarray(torch.load(hf, weights_only=False)).reshape(-1) >= 0.5).astype(int)
            if hb.sum() == 0: continue
            if it is None: it = ds[o2r[o]]
            V = len(hb); sel = rng.choice(V, min(SUB, V), replace=False); idx = torch.as_tensor(sel, dtype=torch.long)
            sl = lambda key: (it.get(key).float()[idx] if it.get(key) is not None else None)
            sf, sc, sk, v2s = sdata(o)
            with torch.no_grad():
                lb = bm(emb[v], slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"), dino_cls=f(it.get("dino_cls")),
                        ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"), vertex_positions=None, dino_vertex=sl("dino_vertex_features"))
                ls = sm(emb[v], slat_feats=sf, slat_coords=sc, slat_knn=sk, vertex_to_slat=v2s[idx], vlm_features=sl("vertex_features"),
                        dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"), dino_vertex=sl("dino_vertex_features"))
            yb = hb[sel]
            if yb.sum() == 0 or (yb == 1).all(): continue
            ens = ((torch.sigmoid(lb) + torch.sigmoid(ls)) / 2).numpy()
            rows.append((v, o, float(average_precision_score(yb, torch.sigmoid(lb).numpy())),
                         float(average_precision_score(yb, torch.sigmoid(ls).numpy())), float(average_precision_score(yb, ens))))
    print(f"fold {k} scored", flush=True)

json.dump(rows, open("/tmp/slat_anchor.json", "w"))
byv = collections.defaultdict(lambda: {"b": [], "s": [], "e": []})
for v, o, ab, as_, ae in rows:
    byv[v]["b"].append(ab); byv[v]["s"].append(as_); byv[v]["e"].append(ae)
def m(x): return float(np.mean(x)) if x else float("nan")
print(f"\n{'verb':9s} {'n':>4s} {'BASE':>6s} {'SLAT':>6s} {'ENS-avg':>7s}  (d=SLAT-BASE)")
pick = {}
for v in TRAINED:
    d = byv[v]; b, s = m(d["b"]), m(d["s"])
    pick[v] = "SLAT" if s >= b else "BASE"
    print(f"{v:9s} {len(d['b']):4d} {b:6.3f} {s:6.3f} {m(d['e']):7.3f}  d={s-b:+.3f}  pick={pick[v]}")
tb = np.mean([m(byv[v]['b']) for v in TRAINED]); tsv = np.mean([m(byv[v]['s']) for v in TRAINED])
te = np.mean([m(byv[v]['e']) for v in TRAINED]); tp = np.mean([m(byv[v]['s'] if pick[v] == 'SLAT' else byv[v]['b']) for v in TRAINED])
print(f"\nTRAINED-MEAN  BASE={tb:.3f}  SLAT={tsv:.3f}  ENS-avg={te:.3f}  ENS-pick={tp:.3f}")
# bootstrap SLAT-vs-BASE gap over objects
objs = sorted({o for _, o, *_ in rows}); bo = collections.defaultdict(dict)
for v, o, ab, as_, ae in rows: bo[o][v] = (ab, as_)
def tmean(draw, i): return np.mean([np.mean([bo[o][v][i] for o in draw if v in bo[o]]) for v in TRAINED if any(v in bo[o] for o in draw)])
# paired bootstrap: resample objects once, compute both SLAT and BASE trained-means on the same draw
d = np.empty(2000)
for i in range(2000):
    dr = rng.choice(objs, len(objs), replace=True); d[i] = tmean(dr, 1) - tmean(dr, 0)
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"bootstrap SLAT-BASE gap: mean={d.mean():+.3f}  95% CI=[{lo:+.3f},{hi:+.3f}]  P(SLAT>BASE)={(d>0).mean():.3f}")
print("=== SLAT ANCHOR DONE ===", flush=True)
