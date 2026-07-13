"""(1) Bootstrap CI on the FT@100 - GEAL trained-mean gap over val objects (is +0.020 real?).
(2) Visibility pre-check: of currently-INVISIBLE vertices (and grasp/contain positives among them),
    what fraction face DOWN (recoverable by a lower camera band) vs UP/HORIZ (occluded/interior).
CPU only. FT preds on CPU."""
import os, sys, json, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

ROOT = "/home/datasets/customDatasets/cmr2"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
val = sorted(set(json.load(open("human_split.json"))["val"]))

# val-only manifest (original features) for a fast eager load
rows = [json.loads(l) for l in open(f"{ROOT}/manifest.finepatch.jsonl")]
vrows = [r for r in rows if r["sam3d_reconstruction_dir"].split("/")[-1] in set(val)]
VM = "/tmp/manifest.valonly.jsonl"; open(VM, "w").write("".join(json.dumps(r) + "\n" for r in vrows))

ck = torch.load("outputs/lc_head_n100/best.pt", map_location="cpu", weights_only=False)
cfg = mlp_head_config_from_model_cfg(ck["model_cfg"])
ft = AffordanceMLP(cfg); ft.load_state_dict(ck["model"]); ft.eval()
ds = DataRootDataset(manifest_path=VM, load_vertex_labels_eager=False, load_vertex_semantics_eager=True,
                     load_vertex_dino=True, dino_filename=cfg.dino_filename)
obj2row = {}
for i, r in enumerate(ds.rows):
    obj2row.setdefault(os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")), i)
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in TRAINED}

def predict(it, e):
    f = lambda t: t.float() if t is not None else None
    with torch.no_grad():
        lo = ft(e, slat_vertex=f(it.get("slat_vertex_features")), vlm_features=f(it.get("vertex_features")),
                dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")),
                vertex_normals=f(it.get("vertex_normals")), vertex_positions=None,
                dino_vertex=f(it.get("dino_vertex_features")), vertex_geom=None)
    return torch.sigmoid(lo).numpy()

ftmap, gmap = collections.defaultdict(dict), collections.defaultdict(dict)   # obj -> {verb: AUPRC}
visbin = collections.defaultdict(lambda: np.zeros(3))   # (verb,'inv'/'pos') -> [down,horiz,up] counts
for o in val:
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
        pf = predict(it, emb[v])
        if not (len(pf) == len(geal) == len(hb)):
            continue
        ftmap[o][v] = average_precision_score(hb, pf); gmap[o][v] = average_precision_score(hb, geal)
        # visibility / normal orientation
        vm = np.asarray(it["vertex_visible_mask"]).astype(bool); nrm = it["vertex_normals"].float().numpy()
        up = nrm[:, 1] / (np.linalg.norm(nrm, axis=1) + 1e-9)
        for tag, m in [("inv", ~vm), ("pos", (~vm) & (hb > 0))]:
            u = up[m]
            visbin[(v, tag)] += np.array([(u < -0.3).sum(), ((u >= -0.3) & (u < 0.3)).sum(), (u >= 0.3).sum()])

# ---- (1) bootstrap CI ----
objs = np.array(sorted(ftmap.keys()))
def trained_mean(draw, M):
    pv = [np.mean([M[o][v] for o in draw if v in M[o]]) for v in TRAINED]
    pv = [x for x in pv if not np.isnan(x)]
    return float(np.mean(pv))
pt_ft, pt_g = trained_mean(objs, ftmap), trained_mean(objs, gmap)
rng = np.random.default_rng(0); B = 3000; dffs = np.empty(B)
for b in range(B):
    d = rng.choice(objs, size=len(objs), replace=True)
    dffs[b] = trained_mean(d, ftmap) - trained_mean(d, gmap)
lo, hi = np.percentile(dffs, [2.5, 97.5])
print("=== (1) ROBUSTNESS: bootstrap over %d val objects (B=%d) ===" % (len(objs), B))
print(f"  point: FT trained-mean={pt_ft:.3f}  GEAL={pt_g:.3f}  gap=+{pt_ft-pt_g:.3f}")
print(f"  bootstrap gap mean={dffs.mean():+.3f}  95%% CI=[{lo:+.3f}, {hi:+.3f}]  P(FT>GEAL)={(dffs>0).mean():.3f}")

# ---- (2) visibility pre-check ----
print("\n=== (2) VISIBILITY: orientation of currently-INVISIBLE vertices (down=lower-band recoverable; up/horiz=occluded/interior) ===")
print(f"{'verb':9s} | {'invisible verts %down/%horiz/%up':>34s} | {'invisible POSITIVES %down/%horiz/%up':>36s}")
for v in TRAINED:
    iv, pv = visbin[(v, "inv")], visbin[(v, "pos")]
    f = lambda a: (a / a.sum() * 100) if a.sum() else np.zeros(3)
    fi, fp = f(iv), f(pv)
    print(f"{v:9s} | {fi[0]:9.0f} {fi[1]:6.0f} {fi[2]:6.0f}            | {fp[0]:9.0f} {fp[1]:6.0f} {fp[2]:6.0f}")
print("  (down = normal faces down, ~recoverable by adding lower-hemisphere views; up/horiz invisible = occlusion/interior, NOT recoverable by more external views)")
