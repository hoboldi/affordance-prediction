"""Diagnose the 'upward-facing shortcut': does FT@100's prediction correlate with the up-normal MORE
than human GT does? And is the prediction driven by render-view feature coverage (||CLIP|| per vertex)?
Per verb, over val objects with human labels. CPU only."""
import os, sys, json, collections
sys.path.insert(0, "src")
import numpy as np, torch
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

ROOT = "/home/datasets/customDatasets/cmr2"; MAN = f"{ROOT}/manifest.finepatch.jsonl"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]

ck = torch.load("outputs/lc_head_n100/best.pt", map_location="cpu", weights_only=False)
cfg = mlp_head_config_from_model_cfg(ck["model_cfg"])
ft = AffordanceMLP(cfg); ft.load_state_dict(ck["model"]); ft.eval()
ds = DataRootDataset(manifest_path=MAN, load_vertex_labels_eager=False, load_vertex_semantics_eager=True,
                     load_vertex_dino=True, dino_filename=cfg.dino_filename)
obj2row = {}
for i, r in enumerate(ds.rows):
    obj2row.setdefault(os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")), i)
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in TRAINED}
val = set(json.load(open("human_split.json"))["val"])

def predict(it, e):
    f = lambda t: t.float() if t is not None else None
    with torch.no_grad():
        lo = ft(e, slat_vertex=f(it.get("slat_vertex_features")), vlm_features=f(it.get("vertex_features")),
                dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")),
                vertex_normals=f(it.get("vertex_normals")), vertex_positions=None,
                dino_vertex=f(it.get("dino_vertex_features")), vertex_geom=None)
    return torch.sigmoid(lo).numpy()

def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

agg = collections.defaultdict(lambda: collections.defaultdict(list))
for o in sorted(val):
    if o not in obj2row:
        continue
    it = None
    for v in TRAINED:
        hf = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
        if not os.path.exists(hf):
            continue
        hum = np.asarray(torch.load(hf, weights_only=False)).reshape(-1)
        hb = (hum >= 0.5).astype(float)
        if hb.sum() == 0:
            continue
        if it is None:
            it = ds[obj2row[o]]
        nrm = it.get("vertex_normals")
        if nrm is None:
            continue
        nrm = nrm.float().numpy()
        up = nrm[:, 1] / (np.linalg.norm(nrm, axis=1) + 1e-9)   # up-component of unit normal (y = world up)
        pf = predict(it, emb[v])
        clip = it.get("vertex_features").float().numpy()
        fnorm = np.linalg.norm(clip, axis=1)
        if len(up) != len(hb):
            continue
        agg[v]["pred_up"].append(corr(pf, up))           # model vs "faces up"
        agg[v]["hum_up"].append(corr(hb, up))            # human GT vs "faces up"
        agg[v]["pred_fnorm"].append(corr(pf, fnorm))     # model vs feature coverage
        agg[v]["uncov"].append(float((fnorm < 0.1 * np.median(fnorm[fnorm > 0])).mean()))  # frac near-empty feats

print(f"{'verb':9s} {'n':>3s} | {'pred~up':>8s} {'human~up':>9s} {'EXCESS':>7s} | {'pred~featnorm':>13s} {'frac_uncov':>10s}")
print("-" * 70)
nm = lambda x: float(np.nanmean(x)) if x else float("nan")
for v in TRAINED:
    if not agg[v]["pred_up"]:
        continue
    pu, hu = nm(agg[v]["pred_up"]), nm(agg[v]["hum_up"])
    print(f"{v:9s} {len(agg[v]['pred_up']):3d} | {pu:8.3f} {hu:9.3f} {pu-hu:7.3f} | {nm(agg[v]['pred_fnorm']):13.3f} {nm(agg[v]['uncov']):10.3f}")
print("\npred~up = corr(prediction, normal-up).  human~up = corr(human label, normal-up).")
print("EXCESS = pred~up - human~up  (positive & large => model over-relies on 'faces up' beyond what humans warrant).")
print("pred~featnorm = corr(prediction, ||CLIP||): high => prediction follows render-view feature COVERAGE.")
