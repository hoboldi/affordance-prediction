"""Render the full-FT SLAT model (0.823 3-fold, source of the headline) on its best object per verb.
Each object is scored OUT-OF-FOLD by the fold model (cv_ft3_fold{k}.pt) that held it out — honest, no leakage.
Two columns: Human GT | full-FT prediction, six rows (one per trained verb). CPU only."""
import os, sys, json, dataclasses
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import mlp_head_config_from_model_cfg
from models.slat_encoder import SlatEncoderConfig, build_slat_knn, build_vertex_to_slat
from models.mlp_slat import AffordanceMLPSlat
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

ROOT = "/home/datasets/customDatasets/cmr2"
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
SUB = 16000; rng = np.random.default_rng(0)
f = lambda t: t.float() if t is not None else None

# MEDIAN (typical, non-cherry-picked) object per verb from /tmp/slat_cv_ft3.json; (verb, object, holdout-fold, verb-mean)
PICKS = [
    ("contain", "bowl__69_5475_12824__v000",       1, 0.954),
    ("sit",     "chair__555_79602_154353__v000",   2, 0.880),
    ("display", "tv__396_49386_97450__v000",       1, 0.867),
    ("pour",    "bottle__34_1435_4403__v000",      2, 0.812),
    ("move",    "chair__108_12872_24156__v000",    1, 0.832),
    ("grasp",   "handbag__20_743_1438__v000",      0, 0.594),
]

# shared config from a base ckpt
b0 = torch.load("outputs/cv_fold0/last.pt", map_location="cpu", weights_only=False)
bcfg = mlp_head_config_from_model_cfg(b0["model_cfg"])
slat_cfg = SlatEncoderConfig(in_dim=8, hidden=64, layers=3, out_dim=64, knn_k=8)
mlp_cfg = dataclasses.replace(bcfg, sam3d_dim=64)
vte = b0["model"].get("verb_text")

ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=bcfg.dino_filename)
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v, *_ in PICKS}

model_cache = {}
def fold_model(k):
    if k not in model_cache:
        m = AffordanceMLPSlat(mlp_cfg, slat_cfg, verb_text_embeddings=vte)
        m.load_state_dict(torch.load(f"outputs/cv_ft3_fold{k}.pt", map_location="cpu", weights_only=False)["model"])
        m.eval(); model_cache[k] = m
    return model_cache[k]

slat_cache = {}
def sdata(o):
    if o not in slat_cache:
        rd = f"{ROOT}/reconstructions/{o}"
        sf = torch.load(f"{rd}/slat_feats.pt", weights_only=False).float()
        sc = torch.load(f"{rd}/slat_coords.pt", weights_only=False).float()
        pos = torch.as_tensor(np.asarray(torch.load(f"{rd}/vertex_positions.pt", weights_only=False)), dtype=torch.float32)
        slat_cache[o] = (sf, sc, build_slat_knn(sc, 8), build_vertex_to_slat(pos, sc))
    return slat_cache[o]

def predict(o, v, k):
    it = ds[o2r[o]]; m = fold_model(k)
    sf, sc, sk, v2s = sdata(o)
    with torch.no_grad():
        p = torch.sigmoid(m(emb[v], slat_feats=sf, slat_coords=sc, slat_knn=sk, vertex_to_slat=v2s,
                            vlm_features=f(it.get("vertex_features")), dino_cls=f(it.get("dino_cls")),
                            ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=f(it.get("vertex_normals")),
                            dino_vertex=f(it.get("dino_vertex_features")))).numpy()
    return p

fig = plt.figure(figsize=(2 * 3.0, len(PICKS) * 2.9))
for ri, (v, o, k, vmean) in enumerate(PICKS):
    it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy()
    hb = np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)
    pr = predict(o, v, k)
    yb = (hb >= 0.5).astype(int)
    ap = average_precision_score(yb, pr) if 0 < yb.sum() < len(yb) else float("nan")
    sel = rng.choice(len(xyz), min(SUB, len(xyz)), replace=False)
    for ci, (name, fld, show_ap) in enumerate([("Human GT", hb, None), ("full-FT prediction", pr, ap)]):
        ax = fig.add_subplot(len(PICKS), 2, ri * 2 + ci + 1, projection="3d")
        vmax = 1.0 if ci == 0 else float(max(0.05, fld[sel].max()))
        ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=fld[sel], cmap="turbo", s=2, vmin=0, vmax=vmax)
        h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
        ax.set_xlim(mid[0] - h, mid[0] + h); ax.set_ylim(mid[2] - h, mid[2] + h); ax.set_zlim(mid[1] - h, mid[1] + h)
        ax.view_init(elev=16, azim=-60); ax.set_axis_off()
        ax.set_title(name + (f"\nAUPRC = {show_ap:.2f}" if show_ap is not None else ""), fontsize=9,
                     color=("tab:green" if ci == 1 else "black"), fontweight=("bold" if ci == 1 else "normal"))
        if ci == 0:
            ax.text2D(-0.12, 0.5, f"{v}\n{o.split('__')[0]}\n(verb mean {vmean:.2f})", transform=ax.transAxes,
                      fontsize=10, fontweight="bold", rotation=90, va="center")
    print(f"{v:9s} {o.split('__')[0]:8s} fold{k}  AUPRC={ap:.3f}  (verb mean {vmean:.3f})", flush=True)

fig.suptitle("Full-FT SLAT model (human-GT fine-tuned) — MEDIAN (typical) object per verb, scored out-of-fold\n"
             "Human GT  |  model prediction     (turbo: blue = 0, red = 1)     3-fold trained-mean AUPRC = 0.823",
             fontsize=10, y=0.995)
out = "outputs/renders/fullft_median.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=125, bbox_inches="tight"); print("saved", out)
