"""Render where the SLAT encoder gains over the baseline (fold 0, display+contain): pick objects with
the biggest SLAT-vs-baseline AUPRC gain, show Human GT | Baseline | SLAT per-vertex heatmaps. CPU only."""
import os, sys, json, dataclasses, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from models.slat_encoder import SlatEncoderConfig, build_slat_knn, build_vertex_to_slat
from models.mlp_slat import AffordanceMLPSlat
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

ROOT = "/home/datasets/customDatasets/cmr2"
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
SUB = 14000; rng = np.random.default_rng(0)
val = json.load(open(f"{SCR}/cv_fold0.json"))["val"]

# baseline
bck = torch.load("outputs/cv_fold0/last.pt", map_location="cpu", weights_only=False)
bcfg = mlp_head_config_from_model_cfg(bck["model_cfg"])
base = AffordanceMLP(bcfg); base.load_state_dict(bck["model"]); base.eval()
# slat model
sd = torch.load("outputs/cv_slat_cv_fold0.pt", map_location="cpu", weights_only=False)["model"]
slat_cfg = SlatEncoderConfig(in_dim=8, hidden=64, layers=3, out_dim=64, knn_k=8)
mlp_cfg = dataclasses.replace(bcfg, sam3d_dim=64)
slat = AffordanceMLPSlat(mlp_cfg, slat_cfg, verb_text_embeddings=bck["model"].get("verb_text"))
slat.load_state_dict(sd); slat.eval()

ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=bcfg.dino_filename)
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in ["display", "contain"]}
f = lambda t: t.float() if t is not None else None

slat_cache = {}
def sdata(o):
    if o not in slat_cache:
        rd = f"{ROOT}/reconstructions/{o}"
        sf = torch.load(f"{rd}/slat_feats.pt", weights_only=False).float()
        sc = torch.load(f"{rd}/slat_coords.pt", weights_only=False).float()
        pos = torch.as_tensor(np.asarray(torch.load(f"{rd}/vertex_positions.pt", weights_only=False)), dtype=torch.float32)
        slat_cache[o] = (sf, sc, build_slat_knn(sc, 8), build_vertex_to_slat(pos, sc))
    return slat_cache[o]

def preds(o, v):
    it = ds[o2r[o]]
    with torch.no_grad():
        pb = torch.sigmoid(base(emb[v], slat_vertex=f(it.get("slat_vertex_features")), vlm_features=f(it.get("vertex_features")),
                                dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")),
                                vertex_normals=f(it.get("vertex_normals")), vertex_positions=None,
                                dino_vertex=f(it.get("dino_vertex_features")), vertex_geom=None)).numpy()
        sf, sc, sk, v2s = sdata(o)
        ps = torch.sigmoid(slat(emb[v], slat_feats=sf, slat_coords=sc, slat_knn=sk, vertex_to_slat=v2s,
                                vlm_features=f(it.get("vertex_features")), dino_cls=f(it.get("dino_cls")),
                                ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=f(it.get("vertex_normals")),
                                dino_vertex=f(it.get("dino_vertex_features")))).numpy()
    return pb, ps

# scan for biggest gains
cand = []
for v in ["display", "contain"]:
    for o in val:
        hf = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
        if not (o in o2r and os.path.exists(hf)):
            continue
        hb = (np.asarray(torch.load(hf, weights_only=False)).reshape(-1) >= 0.5).astype(int)
        if hb.sum() == 0:
            continue
        pb, ps = preds(o, v)
        if not (len(pb) == len(ps) == len(hb)):
            continue
        ab, as_ = average_precision_score(hb, pb), average_precision_score(hb, ps)
        cand.append((as_ - ab, ab, as_, v, o))
cand.sort(reverse=True)
seen = collections.Counter(); chosen = []
for g in cand:
    v, o = g[3], g[4].split("__")[0]
    if g[0] < 0.03 or seen[(g[3], o)] >= 1:
        continue
    chosen.append(g); seen[(g[3], o)] += 1
    if len(chosen) >= 6:
        break
print("chosen:", [(g[3], g[4].split("__")[0], f"b{g[1]:.2f}->s{g[2]:.2f}") for g in chosen])

fig = plt.figure(figsize=(3 * 2.7, len(chosen) * 2.8))
for ri, (gain, ab, as_, v, o) in enumerate(chosen):
    it = ds[o2r[o]]; xyz = it["vertex_positions"].numpy()
    sel = rng.choice(len(xyz), min(SUB, len(xyz)), replace=False)
    hb = np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1)
    pb, ps = preds(o, v)
    for ci, (name, fld, ap) in enumerate([("Human GT", hb, None), ("Baseline", pb, ab), ("SLAT encoder", ps, as_)]):
        ax = fig.add_subplot(len(chosen), 3, ri * 3 + ci + 1, projection="3d")
        vmax = 1.0 if ci == 0 else float(max(0.05, fld[sel].max()))
        ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=fld[sel], cmap="turbo", s=2, vmin=0, vmax=vmax)
        h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6; mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
        ax.set_xlim(mid[0] - h, mid[0] + h); ax.set_ylim(mid[2] - h, mid[2] + h); ax.set_zlim(mid[1] - h, mid[1] + h)
        ax.view_init(elev=14, azim=-60); ax.set_axis_off()
        ax.set_title(name + (f"\nAUPRC={ap:.2f}" if ap is not None else ""), fontsize=8,
                     color=("tab:green" if ci == 2 else "black"))
        if ci == 0:
            ax.text2D(-0.16, 0.5, f"{v}\n{o.split('__')[0]}", transform=ax.transAxes, fontsize=10, fontweight="bold", rotation=90, va="center")
fig.suptitle("Where the SLAT encoder gains (fold 0): Human GT | Baseline | SLAT encoder  —  turbo blue=0 red=1; AUPRC vs human", fontsize=10)
out = "outputs/renders/slat_gains.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight"); print("saved", out)
