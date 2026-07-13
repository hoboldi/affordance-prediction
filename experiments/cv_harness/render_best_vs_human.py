"""Render the NEW BEST model (head-only FT on 100 human-labeled objects) against HUMAN GT and the
base open-vocab model. Rows = (verb, object) examples from the val split; columns = Human GT | FT@100 | Base.
CPU + matplotlib only. Per-panel AUPRC (vs human) in titles."""
import os, sys, json, glob
sys.path.insert(0, "src")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

ROOT = "/home/datasets/customDatasets/cmr2"
MAN = f"{ROOT}/manifest.finepatch.jsonl"
SUB = 12000
rng = np.random.default_rng(0)

# (verb, category) — pick the best-painted val object of that category for that verb
CHOSEN = [("contain", "bowl"), ("sit", "chair"), ("pour", "bottle"),
          ("display", "tv"), ("grasp", "handbag"), ("move", "chair")]
val = set(json.load(open("human_split.json"))["val"])

def best_obj(verb, cat):
    best, bp = None, -1.0
    for hf in glob.glob(f"human_gt_labels/{cat}__*/vertex_manuallabels_{verb}.pt"):
        o = hf.split("/")[1]
        if o not in val or not os.path.exists(f"{ROOT}/reconstructions/{o}/vertex_positions.pt"):
            continue
        lab = np.asarray(torch.load(hf, weights_only=False)).reshape(-1)
        pr = float((lab >= 0.5).mean())
        if pr > bp:
            bp, best = pr, o
    return best

items = [(v, best_obj(v, c)) for v, c in CHOSEN]
items = [(v, o) for v, o in items if o]
print("examples:", [(v, o) for v, o in items])

def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = mlp_head_config_from_model_cfg(ck["model_cfg"])
    net = AffordanceMLP(cfg); net.load_state_dict(ck["model"]); net.eval()
    return net, cfg

ft, cfg = load("outputs/lc_head_n100/best.pt")
base, _ = load("outputs/ov_concat_finepatch/best.pt")

ds = DataRootDataset(manifest_path=MAN, load_vertex_labels_eager=False, load_vertex_semantics_eager=True,
                     load_vertex_dino=True, dino_filename=cfg.dino_filename)
obj2row = {}
for i, r in enumerate(ds.rows):
    obj2row.setdefault(os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")), i)

vlm = VLMWrapper(VLMConfig(device="cpu"))
emb = {v: vlm.encode_text([v])[0] for v, _ in items}

def predict(net, it, e):
    f = lambda t: t.float() if t is not None else None
    with torch.no_grad():
        lo = net(e, slat_vertex=f(it.get("slat_vertex_features")), vlm_features=f(it.get("vertex_features")),
                 dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")),
                 vertex_normals=f(it.get("vertex_normals")), vertex_positions=None,
                 dino_vertex=f(it.get("dino_vertex_features")), vertex_geom=None)
    return torch.sigmoid(lo).numpy()

NCOL = 4
fig = plt.figure(figsize=(NCOL * 2.7, len(items) * 2.8))
for ri, (verb, o) in enumerate(items):
    it = ds[obj2row[o]]
    xyz = it["vertex_positions"].numpy()
    sel = rng.choice(len(xyz), min(SUB, len(xyz)), replace=False)
    hum = np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{verb}.pt", weights_only=False)).reshape(-1)
    hb = (hum >= 0.5).astype(int)
    pf, pb = predict(ft, it, emb[verb]), predict(base, it, emb[verb])
    gf = f"{ROOT}/reconstructions/{o}/vertex_pseudolabels_{verb}.pt"
    geal = np.asarray(torch.load(gf, weights_only=False)).reshape(-1) if os.path.exists(gf) else None
    ap = lambda p: (average_precision_score(hb, p) if (hb.sum() and p is not None and len(p) == len(hb)) else None)
    apf, apb, apg = ap(pf), ap(pb), ap(geal)
    panels = [("Human GT", hum, None), ("FT@100 (new best)", pf, apf),
              ("Base open-vocab", pb, apb), ("GEAL teacher", geal, apg)]
    for ci, (name, field, apv) in enumerate(panels):
        ax = fig.add_subplot(len(items), NCOL, ri * NCOL + ci + 1, projection="3d")
        if field is None:
            ax.text2D(0.5, 0.5, "GEAL\nabsent", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="gray")
            ax.set_axis_off(); ax.set_title(name, fontsize=8); continue
        vmax = 1.0 if ci == 0 else float(max(0.05, field[sel].max()))
        ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=field[sel], cmap="turbo", s=2, vmin=0.0, vmax=vmax)
        h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6
        mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
        ax.set_xlim(mid[0] - h, mid[0] + h); ax.set_ylim(mid[2] - h, mid[2] + h); ax.set_zlim(mid[1] - h, mid[1] + h)
        ax.view_init(elev=14, azim=-60); ax.set_axis_off()
        # FT title green when it beats the GEAL teacher (the headline claim)
        green = ci == 1 and apf is not None and apg is not None and apf >= apg
        ax.set_title(name + (f"\nAUPRC={apv:.2f}" if apv is not None else ""), fontsize=8,
                     color=("tab:green" if green else "black"))
        if ci == 0:
            ax.text2D(-0.16, 0.5, f"{verb}\n{o.split('__')[0]}", transform=ax.transAxes,
                      fontsize=10, fontweight="bold", rotation=90, va="center")
fig.suptitle("New best model = head-only FT on 100 human-labeled objects (beats GEAL teacher: 0.613 vs 0.593 mean)\n"
             "Human GT  |  FT@100 (new best)  |  Base open-vocab  |  GEAL teacher   —   turbo: blue=0, red=1; AUPRC vs human "
             "(FT title green = beats teacher)", fontsize=9.5)
out = "outputs/renders/best_n100_vs_human.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print("saved", out)
