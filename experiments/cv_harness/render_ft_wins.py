"""Render val objects where the new best model (FT@100) ACTUALLY BEATS the GEAL teacher per-object.
Scans all val objects, ranks by per-object (FT AUPRC - GEAL AUPRC) vs human, picks clean diverse winners.
Columns: Human GT | FT@100 | Base | GEAL teacher. CPU + matplotlib only."""
import os, sys, json, glob, collections
sys.path.insert(0, "src")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

ROOT = "/home/datasets/customDatasets/cmr2"; MAN = f"{ROOT}/manifest.finepatch.jsonl"
SUB = 12000; rng = np.random.default_rng(0)
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]

def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
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
emb = {v: vlm.encode_text([v])[0] for v in TRAINED}
val = set(json.load(open("human_split.json"))["val"])

def predict(net, it, e):
    f = lambda t: t.float() if t is not None else None
    with torch.no_grad():
        lo = net(e, slat_vertex=f(it.get("slat_vertex_features")), vlm_features=f(it.get("vertex_features")),
                 dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")),
                 vertex_normals=f(it.get("vertex_normals")), vertex_positions=None,
                 dino_vertex=f(it.get("dino_vertex_features")), vertex_geom=None)
    return torch.sigmoid(lo).numpy()

# ---- scan all val objects ----
cand = []
for o in sorted(val):
    if o not in obj2row:
        continue
    itc = None
    for v in TRAINED:
        hf = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
        gf = f"{ROOT}/reconstructions/{o}/vertex_pseudolabels_{v}.pt"
        if not (os.path.exists(hf) and os.path.exists(gf)):
            continue
        hum = np.asarray(torch.load(hf, weights_only=False)).reshape(-1)
        hb = (hum >= 0.5).astype(int)
        geal = np.asarray(torch.load(gf, weights_only=False)).reshape(-1)
        if hb.sum() == 0 or len(geal) != len(hb):
            continue
        if itc is None:
            itc = ds[obj2row[o]]
        pf = predict(ft, itc, emb[v])
        apf = average_precision_score(hb, pf); apg = average_precision_score(hb, geal)
        cand.append((apf - apg, apf, apg, float(hb.mean()), v, o))
print(f"scanned {len(cand)} (object,verb) pairs")

# ---- select: FT beats GEAL with margin, clean positive-rate, diverse verbs+categories ----
good = sorted([c for c in cand if c[0] > 0.05 and 0.04 <= c[3] <= 0.6], key=lambda c: -c[0])
# at most 2 examples per verb, unique categories, highest FT−GEAL gap first
chosen, vcount, seen_cat = [], collections.Counter(), set()
for c in good:
    v, cat = c[4], c[5].split("__")[0]
    if vcount[v] >= 2 or cat in seen_cat:
        continue
    chosen.append(c); vcount[v] += 1; seen_cat.add(cat)
    if len(chosen) >= 6:
        break
print("CHOSEN:", [(c[4], c[5].split('__')[0], f"FT{c[1]:.2f}/GEAL{c[2]:.2f}") for c in chosen])
items = [(c[4], c[5]) for c in chosen]

# ---- render ----
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
    geal = np.asarray(torch.load(gf, weights_only=False)).reshape(-1)
    ap = lambda p: average_precision_score(hb, p) if (hb.sum() and len(p) == len(hb)) else None
    apf, apb, apg = ap(pf), ap(pb), ap(geal)
    panels = [("Human GT", hum, None), ("FT@100 (new best)", pf, apf),
              ("Base open-vocab", pb, apb), ("GEAL teacher", geal, apg)]
    for ci, (name, field, apv) in enumerate(panels):
        ax = fig.add_subplot(len(items), NCOL, ri * NCOL + ci + 1, projection="3d")
        vmax = 1.0 if ci == 0 else float(max(0.05, field[sel].max()))
        ax.scatter(xyz[sel, 0], xyz[sel, 2], xyz[sel, 1], c=field[sel], cmap="turbo", s=2, vmin=0.0, vmax=vmax)
        h = (xyz[sel].max(0) - xyz[sel].min(0)).max() / 2 + 1e-6
        mid = (xyz[sel].max(0) + xyz[sel].min(0)) / 2
        ax.set_xlim(mid[0] - h, mid[0] + h); ax.set_ylim(mid[2] - h, mid[2] + h); ax.set_zlim(mid[1] - h, mid[1] + h)
        ax.view_init(elev=14, azim=-60); ax.set_axis_off()
        green = ci == 1 and apf is not None and apg is not None and apf >= apg
        ax.set_title(name + (f"\nAUPRC={apv:.2f}" if apv is not None else ""), fontsize=8,
                     color=("tab:green" if green else "black"))
        if ci == 0:
            ax.text2D(-0.16, 0.5, f"{verb}\n{o.split('__')[0]}", transform=ax.transAxes,
                      fontsize=10, fontweight="bold", rotation=90, va="center")
fig.suptitle("Where the new best model beats the teacher: FT@100 (human-labeled) vs GEAL on val objects FT wins\n"
             "Human GT | FT@100 (new best) | Base open-vocab | GEAL teacher   —   turbo: blue=0, red=1; AUPRC vs human "
             "(green = FT beats GEAL)", fontsize=9.5)
out = "outputs/renders/best_n100_ft_wins.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print("saved", out)
