"""Does FT@100 fail specifically on INVISIBLE vertices? Per verb, pooled over val objects:
AUPRC on visible-only vs invisible-only vertices, and the fraction of human-positives that are invisible.
Uses the exact visible_in_any_view mask. CPU only."""
import os, sys, json, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
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

P = collections.defaultdict(lambda: {"vis": ([], []), "inv": ([], []), "pos_inv": [], "frac_inv": []})
for o in sorted(val):
    if o not in obj2row:
        continue
    it = None
    for v in TRAINED:
        hf = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
        if not os.path.exists(hf):
            continue
        hum = np.asarray(torch.load(hf, weights_only=False)).reshape(-1)
        hb = (hum >= 0.5).astype(int)
        if hb.sum() == 0:
            continue
        if it is None:
            it = ds[obj2row[o]]
        vm = it.get("vertex_visible_mask")
        if vm is None:
            continue
        vis = np.asarray(vm).astype(bool)
        if len(vis) != len(hb):
            continue
        pf = predict(it, emb[v])
        P[v]["vis"][0].append(pf[vis]); P[v]["vis"][1].append(hb[vis])
        P[v]["inv"][0].append(pf[~vis]); P[v]["inv"][1].append(hb[~vis])
        P[v]["frac_inv"].append(float((~vis).mean()))
        P[v]["pos_inv"].append(float(hb[~vis].sum() / max(1, hb.sum())))  # frac of positives that are invisible

def auprc(ps, ys):
    p, y = np.concatenate(ps), np.concatenate(ys)
    return average_precision_score(y, p) if y.sum() and (y == 0).any() else float("nan")

print(f"{'verb':9s} | {'AUPRC_visible':>13s} {'AUPRC_invis':>11s} | {'frac_verts_inv':>14s} {'frac_POS_inv':>12s}")
print("-" * 70)
for v in TRAINED:
    if not P[v]["vis"][0]:
        continue
    av, ai = auprc(*P[v]["vis"]), auprc(*P[v]["inv"])
    print(f"{v:9s} | {av:13.3f} {ai:11.3f} | {np.mean(P[v]['frac_inv']):14.3f} {np.mean(P[v]['pos_inv']):12.3f}")
print("\nfrac_POS_inv = share of HUMAN-POSITIVE vertices that are invisible (no real features).")
print("If high AND AUPRC_invis << AUPRC_visible -> model is blind exactly where the affordance is -> coverage caps it.")
