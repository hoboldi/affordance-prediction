"""Leave-one-verb-out eval: for each verb V, the model trained with V HELD OUT (lovo_V) predicts V
zero-shot via its CLIP-text embedding. Compare to BASE (no human-FT) and FULL-CV (V in training).
LOVO≈FULL => human-FT transfers across verbs; LOVO≈BASE => human-FT is verb-specific. CPU, 25k subsample."""
import os, sys, json, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
FULL_CV = {"contain": 0.748, "sit": 0.818, "pour": 0.762, "move": 0.707, "display": 0.543, "grasp": 0.321}
SUB = 25000; rng = np.random.default_rng(0)
objs = json.load(open(f"{SCR}/split_all.json"))["train"]

cfg0 = mlp_head_config_from_model_cfg(torch.load("outputs/ov_concat_finepatch/best.pt", map_location="cpu", weights_only=False)["model_cfg"])
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=cfg0.dino_filename)
obj2row = {}
for i, r in enumerate(ds.rows):
    obj2row.setdefault(os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")), i)
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in TRAINED}

def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = AffordanceMLP(mlp_head_config_from_model_cfg(ck["model_cfg"])); m.load_state_dict(ck["model"]); m.eval()
    return m

def score(model, verb):  # mean per-object AUPRC of `verb` over all objects that have it
    aps = []
    for o in objs:
        if o not in obj2row: continue
        hf = f"human_gt_labels/{o}/vertex_manuallabels_{verb}.pt"
        if not os.path.exists(hf): continue
        hb = (np.asarray(torch.load(hf, weights_only=False)).reshape(-1) >= 0.5).astype(int)
        if hb.sum() == 0: continue
        it = ds[obj2row[o]]; clip = it.get("vertex_features")
        if clip is None or len(clip) != len(hb): continue
        V = len(hb); sel = rng.choice(V, min(SUB, V), replace=False)
        if hb[sel].sum() == 0: continue
        f = lambda t: (t.float()[sel] if t is not None else None)
        with torch.no_grad():
            lo = model(emb[verb], slat_vertex=f(it.get("slat_vertex_features")), vlm_features=f(clip),
                       dino_cls=(it.get("dino_cls").float() if it.get("dino_cls") is not None else None),
                       ss_dino_cls=(it.get("ss_dino_cls").float() if it.get("ss_dino_cls") is not None else None),
                       vertex_normals=f(it.get("vertex_normals")), vertex_positions=None,
                       dino_vertex=f(it.get("dino_vertex_features")), vertex_geom=None)
        aps.append(average_precision_score(hb[sel], torch.sigmoid(lo).numpy()))
    return float(np.mean(aps)) if aps else float("nan"), len(aps)

base = load("outputs/ov_concat_finepatch/best.pt")
print(f"{'verb':9s} {'n':>4s} {'BASE':>6s} {'LOVO':>6s} {'FULL-CV':>7s} | transfer")
base_row, lovo_row = [], []
for v in TRAINED:
    b, n = score(base, v)
    lp = f"outputs/lovo_{v}/last.pt"
    if os.path.exists(lp):
        l, _ = score(load(lp), v)
    else:
        l = float("nan")
    base_row.append(b); lovo_row.append(l)
    # interpret: closer to FULL => transfers; closer to BASE => verb-specific
    if not np.isnan(l):
        frac = (l - b) / (FULL_CV[v] - b) if abs(FULL_CV[v] - b) > 1e-6 else float("nan")
        tag = f"{frac*100:3.0f}% of full-FT gain recovered zero-shot"
    else:
        tag = "(lovo ckpt missing)"
    print(f"{v:9s} {n:4d} {b:6.3f} {l:6.3f} {FULL_CV[v]:7.3f} | {tag}")
bm = np.nanmean(base_row); lm = np.nanmean(lovo_row); fm = np.mean(list(FULL_CV.values()))
print(f"\nMEAN     BASE={bm:.3f}  LOVO(held-out)={lm:.3f}  FULL-CV={fm:.3f}")
print("LOVO near FULL => human-FT generalizes across verbs (open-vocab transfer works).")
print("LOVO near BASE => human-FT only helps verbs it trained on (poor transfer).")
