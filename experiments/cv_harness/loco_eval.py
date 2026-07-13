"""LOCO transfer eval for one held-out category. Score the held-out category's objects, per verb, with:
  LOCO   = outputs/loco_<CAT>/last.pt        (full-FT model that NEVER saw the category)
  IN-DIST= outputs/basefull_fold{k}/last.pt  (full-FT that saw the category; each obj scored by the fold that held IT out)
gap = IN-DIST - LOCO = cost of never seeing the category. CAT passed as argv[1]. CPU."""
import os, sys, json, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

CAT = sys.argv[1]
REGIME = sys.argv[2] if len(sys.argv) > 2 else "full"   # "full" -> loco_*/basefull_fold*(0-2); "head" -> loco_ho_*/cv_fold*(0-4)
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
SUB = 25000; rng = np.random.default_rng(0); f = lambda t: t.float() if t is not None else None
LOCO_DIR = f"outputs/loco_ho_{CAT}" if REGIME == "head" else f"outputs/loco_{CAT}"
IND_FOLDS = [0, 1, 2, 3, 4] if REGIME == "head" else [0, 1, 2]
ind_path = (lambda k: f"outputs/cv_fold{k}/last.pt") if REGIME == "head" else (lambda k: f"outputs/basefull_fold{k}/last.pt")

def load(p):
    c = torch.load(p, map_location="cpu", weights_only=False); m = AffordanceMLP(mlp_head_config_from_model_cfg(c["model_cfg"])); m.load_state_dict(c["model"]); m.eval(); return m
loco = load(f"{LOCO_DIR}/last.pt")
indist = {k: load(ind_path(k)) for k in IND_FOLDS}
# which fold held out each object
fold_of = {}
for k in IND_FOLDS:
    for o in json.load(open(f"{SCR}/cv_fold{k}.json"))["val"]: fold_of[o] = k

bcfg = mlp_head_config_from_model_cfg(torch.load(f"{LOCO_DIR}/last.pt", map_location="cpu", weights_only=False)["model_cfg"])
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=bcfg.dino_filename)
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in TRAINED}
cat_objs = [o for o in o2r if o.split("__")[0] == CAT]

def score(model, o, v, sel, idx, it):
    sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
    with torch.no_grad():
        p = torch.sigmoid(model(emb[v], slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"),
                                dino_cls=f(it.get("dino_cls")), ss_dino_cls=f(it.get("ss_dino_cls")),
                                vertex_normals=sl("vertex_normals"), vertex_positions=None, dino_vertex=sl("dino_vertex_features"))).numpy()
    return average_precision_score, p

byv = collections.defaultdict(lambda: {"loco": [], "ind": []})
for o in cat_objs:
    it = None
    for v in TRAINED:
        hf = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
        if not os.path.exists(hf): continue
        hb = (np.asarray(torch.load(hf, weights_only=False)).reshape(-1) >= 0.5).astype(int)
        if not (0 < hb.sum() < len(hb)): continue
        if it is None: it = ds[o2r[o]]
        V = len(hb); sel = rng.choice(V, min(SUB, V), replace=False); idx = torch.as_tensor(sel, dtype=torch.long); yb = hb[sel]
        sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
        with torch.no_grad():
            pl = torch.sigmoid(loco(emb[v], slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"), dino_cls=f(it.get("dino_cls")),
                                    ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"), vertex_positions=None, dino_vertex=sl("dino_vertex_features"))).numpy()
        byv[v]["loco"].append(average_precision_score(yb, pl))
        if o in fold_of:  # in-dist: model that saw CAT, scored out-of-fold
            m = indist[fold_of[o]]
            with torch.no_grad():
                pi = torch.sigmoid(m(emb[v], slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"), dino_cls=f(it.get("dino_cls")),
                                     ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"), vertex_positions=None, dino_vertex=sl("dino_vertex_features"))).numpy()
            byv[v]["ind"].append(average_precision_score(yb, pi))

print(f"=== LOCO transfer [{REGIME}]: held-out '{CAT}' ({len(cat_objs)} objects) ===")
print(f"{'verb':9s} {'LOCO(unseen cat)':>16s} {'IN-DIST(seen cat)':>18s} {'gap':>8s}  n")
for v in TRAINED:
    if byv[v]["loco"]:
        lo = np.mean(byv[v]["loco"]); ind = np.mean(byv[v]["ind"]) if byv[v]["ind"] else float("nan")
        print(f"{v:9s} {lo:16.3f} {ind:18.3f} {ind-lo:+8.3f}  n={len(byv[v]['loco'])}(ind {len(byv[v]['ind'])})")
