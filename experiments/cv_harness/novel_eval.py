"""Does verb augmentation improve ZERO-SHOT novel-verb generalization? Score lift/open/press (never trained)
on ALL objects that have those human labels, for two fold0 full-FT models:
  NO-AUG  outputs/basefull_fold0/last.pt
  +AUG    outputs/vaug_fold0/last.pt
Novel verbs encoded via CLIP-text (open-vocab). Per-verb AUPRC vs human, SUB=25000. CPU."""
import os, sys, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
NOVEL = ["lift", "open", "press"]
SUB = 25000; rng = np.random.default_rng(0); f = lambda t: t.float() if t is not None else None

def load(p):
    c = torch.load(p, map_location="cpu", weights_only=False); m = AffordanceMLP(mlp_head_config_from_model_cfg(c["model_cfg"])); m.load_state_dict(c["model"]); m.eval(); return m
TREAT = sys.argv[1] if len(sys.argv) > 1 else "outputs/vaug_fold0/last.pt"
DESC = len(sys.argv) > 2 and sys.argv[2] == "desc"   # eval the treatment model with the verb DESCRIPTION
VERB_DESC = {"lift": "grip and raise this object up off the surface",
             "open": "open this object by moving its lid or door",
             "press": "press down on the surface of this object with a finger"}
noaug = load("outputs/basefull_fold0/last.pt")
aug = load(TREAT)
print(f"treatment model: {TREAT}  (desc-encoding: {DESC})")
bcfg = mlp_head_config_from_model_cfg(torch.load("outputs/basefull_fold0/last.pt", map_location="cpu", weights_only=False)["model_cfg"])
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=bcfg.dino_filename)
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu"))
emb = {v: vlm.encode_text([v])[0] for v in NOVEL}                          # bare word (baseline)
emb_t = {v: vlm.encode_text([VERB_DESC[v]])[0] for v in NOVEL} if DESC else emb  # description (treatment, if desc)

byv = collections.defaultdict(lambda: {"na": [], "a": []})
for o in sorted(o2r):
    it = None
    for v in NOVEL:
        hf = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
        if not os.path.exists(hf): continue
        hb = (np.asarray(torch.load(hf, weights_only=False)).reshape(-1) >= 0.5).astype(int)
        if not (0 < hb.sum() < len(hb)): continue
        if it is None: it = ds[o2r[o]]
        V = len(hb); sel = rng.choice(V, min(SUB, V), replace=False); idx = torch.as_tensor(sel, dtype=torch.long)
        sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
        kw = dict(slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"), dino_cls=f(it.get("dino_cls")),
                  ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"), vertex_positions=None, dino_vertex=sl("dino_vertex_features"))
        with torch.no_grad():
            pna = torch.sigmoid(noaug(emb[v], **kw)).numpy()      # baseline: bare-word verb
            pa = torch.sigmoid(aug(emb_t[v], **kw)).numpy()       # treatment: description verb if --desc
        byv[v]["na"].append(average_precision_score(hb[sel], pna))
        byv[v]["a"].append(average_precision_score(hb[sel], pa))

print("=== ZERO-SHOT novel-verb generalization (all objects with the label) ===")
print(f"{'verb':7s} {'n':>3s} {'NO-AUG':>7s} {'+AUG':>7s} {'delta':>7s}")
na_all, a_all = [], []
for v in NOVEL:
    if byv[v]["na"]:
        na, a = np.mean(byv[v]["na"]), np.mean(byv[v]["a"]); na_all.append(na); a_all.append(a)
        print(f"{v:7s} {len(byv[v]['na']):3d} {na:7.3f} {a:7.3f} {a-na:+7.3f}")
print(f"{'MEAN':7s}     {np.mean(na_all):7.3f} {np.mean(a_all):7.3f} {np.mean(a_all)-np.mean(na_all):+7.3f}")
