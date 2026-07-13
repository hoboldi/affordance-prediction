"""DECISIVE test: does a NONSENSE word score as well as a real synonym against a verb's labels?
If nonsense ~ synonym -> the 'synonym robustness' is a verb-agnostic fallback, not real routing.
If nonsense << synonym -> synonyms genuinely steer to the right region. +aug model, fold0-val. CPU."""
import os, sys, json, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
SYN = {"contain": ["put objects into", "keep things in"], "sit": ["rest on", "seat yourself on"],
       "pour": ["decant from", "dispense from"], "move": ["shove", "slide"],
       "display": ["showcase", "put on view"], "grasp": ["clutch", "seize"]}
NONSENSE = ["banana", "xqzptv", "the weather", "a chair", "purple"]   # non-verb / off-affordance controls
SUB = 25000; rng = np.random.default_rng(0); f = lambda t: t.float() if t is not None else None
c = torch.load("outputs/vaug_fold0/last.pt", map_location="cpu", weights_only=False)
m = AffordanceMLP(mlp_head_config_from_model_cfg(c["model_cfg"])); m.load_state_dict(c["model"]); m.eval()
bcfg = mlp_head_config_from_model_cfg(c["model_cfg"])
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=bcfg.dino_filename)
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu"))
words = TRAINED + [s for v in SYN for s in SYN[v]] + NONSENSE
emb = {w: vlm.encode_text([w])[0] for w in words}
val = json.load(open(f"{SCR}/cv_fold0.json"))["val"]

# for each (obj,verb): AUPRC of canonical, synonyms, and EACH nonsense word vs that verb's labels
cat = collections.defaultdict(list)  # category -> list of AUPRC (vs the correct verb's labels)
for o in val:
    if o not in o2r: continue
    it = None
    for v in TRAINED:
        hf = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
        if not os.path.exists(hf): continue
        hb = (np.asarray(torch.load(hf, weights_only=False)).reshape(-1) >= 0.5).astype(int)
        if not (0 < hb.sum() < len(hb)): continue
        if it is None: it = ds[o2r[o]]
        Vn = len(hb); sel = rng.choice(Vn, min(SUB, Vn), replace=False); idx = torch.as_tensor(sel, dtype=torch.long); yb = hb[sel]
        sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
        kw = dict(slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"), dino_cls=f(it.get("dino_cls")),
                  ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"), vertex_positions=None, dino_vertex=sl("dino_vertex_features"))
        ap = lambda w: average_precision_score(yb, torch.sigmoid(m(emb[w], **kw)).detach().numpy())
        with torch.no_grad():
            cat["canonical"].append(ap(v))
            for s in SYN[v]: cat["synonym(held-out)"].append(ap(s))
            for nw in NONSENSE: cat[f"nonsense:{nw}"].append(ap(nw))
        # base rate (a constant map = positive fraction) as the true floor
        cat["base-rate(floor)"].append(yb.mean())

print(f"=== AUPRC vs the CORRECT verb's labels (+aug, fold0-val), by prompt type ===")
for k in ["canonical", "synonym(held-out)"] + [f"nonsense:{nw}" for nw in NONSENSE] + ["base-rate(floor)"]:
    print(f"  {k:22s} {np.mean(cat[k]):.3f}  (n={len(cat[k])})")
print("\nIf nonsense ~= synonym => fallback. If synonym >> nonsense (and > base-rate) => real routing.")
