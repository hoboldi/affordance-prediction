"""The RIGHT open-vocab test: synonym robustness on well-supported TRAINED verbs (not novel lift/open).
For each trained verb, prompt the model with SYNONYMS (zero-shot CLIP-text) and score against that verb's
labels. Does the model hold up under paraphrase, and does verb-augmentation training improve it?
Compares no-aug (basefull_fold0) vs +aug (vaug_fold0) on fold0-val. CPU."""
import os, sys, json, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
# HELD-OUT synonyms — genuine paraphrases NOT in the training augmentation set (VERB_SYN), so this tests
# zero-shot phrasing GENERALIZATION, not memorization of the trained paraphrases.
# (trained aug was: contain[store,fill,hold-things-inside] sit[sit-on,be-seated-on,perch-on]
#  pour[pour-out,tip-out,empty] move[push,drag,relocate] display[show,present,exhibit] grasp[grab,grip,pick-up])
SYN = {
    "contain": ["put objects into", "keep things in", "load with items"],
    "sit":     ["rest on", "seat yourself on", "plop down on"],
    "pour":    ["dispense from", "decant from", "spill out of"],
    "move":    ["shove", "slide", "nudge"],
    "display": ["showcase", "put on view", "project onto"],
    "grasp":   ["clutch", "seize", "take hold of"],
}
SUB = 25000; rng = np.random.default_rng(0); f = lambda t: t.float() if t is not None else None
def load(p):
    c = torch.load(p, map_location="cpu", weights_only=False); m = AffordanceMLP(mlp_head_config_from_model_cfg(c["model_cfg"])); m.load_state_dict(c["model"]); m.eval(); return m
models = {"no-aug": load("outputs/basefull_fold0/last.pt"), "+aug": load("outputs/vaug_fold0/last.pt")}
bcfg = mlp_head_config_from_model_cfg(torch.load("outputs/basefull_fold0/last.pt", map_location="cpu", weights_only=False)["model_cfg"])
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=bcfg.dino_filename)
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu"))
words = sorted(set(TRAINED) | {s for v in TRAINED for s in SYN[v]})
emb = {w: vlm.encode_text([w])[0] for w in words}
val = json.load(open(f"{SCR}/cv_fold0.json"))["val"]

# per model, per verb: canonical AUPRC and per-synonym AUPRC (vs the verb's labels)
res = {mn: collections.defaultdict(lambda: {"canon": [], "syn": []}) for mn in models}
for o in val:
    if o not in o2r: continue
    it = None
    for v in TRAINED:
        hf = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
        if not os.path.exists(hf): continue
        hb = (np.asarray(torch.load(hf, weights_only=False)).reshape(-1) >= 0.5).astype(int)
        if not (0 < hb.sum() < len(hb)): continue
        if it is None: it = ds[o2r[o]]
        V = len(hb); sel = rng.choice(V, min(SUB, V), replace=False); idx = torch.as_tensor(sel, dtype=torch.long); yb = hb[sel]
        sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
        kw = dict(slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"), dino_cls=f(it.get("dino_cls")),
                  ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"), vertex_positions=None, dino_vertex=sl("dino_vertex_features"))
        for mn, m in models.items():
            with torch.no_grad():
                res[mn][v]["canon"].append(average_precision_score(yb, torch.sigmoid(m(emb[v], **kw)).numpy()))
                for s in SYN[v]:
                    res[mn][v]["syn"].append(average_precision_score(yb, torch.sigmoid(m(emb[s], **kw)).numpy()))

print("=== synonym robustness (fold0-val): canonical verb vs its synonyms ===")
for mn in models:
    print(f"\n--- {mn} ---")
    print(f"{'verb':9s} {'canon':>6s} {'syn':>6s} {'drop':>6s}")
    cs, ss = [], []
    for v in TRAINED:
        c = np.mean(res[mn][v]["canon"]); s = np.mean(res[mn][v]["syn"]); cs.append(c); ss.append(s)
        print(f"{v:9s} {c:6.3f} {s:6.3f} {c-s:+6.3f}")
    print(f"{'MEAN':9s} {np.mean(cs):6.3f} {np.mean(ss):6.3f} {np.mean(cs)-np.mean(ss):+6.3f}   (drop = how much synonyms lose vs canonical)")
