"""Is the verb REALLY steering the prediction, or is there a generic fallback that happens to score OK?
On multi-verb objects, with the +aug model:
 (1) cross-verb distinctness: correlation between DIFFERENT canonical verbs' maps (low => verbs matter).
 (2) synonym routing: for each held-out synonym, is its map most-correlated with its OWN verb (not others)?
 (3) nonsense control: does a non-verb word ("banana"/"the") produce a real-verb-like map or a degenerate one?
CPU, aug model = vaug_fold0."""
import os, sys, json, collections
sys.path.insert(0, "src")
import numpy as np, torch
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
SYN = {"contain": "put objects into", "sit": "rest on", "pour": "decant from",
       "move": "shove", "display": "showcase", "grasp": "clutch"}      # held-out synonyms
NONSENSE = ["banana", "the weather", "xqzptv"]
SUB = 20000; rng = np.random.default_rng(0); f = lambda t: t.float() if t is not None else None
c = torch.load("outputs/vaug_fold0/last.pt", map_location="cpu", weights_only=False)
m = AffordanceMLP(mlp_head_config_from_model_cfg(c["model_cfg"])); m.load_state_dict(c["model"]); m.eval()
bcfg = mlp_head_config_from_model_cfg(c["model_cfg"])
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=bcfg.dino_filename)
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu"))
words = TRAINED + list(SYN.values()) + NONSENSE
emb = {w: vlm.encode_text([w])[0] for w in words}
def verbs_of(o): return [v for v in TRAINED if os.path.exists(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt")]
# multi-verb objects (>=2 trained verbs), from fold0 val
val = json.load(open(f"{SCR}/cv_fold0.json"))["val"]
objs = [o for o in val if o in o2r and len(verbs_of(o)) >= 2][:20]

def cc(a, b): return float(np.corrcoef(a, b)[0, 1])
offdiag, syn_own, syn_other, syn_rank1, nons = [], [], [], [], collections.defaultdict(list)
for o in objs:
    it = ds[o2r[o]]; V = len(it["vertex_positions"]); sel = rng.choice(V, min(SUB, V), replace=False); idx = torch.as_tensor(sel, dtype=torch.long)
    sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
    kw = dict(slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"), dino_cls=f(it.get("dino_cls")),
              ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"), vertex_positions=None, dino_vertex=sl("dino_vertex_features"))
    with torch.no_grad():
        P = {w: torch.sigmoid(m(emb[w], **kw)).numpy() for w in words}
    vs = verbs_of(o)
    for i in range(len(vs)):                                    # (1) canonical cross-verb corr
        for j in range(i + 1, len(vs)):
            offdiag.append(cc(P[vs[i]], P[vs[j]]))
    for v in vs:                                                # (2) synonym routing
        s = SYN[v]; own = cc(P[s], P[v]); others = [cc(P[s], P[u]) for u in TRAINED if u != v]
        syn_own.append(own); syn_other.append(float(np.mean(others)))
        allv = sorted(TRAINED, key=lambda u: -cc(P[s], P[u])); syn_rank1.append(allv[0] == v)
    for nw in NONSENSE:                                         # (3) nonsense: corr to nearest real verb
        nons[nw].append(max(cc(P[nw], P[v]) for v in vs))

print(f"objects: {len(objs)} (multi-verb)")
print(f"\n(1) CROSS-VERB DISTINCTNESS: mean corr between DIFFERENT canonical verbs = {np.mean(offdiag):+.3f}")
print(f"    (near 0 or negative => verbs produce DISTINCT maps, no generic fallback; ~1 => collapse)")
print(f"\n(2) SYNONYM ROUTING: held-out synonym's map correlation to...")
print(f"    its OWN verb   = {np.mean(syn_own):+.3f}")
print(f"    OTHER verbs    = {np.mean(syn_other):+.3f}   (gap = {np.mean(syn_own)-np.mean(syn_other):+.3f})")
print(f"    synonym's #1 most-correlated verb IS its own verb: {100*np.mean(syn_rank1):.0f}% of cases")
print(f"\n(3) NONSENSE CONTROL: a non-verb word's max corr to any real verb map:")
for nw in NONSENSE: print(f"    '{nw:12s}' = {np.mean(nons[nw]):+.3f}")
