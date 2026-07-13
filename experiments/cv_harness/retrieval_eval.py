"""C1 — test-time verb retrieval: for a NOVEL verb, don't feed its raw embedding; route it through the
TRAINED verbs. For each novel object, predict with every trained verb, then aggregate those predictions by
the novel verb's similarity to each trained verb (in verb_proj space, where lift is 0.79-closest to grasp).
Aggregation rules swept: hard-nearest, softmax(sim/temp) over temps, sim-weighted. No training. CPU."""
import os, sys, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
NOVEL = ["lift", "open", "press"]
SUB = 25000; rng = np.random.default_rng(0); f = lambda t: t.float() if t is not None else None

ck = torch.load("outputs/basefull_fold0/last.pt", map_location="cpu", weights_only=False)
cfg = mlp_head_config_from_model_cfg(ck["model_cfg"])
m = AffordanceMLP(cfg); m.load_state_dict(ck["model"]); m.eval(); v2i = ck["verb_to_idx"]
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=cfg.dino_filename)
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
vlm = VLMWrapper(VLMConfig(device="cpu"))
clip = {v: vlm.encode_text([v])[0].float() for v in TRAINED + NOVEL}
with torch.no_grad():
    proj = {v: m.verb_proj(clip[v]).reshape(-1) for v in TRAINED + NOVEL}
def cos(a, b): return torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
# novel -> projected-space similarity to each trained verb
sim = {nv: torch.tensor([cos(proj[nv], proj[tv]) for tv in TRAINED]) for nv in NOVEL}
for nv in NOVEL:
    near = TRAINED[int(sim[nv].argmax())]
    print(f"{nv}: nearest trained = {near} (proj-cos {sim[nv].max():.2f})")

RULES = ["bare", "hard", "sm0.05", "sm0.1", "sm0.2", "simw"]
byv = collections.defaultdict(lambda: collections.defaultdict(list))
for o in sorted(o2r):
    it = None; preds_t = None
    for nv in NOVEL:
        hf = f"human_gt_labels/{o}/vertex_manuallabels_{nv}.pt"
        if not os.path.exists(hf): continue
        hb = (np.asarray(torch.load(hf, weights_only=False)).reshape(-1) >= 0.5).astype(int)
        if not (0 < hb.sum() < len(hb)): continue
        if it is None:
            it = ds[o2r[o]]; V = len(hb); sel = rng.choice(V, min(SUB, V), replace=False); idx = torch.as_tensor(sel, dtype=torch.long)
            sl = lambda k: (it.get(k).float()[idx] if it.get(k) is not None else None)
            kw = dict(slat_vertex=sl("slat_vertex_features"), vlm_features=sl("vertex_features"), dino_cls=f(it.get("dino_cls")),
                      ss_dino_cls=f(it.get("ss_dino_cls")), vertex_normals=sl("vertex_normals"), vertex_positions=None, dino_vertex=sl("dino_vertex_features"))
            with torch.no_grad():
                preds_t = torch.stack([torch.sigmoid(m(clip[tv], **kw)) for tv in TRAINED])  # (6, Vsub)
        yb = hb[sel]
        s = sim[nv]
        for rule in RULES:
            if rule == "bare":
                with torch.no_grad(): p = torch.sigmoid(m(clip[nv], **kw)).numpy()
            elif rule == "hard":
                p = preds_t[int(s.argmax())].numpy()
            elif rule == "simw":
                w = (s / s.sum()).reshape(-1, 1); p = (w * preds_t).sum(0).numpy()
            else:
                temp = float(rule[2:]); w = torch.softmax(s / temp, 0).reshape(-1, 1); p = (w * preds_t).sum(0).numpy()
            byv[nv][rule].append(average_precision_score(yb, p))

print(f"\n=== C1 test-time retrieval (novel verbs, all objects) ===")
print(f"{'verb':6s} {'n':>3s} " + " ".join(f"{r:>7s}" for r in RULES))
for nv in NOVEL:
    print(f"{nv:6s} {len(byv[nv]['bare']):3d} " + " ".join(f"{np.mean(byv[nv][r]):7.3f}" for r in RULES))
print("(bare = raw novel-verb embedding baseline; grasp target for lift ~0.46)")
