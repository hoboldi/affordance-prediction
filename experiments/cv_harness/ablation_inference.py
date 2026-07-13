"""Feature-channel ablation via inference-zeroing on the 5 CV fold models (out-of-fold, human GT).
Zero a channel at inference, measure the drop in trained-mean + per-verb AUPRC vs full. CPU only.
Vertex-subsampled (25k) for speed. Channels: SLAT(8), CLIP-vertex(128), DINO-vertex(128), DINO-cls(2048), normals(3)."""
import os, sys, json, collections
sys.path.insert(0, "src")
import numpy as np, torch
from sklearn.metrics import average_precision_score
from datasets.data_root_dataset import DataRootDataset
from models.mlp_head import AffordanceMLP, mlp_head_config_from_model_cfg
from vlm.vlm_wrapper import VLMWrapper, VLMConfig

SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
MODES = ["full", "no_slat", "no_clip", "no_dino", "no_dinocls", "no_normals"]
K, SUB = 5, 25000
rng = np.random.default_rng(0)
cfg0 = mlp_head_config_from_model_cfg(torch.load("outputs/cv_fold0/last.pt", map_location="cpu", weights_only=False)["model_cfg"])
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False,
                     load_vertex_semantics_eager=True, load_vertex_dino=True, dino_filename=cfg0.dino_filename)
obj2row = {}
for i, r in enumerate(ds.rows):
    obj2row.setdefault(os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")), i)
vlm = VLMWrapper(VLMConfig(device="cpu")); emb = {v: vlm.encode_text([v])[0] for v in TRAINED}

def run(model, F, e, mode):
    slat, clip, dino, dcls, sdcls, nrm = F["slat"], F["clip"], F["dino"], F["dcls"], F["sdcls"], F["nrm"]
    z = lambda t: torch.zeros_like(t) if t is not None else None
    if mode == "no_slat": slat = z(slat)
    if mode == "no_clip": clip = z(clip)
    if mode == "no_dino": dino = z(dino)
    if mode == "no_dinocls": dcls, sdcls = z(dcls), z(sdcls)
    if mode == "no_normals": nrm = z(nrm)
    with torch.no_grad():
        lo = model(e, slat_vertex=slat, vlm_features=clip, dino_cls=dcls, ss_dino_cls=sdcls,
                   vertex_normals=nrm, vertex_positions=None, dino_vertex=dino, vertex_geom=None)
    return torch.sigmoid(lo).numpy()

ap = {m: collections.defaultdict(dict) for m in MODES}
for k in range(K):
    ck = torch.load(f"outputs/cv_fold{k}/last.pt", map_location="cpu", weights_only=False)
    m = AffordanceMLP(mlp_head_config_from_model_cfg(ck["model_cfg"])); m.load_state_dict(ck["model"]); m.eval()
    for o in json.load(open(f"{SCR}/cv_fold{k}.json"))["val"]:
        if o not in obj2row: continue
        verbs = [v for v in TRAINED if os.path.exists(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt")]
        if not verbs: continue
        it = ds[obj2row[o]]
        clip_t = it.get("vertex_features")
        if clip_t is None: continue
        V = len(clip_t); sel = rng.choice(V, min(SUB, V), replace=False)
        sl = lambda t: (t.float()[sel] if t is not None else None)
        gl = lambda t: (t.float() if t is not None else None)
        F = dict(slat=sl(it.get("slat_vertex_features")), clip=sl(clip_t), dino=sl(it.get("dino_vertex_features")),
                 nrm=sl(it.get("vertex_normals")), dcls=gl(it.get("dino_cls")), sdcls=gl(it.get("ss_dino_cls")))
        for v in verbs:
            hb = (np.asarray(torch.load(f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt", weights_only=False)).reshape(-1) >= 0.5).astype(int)
            if hb.sum() == 0 or len(hb) != V: continue
            hbs = hb[sel]
            if hbs.sum() == 0 or hbs.sum() == len(hbs): continue
            for mode in MODES:
                ap[mode][o][v] = average_precision_score(hbs, run(m, F, emb[v], mode))
    print(f"fold {k} scored")

objs = sorted(ap["full"].keys())
tmean = lambda M: float(np.mean([np.mean([M[o][v] for o in objs if v in M[o]]) for v in TRAINED]))
pv = lambda M, v: float(np.mean([M[o][v] for o in objs if v in M[o]]))
full_tm = tmean(ap["full"])
print(f"\n{'mode':11s} " + " ".join(f"{v[:7]:>7s}" for v in TRAINED) + f" | {'MEAN':>6s} {'Δmean':>7s}")
for mode in MODES:
    tm = tmean(ap[mode])
    print(f"{mode:11s} " + " ".join(f"{pv(ap[mode], v):7.3f}" for v in TRAINED) +
          f" | {tm:6.3f} {'' if mode=='full' else f'{tm-full_tm:+.3f}'}")
print(f"\nΔmean = trained-mean change when zeroed (more negative = contributes more). full={full_tm:.3f} (~CV 0.650).")
print(f"SLAT contribution ≈ full − no_slat.  (25k-vertex subsample; inference-zeroing = reliance, not retrain-without.)")
