"""Build 5-fold stratified CV over the labeled objects (balanced per verb), + a 241-object manifest
(original features) for a light eager-load. Writes cv_fold{0..4}.json and manifest.cv.jsonl."""
import json, glob, collections, numpy as np
ROOT = "/home/datasets/customDatasets/cmr2"
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
K = 5

# manifest objects (must have features)
man_rows = [json.loads(l) for l in open(f"{ROOT}/manifest.finepatch.jsonl")]
man_objs = {r["sam3d_reconstruction_dir"].split("/")[-1] for r in man_rows}

# labeled objects -> verbs with positives
import torch
objverbs = collections.defaultdict(set)
for hf in glob.glob("human_gt_labels/*/vertex_manuallabels_*.pt"):
    o = hf.split("/")[1]; v = hf.split("/")[-1].replace("vertex_manuallabels_", "").replace(".pt", "")
    if o not in man_objs:
        continue
    lab = np.asarray(torch.load(hf, weights_only=False)).reshape(-1)
    if (lab >= 0.5).sum() > 0:
        objverbs[o].add(v)
objs = sorted(objverbs)
gcount = collections.Counter(v for o in objs for v in objverbs[o])
print(f"{len(objs)} labeled objects in manifest | verb totals: {dict(gcount)}")

# greedy stratified assignment: place rare-verb objects first, into the fold with fewest of that verb
fold_vc = [collections.Counter() for _ in range(K)]
fold_sz = [0] * K
assign = {}
for o in sorted(objs, key=lambda o: (min(gcount[v] for v in objverbs[o]), o)):
    rv = min(objverbs[o], key=lambda v: gcount[v])
    k = min(range(K), key=lambda f: (fold_vc[f][rv], fold_sz[f]))
    assign[o] = k; fold_sz[k] += 1
    for v in objverbs[o]:
        fold_vc[k][v] += 1

# write splits + report balance
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]
print(f"\nper-fold held-out (val) counts:\n{'fold':5s} {'size':>5s} " + " ".join(f"{v:>8s}" for v in TRAINED))
for k in range(K):
    valk = sorted(o for o in objs if assign[o] == k)
    traink = sorted(o for o in objs if assign[o] != k)
    json.dump({"train": traink, "val": valk}, open(f"{SCR}/cv_fold{k}.json", "w"))
    print(f"{k:5d} {len(valk):5d} " + " ".join(f"{fold_vc[k][v]:8d}" for v in TRAINED))

# 241-object manifest (original features)
cv_rows = [r for r in man_rows if r["sam3d_reconstruction_dir"].split("/")[-1] in set(objs)]
open(f"{SCR}/manifest.cv.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in cv_rows))
print(f"\nmanifest.cv.jsonl: {len(cv_rows)} rows, {len(set(r['sam3d_reconstruction_dir'] for r in cv_rows))} objects")
