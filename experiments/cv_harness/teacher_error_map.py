"""CPU-only test: GEAL-vs-human agreement per verb (teacher error map) + label stats.
No model, no GPU — just compares the on-disk GEAL pseudolabels to human labels per (object,verb)."""
import torch, glob, os, collections, numpy as np
from sklearn.metrics import average_precision_score

ROOT = "/home/datasets/customDatasets/cmr2"
human = sorted(glob.glob("human_gt_labels/*/vertex_manuallabels_*.pt"))
gv = collections.defaultdict(list)    # GEAL-vs-human AUPRC per verb
hpr = collections.defaultdict(list)   # human positive-rate per verb
gpr = collections.defaultdict(list)   # GEAL positive-rate per verb (where GEAL exists)
only_h = collections.Counter()        # verbs with no GEAL pseudolabel

for hf in human:
    obj = hf.split("/")[1]
    verb = hf.split("/")[-1].replace("vertex_manuallabels_", "").replace(".pt", "")
    h = np.asarray(torch.load(hf, weights_only=False)).reshape(-1)
    hb = (h >= 0.5).astype(int)
    hpr[verb].append(float(hb.mean()))
    gf = f"{ROOT}/reconstructions/{obj}/vertex_pseudolabels_{verb}.pt"
    if os.path.exists(gf):
        g = np.asarray(torch.load(gf, weights_only=False)).reshape(-1)
        if hb.sum() > 0 and len(g) == len(hb):
            gv[verb].append(float(average_precision_score(hb, g)))
            gpr[verb].append(float((g >= 0.5).mean()))
    else:
        only_h[verb] += 1

print("=== TEACHER ERROR MAP — GEAL pseudolabel vs HUMAN, per verb (lower AUPRC = GEAL more wrong = more headroom) ===")
rows = []
for v in sorted(gv):
    rows.append((v, np.mean(gv[v]), len(gv[v]), np.mean(hpr[v]), np.mean(gpr[v])))
for v, ap, n, hp, gp in sorted(rows, key=lambda r: r[1]):
    print(f"  {v:9s} GEAL-vs-human AUPRC={ap:.3f} (n={n:3d}) | human_posrate={hp:.3f}  GEAL_posrate={gp:.3f}  ({'GEAL OVER-segments' if gp>hp*1.3 else 'GEAL UNDER-segments' if gp<hp*0.7 else 'rate~ok'})")
if gv:
    allap = [a for v in gv for a in gv[v]]
    print(f"  OVERALL GEAL-vs-human AUPRC = {np.mean(allap):.3f} over {len(allap)} (object,verb) pairs")
print("=== verbs with NO GEAL pseudolabel (human-only -> open-vocab only, no teacher signal) ===")
for v, c in only_h.most_common():
    print(f"  {v}: {c} human labels, GEAL absent | human_posrate={np.mean(hpr[v]):.3f}")
