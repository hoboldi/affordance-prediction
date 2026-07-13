"""Verb-region overlap heatmap (the 'cross-verb zero-shot is impossible' figure): pairwise mean IoU of
human-label regions over objects labeled for both verbs. Max ~0.06 = mutually near-disjoint islands. CPU."""
import os, glob, itertools, collections
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
TRAINED = ["contain", "sit", "pour", "move", "display", "grasp"]; NOVEL = ["lift", "open", "press"]
V = TRAINED + NOVEL
def load(o, v):
    fp = f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
    return (np.asarray(torch.load(fp, weights_only=False)).reshape(-1) >= 0.5) if os.path.exists(fp) else None
objs = sorted({fp.split("/")[-2] for fp in glob.glob("human_gt_labels/*/vertex_manuallabels_*.pt")})
pair = collections.defaultdict(list)
for o in objs:
    labs = {v: load(o, v) for v in V}; labs = {v: a for v, a in labs.items() if a is not None and 0 < a.sum() < len(a)}
    for a, b in itertools.combinations(labs, 2):
        if len(labs[a]) == len(labs[b]):
            u = (labs[a] | labs[b]).sum()
            if u > 0: pair[frozenset((a, b))].append(float((labs[a] & labs[b]).sum() / u))
M = np.full((len(V), len(V)), np.nan)
for i, a in enumerate(V):
    for j, b in enumerate(V):
        if a == b: M[i, j] = 1.0
        else:
            xs = pair.get(frozenset((a, b)), [])
            if xs: M[i, j] = np.mean(xs)
fig, ax = plt.subplots(figsize=(7.5, 6.5))
Mm = np.ma.masked_invalid(M)
im = ax.imshow(Mm, cmap="magma", vmin=0, vmax=0.3)
ax.set_xticks(range(len(V))); ax.set_yticks(range(len(V)))
ax.set_xticklabels(V, rotation=45, ha="right"); ax.set_yticklabels(V)
for i in range(len(V)):
    for j in range(len(V)):
        if not np.isnan(M[i, j]) and i != j:
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8, color=("white" if M[i, j] < 0.2 else "black"))
# outline novel verbs
for k in range(len(TRAINED), len(V)):
    ax.add_patch(plt.Rectangle((k-0.5, -0.5), 1, len(V), fill=False, edgecolor="cyan", lw=2))
    ax.add_patch(plt.Rectangle((-0.5, k-0.5), len(V), 1, fill=False, edgecolor="cyan", lw=2))
fig.colorbar(im, ax=ax, fraction=0.046, label="region IoU (mean over shared objects)")
# (no title — message lives in the poster/paper caption)
plt.tight_layout()
out = "outputs/renders/verb_overlap_heatmap.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=130, bbox_inches="tight"); print("saved", out, "max off-diag IoU", round(float(np.nanmax(M[M < 0.999])), 3))
