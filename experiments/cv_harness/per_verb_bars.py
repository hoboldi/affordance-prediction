"""Per-verb mean AUPRC vs HUMAN over the full 58-object val split: base vs GEAL teacher vs FT@100.
The rigorous version of the qualitative render (single-object AUPRCs are noisy)."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

verbs = ["contain", "sit", "pour", "move", "display", "grasp"]
n     = [40, 7, 15, 7, 8, 12]
base  = [0.552, 0.717, 0.644, 0.603, 0.556, 0.410]
geal  = [0.483, 0.532, 0.655, 0.657, 0.646, 0.584]
ft    = [0.619, 0.779, 0.685, 0.645, 0.550, 0.401]
m_base, m_geal, m_ft = 0.580, 0.593, 0.613

labels = [f"{v}\n(n={k})" for v, k in zip(verbs, n)] + ["MEAN"]
base_p, geal_p, ft_p = base + [m_base], geal + [m_geal], ft + [m_ft]
x = np.arange(len(labels)); w = 0.26
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w, base_p, w, label="base (open-vocab)", color="#bbbbbb")
ax.bar(x,      geal_p, w, label="GEAL teacher",      color="#e08a3c")
ax.bar(x + w,  ft_p,   w, label="FT@100 (new best)", color="#2e8b57")
for xi, (b, g, f) in enumerate(zip(base_p, geal_p, ft_p)):
    for dx, val in ((-w, b), (0, g), (w, f)):
        ax.text(xi + dx, val + 0.008, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
ax.axvline(len(verbs) - 0.5, ls="--", lw=1, color="0.6")
ax.set_ylabel("mean AUPRC vs HUMAN (58-obj val)"); ax.set_ylim(0, 0.92)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_title("Per-verb affordance accuracy vs HUMAN GT: head-only FT on 100 human-labeled objects beats the GEAL teacher\n"
             "FT wins the high-volume geometric verbs (contain/sit/pour); GEAL keeps an edge on grasp & display", fontsize=10.5)
ax.legend(loc="upper right", fontsize=9); ax.grid(axis="y", alpha=0.3)
out = "outputs/renders/per_verb_means.png"; os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout(); fig.savefig(out, dpi=130)
print("saved", out)
