"""Training pipeline diagram (horizontal banner) — matches method_pipeline.png style.
Two stages: (1) distill a FROZEN GEAL teacher into the GNN (pretrain on pseudo-labels),
(2) fine-tune on human labels (5-fold CV). Warm start, not the source of performance."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NEUT_F, NEUT_E = "#eef1f4", "#3f4a5a"
HERO = "#1d3557"          # the GNN (same navy as inference diagram)
TEACH = "#5c6b7a"         # frozen teacher (gray-blue)
HUMAN = "#2a9d8f"         # human labels (teal)
FINAL = "#b5842a"         # final model (gold accent)
S1_BG, S2_BG = "#eaf2fb", "#eaf6f2"
TXT = "#14213d"

fig, ax = plt.subplots(figsize=(17.5, 5.4))
ax.set_xlim(0, 180); ax.set_ylim(0, 54); ax.axis("off")

def box(cx, cy, w, h, text, fc=NEUT_F, ec=NEUT_E, tc=TXT, fs=11.5, bold=False, lw=1.8, r=2.0, z=3):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight=("bold" if bold else "normal"), zorder=z+1, linespacing=1.3)
    return dict(cx=cx, cy=cy, w=w, h=h)

def edge(b, s):
    cx, cy, w, h = b["cx"], b["cy"], b["w"], b["h"]
    return {"r": (cx+w/2, cy), "l": (cx-w/2, cy), "t": (cx, cy+h/2), "b": (cx, cy-h/2)}[s]

def arr(a, b, sa="b", sb="t", color=NEUT_E, lw=2.3, ls="-", rad=0.0, label=None, lfs=10, ldx=0, ldy=0, la="center"):
    ax_, ay_ = edge(a, sa); bx_, by_ = edge(b, sb)
    ax.add_patch(FancyArrowPatch((ax_, ay_), (bx_, by_), arrowstyle="-|>", mutation_scale=17,
                                 lw=lw, color=color, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        ax.text((ax_+bx_)/2+ldx, (ay_+by_)/2+ldy, label, ha=la, va="center", fontsize=lfs,
                color=color, style="italic", zorder=5, linespacing=1.2)

# ---- stage background panels ----
def panel(x0, x1, fc, title):
    ax.add_patch(FancyBboxPatch((x0, 4), x1-x0, 45, boxstyle="round,pad=0.02,rounding_size=2.5",
                                fc=fc, ec="none", zorder=0))
    ax.text((x0+x1)/2, 46.2, title, ha="center", fontsize=13.5, color=TXT, fontweight="bold", zorder=1)
panel(4, 74, S1_BG, "① Pretrain — distill from a frozen teacher")
panel(80, 150, S2_BG, "② Fine-tune — human labels")

# ================= STAGE 1 =================
b_teach = box(28, 37, 40, 13, "GEAL teacher   ❄ frozen\nPointNet++ + RoBERTa", fc="#e3e8ee", ec=TEACH, tc="#33404f", fs=11.5)
b_pre = box(28, 15, 40, 15, "Pretrain AffordanceGNN\nsoft-BCE on pseudo-labels", fc=HERO, ec=HERO, tc="white", bold=True, fs=12)
arr(b_teach, b_pre, "b", "t", color=TEACH, label="pseudo-labels\n667 obj · 1227 pairs", ldx=15, ldy=0, la="left")
ax.text(28, 5.9, "pure-GEAL alone → 0.613 on human data", ha="center", fontsize=9.5, color="#7a3b3b", style="italic", zorder=5)

# ================= STAGE 2 =================
b_hum = box(112, 37, 46, 13, "226 human-labeled meshes\n(54 contaminated files removed)", fc="#e2f3ef", ec=HUMAN, tc="#155e54", fs=11.5)
b_ft = box(112, 15, 46, 15, "Fine-tune AffordanceGNN\nclass-balanced BCE · 5-fold CV", fc=HERO, ec=HERO, tc="white", bold=True, fs=12)
arr(b_hum, b_ft, "b", "t", color=HUMAN, label="human GT", ldx=13, ldy=0, la="left")

# ============ WARM-START CONNECTOR (stage1 -> stage2) ============
arr(b_pre, b_ft, "r", "l", color=HERO, lw=3.0)
ax.text(70, 19.6, "warm start\n(init weights)", ha="center", fontsize=11, color=HERO, fontweight="bold", zorder=5, linespacing=1.15)
ax.text(70, 9.4, "GNN 0.870 (scratch)\n→ 0.895 (+0.025)", ha="center", fontsize=9.5, color=HERO, style="italic", zorder=5, linespacing=1.15)

# ================= FINAL MODEL =================
b_final = box(166, 15, 24, 20, "Final model\n\n0.895 AUPRC\nbeats teacher\n+0.058", fc="#fbf3e0", ec=FINAL, tc="#6b4e13", bold=True, fs=11.5)
arr(b_ft, b_final, "r", "l", color=FINAL, lw=2.6)

ax.text(90, 51.6, "Training: distill a frozen GEAL teacher, then fine-tune on human labels",
        ha="center", fontsize=14.5, color=TXT, fontweight="bold")

plt.tight_layout()
out = "final_figures/train_pipeline.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("saved", out)
