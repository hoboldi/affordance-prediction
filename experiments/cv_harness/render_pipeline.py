"""Methodology pipeline diagram (horizontal banner) for the poster.
Two inputs (RGB image, verb text) -> three color-coded branches (DINO / geometry / verb)
-> concat -> verb-conditioned GNN head (highlighted) -> per-vertex affordance map."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon

# palette
NEUT_F, NEUT_E = "#eef1f4", "#3f4a5a"
DINO, GEOM, VERB = "#2a9d8f", "#e08a00", "#7b4ea3"
HERO, OUT = "#1d3557", "#43aa8b"
TXT = "#14213d"

fig, ax = plt.subplots(figsize=(18, 5.2))
ax.set_xlim(0, 180); ax.set_ylim(0, 52); ax.axis("off")

def box(cx, cy, w, h, text, fc=NEUT_F, ec=NEUT_E, tc=TXT, fs=12.5, bold=False, lw=1.8, r=0.02):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle=f"round,pad=0.02,rounding_size={r*100}",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight=("bold" if bold else "normal"), zorder=4, linespacing=1.25)
    return dict(cx=cx, cy=cy, w=w, h=h)

def arr(a, b, side_a="r", side_b="l", color="#3f4a5a", lw=2.2, ls="-", rad=0.0, label=None, lfs=10, ldy=2.6):
    ax_, ay_ = edge(a, side_a); bx_, by_ = edge(b, side_b)
    ax.add_patch(FancyArrowPatch((ax_, ay_), (bx_, by_), arrowstyle="-|>", mutation_scale=17,
                                 lw=lw, color=color, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        ax.text((ax_+bx_)/2, (ay_+by_)/2+ldy, label, ha="center", va="center", fontsize=lfs,
                color=color, style="italic", zorder=5)

def edge(b, side):
    cx, cy, w, h = b["cx"], b["cy"], b["w"], b["h"]
    return {"r": (cx+w/2, cy), "l": (cx-w/2, cy), "t": (cx, cy+h/2), "b": (cx, cy-h/2)}[side]

# ---- input glyphs ----
def photo_glyph(cx, cy):
    ax.add_patch(Circle((cx-3.2, cy+2.2), 1.1, fc="#ffd166", ec="none", zorder=5))          # sun
    ax.add_patch(Polygon([[cx-5.5, cy-2.6],[cx-1.5, cy+1.2],[cx+2.5, cy-2.6]], closed=True,  # mountain
                         fc="#8ecae6", ec="none", zorder=5))
    ax.add_patch(Polygon([[cx-0.5, cy-2.6],[cx+3, cy+1.8],[cx+6, cy-2.6]], closed=True,
                         fc="#219ebc", ec="none", zorder=5))
def mesh_glyph(cx, cy):
    th = np.linspace(0, 2*np.pi, 9)
    pts = np.c_[cx+3.3*np.cos(th), cy+3.3*np.sin(th)]
    for i in range(len(th)):
        for j in range(i+1, len(th)):
            ax.plot([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]], color="#5c6b7a", lw=0.4, zorder=5)
    ax.scatter(pts[:,0], pts[:,1], s=8, color="#1d3557", zorder=6)
def heat_glyph(cx, cy):
    rng = np.random.default_rng(3)
    n = 140; a = rng.uniform(0, 2*np.pi, n); r = rng.uniform(0, 1, n)**0.5*3.4
    x = cx + r*np.cos(a); y = cy + r*np.sin(a)
    val = np.exp(-((x-(cx+1.5))**2+(y-(cy+1))**2)/6)     # a hot lobe
    ax.scatter(x, y, c=val, cmap="turbo", s=9, zorder=5)

# ============ INPUTS ============
b_img = box(13, 40, 22, 15, "", lw=2.0); photo_glyph(13, 41.5)
ax.text(13, 33.7, "Single RGB image", ha="center", fontsize=12, color=TXT, fontweight="bold")
b_verb = box(13, 9, 22, 11, 'Verb query\n“pour” / “grasp”', fs=12.5, bold=True, fc="#f3ecf8", ec=VERB)

# ============ RECONSTRUCTION ============
b_mesh = box(45, 32, 22, 16, "", lw=2.0); mesh_glyph(45, 34.5)
ax.text(45, 25.8, "3D mesh", ha="center", fontsize=12, color=TXT, fontweight="bold")
arr(b_img, b_mesh, "r", "l", rad=-0.12, label="SAM3D /\nTRELLIS", ldy=3.4)

# ============ BRANCHES ============
# DINO (appearance)
b_dino = box(84, 44, 30, 12, "DINOv2 patches\n→ project to vertices", fc="#e6f4f2", ec=DINO, tc="#155e54", fs=12)
b_dinoc = box(116, 44, 20, 11, "DINOv2\n128-d", fc=DINO, ec=DINO, tc="white", bold=True, fs=13)
arr(b_mesh, b_dino, "r", "l", color=DINO, rad=0.18, label="render 16 views", ldy=3.0, lfs=9.5)
arr(b_dino, b_dinoc, "r", "l", color=DINO)
# geometry (shape)
b_geom = box(84, 27, 30, 12, "Per-vertex geometry\n(height/concavity/curv/\nup/radial)", fc="#fdf1df", ec=GEOM, tc="#8a5300", fs=11)
b_geomc = box(116, 27, 20, 11, "geometry\n5-d", fc=GEOM, ec=GEOM, tc="white", bold=True, fs=13)
arr(b_mesh, b_geom, "r", "l", color=GEOM, rad=0.0, label="no render", ldy=2.6, lfs=9.5)
arr(b_geom, b_geomc, "r", "l", color=GEOM)
# verb (text)
b_clip = box(84, 9, 30, 12, "CLIP text encoder", fc="#f3ecf8", ec=VERB, tc="#4a2d63", fs=12.5)
b_verbc = box(116, 9, 20, 11, "verb\n128-d", fc=VERB, ec=VERB, tc="white", bold=True, fs=13)
arr(b_verb, b_clip, "r", "l", color=VERB)
arr(b_clip, b_verbc, "r", "l", color=VERB)

# ============ CONCAT ============
b_cat = box(138, 27, 15, 44, "Concatenate\nper vertex\n\n133-d\n+ verb", fs=12, bold=True, fc="#f7f9fb", ec="#3f4a5a", r=0.015)
for chip in (b_dinoc, b_geomc, b_verbc):
    arr(chip, b_cat, "r", "l", color="#6b7684", lw=2.0, rad=(0.0 if abs(chip["cy"]-27)<1 else (-0.10 if chip["cy"]>27 else 0.10)))

# ============ GNN HEAD (hero) ============
b_gnn = box(160, 27, 20, 20, "Verb-\nconditioned\nGNN head\n\nEdgeConv over\nmesh kNN", fc=HERO, ec=HERO, tc="white", bold=True, fs=12)
arr(b_cat, b_gnn, "r", "l", color=HERO, lw=2.6)
# spatial-graph cue: local annotation on the mesh (which yields the kNN graph the GNN runs on)
ax.add_patch(FancyArrowPatch((45, 24), (45, 20.5), arrowstyle="-|>", mutation_scale=13,
                             lw=1.5, color=HERO, ls=(0, (4, 2.5)), zorder=1))
ax.text(45, 19.4, "mesh kNN graph\n(spatial message passing)", ha="center", va="top",
        fontsize=10, color=HERO, style="italic", zorder=5, linespacing=1.2)

# ============ OUTPUT (affordance map, above the GNN head) ============
b_map = box(160, 45, 22, 12, "", fc="white", ec=OUT, lw=2.4); heat_glyph(160, 46.4)
ax.text(160, 38.7, "Per-vertex affordance map", ha="center", fontsize=11.5, color="#2c6e57", fontweight="bold")
arr(b_gnn, b_map, "t", "b", color=OUT, lw=2.4)

ax.text(90, 50.4, "From a single image to a verb-conditioned 3D affordance map",
        ha="center", fontsize=14.5, color=TXT, fontweight="bold")

plt.tight_layout()
out = "final_figures/method_pipeline.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("saved", out)
