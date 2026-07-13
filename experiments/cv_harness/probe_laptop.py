"""Probe view angles for the display-on-laptop GT so the screen region faces the camera."""
import os, sys, hashlib
sys.path.insert(0, "src")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from datasets.data_root_dataset import DataRootDataset
SCR = "/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad"
O = "laptop__122_14291_29271__v000"; V_ = "display"
def seed_of(o): return int(hashlib.md5(o.encode()).hexdigest()[:8], 16)
ds = DataRootDataset(manifest_path=f"{SCR}/manifest.cv.jsonl", load_vertex_labels_eager=False, load_vertex_semantics_eager=False)
o2r = {os.path.basename(str(r.sam3d_reconstruction_dir).rstrip("/")): i for i, r in enumerate(ds.rows)}
it = ds[o2r[O]]; xyz = it["vertex_positions"].numpy(); V = len(xyz)
r = np.random.default_rng(seed_of(O)); sel = np.sort(r.choice(V, min(24000, V), replace=False))
g = np.asarray(torch.load(f"human_gt_labels/{O}/vertex_manuallabels_{V_}.pt", weights_only=False)).reshape(-1)[sel]
angles = [(16,-60),(20,20),(25,110),(30,-120),(35,70),(18,150)]
fig = plt.figure(figsize=(len(angles)*2.4, 2.6))
for i,(el,az) in enumerate(angles):
    ax = fig.add_subplot(1,len(angles),i+1,projection="3d")
    ax.scatter(xyz[sel,0],xyz[sel,2],xyz[sel,1],c=g,cmap="turbo",s=3,vmin=0,vmax=1)
    h=(xyz[sel].max(0)-xyz[sel].min(0)).max()/2+1e-6; mid=(xyz[sel].max(0)+xyz[sel].min(0))/2
    ax.set_xlim(mid[0]-h,mid[0]+h); ax.set_ylim(mid[2]-h,mid[2]+h); ax.set_zlim(mid[1]-h,mid[1]+h)
    ax.view_init(elev=el,azim=az); ax.set_axis_off(); ax.set_title(f"elev={el} azim={az}",fontsize=8)
out=f"{SCR}/probe_laptop.png"; fig.savefig(out,dpi=115,bbox_inches="tight"); print("saved",out)
