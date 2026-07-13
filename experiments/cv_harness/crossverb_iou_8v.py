import os,glob,itertools,collections
import numpy as np,torch
V=["contain","pour","sit","move","display","grasp","press","lift"]
def load(o,v):
    fp=f"human_gt_labels/{o}/vertex_manuallabels_{v}.pt"
    return (np.asarray(torch.load(fp,weights_only=False)).reshape(-1)>=0.5) if os.path.exists(fp) else None
objs=sorted({fp.split("/")[-2] for fp in glob.glob("human_gt_labels/*/vertex_manuallabels_*.pt")})
pair=collections.defaultdict(list)
for o in objs:
    labs={v:load(o,v) for v in V}; labs={v:a for v,a in labs.items() if a is not None and 0<a.sum()<len(a)}
    for a,b in itertools.combinations(labs,2):
        if len(labs[a])==len(labs[b]):
            u=(labs[a]|labs[b]).sum()
            if u>0: pair[frozenset((a,b))].append(float((labs[a]&labs[b]).sum()/u))
print("=== 8-verb cross-verb region IoU (mean over shared objects) ===")
rows=[]
for a,b in itertools.combinations(V,2):
    xs=pair.get(frozenset((a,b)),[])
    if xs: rows.append((np.mean(xs),len(xs),a,b))
rows.sort(reverse=True)
print("  top overlaps:")
for iou,n,a,b in rows[:8]: print(f"    {a}~{b}: IoU={iou:.3f} (n={n})")
allious=[r[0] for r in rows]
print(f"  MAX off-diagonal IoU = {max(allious):.3f}  (over {len(rows)} verb pairs)")
lg=[r for r in rows if set((r[2],r[3]))=={'lift','grasp'}]
if lg: print(f"  lift~grasp: IoU={lg[0][0]:.3f} (n={lg[0][1]})")
