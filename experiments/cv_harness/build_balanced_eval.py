"""Build a bigger/balanced eval manifest: all multi-verb chairs (move/sit) + N each of the
other multi-verb classes, so every verb has n~90-120 instead of the current n=10."""
import json, collections, sys
ROOT = "/home/datasets/customDatasets/cmr2"
SRC = f"{ROOT}/manifest.pseudolabeled.clean.dino.NEW.jsonl"
OUT = f"{ROOT}/manifest.eval_balanced.jsonl"
N_PER_OTHER = 30  # per non-chair multi-verb class

rows = [json.loads(l) for l in open(SRC)]
byobj = collections.defaultdict(list)
cls = {}
for r in rows:
    o = r["sam3d_reconstruction_dir"]
    byobj[o].append(r)
    cls[o] = r["object_class"]
verbs = {o: {x["verb"] for x in rs} for o, rs in byobj.items()}
mv = [o for o in byobj if len(verbs[o]) >= 2]               # multi-verb objects only

bycat = collections.defaultdict(list)
for o in sorted(mv):                                        # deterministic
    bycat[cls[o]].append(o)

sel = list(bycat.get("Chair", []))                          # ALL chairs -> move/sit
for c in ["Bottle", "Mug", "Vase", "Bag"]:                  # contain/grasp/pour balanced
    sel += bycat.get(c, [])[:N_PER_OTHER]

selset = set(sel)
out_rows = [r for o in selset for r in byobj[o]]
with open(OUT, "w") as f:
    for r in out_rows:
        f.write(json.dumps(r) + "\n")

vc = collections.Counter(r["verb"] for r in out_rows)
catc = collections.Counter(cls[o] for o in selset)
print(f"wrote {OUT}: {len(selset)} objects, {len(out_rows)} rows")
print("objects by class:", dict(catc))
print("per-verb object count:", dict(vc))
