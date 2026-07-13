"""Build dual-scale DINO: concat coarse 16x16 (vertex_dino.pt) + fine 32x32 (vertex_dino_fine.pt)
-> 256-d vertex_dino_both.pt per object. Lets the head pick scale per verb (recover contain,
keep grasp/pour/move wins). Idempotent: skips objects that already have the combined file."""
import json, os, sys, torch

ROOT = "/home/datasets/customDatasets/cmr2"
rows = [json.loads(l) for l in open(f"{ROOT}/manifest.finepatch.jsonl")]
dirs = sorted({r["sam3d_reconstruction_dir"] for r in rows})
done = skip = err = 0
for i, d in enumerate(dirs):
    base = f"{ROOT}/{d}"
    out = f"{base}/vertex_dino_both.pt"
    if os.path.exists(out):
        skip += 1
        continue
    try:
        a = torch.load(f"{base}/vertex_dino.pt", weights_only=False)        # coarse 16x16
        b = torch.load(f"{base}/vertex_dino_fine.pt", weights_only=False)   # fine 32x32
        feat = torch.cat([a["features"], b["features"]], dim=1)             # [N, 256]
        vis = a["visible_in_any_view"] | b["visible_in_any_view"]           # union (same render -> ~identical)
        torch.save({"features": feat, "visible_in_any_view": vis}, out)
        done += 1
    except Exception as e:
        err += 1
        print(f"ERR {d}: {e}", flush=True)
    if i % 200 == 0:
        print(f"{i}/{len(dirs)} done={done} skip={skip} err={err}", flush=True)
print(f"DONE: {done} written, {skip} skipped, {err} errors (of {len(dirs)} dirs)", flush=True)
