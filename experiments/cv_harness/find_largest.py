import json
from pathlib import Path

DATA_ROOT = Path("/home/datasets/customDatasets/cmr2")
MANIFEST = DATA_ROOT / "manifest.finepatch.jsonl"

# Unique recon dirs from the manifest
recon_dirs = []
seen = set()
with MANIFEST.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        d = r.get("sam3d_reconstruction_dir")
        if d and d not in seen:
            seen.add(d)
            recon_dirs.append(d)

print(f"{len(recon_dirs)} unique objects")

# Estimate V from vertex_positions.pt file size (V*3*float32 + small torch header).
# Cheap proxy that avoids loading 2306 tensors; we confirm the top one exactly afterward.
best = []
for d in recon_dirs:
    p = DATA_ROOT / d / "vertex_positions.pt"
    if p.is_file():
        sz = p.stat().st_size
        best.append((sz, d))
best.sort(reverse=True)
print("Top 5 by vertex_positions.pt size (bytes):")
for sz, d in best[:5]:
    v_est = (sz - 600) // 12  # ~V from 3*float32 per vertex
    print(f"  {sz:>12}  V~{v_est:>9}  {d}")

# Write the top recon dir for the smoke test to pick up
top_dir = best[0][1]
Path("/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad/top_dir.txt").write_text(top_dir)
print("TOP_DIR:", top_dir)
