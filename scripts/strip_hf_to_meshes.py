"""Slim MattisKr/co3d-sam3d-affordance to meshes-only-for-labeling: delete heavy latent/feature files,
KEEP mesh.glb + vertex_positions.pt + vertex_normals.pt + meta.json + GEAL vertex_pseudolabels_*.pt +
manifests. Add ANNOTATION_SPEC.md + alignment_manifest.json + human_gt_labels/ + a README banner.
Deletes are batched commits (metadata ops). A hard assertion guarantees no mesh/label file is deleted.
"""
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)

REPO = "MattisKr/co3d-sam3d-affordance"
STAGE = Path("/home/datasets/customDatasets/cmr2/hf_meshes_only")
DELETE_BASENAMES = {
    "dino_cls.pt", "dino_patches.pt", "ss_dino_cls.pt", "ss_dino_patches.pt",
    "gaussian.ply", "global_latent.pt", "shape_latent.pt",
    "slat_coords.pt", "slat_feats.pt", "slat_vertex_features.pt", "vertex_semantics.pt",
}
KEEP_GUARD = {"mesh.glb", "vertex_positions.pt", "vertex_normals.pt", "meta.json"}

api = HfApi()
files = api.list_repo_files(REPO, repo_type="dataset")
to_delete = sorted(f for f in files if f.split("/")[-1] in DELETE_BASENAMES)
print(f"repo files: {len(files)} | to delete: {len(to_delete)}", flush=True)

# HARD SAFETY: never delete a mesh/positions/normals/meta or any GEAL pseudolabel
bad = [f for f in to_delete if f.split("/")[-1] in KEEP_GUARD or "pseudolabel" in f.lower()]
assert not bad, f"REFUSING: delete set touched protected files: {bad[:5]}"
print("safety check OK — delete set is latents only", flush=True)

# 1) docs + README banner + labels folder placeholder
cur = ""
try:
    cur = Path(hf_hub_download(REPO, "README.md", repo_type="dataset")).read_text()
except Exception as e:  # noqa: BLE001
    print("(no existing README fetched:", e, ")")
banner = (
    "> ⚠️ **Slimmed to meshes-only for human affordance labeling.** Heavy latent/feature files "
    "were removed to keep this lean; `mesh.glb`, `vertex_positions.pt`, `vertex_normals.pt`, the GEAL "
    "`vertex_pseudolabels_*.pt`, and the manifests are kept. See **ANNOTATION_SPEC.md** before labeling.\n\n"
)
add_ops = [
    CommitOperationAdd("README.md", (banner + cur).encode()),
    CommitOperationAdd("ANNOTATION_SPEC.md", (STAGE / "ANNOTATION_SPEC.md").read_bytes()),
    CommitOperationAdd("alignment_manifest.json", (STAGE / "alignment_manifest.json").read_bytes()),
    CommitOperationAdd("human_gt_labels/.gitkeep", b""),
]
api.create_commit(REPO, repo_type="dataset", operations=add_ops,
                  commit_message="meshes-only: add annotation spec + alignment manifest + labels folder")
print("docs/spec/manifest added", flush=True)

# 2) batched deletes
B = 1500
for i in range(0, len(to_delete), B):
    chunk = to_delete[i:i + B]
    api.create_commit(REPO, repo_type="dataset",
                      operations=[CommitOperationDelete(path_in_repo=p) for p in chunk],
                      commit_message=f"meshes-only: drop heavy latents [{i // B + 1}]")
    print(f"deleted batch {i // B + 1}: {len(chunk)} files", flush=True)
print("DONE strip", flush=True)
