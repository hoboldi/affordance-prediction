import os
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

from huggingface_hub import snapshot_download
from pathlib import Path

snapshot_download(
    repo_id="MattisKr/co3d-sam3d-affordance",
    repo_type="dataset",
    local_dir=Path(__file__).parent.parent / "data",
)
