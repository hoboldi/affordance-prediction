from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _scalar_to_rgb(values: np.ndarray, cmap_name: str = "hot") -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    return (cmap(values)[:, :3] * 255).astype(np.uint8)


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    n = len(xyz)
    header = (
        f"ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"end_header\n"
    )
    data = np.hstack([xyz.astype(np.float32), rgb.astype(np.float32)])
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, data, fmt=["%f", "%f", "%f", "%d", "%d", "%d"])


def save_prediction_plys(
    xyz: torch.Tensor,
    gt: torch.Tensor,
    pred: torch.Tensor,
    visible: torch.Tensor,
    output_dir: Path,
    name: str,
) -> None:
    """
    Saves {name}_gt.ply and {name}_pred.ply.
    Visible vertices are coloured by affordance value (hot colormap, 0→black, 1→white).
    Invisible vertices are grey.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    xyz_np = xyz.cpu().float().numpy()
    gt_np = gt.cpu().float().numpy()
    pred_np = pred.cpu().float().numpy()
    vis_np = visible.cpu().bool().numpy()

    grey = np.full((len(xyz_np), 3), 180, dtype=np.uint8)

    for values, suffix in [(gt_np, "gt"), (pred_np, "pred")]:
        rgb = grey.copy()
        if vis_np.any():
            rgb[vis_np] = _scalar_to_rgb(values[vis_np])
        _write_ply(output_dir / f"{name}_{suffix}.ply", xyz_np, rgb)
