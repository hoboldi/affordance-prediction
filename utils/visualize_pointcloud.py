"""
Point cloud visualizer for .npy files.

Expected array shapes:
  (N, 3)  — XYZ coordinates, uniform color
  (N, 4)  — XYZ + scalar (e.g. affordance score or label), color-mapped
  (N, 6)  — XYZ + RGB (values in [0, 1] or [0, 255])

Usage (CLI):
    python utils/visualize_pointcloud.py path/to/file.npy [--cmap plasma] [--size 3]

Usage (library):
    from utils.visualize_pointcloud import plot_pointcloud
    plot_pointcloud("path/to/file.npy")
    plot_pointcloud(array)          # pass a numpy array directly
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt


def plot_pointcloud(
    source,
    *,
    cmap: str = "plasma",
    point_size: float = 3.0,
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a point cloud from a .npy file path or a numpy array.

    Args:
        source:     Path to a .npy file, or an ndarray of shape (N,3), (N,4), or (N,6).
        cmap:       Matplotlib colormap used when a scalar channel is present.
        point_size: Marker size passed to scatter.
        title:      Figure title. Defaults to the file name when source is a path.
        show:       Call plt.show() after plotting.
        save_path:  If given, save the figure to this path.

    Returns:
        The matplotlib Figure.
    """
    if isinstance(source, (str,)):
        data = np.load(source, allow_pickle=False)
        if title is None:
            import os
            title = os.path.basename(source)
    else:
        data = np.asarray(source, dtype=np.float64)

    if data.ndim != 2 or data.shape[1] not in (3, 4, 6):
        raise ValueError(
            f"Expected shape (N, 3|4|6), got {data.shape}. "
            "Columns must be XYZ, XYZ+scalar, or XYZ+RGB."
        )

    xyz = data[:, :3]
    n_cols = data.shape[1]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if n_cols == 3:
        sc = ax.scatter(*xyz.T, s=point_size, c="steelblue", depthshade=True)

    elif n_cols == 4:
        scalar = data[:, 3]
        sc = ax.scatter(*xyz.T, s=point_size, c=scalar, cmap=cmap, depthshade=True)
        fig.colorbar(sc, ax=ax, shrink=0.6, label="scalar")

    else:  # 6 columns — RGB
        rgb = data[:, 3:6]
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        sc = ax.scatter(*xyz.T, s=point_size, c=rgb, depthshade=True)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        ax.set_title(title)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a .npy point cloud file.")
    parser.add_argument("file", help="Path to the .npy file")
    parser.add_argument("--cmap", default="plasma", help="Colormap for scalar channel (default: plasma)")
    parser.add_argument("--size", type=float, default=3.0, help="Point size (default: 3.0)")
    parser.add_argument("--save", default=None, help="Save figure to this path instead of displaying")
    args = parser.parse_args()

    plot_pointcloud(
        args.file,
        cmap=args.cmap,
        point_size=args.size,
        show=args.save is None,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
