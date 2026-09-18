import random
from pathlib import Path

import h5py
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui

if __package__:
    from .ABCViewer import ABCViewer, categorical_colors
else:
    from ABCViewer import ABCViewer, categorical_colors


DATASET_DIR = Path(r"C:\ABC_final")


def load_part(filepath):
    """Load and validate the point-cloud arrays required by the viewer."""
    required_keys = ("points", "normals", "labels", "prim", "T_param")
    with h5py.File(filepath, "r") as h5_file:
        missing_keys = [key for key in required_keys if key not in h5_file]
        if missing_keys:
            raise KeyError(f"{filepath} is missing dataset(s): {', '.join(missing_keys)}")

        print("Keys:", list(h5_file.keys()))
        for key in h5_file.keys():
            data = h5_file[key]
            print(f"{key:10s}", "shape =", data.shape, "dtype =", data.dtype)

        points = np.asarray(h5_file["points"][:], dtype=np.float64)
        normals = np.asarray(h5_file["normals"][:], dtype=np.float64)
        t_param = np.asarray(h5_file["T_param"][:], dtype=np.float64)
        labels = np.asarray(h5_file["labels"][:]).reshape(-1)
        prim = np.asarray(h5_file["prim"][:]).reshape(-1)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}.")
    if normals.shape != points.shape:
        raise ValueError(
            f"normals must have shape {points.shape}, got {normals.shape}."
        )
    if len(labels) != len(points) or len(prim) != len(points):
        raise ValueError(
            "labels and prim must contain one value per point "
            f"({len(points)}); got {len(labels)} and {len(prim)}."
        )
    return points, normals, labels, prim, t_param


def main():
    files = [path for path in DATASET_DIR.iterdir() if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No files found in {DATASET_DIR}")

    filepath = random.choice(files)
    print(f"Selected: {filepath.name}")
    points, normals, labels, prim, t_param = load_part(filepath)

    print("\nNumber of points:", len(points))
    print("Primitive instances:", np.unique(labels))
    print("Number of instances:", len(np.unique(labels)))
    print("Number of primitives:", len(np.unique(prim)))
    for label in np.unique(labels):
        mask = labels == label
        print(
            f"Instance {label}: {mask.sum()} points, "
            f"primitive type = {np.unique(prim[mask])}"
        )

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)

    label_colors = categorical_colors(labels, seed=42)
    prim_colors = categorical_colors(prim, seed=10)
    normal_colors = np.clip((normals + 1.0) / 2.0, 0.0, 1.0)
    point_cloud.colors = o3d.utility.Vector3dVector(label_colors)

    app = gui.Application.instance
    app.initialize()
    viewer = ABCViewer(
        point_cloud,
        label_colors,
        prim_colors,
        normal_colors,
        points,
        normals,
        labels,
        prim,
        t_param,    
    )
    app.run()


if __name__ == "__main__":
    main()
