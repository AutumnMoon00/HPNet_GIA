import h5py
import numpy as np
import open3d as o3d
import random
from pathlib import Path

# select a random file from the ABCParts dataset
dir = Path("C:\\ABC_final")
files = [f for f in dir.iterdir() if f.is_file()]
filepath = random.choice(files)     # randomly select a file
print(f"Selected: {filepath.name}")

with h5py.File(filepath, "r") as f:
    print("Keys:", list(f.keys()))
    for key in f.keys():
        data = f[key]
        print(
            f"{key:10s}",
            "shape =", data.shape,
            "dtype =", data.dtype
        )

    points = f["points"][:]
    normals = f["normals"][:]
    labels = f["labels"][:]
    prim = f["prim"][:]

print()
print("Number of points:", len(points))
print("Primitive instances:", np.unique(labels))
print("Number of instances:", len(np.unique(labels)))
print("Number of primitives:", len(np.unique(prim)))



# generate one deterministic color per instance
rng = np.random.default_rng(42)

unique_labels = np.unique(labels)

color_table = {
    label: rng.random(3)
    for label in unique_labels
}

colors = np.array([
    color_table[label]
    for label in labels
])

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd])
