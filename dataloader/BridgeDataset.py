"""Read a complete bridge point cloud without requiring any target labels."""

from pathlib import Path

import h5py
import numpy as np
from torch.utils.data import Dataset


class BridgeDataset(Dataset):
    def __init__(self, path, cloud=None):
        self.path = Path(path).expanduser().resolve()
        with h5py.File(self.path, "r") as source:
            if cloud is not None:
                self.cloud_path = cloud.strip("/")
            elif "points" in source:
                self.cloud_path = ""
            else:
                candidates = []
                source.visititems(
                    lambda name, obj: candidates.append(name.rsplit("/", 1)[0])
                    if isinstance(obj, h5py.Dataset) and name.endswith("/points")
                    else None
                )
                if len(candidates) != 1:
                    raise ValueError(
                        f"Expected one point cloud, found {candidates}; specify --cloud."
                    )
                self.cloud_path = candidates[0]
            group = source[self.cloud_path] if self.cloud_path else source
            points = np.asarray(group["points"], dtype=np.float32)
            normal_key = "normals" if "normals" in group else "fields/normals/data"
            if normal_key not in group:
                raise ValueError("The pretrained model requires input normals.")
            normals = np.asarray(group[normal_key], dtype=np.float32)

        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError(f"Expected a nonempty (N, 3) point array, got {points.shape}.")
        if normals.shape != points.shape:
            raise ValueError("Normals must have one 3D vector per point.")
        if not np.isfinite(points).all() or not np.isfinite(normals).all():
            raise ValueError("Points and normals must contain only finite values.")
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        if (lengths < 1e-8).any():
            raise ValueError("Input contains zero-length normals.")
        self.points = np.ascontiguousarray(points)
        self.normals = np.ascontiguousarray(normals / lengths)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return {"points": self.points, "normals": self.normals}
