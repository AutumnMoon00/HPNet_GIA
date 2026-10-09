"""Save complete point-aligned bridge predictions and colored inspection files."""

import json
from pathlib import Path

import h5py
import numpy as np


def mapped_primitives(raw):
    result = np.asarray(raw, dtype=np.int64).copy()
    result[np.isin(result, [6, 7, 9])] = 0
    result[result == 8] = 2
    return result


def write_ply(path, points, labels):
    unique, inverse = np.unique(labels, return_inverse=True)
    colors = np.random.default_rng(42).integers(40, 256, size=(len(unique), 3))[inverse]
    with open(path, "x", encoding="ascii") as output:
        output.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        np.savetxt(output, np.column_stack([points, colors]), fmt="%.9g %.9g %.9g %d %d %d")


def export_bridge_prediction(dataset, output_dir, predictions, metadata):
    count = len(dataset.points)
    expected = np.arange(count, dtype=np.int64)
    if not np.array_equal(predictions["source_index"], expected):
        raise ValueError("Predictions must cover every source row once, in its original order.")
    for name, values in predictions.items():
        if len(values) != count or not np.isfinite(values).all():
            raise ValueError(f"Invalid point-aligned prediction: {name}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset.path.stem + "_predictions"
    h5_path = output_dir / (stem + ".h5")
    ply_paths = [output_dir / (stem + suffix) for suffix in ("_instances.ply", "_primitives.ply")]
    summary_path = output_dir / (stem + ".json")
    for path in [h5_path, *ply_paths, summary_path]:
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}; choose a new --output_dir.")
    with h5py.File(dataset.path, "r") as source, h5py.File(h5_path, "x") as output:
        if any(name in source for name in predictions):
            raise ValueError("Source already contains prediction fields.")
        for key, value in source.attrs.items():
            output.attrs[key] = value
        for key in source:
            source.copy(key, output)
        if "points" not in output:
            output.create_dataset("points", data=dataset.points)
        if "normals" not in output:
            output.create_dataset("normals", data=dataset.normals)
        for name, values in predictions.items():
            output.create_dataset(name, data=values, compression="gzip", shuffle=True)
        output.attrs["source_file"] = str(dataset.path)
        output.attrs["source_cloud"] = dataset.cloud_path
        output.attrs["source_point_count"] = count
        output.attrs["checkpoint"] = metadata["checkpoint"]
        output.attrs["postprocess"] = metadata["postprocess"]
        output["labels_pred"].attrs["description"] = "Predicted instance cluster IDs, arbitrary up to permutation"
        output["prim_pred"].attrs["description"] = "ABC primitive IDs: 0 closed spline, 1 plane, 2 open spline, 3 cone, 4 cylinder, 5 sphere"
        output["type_prob_pred"].attrs["description"] = "Ten original model classes, before ABC validation remapping"
        output["T_param_pred"].attrs["description"] = "Sphere [0:4], plane [4:8], cylinder [8:15], cone [15:22]; input coordinate system"
    write_ply(ply_paths[0], dataset.points, predictions["labels_pred"])
    write_ply(ply_paths[1], dataset.points, predictions["prim_pred"])
    types, counts = np.unique(predictions["prim_pred"], return_counts=True)
    summary = {
        **metadata, "source_file": str(dataset.path), "source_cloud": dataset.cloud_path,
        "point_count": count, "predicted_point_count": count,
        "instance_count": int(len(np.unique(predictions["labels_pred"]))),
        "primitive_counts": {str(int(kind)): int(n) for kind, n in zip(types, counts)},
        "prediction_file": str(h5_path.resolve()),
        "ply_files": [str(path.resolve()) for path in ply_paths],
    }
    with open(summary_path, "x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
        output.write("\n")
    return h5_path, summary
