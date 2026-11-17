"""Validate every centroid, label, normal and mapping in ABCParts superpoint files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from geon.data.document import Document
from geon.data.pointcloud import FieldType, SemanticSegmentation


def _flat(array: np.ndarray, n: int, name: str) -> np.ndarray:
    if array.shape not in ((n,), (n, 1)):
        raise ValueError(f"{name}: invalid shape {array.shape}")
    return array.reshape(-1)


def _majority(values: np.ndarray) -> tuple[int, int]:
    ids, counts = np.unique(values, return_counts=True)
    ix = np.flatnonzero(counts == counts.max())[0]
    return int(ids[ix]), int(counts[ix])


def validate_file(path: Path, source_path: Path | None = None,
                  original_path: Path | None = None) -> dict:
    path = Path(path)
    with h5py.File(path, "r") as f:
        extra = f["abcparts"]
        n = int(extra.attrs["original_point_count"])
        m = int(extra.attrs["superpoint_count"])
        point2sp = extra["point2sp"][()]
        source_ids = extra["source_superpoint_id"][()]
        params = extra["T_param"][()]
        if point2sp.dtype != np.int32 or point2sp.shape != (n,):
            raise ValueError("point2sp must be int32 with one row per original point")
        if source_ids.dtype != np.int32 or source_ids.shape != (m,):
            raise ValueError("source_superpoint_id must be int32 with one row per superpoint")
        if np.any(point2sp < 0) or np.any(point2sp >= m):
            raise ValueError("point2sp contains out-of-range indices")
        counts = np.bincount(point2sp, minlength=m)
        if np.any(counts == 0) or len(counts) != m:
            raise ValueError("superpoint IDs are not consecutive and populated")
        if params.ndim != 2 or len(params) != n or not np.isfinite(params).all():
            raise ValueError("original primitive parameters are missing or nonfinite")
        if len(np.unique(source_ids)) != m or not np.array_equal(source_ids, np.sort(source_ids)):
            raise ValueError("source superpoint IDs are not unique and sorted")
        source_path = source_path or Path(str(extra.attrs["source_geon_file"]))
        original_path = original_path or Path(str(extra.attrs["source_abc_file"]))
        expected_hash = str(extra.attrs["source_points_float32_sha256"])
        if str(extra.attrs["point_ordering"]) != "row i matches original ABCParts array index i":
            raise ValueError("unexpected original point ordering convention")
        if float(extra.attrs["graph_cut_regularization"]) != np.float32(0.025):
            raise ValueError("wrong graph-cut regularization")

    doc = Document.load_hdf5(path)
    if len(doc.scene_items) != 1:
        raise ValueError("expected one GEON point cloud")
    cloud = next(iter(doc.scene_items.values()))
    points = np.asarray(cloud.points)
    if points.shape != (m, 3) or points.dtype != np.float32:
        raise ValueError("centroids must be float32 (M, 3)")
    wanted = {"normals": (FieldType.NORMAL, (m, 3), np.float32),
              "primitive_instance": (FieldType.INSTANCE, (m, 1), np.int32),
              "primitive_type": (FieldType.SEMANTIC, (m,), np.int32),
              "sp_count": (FieldType.SCALAR, (m, 1), np.int32),
              "sp_purity": (FieldType.SCALAR, (m, 1), np.float32),
              "sp_normal_valid": (FieldType.SCALAR, (m, 1), np.uint8)}
    if set(cloud.field_names) != set(wanted):
        raise ValueError(f"unexpected GEON fields: {cloud.field_names}")
    fields = {}
    for name, (field_type, shape, dtype) in wanted.items():
        field = cloud.get_fields(name)[0]
        if field.field_type != field_type or field.data.shape != shape or field.data.dtype != dtype:
            raise ValueError(f"{name}: GEON field type, shape or dtype mismatch")
        if not np.isfinite(field.data).all():
            raise ValueError(f"{name}: nonfinite values")
        fields[name] = field
    if not np.isfinite(points).all():
        raise ValueError("nonfinite centroids")
    primitive = fields["primitive_type"]
    if not isinstance(primitive, SemanticSegmentation):
        raise ValueError("primitive_type was not recognized as GEON semantic segmentation")
    from create_superpoint_dataset import PRIMITIVE_NAMES
    schema = {item.id: item.name for item in primitive.schema.semantic_classes}
    if schema != {-1: "_unlabeled", **PRIMITIVE_NAMES}:
        raise ValueError("semantic schema does not match original HPNet primitive IDs")
    if not np.all(np.isin(primitive.data, list(schema))):
        raise ValueError("primitive labels absent from schema")
    if not np.array_equal(fields["sp_count"].data[:, 0], counts):
        raise ValueError("sp_count differs from point2sp")
    purity = fields["sp_purity"].data[:, 0]
    if np.any((purity < 0) | (purity > 1)):
        raise ValueError("purity outside [0, 1]")
    valid = fields["sp_normal_valid"].data[:, 0]
    if np.any((valid != 0) & (valid != 1)):
        raise ValueError("sp_normal_valid must be binary")
    normals = fields["normals"].data
    lengths = np.linalg.norm(normals.astype(np.float64), axis=1)
    if not np.allclose(lengths[valid == 1], 1, atol=2e-6) or np.any(normals[valid == 0] != 0):
        raise ValueError("invalid aggregated normal lengths")

    if not source_path.is_file() or not original_path.is_file():
        raise FileNotFoundError("source GEON or original ABCParts file required to verify point order")
    with h5py.File(source_path, "r") as source, h5py.File(original_path, "r") as original:
        source_clouds = [v for v in source["document"].values() if isinstance(v, h5py.Group) and v.attrs.get("type_id") == "PointCloudData"]
        if len(source_clouds) != 1:
            raise ValueError("source GEON file must have one point cloud")
        src = source_clouds[0]
        src_points = src["points"][()]
        src_normals = src["fields/normals/data"][()]
        src_labels = _flat(src["fields/labels/data"][()], n, "source labels")
        src_types = _flat(src["fields/prim/data"][()], n, "source primitive types")
        src_ids = _flat(src["fields/superpoints_reg_0.025/data"][()], n, "source superpoints")
        if src_points.shape != (n, 3) or src_normals.shape != (n, 3):
            raise ValueError("source geometry dimensions changed")
        if hashlib.sha256(np.ascontiguousarray(src_points, dtype=np.float32).tobytes()).hexdigest() != expected_hash:
            raise ValueError("source point order/hash changed")
        if not np.array_equal(np.asarray(original["points"], dtype=np.float32), src_points):
            raise ValueError("source GEON point order differs from original ABCParts")
        original_normals = original["normals"][()]
        lengths_original = np.linalg.norm(original_normals, axis=1)
        if np.any(lengths_original == 0) or not np.allclose(
            src_normals, (original_normals / lengths_original[:, None]).astype(src_normals.dtype),
            atol=2e-6, rtol=2e-6
        ):
            raise ValueError("source GEON normals differ from original ABCParts point order")
        for name, values in (("labels", src_labels), ("prim", src_types), ("T_param", params)):
            if not np.array_equal(values, original[name][()].reshape(values.shape)):
                raise ValueError(f"{name}: original ABCParts data differs")
        if not np.array_equal(params, src["fields/T_param/data"][()]):
            raise ValueError("preserved T_param differs from source")
        if not np.array_equal(src_ids, source_ids[point2sp]):
            raise ValueError("point2sp does not reproduce original graph-cut IDs")
        reconstructed_centroids = np.zeros((m, 3), dtype=np.float64)
        reconstructed_normals = np.zeros((m, 3), dtype=np.float64)
        np.add.at(reconstructed_centroids, point2sp, src_points)
        np.add.at(reconstructed_normals, point2sp, src_normals)
        reconstructed_centroids = (reconstructed_centroids / counts[:, None]).astype(np.float32)
        reconstructed_normals /= counts[:, None]
        norm_lengths = np.linalg.norm(reconstructed_normals, axis=1)
        expected_valid = norm_lengths > 1e-8
        if not np.array_equal(points, reconstructed_centroids) or not np.array_equal(valid, expected_valid.astype(np.uint8)):
            raise ValueError("centroid or normal validity differs from source")
        expected_normals = np.zeros((m, 3), dtype=np.float32)
        expected_normals[expected_valid] = (reconstructed_normals[expected_valid] / norm_lengths[expected_valid, None]).astype(np.float32)
        if not np.allclose(normals, expected_normals, atol=2e-6):
            raise ValueError("aggregated normal differs from source")
        order = np.argsort(point2sp, kind="stable")
        bounds = np.r_[0, np.cumsum(counts)]
        for sp in range(m):
            ix = order[bounds[sp]:bounds[sp + 1]]
            label, count = _majority(src_labels[ix])
            type_id, _ = _majority(src_types[ix][src_labels[ix] == label])
            if fields["primitive_instance"].data[sp, 0] != label or primitive.data[sp] != type_id:
                raise ValueError(f"superpoint {sp}: majority label/type differs from source")
            if not np.isclose(purity[sp], count / counts[sp], atol=1e-7):
                raise ValueError(f"superpoint {sp}: purity differs from source")
    return {"superpoints": m, "points": n, "mean_size": float(np.mean(counts)),
            "mean_purity": float(np.mean(purity)),
            "point_weighted_purity": float(np.average(purity, weights=counts))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="One output file or directory of *_superpoint_level.h5 files")
    args = parser.parse_args()
    files = sorted(args.path.glob("*_superpoint_level.h5")) if args.path.is_dir() else [args.path]
    if not files:
        parser.error("no output files found")
    results = []
    for path in files:
        result = validate_file(path)
        results.append(result)
        print(json.dumps({"file": path.name, **result}, sort_keys=True), flush=True)
    total_points = sum(r["points"] for r in results)
    total_superpoints = sum(r["superpoints"] for r in results)
    print(json.dumps({"files": len(results), "superpoints": total_superpoints,
                      "mean_superpoints_per_file": total_superpoints / len(results),
                      "mean_size": total_points / total_superpoints,
                      "mean_purity": sum(r["mean_purity"] * r["superpoints"] for r in results) / total_superpoints,
                      "point_weighted_purity": sum(r["point_weighted_purity"] * r["points"] for r in results) / total_points}, sort_keys=True))


if __name__ == "__main__":
    main()
