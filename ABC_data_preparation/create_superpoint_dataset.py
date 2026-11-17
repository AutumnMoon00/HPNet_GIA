"""Convert point-aligned ABCParts GEON documents to centroid-level documents.

Run with the Python environment that contains GEON, NumPy and h5py. This
script only reads existing graph-cut assignments; it never segments points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
from geon.data.document import Document
from geon.data.pointcloud import FieldType, PointCloudData, SemanticClass, SemanticSchema


DEFAULT_INPUT = Path("/data/users/wm-sharath/fib/ABC_final_geon_superpoints_combined")
VERSION = "1.0"
REGULARIZATION = "0.025"
SUPERPOINT_FIELD = f"superpoints_reg_{REGULARIZATION}"
PRIMITIVE_NAMES = {
    0: "Closed B-spline",
    1: "Plane",
    2: "Open B-spline",
    3: "Cone",
    4: "Cylinder",
    5: "Sphere",
    6: "Closed B-spline (6)",
    7: "Closed B-spline (7)",
    8: "Open B-spline (8)",
    9: "Closed B-spline (9)",
}


def _integer_vector(data: np.ndarray, name: str, n: int) -> np.ndarray:
    if data.shape not in ((n,), (n, 1)) or not np.issubdtype(data.dtype, np.integer):
        raise ValueError(f"{name}: expected an integer array of shape ({n},) or ({n}, 1)")
    return data.reshape(-1).astype(np.int64, copy=False)


def _read_source(path: Path, original_dir: Path | None) -> dict:
    with h5py.File(path, "r") as f:
        children = [v for v in f["document"].values() if isinstance(v, h5py.Group) and v.attrs.get("type_id") == "PointCloudData"]
        if len(children) != 1:
            raise ValueError(f"{path}: expected exactly one GEON PointCloudData")
        cloud = children[0]
        fields = cloud["fields"]
        required = ("normals", "labels", "prim", "T_param", SUPERPOINT_FIELD)
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError(f"{path}: missing fields {missing}")
        points = cloud["points"][()]
        normals = fields["normals/data"][()]
        n = len(points)
        labels = _integer_vector(fields["labels/data"][()], "labels", n)
        types = _integer_vector(fields["prim/data"][()], "prim", n)
        source_ids = _integer_vector(fields[f"{SUPERPOINT_FIELD}/data"][()], SUPERPOINT_FIELD, n)
        params = fields["T_param/data"][()]
        settings = dict(fields[SUPERPOINT_FIELD].attrs)
        global_settings = f.attrs.get("superpoint_settings", "")
        original_attr = f.attrs.get("source_abc_file", "")

    if points.shape != (n, 3) or n == 0 or normals.shape != (n, 3):
        raise ValueError(f"{path}: invalid point or normal dimensions")
    if params.ndim != 2 or params.shape[0] != n or params.shape[1] == 0:
        raise ValueError(f"{path}: invalid T_param shape {params.shape}")
    if not all(np.isfinite(a).all() for a in (points, normals, params)):
        raise ValueError(f"{path}: nonfinite input")
    if np.any(source_ids < 0):
        raise ValueError(f"{path}: negative graph-cut superpoint ID")
    if np.any(source_ids > np.iinfo(np.int32).max):
        raise ValueError(f"{path}: graph-cut superpoint ID exceeds int32 range")
    if not np.all(np.isin(types, [-1, *PRIMITIVE_NAMES])):
        raise ValueError(f"{path}: unknown HPNet primitive type")
    if float(settings.get("regularization", np.nan)) != 0.025:
        raise ValueError(f"{path}: wrong graph-cut regularization")

    original_path = ((original_dir / (path.name.replace("_geon_superpoints.h5", ".h5")))
                     if original_dir else Path(str(original_attr)))
    if not original_path.is_file():
        raise FileNotFoundError(f"{path}: original ABCParts file unavailable: {original_path}; supply --original-dir")
    with h5py.File(original_path, "r") as original:
        if not np.array_equal(points, np.asarray(original["points"], dtype=points.dtype)):
            raise ValueError(f"{path}: point order differs from original ABCParts file")
        for field, value in (("labels", labels), ("prim", types), ("T_param", params)):
            other = original[field][()]
            if not np.array_equal(value, other.reshape(value.shape)):
                raise ValueError(f"{path}: {field} differs from original ABCParts file")
        original_normals = original["normals"][()]
        length = np.linalg.norm(original_normals, axis=1)
        if np.any(length == 0):
            raise ValueError(f"{path}: original ABCParts normals contain zero vectors")
        expected_normals = (original_normals / length[:, None]).astype(normals.dtype)
        if not np.allclose(normals, expected_normals, atol=2e-6, rtol=2e-6):
            raise ValueError(f"{path}: normals differ from original ABCParts point order")

    return dict(points=points, normals=normals, labels=labels, types=types,
                source_ids=source_ids, params=params, settings=settings,
                global_settings=str(global_settings), original_path=original_path)


def _majority(values: np.ndarray) -> tuple[int, int]:
    ids, counts = np.unique(values, return_counts=True)
    winner = int(np.flatnonzero(counts == counts.max())[0])
    return int(ids[winner]), int(counts[winner])


def _aggregate(source: dict) -> dict:
    points = source["points"]
    normals = source["normals"]
    source_ids, point2sp, counts = np.unique(source["source_ids"], return_inverse=True, return_counts=True)
    n, m = len(points), len(source_ids)
    if m > np.iinfo(np.int32).max or n > np.iinfo(np.int32).max:
        raise ValueError("int32 mapping or count capacity exceeded")
    point2sp = point2sp.astype(np.int32)
    centroids = np.zeros((m, 3), dtype=np.float64)
    means = np.zeros((m, 3), dtype=np.float64)
    np.add.at(centroids, point2sp, points)
    np.add.at(means, point2sp, normals)
    centroids = (centroids / counts[:, None]).astype(np.float32)
    means /= counts[:, None]
    lengths = np.linalg.norm(means, axis=1)
    valid = lengths > 1e-8
    merged_normals = np.zeros((m, 3), dtype=np.float32)
    merged_normals[valid] = (means[valid] / lengths[valid, None]).astype(np.float32)
    instance = np.empty(m, dtype=np.int32)
    primitive = np.empty(m, dtype=np.int32)
    purity = np.empty(m, dtype=np.float32)
    order = np.argsort(point2sp, kind="stable")
    bounds = np.r_[0, np.cumsum(counts)]
    for sp in range(m):
        ix = order[bounds[sp]:bounds[sp + 1]]
        label, majority_count = _majority(source["labels"][ix])
        if not np.iinfo(np.int32).min <= label <= np.iinfo(np.int32).max:
            raise ValueError("instance ID exceeds GEON int32 range")
        instance[sp] = label
        primitive[sp], _ = _majority(source["types"][ix][source["labels"][ix] == label])
        purity[sp] = np.float32(majority_count / counts[sp])
    return dict(points=centroids, normals=merged_normals, instance=instance,
                primitive=primitive, count=counts.astype(np.int32), purity=purity,
                valid=valid.astype(np.uint8), point2sp=point2sp,
                source_ids=source_ids.astype(np.int32))


def _schema() -> SemanticSchema:
    colors = ((204, 204, 204), (191, 54, 54), (63, 115, 191), (202, 134, 37),
              (112, 77, 173), (55, 151, 111), (191, 86, 155), (109, 120, 42),
              (46, 151, 166), (162, 108, 67), (93, 99, 177))
    classes = [SemanticClass(-1, "_unlabeled", colors[0])]
    classes += [SemanticClass(i, name, colors[i + 1]) for i, name in PRIMITIVE_NAMES.items()]
    return SemanticSchema(name="ABCParts_HPNet_primitive_types", semantic_classes=classes)


def process_file(path: Path, output: Path, original_dir: Path | None = None,
                 overwrite: bool = False) -> dict:
    if output.exists() and not overwrite:
        return {"status": "skipped", "file": output.name}
    if path.resolve() == output.resolve():
        raise ValueError("Output path equals input path")
    source = _read_source(path, original_dir)
    merged = _aggregate(source)
    cloud = PointCloudData(merged["points"])
    cloud.add_field("normals", merged["normals"], FieldType.NORMAL)
    cloud.add_field("primitive_instance", merged["instance"], FieldType.INSTANCE)
    cloud.add_field("primitive_type", merged["primitive"], FieldType.SEMANTIC, schema=_schema())
    for name, key in (("sp_count", "count"), ("sp_purity", "purity"), ("sp_normal_valid", "valid")):
        cloud.add_field(name, merged[key][:, None], FieldType.SCALAR)
    document = Document(name=f"ABC_{path.stem.removesuffix('_geon_superpoints')}_superpoint_level")
    document.add_data(cloud)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.stem}_", suffix=".h5", dir=output.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        document.save_hdf5(temp)
        with h5py.File(temp, "a") as f:
            extra = f.create_group("abcparts")
            extra.create_dataset("point2sp", data=merged["point2sp"], compression="gzip")
            extra.create_dataset("source_superpoint_id", data=merged["source_ids"], compression="gzip")
            extra.create_dataset("T_param", data=source["params"], compression="gzip")
            extra.attrs["source_abc_file"] = str(source["original_path"].resolve())
            extra.attrs["source_geon_file"] = str(path.resolve())
            extra.attrs["original_point_count"] = np.int64(len(source["points"]))
            extra.attrs["superpoint_count"] = np.int64(len(merged["points"]))
            extra.attrs["point_ordering"] = "row i matches original ABCParts array index i"
            extra.attrs["preprocessing_version"] = VERSION
            extra.attrs["source_points_float32_sha256"] = hashlib.sha256(np.ascontiguousarray(source["points"], dtype=np.float32).tobytes()).hexdigest()
            extra.attrs["graph_cut_settings_json"] = source["global_settings"]
            extra.attrs["graph_cut_field_attrs_json"] = json.dumps({k: v.item() if isinstance(v, np.generic) else str(v) for k, v in source["settings"].items()}, sort_keys=True)
            extra.attrs["graph_cut_regularization"] = np.float32(0.025)
        from validate_superpoint_dataset import validate_file
        stats = validate_file(temp, source_path=path, original_path=source["original_path"])
        if output.exists() and not overwrite:
            return {"status": "skipped", "file": output.name}
        os.replace(temp, output)
        return {"status": "written", "file": output.name, **stats}
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path, help="ABCParts original files; defaults to source_abc_file metadata")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-objects", type=int, help="Process at most this many files in sorted order")
    args = parser.parse_args()
    if args.max_objects is not None and args.max_objects < 1:
        parser.error("--max-objects must be positive")
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    files = sorted(args.input_dir.glob("*_geon_superpoints.h5"))
    if not files:
        parser.error("No *_geon_superpoints.h5 files found")
    if args.max_objects:
        files = files[:args.max_objects]
    for i, path in enumerate(files, 1):
        output = args.output_dir / path.name.replace("_geon_superpoints.h5", "_superpoint_level.h5")
        result = process_file(path, output, args.original_dir, args.overwrite)
        print(f"[{i}/{len(files)}] {json.dumps(result, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
