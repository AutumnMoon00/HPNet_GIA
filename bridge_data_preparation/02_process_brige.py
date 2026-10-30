"""Add superpoints using normals and intensity, excluding unlabelled points."""

import argparse
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np

from geon.algorithms.superpoints import segment_superpoints


DATASET_DIR = Path(
    r"C:\Users\Madanu.S\Downloads\01-BridgePatchDataset.geon\01-BridgePatchDataset.geon"
)
INPUT_DIR = DATASET_DIR / "org_copy"
OUTPUT_DIR = DATASET_DIR / "org_copy_superpoints"
EXCLUDED_CLASS_IDS = (-1, 22)
NORMAL_FIELD = "normals_0.100"
INTENSITY_FIELD = "intensity"


def compute_superpoint_ids(points, class_ids, normals, intensity):
    points = np.asarray(points)
    class_ids = np.asarray(class_ids).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points, got {points.shape}.")
    if len(class_ids) != len(points):
        raise ValueError("The 'class' field must contain one ID per point.")
    normals = np.asarray(normals, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32)
    if intensity.ndim == 1:
        intensity = intensity[:, None]
    if normals.shape != (len(points), 3):
        raise ValueError(f"Expected (N, 3) normals, got {normals.shape}.")
    if intensity.shape != (len(points), 1):
        raise ValueError(f"Expected (N, 1) intensity, got {intensity.shape}.")

    valid_mask = ~np.isin(class_ids, EXCLUDED_CLASS_IDS)
    superpoint_ids = np.full(len(points), -1, dtype=np.int32)
    valid_count = int(valid_mask.sum())

    if valid_count == 1:
        superpoint_ids[valid_mask] = 0
    elif valid_count > 1:
        # Coordinates and both feature fields use the same labelled-point mask.
        # Feature columns are [normal_x, normal_y, normal_z, intensity].
        features = np.concatenate((normals[valid_mask], intensity[valid_mask]), axis=1)
        if not np.isfinite(features).all():
            raise ValueError("Normals and intensity must be finite for labelled points.")
        labels, _ = segment_superpoints(
            points[valid_mask],
            features=features,
            k_neighbors=min(10, valid_count - 1),
            regularization=0.05,
            spatial_weight=1.0,
            cutoff=10,
            iterations=10,
            parallel=True,
        )
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != valid_count or np.any(labels < 0):
            raise ValueError("Segmentation must return a nonnegative ID per labelled point.")
        superpoint_ids[valid_mask] = labels

    return superpoint_ids


def process_file(
    source_path, output_path, *, normal_field=NORMAL_FIELD, intensity_field=INTENSITY_FIELD
):
    """Copy an HDF5 file and append only the new superpoints fields."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("The output must be separate from the source file.")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}_",
        suffix=output_path.suffix,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        # A file copy preserves every original dataset, attribute, schema,
        # scene item, and compression setting without reserializing them.
        shutil.copyfile(source_path, temporary_path)
        with h5py.File(temporary_path, "r+") as output:
            document = output["document"]
            cloud_count = 0
            for name, scene_item in document.items():
                if not isinstance(scene_item, h5py.Group):
                    continue
                type_id = scene_item.attrs.get("type_id", scene_item.attrs.get("type"))
                if isinstance(type_id, bytes):
                    type_id = type_id.decode("utf-8")
                if type_id != "PointCloudData":
                    continue

                cloud_count += 1
                fields = scene_item.get("fields")
                if fields is None or "class/data" not in fields:
                    raise ValueError(f"{source_path.name}, cloud {name!r}: no 'class' field.")
                if "superpoints" in fields:
                    raise ValueError(
                        f"{source_path.name}, cloud {name!r}: 'superpoints' already exists."
                    )
                for feature_field in (normal_field, intensity_field):
                    if f"{feature_field}/data" not in fields:
                        raise ValueError(
                            f"{source_path.name}, cloud {name!r}: "
                            f"no feature field {feature_field!r}. "
                            f"Available fields: {list(fields)}"
                        )

                class_ids = fields["class/data"][()]
                superpoint_ids = compute_superpoint_ids(
                    scene_item["points"][()], class_ids,
                    fields[f"{normal_field}/data"][()],
                    fields[f"{intensity_field}/data"][()],
                )

                # Geon's instance field format; signed int32 retains -1.
                superpoints = fields.create_group("superpoints")
                superpoints.attrs["field_type"] = "INSTANCE"
                superpoints.create_dataset("data", data=superpoint_ids[:, None])

                excluded_count = int(np.isin(class_ids, EXCLUDED_CLASS_IDS).sum())
                superpoint_count = len(np.unique(superpoint_ids[superpoint_ids >= 0]))
                print(
                    f"  {name}: {len(superpoint_ids):,} points, "
                    f"{excluded_count:,} excluded, {superpoint_count:,} superpoints",
                    flush=True,
                )

            if cloud_count == 0:
                raise ValueError(f"No point clouds found in {source_path}.")

        # Publish only a fully written file. Existing outputs are not replaced.
        if output_path.exists():
            raise FileExistsError(f"Output already exists: {output_path}")
        temporary_path.rename(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Saved: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--normal-field", default=NORMAL_FIELD,
        help="Existing normals field to use as features (default: %(default)s).",
    )
    parser.add_argument(
        "--intensity-field", default=INTENSITY_FIELD,
        help="Existing intensity field to use as a feature (default: %(default)s).",
    )
    args = parser.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        parser.error("--output-dir must be different from --input-dir.")

    source_paths = sorted(
        path for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
    )
    if not source_paths:
        parser.error(f"No .h5 or .hdf5 files found in {args.input_dir}.")

    for source_path in source_paths:
        print(f"Processing: {source_path}", flush=True)
        process_file(
            source_path, args.output_dir / source_path.name,
            normal_field=args.normal_field, intensity_field=args.intensity_field,
        )


if __name__ == "__main__":
    main()
