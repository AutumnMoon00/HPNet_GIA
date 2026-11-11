"""Rename the existing HDF5 field superpoints to superpoints_1."""

import argparse
from pathlib import Path

import h5py


INPUT_DIR = Path(
    "/data/users/wm-sharath/fib/01-BridgePatchDataset.geon/org_copy_superpoints"
)
OLD_FIELD = "superpoints"
NEW_FIELD = "superpoints_1"


def process_file(file_path, *, dry_run=False):
    """Rename field groups in place without loading or recomputing point data."""
    file_path = Path(file_path)
    with h5py.File(file_path, "r" if dry_run else "r+") as document_file:
        pending = []
        cloud_count = 0
        for name, scene_item in document_file["document"].items():
            if not isinstance(scene_item, h5py.Group):
                continue
            type_id = scene_item.attrs.get("type_id", scene_item.attrs.get("type"))
            if isinstance(type_id, bytes):
                type_id = type_id.decode("utf-8")
            if type_id != "PointCloudData":
                continue

            cloud_count += 1
            fields = scene_item.get("fields")
            if fields is None:
                print(f"  {name}: no fields; skipping.", flush=True)
                continue
            if OLD_FIELD in fields and NEW_FIELD in fields:
                raise ValueError(
                    f"{file_path.name}, cloud {name!r}: both {OLD_FIELD!r} and "
                    f"{NEW_FIELD!r} exist; refusing to overwrite either field."
                )
            if OLD_FIELD not in fields:
                reason = (
                    f"{NEW_FIELD} already exists" if NEW_FIELD in fields
                    else f"no {OLD_FIELD} field"
                )
                print(f"  {name}: {reason}; skipping.", flush=True)
                continue
            pending.append((name, fields))

        if cloud_count == 0:
            raise ValueError(f"No point clouds found in {file_path}.")

        # Check every cloud for name conflicts before changing this file.
        # Moving the group preserves its data, attributes, and compression.
        for name, fields in pending:
            if not dry_run:
                fields.move(OLD_FIELD, NEW_FIELD)
            action = "Would rename" if dry_run else "Renamed"
            print(f"  {name}: {action} {OLD_FIELD} -> {NEW_FIELD}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report planned renames without modifying files.",
    )
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}.")
    file_paths = sorted(
        path for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
    )
    if not file_paths:
        parser.error(f"No .h5 or .hdf5 files found in {args.input_dir}.")
    for file_path in file_paths:
        print(f"Processing: {file_path}", flush=True)
        process_file(file_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
