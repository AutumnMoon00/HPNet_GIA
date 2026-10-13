"""Export per-file primitive instance counts for the ABC dataset.

Run with the defaults:
    python ABC_superpoints_code/bspline_lookup.py

Or select an input directory and output CSV:
    python ABC_superpoints_code/bspline_lookup.py --input-dir D:/GIA/ABC_final \
        --output ABC_superpoints_code/primitive_instance_counts.csv

Counts use unique nonnegative instance IDs in ``labels``, not point counts.
Negative instance IDs are unassigned/background and are excluded. No minimum
point-count filter is applied. Primitive categories follow ABCViewer.py.
"""

import argparse
import csv
from pathlib import Path
import tempfile

import h5py
import numpy as np


DEFAULT_INPUT_DIR = Path(r"D:\GIA\ABC_final")
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("primitive_instance_counts.csv")

# Primitive IDs from my_code/ABCViewer.py, grouped into six categories.
PRIMITIVE_CATEGORIES = {
    "plane": (1,),
    "cylinder": (4,),
    "cone": (3,),
    "sphere": (5,),
    "closed_bspline": (0, 6, 7, 9),
    "open_bspline": (2, 8),
}
CATEGORY_NAMES = tuple(PRIMITIVE_CATEGORIES)
TYPE_TO_CATEGORY = np.empty(10, dtype=np.int64)
for category_index, primitive_ids in enumerate(PRIMITIVE_CATEGORIES.values()):
    TYPE_TO_CATEGORY[list(primitive_ids)] = category_index


def count_instances(file_path):
    """Count each labeled instance once, after grouping B-spline type IDs.

    Reject unknown primitive types and instances spanning different categories
    so that ambiguous data cannot silently produce misleading counts.
    """
    file_path = Path(file_path)
    with h5py.File(file_path, "r") as source:
        labels = source["labels"][()].reshape(-1)
        primitives = source["prim"][()].reshape(-1)

    if labels.shape != primitives.shape:
        raise ValueError(f"{file_path}: labels and prim must have equal lengths.")
    if not all(np.issubdtype(values.dtype, np.integer)
               for values in (labels, primitives)):
        raise ValueError(f"{file_path}: labels and prim must contain integer IDs.")

    assigned = labels >= 0
    labels = labels[assigned]
    primitives = primitives[assigned]
    invalid_types = np.unique(primitives[(primitives < 0) | (primitives >= 10)])
    if invalid_types.size:
        raise ValueError(f"{file_path}: unknown primitive IDs {invalid_types.tolist()}.")

    categories = TYPE_TO_CATEGORY[primitives]
    instance_ids, first_indices, inverse = np.unique(
        labels, return_index=True, return_inverse=True
    )
    instance_categories = categories[first_indices]
    inconsistent = categories != instance_categories[inverse]
    if np.any(inconsistent):
        conflicting_ids = np.unique(labels[inconsistent])
        raise ValueError(
            f"{file_path}: instances span multiple primitive categories: "
            f"{conflicting_ids.tolist()}."
        )

    counts = np.bincount(instance_categories, minlength=len(CATEGORY_NAMES))
    assert int(counts.sum()) == len(instance_ids)
    return dict(zip(CATEGORY_NAMES, map(int, counts)))


def export_counts(input_dir=DEFAULT_INPUT_DIR, output_path=DEFAULT_OUTPUT):
    """Scan HDF5 files recursively and write one CSV row per input file."""
    input_dir = Path(input_dir).resolve()
    output_path = Path(output_path).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {input_dir}")
    files = sorted(
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
    )
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    totals = dict.fromkeys(CATEGORY_NAMES, 0)
    temporary_path = None
    try:
        # Replace the final CSV only after every file has been counted.
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=output_path.parent,
            prefix=output_path.name + ".", suffix=".tmp", delete=False
        ) as output:
            temporary_path = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=("file_name", *CATEGORY_NAMES))
            writer.writeheader()
            for index, file_path in enumerate(files, start=1):
                counts = count_instances(file_path)
                writer.writerow({
                    "file_name": file_path.relative_to(input_dir).as_posix(),
                    **counts,
                })
                for category, count in counts.items():
                    totals[category] += count
                if index % 1000 == 0 or index == len(files):
                    print(f"Processed {index:,}/{len(files):,} files", flush=True)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"CSV saved to: {output_path}")
    print("Total instances: " + ", ".join(
        f"{category}={count:,}" for category, count in totals.items()
    ))
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_counts(args.input_dir, args.output)


if __name__ == "__main__":
    main()
