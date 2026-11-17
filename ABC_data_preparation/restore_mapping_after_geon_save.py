"""Restore /abcparts after GEON resaves a centroid document.

GEON's Document.save_hdf5 writes only /document, so a GEON resave drops the
custom root group. This tool copies that group only when the centroid cloud
and all its fields still match the original output exactly.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np

from validate_superpoint_dataset import validate_file


def restore(original: Path, resaved: Path, output: Path, overwrite: bool = False) -> None:
    if not original.is_file() or not resaved.is_file():
        raise FileNotFoundError("original and resaved GEON files must both exist")
    if output.resolve() == original.resolve():
        raise ValueError("cannot replace the original mapping source")
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    with h5py.File(original, "r") as a, h5py.File(resaved, "r") as b:
        if "abcparts" not in a or "document" not in b:
            raise ValueError("original /abcparts or resaved /document is missing")
        for name in ("points", "fields/normals/data", "fields/primitive_instance/data",
                     "fields/primitive_type/data", "fields/sp_count/data",
                     "fields/sp_purity/data", "fields/sp_normal_valid/data"):
            a_clouds = [v for v in a["document"].values() if isinstance(v, h5py.Group)]
            b_clouds = [v for v in b["document"].values() if isinstance(v, h5py.Group)]
            if len(a_clouds) != 1 or len(b_clouds) != 1 or not np.array_equal(a_clouds[0][name][()], b_clouds[0][name][()]):
                raise ValueError(f"GEON resave changed {name}; mapping cannot be restored safely")
        if "abcparts" in b:
            raise ValueError("resaved document already has /abcparts")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.stem}_", suffix=".h5", dir=output.parent)
    os.close(fd)
    temp = Path(temporary)
    try:
        shutil.copy2(resaved, temp)
        with h5py.File(original, "r") as a, h5py.File(temp, "a") as b:
            a.copy("abcparts", b)
        validate_file(temp)
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True, help="Original superpoint-level output with /abcparts")
    parser.add_argument("--resaved", type=Path, required=True, help="File saved through GEON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    restore(args.original, args.resaved, args.output, args.overwrite)
    print(f"Restored and validated: {args.output}")


if __name__ == "__main__":
    main()
