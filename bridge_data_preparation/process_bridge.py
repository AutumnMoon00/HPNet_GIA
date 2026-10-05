"""Normalize a bridge cloud and save it as a geon HDF5 document.

The output retains document/<cloud ID>/points, fields/<field name>/data,
and document/telemetry, including source field types and semantic schemas.
Points use HPNet's ABC coordinate normalization; normals are normalized to
unit length and rotated with the points. All other source fields are retained.
The normalization transform is stored in the document's metadata.
This geon document is not directly compatible with HPNet's flat ABCDataset.

All merged points are retained. HPNet currently selects at most 7,000 of them
inside models/dgcnn.py, so complete prediction coverage needs an inference
routine that handles that selection separately.
"""

import argparse
from pathlib import Path

import numpy as np

from geon.data.document import Document
from geon.data.pointcloud import PointCloudData


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    REPO_ROOT / "bridge_data_with_primitives" / "superpoints_merged"
    / "BW-0000_SC-00_clipped_superpoints_merged.h5"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "bridge_data_with_primitives" / "sp_merged_normalized"
# Match the denominator epsilon in utils/process_abc.py.
EPS = np.finfo(np.float32).eps


def rotation_matrix_a_to_b(a, b):
    """Return a proper rotation mapping unit vector a to b, including poles."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cross = np.cross(a, b)
    sine = np.linalg.norm(cross)
    cosine = np.clip(np.dot(a, b), -1.0, 1.0)

    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        # A half turn around an axis perpendicular to a maps a to -a.
        axis = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    axis = cross / sine
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


def normalize_points(points):
    """Center, align the minor PCA axis to X, and divide by the largest span.

    Row-vector convention:
        normalized = (original - center) @ rotation.T / scale
        original = (normalized * scale) @ rotation + center
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Expected nonempty Nx3 points, got {points.shape}.")
    if not np.isfinite(points).all():
        raise ValueError("Point positions contain NaN or infinite values.")

    center = points.mean(axis=0)
    centered = points - center
    # eigh is appropriate for the symmetric scatter matrix used by ABC's PCA.
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    minor_axis = eigenvectors[:, np.argmin(eigenvalues)]
    # PCA eigenvector signs are arbitrary; fix the sign for reproducible output.
    if minor_axis[np.argmax(np.abs(minor_axis))] < 0:
        minor_axis = -minor_axis
    rotation = rotation_matrix_a_to_b(minor_axis, [1.0, 0.0, 0.0])
    aligned = centered @ rotation.T
    span = np.ptp(aligned, axis=0)
    if span.max() <= np.finfo(np.float64).eps:
        raise ValueError("The point cloud has zero spatial extent.")
    scale = float(span.max() + EPS)
    return aligned / scale, center, rotation, scale


def normalize_normals(normals, rotation, point_count):
    """Restore unit length after superpoint averaging, then rotate like points."""
    normals = np.asarray(normals, dtype=np.float64)
    if normals.shape != (point_count, 3):
        raise ValueError(f"Expected normals of shape {(point_count, 3)}, got {normals.shape}.")
    lengths = np.linalg.norm(normals, axis=1)
    invalid = ~np.isfinite(normals).all(axis=1) | (lengths <= 1e-12)
    if invalid.any():
        raise ValueError(
            f"Found {invalid.sum()} invalid or zero normals; compute valid surface "
            "normals before preparing inference inputs."
        )
    # Translation and uniform scaling do not change normal directions.
    # Keep the supplied signs; arbitrary flips would change HPNet's normal input.
    return (normals / lengths[:, None]) @ rotation.T


def prepare_bridge(input_path, output_dir):
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_path = output_dir / f"{input_path.stem}_normalized_scaled.h5"
    if output_path == input_path:
        raise ValueError("The output must be separate from the source geon file.")

    doc = Document.load_hdf5(str(input_path))
    clouds = [(name, item) for name, item in doc.scene_items.items()
              if isinstance(item, PointCloudData)]
    if len(clouds) != 1:
        raise ValueError(f"Expected exactly one point cloud, found {len(clouds)}.")
    cloud_name, pcd = clouds[0]
    points = np.asarray(pcd.points, dtype=np.float64)
    normalized_points, center, rotation, scale = normalize_points(points)
    # DGCNN's first neighborhood uses 80 neighbors.
    if len(points) < 80:
        raise ValueError("HPNet's DGCNN requires at least 80 input points.")

    fields = pcd.get_fields()
    normal_fields = [field for field in fields if field.name == "normals"]
    if len(normal_fields) != 1:
        raise ValueError("Expected exactly one 'normals' field in the point cloud.")
    normalized_normals = normalize_normals(normal_fields[0].data, rotation, len(points))
    for field in fields:
        if np.asarray(field.data).ndim == 0 or len(field.data) != len(points):
            raise ValueError(f"Field {field.name!r} does not match the point count.")

    # Update the loaded objects so geon retains field types, semantic schemas,
    # color maps, the cloud ID, document metadata, and telemetry.
    pcd.points = normalized_points.astype(np.float32)
    normal_fields[0].data = normalized_normals.astype(np.float32)
    doc.meta.update({
        "source_input_file": str(input_path),
        "source_cloud_name": cloud_name,
        "source_merged_point_count": len(points),
        "normalization_center": center,
        "normalization_rotation": rotation,
        "normalization_scale": scale,
        "normalization_forward": "(original - center) @ rotation.T / scale",
        "normalization_inverse": "(normalized * scale) @ rotation + center",
        "normal_processing": "Normalize source normals to unit length and apply rotation",
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.h5")
    if temporary_path.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_path}")
    doc.save_hdf5(temporary_path)
    temporary_path.replace(output_path)

    print("Point cloud data name:", cloud_name)
    print("Source fields:", pcd.field_names)
    print("Saved points (no sampling):", len(points))
    print("Original mean position (X, Y, Z):", center)
    print("Original span (X, Y, Z):", np.ptp(points, axis=0))
    print("PCA rotation (minor axis -> X):\n", rotation)
    print("Uniform scale divisor:", scale)
    print("Normalized mean position (X, Y, Z):", normalized_points.mean(axis=0))
    print("Normalized span (X, Y, Z):", np.ptp(normalized_points, axis=0))
    print("Saved geon document:", output_path)
    print("Saved fields:", pcd.field_names)
    print("HPNet selects at most 7,000 points per pass; this file retains the complete merged cloud.")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source merged geon HDF5 file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for normalized geon HDF5 documents")
    args = parser.parse_args()
    prepare_bridge(args.input, args.output_dir)


if __name__ == "__main__":
    main()
