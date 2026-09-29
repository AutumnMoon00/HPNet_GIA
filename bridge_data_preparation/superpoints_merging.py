from pathlib import Path
import numpy as np

from geon.data.boundingbox import BoundingBoxData
from geon.data.cellcomplex import CellComplexData
from geon.data.document import Document
from geon.data.pointcloud import FieldType, PointCloudData

datapath = Path(__file__).resolve().parent.parent / "bridge_data_with_primitives" / "bbox_clipped" / "BW-0000_SC-00_clipped.h5"

# Load the point cloud
doc = Document.load_hdf5(str(datapath))
pcd = None

for name, item in doc.scene_items.items():
    if isinstance(item, PointCloudData):
        pcd = item
        print("Point cloud data name:", name)

if pcd is None:
    raise ValueError("The document does not contain a point cloud.")

pts = pcd.points
if pts.ndim != 2 or pts.shape[1] != 3:
    raise ValueError(f"Expected Nx3 points, got {pts.shape}.")
field_names = pcd.field_names
print("Source fields:", field_names)


# One output point per superpoint ID, at the mean position of its source points.
superpoint_fields = pcd.get_fields(names="superpoints")
if len(superpoint_fields) != 1:
    raise ValueError("Expected exactly one 'superpoints' field in the point cloud.")
superpoint_ids = np.asarray(superpoint_fields[0].data).reshape(-1)
if len(superpoint_ids) != len(pts):
    raise ValueError("The superpoints field must contain one ID per point.")

unique_ids, group_index, group_counts = np.unique(
    superpoint_ids, return_inverse=True, return_counts=True
)
merged_points = np.empty((len(unique_ids), 3), dtype=np.float64)
for axis in range(3):
    coordinate_sums = np.bincount(
        group_index, weights=pts[:, axis], minlength=len(unique_ids)
    )
    merged_points[:, axis] = coordinate_sums / group_counts

# Average continuous fields, vote on categorical fields, and retain the ID
# that identifies each merged superpoint. Rows follow sorted superpoint IDs.
mean_fields = {
    "color", "feat_linearity", "feat_sphericity", "feat_surface_variation",
    "feat_verticallity", "intensity", "normals",
}
majority_fields = {"instance", "parts", "patches", "class"}
required_fields = mean_fields | majority_fields | {"superpoints"}
missing_fields = required_fields - set(field_names)
if missing_fields:
    raise ValueError(f"Missing source fields: {sorted(missing_fields)}")

merged_pcd = PointCloudData(merged_points)
merged_pcd.id = pcd.id
sorted_indices = np.argsort(group_index, kind="stable")
group_offsets = np.concatenate(([0], np.cumsum(group_counts)))

for field in pcd.get_fields():
    source_data = np.asarray(field.data)
    if len(source_data) != len(pts):
        raise ValueError(f"Field {field.name!r} does not match the point count.")

    if field.name in mean_fields:
        values = source_data.reshape(len(pts), -1)
        averaged = np.empty((len(unique_ids), values.shape[1]), dtype=np.float64)
        for axis in range(values.shape[1]):
            sums = np.bincount(
                group_index, weights=values[:, axis], minlength=len(unique_ids)
            )
            averaged[:, axis] = sums / group_counts
        if field.name == "color":
            averaged = np.clip(np.rint(averaged), 0, 255)
        merged_data = averaged.astype(source_data.dtype)
        if source_data.ndim == 1:
            merged_data = merged_data[:, 0]

    elif field.name in majority_fields:
        sorted_labels = source_data.reshape(-1)[sorted_indices]
        voted = np.empty(len(unique_ids), dtype=source_data.dtype)
        for group, (start, stop) in enumerate(
            zip(group_offsets[:-1], group_offsets[1:])
        ):
            labels, counts = np.unique(
                sorted_labels[start:stop], return_counts=True
            )
            # np.unique sorts labels, so a tie selects the smallest label.
            voted[group] = labels[np.argmax(counts)]
        merged_data = voted if source_data.ndim == 1 else voted[:, None]

    elif field.name == "superpoints":
        merged_data = unique_ids.astype(source_data.dtype)
        if source_data.ndim != 1:
            merged_data = merged_data[:, None]
    else:
        print("Skipping unsupported field:", field.name)
        continue

    merged_pcd.add_field(
        name=field.name,
        data=merged_data,
        field_type=field.field_type,
        schema=field.schema if field.field_type == FieldType.SEMANTIC else None,
    )

merged_doc = Document(name=f"{doc.name}_superpoints_merged")
merged_doc.add_data(merged_pcd)

output_dir = datapath.parent.parent / "superpoints_merged"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{datapath.stem}_superpoints_merged.h5"
temporary_path = output_path.with_name(f"{output_path.stem}.tmp.h5")
if temporary_path.exists():
    raise FileExistsError(f"Temporary output already exists: {temporary_path}")
merged_doc.save_hdf5(temporary_path)
temporary_path.replace(output_path)

print("Original points:", len(pts))
print("Unique superpoint IDs:", len(unique_ids))
print("Saved merged points:", len(merged_pcd.points))
print("Saved merged fields:", merged_pcd.field_names)
print("Saved merged point cloud to:", output_path)

