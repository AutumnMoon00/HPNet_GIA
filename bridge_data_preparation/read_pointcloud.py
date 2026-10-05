from pathlib import Path
import numpy as np

from geon.data.boundingbox import BoundingBoxData
from geon.data.cellcomplex import CellComplexData
from geon.data.document import Document
from geon.data.pointcloud import FieldType, PointCloudData

# datapath = r"D:\GIA\fib\HPNet_GIA\bridge_data_with_primitives\superpoints_merged\BW-0000_SC-00_clipped_superpoints_merged.h5"
# datapath = r"D:\GIA\fib\HPNet_GIA\bridge_data_with_primitives\bbox_clipped\BW-0000_SC-00_clipped.h5"
# normalized scaled bridge point cloud
datapath = r"D:\GIA\fib\HPNet_GIA\bridge_data_with_primitives\sp_merged_normalized\BW-0000_SC-00_clipped_superpoints_merged_normalized_scaled.h5"

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
print("field_names: ", field_names)
print("pcd.id", pcd.id)

for field in field_names:
    field_data = pcd.get_fields(names=[field])
    print("field: ", field)
    print(field_data)

    print()


# numver of unique instances
instances = pcd.get_fields(names=["instance"])
instances = np.asarray(instances[0].data).reshape(-1)
print("unique instances", np.unique(instances))
print("num of pts: ", len(pts))

# Span (maximum minus minimum) and mean position along each coordinate axis.
if len(pts) == 0:
    raise ValueError("Cannot compute spans and mean positions for an empty point cloud.")

spans = np.ptp(pts, axis=0)
mean_positions = np.mean(pts, axis=0, dtype=np.float64)
for axis, span, mean_position in zip(("X", "Y", "Z"), spans, mean_positions):
    print(f"{axis} span: {span:.10g}")
    print(f"{axis} mean position: {mean_position:.10g}")
