from pathlib import Path

from geon.data.document import Document
from geon.data.pointcloud import FieldType, PointCloudData


org_data_dir = r"C:\Users\Madanu.S\Downloads\01-BridgePatchDataset.geon\01-BridgePatchDataset.geon\org_copy"


# Check intensity fields without changing the source files.
for file_path in Path(org_data_dir).iterdir():
    if file_path.is_file() and file_path.suffix.lower() in {".h5", ".hdf5"}:
        print()
        print(file_path)

        doc = Document.load_hdf5(str(file_path))
        for name, scene_item in doc.scene_items.items():
            if isinstance(scene_item, PointCloudData):
                pcd = scene_item
                print("\tPoint cloud data name:", name)

                has_intensity = any(
                    field.name.lower() == "intensity"
                    or field.field_type == FieldType.INTENSITY
                    for field in pcd.get_fields()
                )
                print("\tIntensity field exists:", "Yes" if has_intensity else "No")
