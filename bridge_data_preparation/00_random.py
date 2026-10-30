import numpy as np
import random
import os
import h5py
import argparse
import re
from pathlib import Path

from geon.data.document import Document
from geon.data.pointcloud import PointCloudData

## Inspect class IDs and their stored names.
org_data_dir = r"C:\Users\Madanu.S\Downloads\01-BridgePatchDataset.geon\01-BridgePatchDataset.geon\org_copy"


# loop through files in dir
# Read class information without changing the source files.
for file_path in Path(org_data_dir).iterdir():
    if file_path.is_file() and file_path.suffix.lower() in {".h5", ".hdf5"}:
        print()
        print(file_path)

        # read PCD
        doc = Document.load_hdf5(str(file_path))

        # Check every point cloud for class labels and their semantic schema.
        for name, scene_item in doc.scene_items.items():
            if isinstance(scene_item, PointCloudData):
                pcd = scene_item
                print("\tPoint cloud data name:", name)
                
                # print("\t # of fields: ", len(pcd.field_names))
                # print("\tfield_names: ", pcd.field_names)

                class_fields = pcd.get_fields(names=["class"])
                if not class_fields:
                    print("\tNo 'class' field found.")
                    continue

                class_field = class_fields[0]
                unique_cls, counts = np.unique(
                    np.asarray(class_field.data).reshape(-1), return_counts=True
                )
                print("\tNumber of classes present:", len(unique_cls))

                # Semantic fields store an ID/name/color mapping in their schema.
                schema = getattr(class_field, "schema", None)
                class_names = {}
                # if schema is not None:
                class_names = {
                    semantic_class.id: semantic_class.name
                    for semantic_class in schema.semantic_classes
                }
                print("\tSchema name:", schema.name)
                print("\tAll class IDs/names defined in the schema:", class_names)
                # else:
                #     print("\tNo semantic schema attached; class names are unavailable.")

                print("\tClasses present in this point cloud:")
                print(f"\t{'ID':>6}  {'Class name':<24} {'Point count':>12}")
                for class_id, count in zip(unique_cls, counts):
                    class_name = class_names.get(class_id, '<name unavailable>')
                    print(f"\t{class_id:>6}  {class_name:<24} {count:>12,}")



        #         for field_name in list(pcd.field_names):
        #             if "superpoint" in field_name.lower():
        #                 print("\tRemoving field:", field_name)
        #                 pcd.remove_fields(names=[field_name])
        #                 changed = True

        # if changed:
        #     # Finish writing the document before replacing its source file.
        #     temporary_path = file_path.with_name(f"{file_path.name}.tmp")
        #     if temporary_path.exists():
        #         raise FileExistsError(f"Temporary output already exists: {temporary_path}")
        #     doc.save_hdf5(temporary_path)
        #     temporary_path.replace(file_path)
        #     print("\tSaved:", file_path)


# class name and number 
