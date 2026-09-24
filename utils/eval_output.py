"""Write point-aligned ABCParts predictions for inspection."""

import os

import h5py
import numpy as np


def export_abc_prediction(source_path, output_path, source_index, labels_pred, prim_pred):
    """Copy the selected source points and append their predictions to an HDF5 file."""
    source_index = np.asarray(source_index, dtype=np.int64)
    labels_pred = np.asarray(labels_pred, dtype=np.int64)
    prim_pred = np.asarray(prim_pred, dtype=np.int64)

    if source_index.ndim != 1 or labels_pred.shape != source_index.shape or prim_pred.shape != source_index.shape:
        raise ValueError("source_index, labels_pred and prim_pred must be equal-length vectors")

    with h5py.File(source_path, "r") as source:
        source_count = source["points"].shape[0]
        if np.any(source_index < 0) or np.any(source_index >= source_count):
            raise ValueError("source_index contains an out-of-range point")
        if np.unique(source_index).size != source_index.size:
            raise ValueError("source_index contains duplicate points")
        if any(key in source for key in ("source_index", "labels_pred", "prim_pred")):
            raise ValueError("source file already contains an export field")

        with h5py.File(output_path, "x") as output:
            for name, value in source.attrs.items():
                output.attrs[name] = value
            for name, field in source.items():
                if not isinstance(field, h5py.Dataset):
                    raise TypeError("unsupported HDF5 group in source: " + name)
                values = field[()]
                if values.ndim and values.shape[0] == source_count:
                    values = values[source_index]
                copied = output.create_dataset(name, data=values)
                for attr_name, attr_value in field.attrs.items():
                    copied.attrs[attr_name] = attr_value

            output.create_dataset("source_index", data=source_index)
            output.create_dataset("labels_pred", data=labels_pred)
            output.create_dataset("prim_pred", data=prim_pred)
            output.attrs["source_file"] = os.path.abspath(source_path)
            output.attrs["source_point_count"] = source_count
            output["labels_pred"].attrs["description"] = "Instance cluster IDs; IDs are arbitrary up to permutation"
            output["prim_pred"].attrs["description"] = "Predicted ABC primitive IDs after the validation class mapping"
