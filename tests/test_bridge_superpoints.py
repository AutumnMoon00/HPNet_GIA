"""Verify label filtering and preservation of the original HDF5 contents."""

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "bridge_data_preparation" / "02_process_brige.py"
SPEC = importlib.util.spec_from_file_location("bridge_superpoints", SCRIPT)
bridge_superpoints = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge_superpoints)


class BridgeSuperpointTests(unittest.TestCase):
    def test_only_labelled_points_reach_segmentation(self):
        points = np.arange(15, dtype=np.float32).reshape(5, 3)
        classes = np.array([-1, 3, 22, 4, 3])[:, None]
        normals = np.arange(15, dtype=np.float32).reshape(5, 3) / 15
        intensity = np.arange(5, dtype=np.float32)[:, None]
        # Nonfinite features on ignored points must never enter segmentation.
        normals[[0, 2]] = np.nan
        intensity[[0, 2]] = np.nan
        with patch.object(
            bridge_superpoints, "segment_superpoints",
            return_value=(np.array([7, 7, 9], dtype=np.int32), {}),
        ) as segment:
            result = bridge_superpoints.compute_superpoint_ids(points, classes, normals, intensity)
        np.testing.assert_array_equal(segment.call_args.args[0], points[[1, 3, 4]])
        np.testing.assert_array_equal(
            segment.call_args.kwargs["features"],
            np.concatenate((normals[[1, 3, 4]], intensity[[1, 3, 4]]), axis=1),
        )
        np.testing.assert_array_equal(result, [-1, 7, -1, 7, 9])
        self.assertEqual(result.dtype, np.int32)

    def test_empty_unlabelled_and_single_labelled_clouds(self):
        with patch.object(bridge_superpoints, "segment_superpoints") as segment:
            for classes, expected in (
                ([], []), ([-1, 22, -1], [-1, -1, -1]),
                ([22, 5, -1], [-1, 0, -1]),
            ):
                with self.subTest(classes=classes):
                    result = bridge_superpoints.compute_superpoint_ids(
                        np.zeros((len(classes), 3)), np.array(classes),
                        np.zeros((len(classes), 3)), np.zeros(len(classes)),
                    )
                    np.testing.assert_array_equal(result, expected)
            segment.assert_not_called()

    def test_real_segmentation_on_a_small_cloud(self):
        points = np.random.default_rng(42).normal(size=(32, 3)).astype(np.float32)
        classes = np.ones(32, dtype=np.int32)
        classes[::3] = -1
        classes[1::3] = 22
        normals = np.tile([0, 0, 1], (32, 1))
        intensity = np.linspace(0, 1, 32)
        segment = bridge_superpoints.segment_superpoints

        def serial_segment(points, **kwargs):
            # Avoid OpenMP startup overhead for this tiny native smoke test.
            labels, stats = segment(points, **{**kwargs, "parallel": False})
            self.assertEqual(stats["feature_dim"], 7)  # XYZ + normals + intensity.
            return labels, stats

        with patch.object(bridge_superpoints, "segment_superpoints", side_effect=serial_segment):
            result = bridge_superpoints.compute_superpoint_ids(points, classes, normals, intensity)
        self.assertTrue(np.all(result[classes == 1] >= 0))
        self.assertTrue(np.all(result[classes != 1] == -1))

    def make_source(self, path):
        with h5py.File(path, "w") as source:
            source.attrs["custom_root_attribute"] = "preserve me"
            document = source.create_group("document")
            document.attrs["type"] = "Document"
            document.attrs["name"] = "bridge"
            document.create_dataset("telemetry", data=np.array([1, 2]))
            box = document.create_group("box")
            box.attrs["type_id"] = "BoundingBoxData"
            box.create_dataset("custom_geometry", data=np.eye(3))
            for name, classes in (("cloud", [-1, 3, 22, 4, 3]), ("ignored", [22, -1])):
                cloud = document.create_group(name)
                cloud.attrs["type_id"] = "PointCloudData"
                cloud.attrs["id"] = name
                cloud.create_dataset("points", data=np.zeros((len(classes), 3)))
                field = cloud.create_group("fields/class")
                field.attrs["field_type"] = "SCALAR"
                field.create_dataset("data", data=np.array(classes)[:, None])
                normals = cloud.create_group("fields/normals_0.100")
                normals.attrs["field_type"] = "NORMAL"
                normals.create_dataset(
                    "data", data=np.ones((len(classes), 3), dtype=np.float32),
                    compression="gzip", shuffle=True,
                )
                intensity = cloud.create_group("fields/intensity")
                intensity.attrs["field_type"] = "INTENSITY"
                intensity.create_dataset(
                    "data", data=np.arange(len(classes), dtype=np.float32)[:, None]
                )

    def test_copy_adds_only_superpoints_and_loads_in_geon(self):
        from geon.data.pointcloud import FieldType, PointCloudData

        with tempfile.TemporaryDirectory() as temp:
            source_path = Path(temp) / "source.h5"
            output_path = Path(temp) / "output" / "source.h5"
            self.make_source(source_path)
            before = hashlib.sha256(source_path.read_bytes()).digest()
            with patch.object(
                bridge_superpoints, "segment_superpoints",
                return_value=(np.array([0, 0, 1], dtype=np.int32), {}),
            ):
                bridge_superpoints.process_file(source_path, output_path)
            self.assertEqual(hashlib.sha256(source_path.read_bytes()).digest(), before)
            with h5py.File(source_path) as source, h5py.File(output_path) as output:
                original_names, output_names = [], []
                source.visit(original_names.append)
                output.visit(output_names.append)
                self.assertEqual(set(output_names) - set(original_names), {
                    f"document/{name}/fields/superpoints{suffix}"
                    for name in ("cloud", "ignored") for suffix in ("", "/data")
                })
                for name in ["/", *original_names]:
                    original, copied = source[name], output[name]
                    self.assertEqual(dict(original.attrs), dict(copied.attrs))
                    if isinstance(original, h5py.Dataset):
                        np.testing.assert_array_equal(original[()], copied[()])
                        self.assertEqual(original.dtype, copied.dtype)
                        self.assertEqual(original.compression, copied.compression)
                        self.assertEqual(original.chunks, copied.chunks)
                        self.assertEqual(original.shuffle, copied.shuffle)
                cloud = PointCloudData.load_hdf5(output["document/cloud"])
                field = cloud.get_fields(names=["superpoints"])[0]
                self.assertEqual(field.field_type, FieldType.INSTANCE)
                np.testing.assert_array_equal(field.data[:, 0], [-1, 0, -1, 0, 1])
                np.testing.assert_array_equal(
                    output["document/ignored/fields/superpoints/data"][:, 0], [-1, -1]
                )
            with self.assertRaises(FileExistsError):
                bridge_superpoints.process_file(source_path, output_path)
            with self.assertRaises(ValueError):
                bridge_superpoints.process_file(source_path, source_path)

    def test_failed_processing_leaves_no_output_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            source_path = Path(temp) / "source.h5"
            output_dir = Path(temp) / "output"
            self.make_source(source_path)
            with h5py.File(source_path, "r+") as source:
                del source["document/cloud/fields/class"]
            before = hashlib.sha256(source_path.read_bytes()).digest()
            with self.assertRaisesRegex(ValueError, "no 'class' field"):
                bridge_superpoints.process_file(source_path, output_dir / source_path.name)
            self.assertEqual(hashlib.sha256(source_path.read_bytes()).digest(), before)
            self.assertEqual(list(output_dir.iterdir()), [])

    def test_missing_feature_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            source_path = Path(temp) / "source.h5"
            output_dir = Path(temp) / "output"
            self.make_source(source_path)
            with h5py.File(source_path, "r+") as source:
                del source["document/cloud/fields/intensity"]
            before = hashlib.sha256(source_path.read_bytes()).digest()
            with self.assertRaisesRegex(ValueError, "no feature field 'intensity'"):
                bridge_superpoints.process_file(source_path, output_dir / source_path.name)
            self.assertEqual(hashlib.sha256(source_path.read_bytes()).digest(), before)
            self.assertEqual(list(output_dir.iterdir()), [])

    def test_invalid_features_are_rejected(self):
        points = np.zeros((2, 3))
        classes = np.array([1, 2])
        with patch.object(bridge_superpoints, "segment_superpoints") as segment:
            for normals, intensity, message in (
                (np.zeros((1, 3)), np.zeros(2), "normals"),
                (np.zeros((2, 3)), np.zeros((2, 2)), "intensity"),
                (np.full((2, 3), np.nan), np.zeros(2), "finite"),
            ):
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        bridge_superpoints.compute_superpoint_ids(
                            points, classes, normals, intensity
                        )
            segment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
