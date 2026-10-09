"""Coverage, checkpoint compatibility, and source-preservation checks."""

import tempfile
from pathlib import Path
import unittest

import h5py
import numpy as np
import torch

from dataloader.BridgeDataset import BridgeDataset
from models import dgcnn, dgcnn_bridge
from train_bridge import build_option, load_model
from utils.bridge_output import export_bridge_prediction
from utils.bridge_postprocess import compute_entropy, normalize_affinity, mean_shift_labels


class BridgeInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_block_neighbors_match_original(self):
        torch.manual_seed(13)
        points = torch.randn(2, 6, 97)
        points[:, 3:] = torch.nn.functional.normalize(points[:, 3:], dim=1)
        for original, bridge in (
            (dgcnn.knn, dgcnn_bridge.knn),
            (dgcnn.knn_points_normals, dgcnn_bridge.knn_points_normals),
        ):
            expected = original(points, 20, 20)
            actual = bridge(points, 20, 20, query_chunk_size=23)
            # Neighbor order can differ on ties, but membership must match.
            torch.testing.assert_close(actual.sort(-1).values, expected.sort(-1).values)

    def test_entropy_uses_partial_blocks_and_constant_channels(self):
        torch.manual_seed(17)
        features = torch.randn(1, 37, 5)
        features[:, :, 0] = 0
        interval = 2 * (features[0].amax(0) - features[0].amin(0))
        values = features[0] / interval.clamp_min(1e-12)
        distances = (values[:, None] - values[None, :]).norm(dim=-1)
        similarity = torch.exp(np.log(0.5) * distances / distances.mean())
        expected = (
            -similarity * torch.log(similarity + 1e-7)
            -(1 - similarity) * torch.log(1 - similarity + 1e-7)
        ).mean()
        torch.testing.assert_close(compute_entropy(features, 8), expected, atol=2e-5, rtol=1e-5)
        self.assertEqual(compute_entropy(torch.zeros(1, 7, 3), 3).item(), 0)

    def test_normalization_matches_dense_formula(self):
        torch.manual_seed(18)
        affinity = torch.rand(11, 11) + 1e-12
        diagonal = torch.diag(affinity.sum(-1).rsqrt())
        expected = diagonal @ affinity @ diagonal
        expected = ((expected + expected.T) * 0.5)[None]
        torch.testing.assert_close(normalize_affinity(affinity.clone()), expected)

    def test_bin_seeds_remain_on_sparse_high_dimensional_data(self):
        # Grid-center seeds are too far from both clouds at this bandwidth.
        features = np.concatenate([np.full((3, 128), 0.3), np.full((3, 128), 1.3)])
        labels = mean_shift_labels(features, bandwidth=0.85, bin_seeding=True, workers=1)
        self.assertEqual(len(labels), 6)
        self.assertEqual(len(np.unique(labels)), 2)
        self.assertTrue((labels[:3] == labels[0]).all())
        self.assertTrue((labels[3:] == labels[3]).all())

    def test_checkpoint_compatible_predictions_in_source_order(self):
        opt = build_option(["--knn_chunk_size", "23"])
        bridge = load_model(opt, torch.device("cpu"))
        original = dgcnn.PrimitiveNet(opt).eval()
        original.load_state_dict(bridge.state_dict(), strict=True)
        torch.manual_seed(19)
        points = torch.randn(1, 3, 97)
        normals = torch.nn.functional.normalize(torch.randn_like(points), dim=1)
        with torch.no_grad():
            result = bridge(points, normals, postprocess=True)
            reference = original(points, normals, postprocess=True)
        torch.testing.assert_close(result[-1], torch.arange(97)[None])
        order = reference[-1][0].argsort()
        for actual, expected in zip(result[:-1], reference[:-1]):
            torch.testing.assert_close(actual, expected[:, order], atol=1e-4, rtol=1e-4)

    def test_more_than_7000_points_are_never_sampled(self):
        class FakeEncoder(torch.nn.Module):
            def forward(self, values):
                self.point_count = values.shape[-1]
                return values.new_zeros((1, 1024)), values.new_zeros((1, 256, self.point_count))

        model = dgcnn_bridge.PrimitiveNet(build_option([])).eval()
        encoder = FakeEncoder()
        model.affinitynet.encoder = encoder
        with torch.no_grad():
            result = model(torch.zeros(1, 3, 7003), torch.ones(1, 3, 7003), postprocess=True)
        self.assertEqual(encoder.point_count, 7003)
        for values in result:
            self.assertEqual(values.shape[1], 7003)
        torch.testing.assert_close(result[-1], torch.arange(7003)[None])

    def test_nested_source_without_targets_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bridge.h5"
            points = np.arange(27, dtype=np.float32).reshape(9, 3)
            with h5py.File(path, "w") as source:
                source.attrs["title"] = "source"
                group = source.create_group("document/cloud")
                group.create_dataset("points", data=points)
                group.create_dataset("fields/normals/data", data=np.tile([0., 0., 2.], (9, 1)))
                group.create_dataset("fields/superpoints/data", data=np.arange(9))
                source.create_dataset("document/telemetry", data=np.zeros(0))
            dataset = BridgeDataset(path)
            np.testing.assert_array_equal(dataset.points, points)
            np.testing.assert_allclose(dataset.normals, np.tile([0., 0., 1.], (9, 1)))
            predictions = {
                "source_index": np.arange(9), "labels_pred": np.arange(9) % 2,
                "prim_pred": np.ones(9, dtype=np.int64),
                "type_prob_pred": np.ones((9, 10)) / 10,
                "T_param_pred": np.zeros((9, 22)),
            }
            output_path, summary = export_bridge_prediction(
                dataset, Path(temp) / "output", predictions,
                {"checkpoint": "test", "postprocess": "embedding"},
            )
            self.assertEqual(summary["predicted_point_count"], 9)
            with h5py.File(output_path) as output, h5py.File(path) as source:
                np.testing.assert_array_equal(output["document/cloud/points"], source["document/cloud/points"])
                np.testing.assert_array_equal(output["document/cloud/fields/superpoints/data"], np.arange(9))
                self.assertEqual(output.attrs["title"], "source")
                self.assertNotIn("labels_pred", source)
                self.assertEqual(output["points"].shape, (9, 3))
            with self.assertRaises(FileExistsError):
                export_bridge_prediction(dataset, Path(temp) / "output", predictions, summary)
            predictions["source_index"] = np.zeros(9, dtype=np.int64)
            with self.assertRaises(ValueError):
                export_bridge_prediction(dataset, Path(temp) / "invalid", predictions, summary)


if __name__ == "__main__":
    unittest.main()
