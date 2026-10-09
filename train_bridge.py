"""Run pretrained HPNet on every point of a bridge HDF5 file, without ground truth."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from dataloader.BridgeDataset import BridgeDataset
from models.dgcnn_bridge import PrimitiveNet
from utils.bridge_output import export_bridge_prediction, mapped_primitives
from utils.bridge_postprocess import cluster_predictions


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = "/data/users/wm-sharath/fib/bridge_data_with_primitives/sp_merged_normalized/BW-0000_SC-00_clipped_superpoints_merged_normalized_scaled.h5"


def build_option(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default=DEFAULT_DATA, help="One HDF5 file, not a dataset list")
    parser.add_argument("--cloud", default=None, help="HDF5 point cloud group, auto-detected by default")
    parser.add_argument("--checkpoint_path", default=str(ROOT / "model_ABCParts/abc_normal/abc_normal"))
    parser.add_argument("--output_dir", default=str(ROOT / "outputs/bridge_inference"))
    parser.add_argument("--device", default="cuda:0", help="PyTorch device, e.g. cuda:0 or cpu")
    parser.add_argument("--knn_chunk_size", type=int, default=1024, help="Query block size; neighbors always use all points")
    parser.add_argument("--postprocess_chunk_size", type=int, default=1024)
    parser.add_argument("--postprocess", choices=("hybrid", "embedding"), default="hybrid", help="hybrid matches HPNet; embedding uses only network features")
    parser.add_argument("--bandwidth", type=float, default=0.85)
    parser.add_argument("--bin_seeding", action="store_true", help="Accelerate mean shift using feature-bin seeds; all points still receive predictions")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--normal_sigma", type=float, default=0.1)
    parser.add_argument("--edge_knn", type=int, default=50)
    parser.add_argument("--topK", type=int, default=10)
    parser.add_argument("--edge_topK", type=int, default=12)
    parser.add_argument("--feat_ent_weight", type=float, default=1.70)
    parser.add_argument("--dis_ent_weight", type=float, default=1.10)
    parser.add_argument("--edge_ent_weight", type=float, default=1.23)
    opt = parser.parse_args(argv)
    for name in ("knn_chunk_size", "postprocess_chunk_size", "workers", "edge_knn", "topK", "edge_topK"):
        if getattr(opt, name) < 1:
            parser.error(f"--{name} must be positive")
    for name in ("bandwidth", "sigma", "normal_sigma"):
        if not np.isfinite(getattr(opt, name)) or getattr(opt, name) <= 0:
            parser.error(f"--{name} must be positive and finite")
    # Match the supplied ABC normal checkpoint, including all prediction heads.
    opt.backbone, opt.input_normal, opt.out_dim, opt.loss_class = "DGCNN", 1, 128, "frp"
    return opt


def load_model(opt, device):
    checkpoint_path = Path(opt.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    # Accept the DataParallel format while loading one complete cloud on one GPU.
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model = PrimitiveNet(opt)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f"Loaded checkpoint {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')})", flush=True)
    return model


def main(argv=None):
    opt = build_option(argv)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    device = torch.device(opt.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use a GPU allocation or --device cpu.")
    # Existing spline checkpoint paths are relative to the repository root.
    # Resolve these by requiring the project working directory for hybrid runs.
    if opt.postprocess == "hybrid" and Path.cwd().resolve() != ROOT:
        import os
        opt.data_path = str(Path(opt.data_path).expanduser().resolve())
        opt.checkpoint_path = str(Path(opt.checkpoint_path).expanduser().resolve())
        opt.output_dir = str(Path(opt.output_dir).expanduser().resolve())
        os.chdir(ROOT)
    if opt.postprocess == "hybrid" and device.type != "cuda":
        raise ValueError("Hybrid spline fitting requires CUDA; use --postprocess embedding on CPU.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    dataset = BridgeDataset(opt.data_path, cloud=opt.cloud)
    print(f"Cloud: {dataset.cloud_path or '/'}; predicting all {len(dataset.points)} points", flush=True)
    model = load_model(opt, device)
    points = torch.from_numpy(dataset.points).to(device).T.unsqueeze(0)
    normals = torch.from_numpy(dataset.normals).to(device).T.unsqueeze(0)
    start = time.monotonic()
    with torch.inference_mode():
        embedding, scores, predicted_normals, parameters, source_index = model(points, normals, postprocess=True)
        expected = torch.arange(len(dataset.points), device=device).unsqueeze(0)
        if not torch.equal(source_index, expected):
            raise RuntimeError("Bridge model did not preserve complete point coverage and order.")
        print(f"Network predicted {scores.shape[1]} points in {time.monotonic() - start:.1f}s", flush=True)
        # Save every direct model output before the more expensive clustering stage.
        output_dir = Path(opt.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = output_dir / (dataset.path.stem + "_network.npz")
        predictions = {
            "source_index": source_index[0].cpu().numpy(),
            "embedding_pred": embedding[0].cpu().numpy(),
            "type_prob_pred": scores[0].exp().cpu().numpy(),
            "prim_pred_raw": scores[0].argmax(-1).cpu().numpy(),
            "normals_pred": predicted_normals[0].cpu().numpy(),
            "T_param_pred": parameters[0].cpu().numpy(),
        }
        predictions["prim_pred"] = mapped_primitives(predictions["prim_pred_raw"])
        with open(raw_path, "xb") as output:
            np.savez_compressed(output, points=dataset.points, normals=dataset.normals, **predictions)
        print(f"Direct network outputs saved: {raw_path}", flush=True)
        labels, weights = cluster_predictions(points, normals, embedding, scores, parameters, opt)
        predictions["labels_pred"] = labels
    h5_path, summary = export_bridge_prediction(dataset, opt.output_dir, predictions, {
        "checkpoint": str(Path(opt.checkpoint_path).resolve()), "postprocess": opt.postprocess,
        "bandwidth": opt.bandwidth, "bin_seeding": opt.bin_seeding,
        "seed": opt.seed, "entropy_weights": weights,
        "elapsed_seconds": time.monotonic() - start,
    })
    print(f"Saved {summary['predicted_point_count']} predictions, {summary['instance_count']} instances: {h5_path}", flush=True)
    print(f"Primitive counts: {summary['primitive_counts']}", flush=True)


if __name__ == "__main__":
    main()
