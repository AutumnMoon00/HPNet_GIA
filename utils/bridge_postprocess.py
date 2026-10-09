"""HPNet hybrid postprocessing for arbitrary point counts, without targets."""

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.cluster import MeanShift


def compute_entropy(features, chunk_size=1024):
    """Original pairwise entropy formula, using every row and bounded blocks."""
    if features.shape[0] != 1:
        raise ValueError("Bridge entropy expects a single cloud.")
    feat = features[0]
    interval = 2 * (feat.amax(0) - feat.amin(0))
    feat = feat / interval.clamp_min(1e-12)
    count = len(feat)
    total = feat.new_zeros(())
    for begin in range(0, count, chunk_size):
        distance = torch.cdist(feat[begin:begin + chunk_size], feat)
        total += distance.sum()
    average = total / (count * count)
    if average <= 1e-12:
        return feat.new_zeros(())
    alpha = -np.log(0.5) / average
    entropy = feat.new_zeros(())
    for begin in range(0, count, chunk_size):
        distance = torch.cdist(feat[begin:begin + chunk_size], feat)
        similarity = torch.exp(-alpha * distance)
        entropy += (
            -similarity * torch.log(similarity + 1e-7)
            -(1 - similarity) * torch.log(1 - similarity + 1e-7)
        ).sum()
    return entropy / (count * count)


def normalize_affinity(affinity):
    """Apply D^-1/2 A D^-1/2 without allocating dense diagonal matrices."""
    inverse = affinity.sum(-1).clamp_min(1e-24).rsqrt()
    affinity *= inverse[:, None]
    affinity *= inverse[None, :]
    return ((affinity + affinity.T) * 0.5).unsqueeze(0)


def type_affinity(points, scores, parameters, sigma, chunk_size):
    # Reuse the repository's primitive distances and pretrained spline fitting.
    from utils.primitive_dis import ComputePrimitiveDistance
    from utils.abc_utils import get_fitting_module

    distance = ComputePrimitiveDistance(reduce=False, one_side=True)
    routines = {
        1: distance.distance_from_plane, 3: distance.distance_from_cone,
        4: distance.distance_from_cylinder, 5: distance.distance_from_sphere,
        2: distance.distance_from_bspline, 9: distance.distance_from_bspline,
    }
    slices = {1: (4, 8), 3: (15, 22), 4: (8, 15), 5: (0, 4)}
    types = scores[0].argmax(-1).clone()
    types[(types == 0) | (types == 6) | (types == 7)] = 9
    types[types == 8] = 2
    xyz = points[0].T
    affinity = xyz.new_full((len(xyz), len(xyz)), 1e-12)
    for kind in types.unique().tolist():
        columns = torch.where(types == kind)[0]
        if len(columns) < 30:
            continue
        print(f"Geometry affinity: type {kind}, {len(columns)} points", flush=True)
        if kind in (2, 9):
            fitter = get_fitting_module()
            weights = xyz.new_ones((len(columns), 1))
            if kind == 2:
                params = fitter.forward_pass_open_spline(xyz[columns], weights=weights)
            else:
                try:
                    params = fitter.forward_pass_closed_spline(
                        xyz[columns], weights=weights, if_optimize=True
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                    print(f"Spline optimization fallback: {error}", flush=True)
                    params = fitter.forward_pass_closed_spline(xyz[columns], weights=weights)
        else:
            lo, hi = slices[kind]
            params = parameters[0, columns, lo:hi]
        for begin in range(0, len(xyz), chunk_size):
            stop = min(begin + chunk_size, len(xyz))
            values = routines[kind](points=xyz[begin:stop], params=params)
            if kind in (2, 9):
                values = values[:, None].expand(-1, len(columns))
            affinity[begin:stop, columns] = torch.exp(-values.square() / (2 * sigma**2))
    if not torch.isfinite(affinity).all():
        raise ValueError("Geometry affinity contains nonfinite values.")
    return normalize_affinity(affinity)


def normal_affinity(points, normals, sigma, knn):
    xyz = points[0].T.detach().cpu().numpy()
    k = min(knn, len(xyz))
    _, neighbors = cKDTree(xyz).query(xyz, k=k)
    neighbors = np.asarray(neighbors).reshape(len(xyz), k)
    neighbors = torch.as_tensor(neighbors, device=normals.device, dtype=torch.long)
    normal = normals[0].T
    angles = torch.acos((normal[:, None] * normal[neighbors]).sum(-1).clamp(-0.99, 0.99))
    weights = torch.exp(-angles.square() / (2 * sigma**2))
    affinity = normal.new_zeros((len(xyz), len(xyz)))
    affinity.scatter_add_(1, neighbors, weights)
    affinity.masked_fill_(affinity == 0, 1e-12)
    return normalize_affinity(affinity)


def mean_shift_labels(values, bandwidth, bin_seeding, workers):
    # Grid centers can lie outside every point's bandwidth in the model's
    # high-dimensional feature space. Use a real vector from each occupied
    # bin so every seed has at least one neighbor, even for sparse features.
    seeds = None
    if bin_seeding:
        _, indices = np.unique(np.round(values / bandwidth), axis=0, return_index=True)
        seeds = values[np.sort(indices)]
        print(f"Mean shift: {len(seeds)} point-backed feature-bin seeds", flush=True)
    return MeanShift(bandwidth=bandwidth, seeds=seeds, n_jobs=workers).fit_predict(values)


def cluster_predictions(points, normals, embedding, scores, parameters, opt):
    features = [embedding]
    weights = [opt.feat_ent_weight - compute_entropy(embedding, opt.postprocess_chunk_size).item()]
    if opt.postprocess == "hybrid":
        for name, builder, topk, entropy_weight in (
            ("geometry", lambda: type_affinity(
                points, scores, parameters, opt.sigma, opt.postprocess_chunk_size
            ), opt.topK, opt.dis_ent_weight),
            ("normals", lambda: normal_affinity(
                points, normals, opt.normal_sigma, opt.edge_knn
            ), opt.edge_topK, opt.edge_ent_weight),
        ):
            print(f"Building {name} spectral features for every point", flush=True)
            affinity = builder()
            if 3 * topk > points.shape[-1]:
                raise ValueError(f"{name} eigenvector count requires at least {3 * topk} points.")
            _, vectors = torch.lobpcg(affinity, k=topk, niter=10)
            del affinity
            vectors = vectors / (vectors.norm(dim=-1, keepdim=True) + 1e-16)
            weights.append(entropy_weight - compute_entropy(vectors, opt.postprocess_chunk_size).item())
            features.append(vectors)
    combined = torch.cat([feat * weight for feat, weight in zip(features, weights)], dim=-1)
    values = combined[0].detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("Clustering features contain nonfinite values.")
    print(f"Mean shift over all {len(values)} points; weights={weights}", flush=True)
    labels = mean_shift_labels(values, opt.bandwidth, opt.bin_seeding, opt.workers)
    return labels.astype(np.int64), weights
