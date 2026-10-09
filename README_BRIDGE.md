# Bridge inference

The bridge pipeline is separate from the ABC training and evaluation pipeline.
All existing files are unchanged. `train_bridge.py` runs inference only: it
does not train, calculate losses, or require ground truth.

The defaults use the supplied bridge HDF5 file and the pretrained
`model_ABCParts/abc_normal/abc_normal` checkpoint. The bridge file contains
20,692 points in `document/PCD_0003/points`, with input normals in
`document/PCD_0003/fields/normals/data`. The loader detects this layout and
also supports flat HDF5 files with `points` and `normals` datasets. Use
`--cloud document/PCD_0003` to select a cloud when a file contains several.
Coordinates are used as supplied; normals are normalized to unit length.
Existing class, instance, and superpoint fields are retained as source data
and are never used as targets or model inputs.

Run on the cluster from this repository:

```bash
mkdir -p logs
sbatch infer_bridge.sh
```

For faster mean shift, use one actual feature vector from each occupied bin
as a seed:

```bash
sbatch infer_bridge.sh --bin_seeding
```

This changes the initialization of clustering and can change instance IDs
and boundaries. Every point still receives its own network prediction and
cluster assignment. The default uses every feature vector as a clustering
seed, matching the ABC pipeline.

To run directly in an allocated GPU environment:

```bash
conda activate hpnet_l40s
python train_bridge.py --output_dir outputs/bridge_inference
```

Override `--data_path /path/to/cloud.h5` or
`--checkpoint_path /path/to/checkpoint` as needed. Output files are never
overwritten; choose a new output directory for another run.

`models/dgcnn_bridge.py` retains the original model architecture and loads
the checkpoint strictly. It removes the random 7,000-point selection and
returns all input rows in their original order. Nearest neighbors are exact
and search the entire cloud; `--knn_chunk_size` controls only memory use.
The full cloud also participates in GroupNorm and global pooling, rather
than being split into separate inference clouds.

`utils/bridge_postprocess.py` combines network embeddings, geometric
primitive affinity, and normal affinity using the HPNet spectral/mean shift
pipeline. Its entropy calculation handles the actual point count, including
the last partial block, and diagonal affinity normalization avoids cubic
dense matrix products. Hybrid postprocessing still needs quadratic storage
for affinities and CUDA for the original spline fitting helpers. The spline
checkpoints must exist in `log/pretrained_models/`.

`--postprocess embedding` uses network features alone for instance clustering;
it omits the two spectral geometry features and therefore produces different
segmentation. This mode also supports `--device cpu`. Smaller
`--postprocess_chunk_size` values reduce temporary memory use. Neither option
reduces point coverage.

Each successful run produces:

- `*_network.npz`: all direct network predictions, saved before clustering.
- `*_predictions.h5`: the complete source HDF5 hierarchy plus root-level
  `points`, `normals`, and the prediction fields below.
- `*_predictions_instances.ply`: all points colored by predicted instance.
- `*_predictions_primitives.ply`: all points colored by primitive class.
- `*_predictions.json`: point counts, primitive counts, instance count,
  checkpoint, clustering settings, and elapsed time.

Open the colored PLY files in CloudCompare, MeshLab, or Open3D to inspect
the result. Logs from cluster runs are `logs/hpnet_bridge_<job-id>.out`
and `.err`; outputs go to `outputs/bridge_inference_<job-id>/`.

Prediction fields in HDF5 and NPZ are point aligned:

| Field | Shape | Meaning |
| --- | --- | --- |
| `source_index` | N | Original row indices, exactly `0..N-1` |
| `labels_pred` | N | Arbitrary predicted instance IDs (HDF5 only) |
| `prim_pred` | N | ABC validation primitive class IDs |
| `prim_pred_raw` | N | Original ten-class model argmax |
| `type_prob_pred` | N × 10 | Probabilities before class remapping |
| `embedding_pred` | N × 128 | Network embeddings |
| `normals_pred` | N × 3 | Predicted unit normals |
| `T_param_pred` | N × 22 | Predicted parameters in input coordinates |

Mapped primitive IDs are 0 = closed B-spline related, 1 = plane,
2 = open B-spline related, 3 = cone, 4 = cylinder, and 5 = sphere.
The parameter vector contains sphere `[0:4]`, plane `[4:8]`, cylinder
`[8:15]`, and cone `[15:22]` parameters. This checkpoint predicts ABC
geometric primitives, rather than bridge semantic component names.
