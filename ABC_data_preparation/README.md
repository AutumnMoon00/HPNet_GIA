# ABCParts superpoint-level dataset (experiment E0)

`create_superpoint_dataset.py` converts each supplied GEON ABCParts document
into a new GEON document with one centroid per precomputed graph-cut superpoint
at regularization **0.025**. It does not run graph cut or change HPNet training.
Use the existing `hpnet_l40s` Conda environment (Python 3.10, GEON, NumPy,
h5py); the base Python environment does not include these packages.

## Inputs and point order

The input is
`/data/users/wm-sharath/fib/ABC_final_geon_superpoints_combined/*_geon_superpoints.h5`.
Its `document/<pointcloud>/points` and `fields/{normals,labels,prim,T_param,
superpoints_reg_0.025}/data` datasets each have one row per original ABCParts
point. The original ABC file uses root datasets `points`, `normals`, `labels`,
`prim`, and `T_param`. The preprocessor verifies point order and exact label,
primitive type, and parameter correspondence against that original file. It
also checks that input normals match the original normals after normalization.
Use `--original-dir` when the `source_abc_file` metadata path is unavailable.

## Output format

Each `*_superpoint_level.h5` has a native GEON `/document` and a separate
`/abcparts` root group. GEON loads `/document` and ignores `/abcparts`.
`M` varies by object and `N` is its original point count.

| Path within `/document/<pointcloud>` | Shape, dtype | Meaning |
| --- | --- | --- |
| `points` | `(M, 3)`, float32 | Mean XYZ of each superpoint; GEON centroid geometry. |
| `fields/normals/data` | `(M, 3)`, float32, NORMAL | Unit length mean source normal, with orientation unchanged. Zero when the mean magnitude is at most `1e-8`. |
| `fields/primitive_instance/data` | `(M, 1)`, int32, INSTANCE | Majority original `labels` ID. A tie selects the smallest ID. |
| `fields/primitive_type/data` | `(M,)`, int32, SEMANTIC | Majority `prim` type among points with the winning instance label. A tie selects the smallest type ID. The GEON field includes a `semantic_schema` with HPNet IDs and names. |
| `fields/sp_count/data` | `(M, 1)`, int32, SCALAR | Number of original points in the superpoint. |
| `fields/sp_purity/data` | `(M, 1)`, float32, SCALAR | Winning instance count divided by `sp_count`, from 0 to 1. |
| `fields/sp_normal_valid/data` | `(M, 1)`, uint8, SCALAR | 1 for a unit length mean normal; 0 for a zero mean normal. |

Primitive IDs retain the source HPNet encoding: 0/6/7/9 = closed B-spline,
1 = plane, 2/8 = open B-spline, 3 = cone, 4 = cylinder, and 5 = sphere.
The schema also defines `-1` as unlabeled. No type IDs are remapped.

| Path within `/abcparts` | Shape, dtype | Meaning |
| --- | --- | --- |
| `point2sp` | `(N,)`, int32 | At original point-array index `i`, the new superpoint ID, consecutive from 0 to M-1. |
| `source_superpoint_id` | `(M,)`, int32 | Original graph-cut ID corresponding to each new superpoint ID. IDs are sorted. |
| `T_param` | `(N, F)`, original numeric dtype | Original per-point primitive parameters in original point order, without aggregation. Current ABCParts uses F=22. |

`/abcparts` attributes record the source ABC and GEON paths, N, M, the point
ordering convention, preprocessing version, graph-cut regularization and
settings, and a SHA-256 hash of the source float32 point array. Parameters
remain point-aligned so later training can access them via `point2sp`.

## Commands

From the repository root, in the GEON environment:

```bash
conda activate hpnet_l40s
python ABC_data_preparation/create_superpoint_dataset.py \
  --input-dir /data/users/wm-sharath/fib/ABC_final_geon_superpoints_combined \
  --output-dir ABC_data_preparation/examples --max-objects 10
python ABC_data_preparation/validate_superpoint_dataset.py ABC_data_preparation/examples
```

For the complete dataset, run this command deliberately when ready; it is
**not** started by this work:

```bash
python ABC_data_preparation/create_superpoint_dataset.py \
  --input-dir /data/users/wm-sharath/fib/ABC_final_geon_superpoints_combined \
  --output-dir /data/users/wm-sharath/fib/ABC_superpoint_level_reg_0.025
```

Existing outputs are skipped; `--overwrite` replaces them atomically after
full validation. `--max-objects` limits the sorted input list. For Slurm, run
`mkdir -p logs` and submit
`sbatch ABC_data_preparation/run_superpoint_dataset.sh --max-objects 10` for
a trial, or omit `--max-objects` for the full run. `INPUT_DIR`, `OUTPUT_DIR`,
`CONDA_ENV`, and `PROJECT_DIR` may be set when submitting.

## Validation and GEON resaves

The validator opens every file with `Document.load_hdf5`, checks GEON field
types and the semantic schema, and independently recomputes centroids,
normals, labels, purity, and counts from the source. It verifies the original
ABC point ordering and that `point2sp` reproduces the graph-cut IDs. The
source ABC and GEON files must remain accessible to run this full check.

GEON's `Document.save_hdf5` writes only `/document`, so resaving a file through
GEON drops `/abcparts`. Keep the original output. If GEON did not change the
cloud data, restore the mapping to a separate file and validate it:

```bash
python ABC_data_preparation/restore_mapping_after_geon_save.py \
  --original ABC_data_preparation/examples/00000_superpoint_level.h5 \
  --resaved /path/to/geon_resaved.h5 \
  --output /path/to/geon_resaved_with_mapping.h5
```

The helper rejects changed geometry or fields and writes the restored file
atomically. It never replaces the mapping source.

## Ten-file trial

The ten examples are in `ABC_data_preparation/examples/`. All loaded and
passed the validator: 1,397 total superpoints, 139.7 per object, 71.58 mean
points per superpoint, 0.9657 mean superpoint purity, and 0.9827 point-weighted
purity. GEON's Python loader recognized the point cloud, NORMAL, INSTANCE,
and SEMANTIC fields. A GEON resave and mapping restoration round trip passed.
An interactive GEON GUI window was not available in this headless session, so
the on-screen display itself was not verified.
