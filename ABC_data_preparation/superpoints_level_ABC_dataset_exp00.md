Task: Create a Superpoint-Level ABCParts Dataset for HPNet Experiment E0

Context

I am working on adapting HPNet for primitive instance segmentation using superpoints instead of individual points.

My repository is HPNet_GIA, and I am currently working on the experimental branch exp-00.

I already have:

1. The original ABCParts dataset, with approximately 10,000 points per CAD object.
2. Point-level XYZ coordinates, surface normals, primitive-instance labels, primitive-type labels, primitive parameters, purity (at 3 regularizations), superpoint ids (at 3 regularizations)
3. Precomputed graph-cut superpoint segmentation results for the ABCParts dataset.
4. A requirement to visualize and inspect the generated data using GEON: https://github.com/biophase/geon

Step 1: Inspect the Existing Repository

Before implementing anything:

- Examine the existing ABCParts data-loading code.
- Identify how the original point clouds and their labels are stored. I want superpoints at regularization of 0.025
- Identify where and how the graph-cut superpoint assignments are stored.
- Verify the correspondence between original point indices and graph-cut superpoint IDs.
- Identify existing preprocessing scripts, utility functions, and naming conventions that can be reused.
- Examine GEON's HDF5 implementation, particularly: 
    - src/geon/data/document.py
    - src/geon/data/pointcloud.py
    - src/geon/data/base.py
    - src/geon/data/registry.py
- Check how primitive parameters are represented.
- Identify the actual GEON-compatible HDF5 structure, supported field types, and serialization methods.

Do not assume arbitrary HDF5 structures can be loaded by GEON.


Step 2: Generate Superpoint-Level Data

For each ABCParts object, calculate the following:

Superpoint coordinates:
- Calculate the centroid of all original points belonging to each superpoint.
- Shape: (M, 3).
- Type: float32.
- Store as the GEON point-cloud points dataset.

Superpoint normals:
- Average the original point normals within each superpoint.
- Normalize the resulting mean vector to unit length.
- If the mean magnitude is close to zero, store a zero vector and mark the normal as invalid.
- Shape: (M, 3).
- Field name: normals.
- GEON FieldType: NORMAL.

Do not flip normals arbitrarily. Preserve the original orientation convention.

Primitive-instance labels:
- Assign the majority primitive-instance label to each superpoint.
- Preserve the original instance-label identifiers.
- Use deterministic tie-breaking.
- Field name: primitive_instance.
- GEON FieldType: INSTANCE.

Primitive-type labels:
- Determine the primitive type associated with the majority primitive instance.
- Preserve the original HPNet primitive-type encoding.
- Field name: primitive_type.
- GEON FieldType: SEMANTIC.
- Construct a valid GEON SemanticSchema using the actual primitive-type IDs and class names in the original dataset.


Additional fields
Store the following as GEON scalar fields:


| Field | Description |
|---|---|
| `sp_count` | Number of original points in the superpoint |
| `sp_purity` | Fraction of original points belonging to the majority primitive instance |
| `sp_normal_valid` | Whether the aggregated normal is valid |

Use the appropriate numeric data types.

Ensure fields have compatible dimensions for GEON's PointCloudData.add_field() method. In particular, use (M, 1) for scalar fields where required.

Do not average primitive parameters blindly. Preserve access to the original primitive parameters for later HPNet training.

Step 3: Store Point-to-Superpoint Mapping

The original ABCParts points do not have explicit point IDs, so their original array indices must be used.

Example:

point2sp = np.array([0, 0, 1, 1, 2, 2])

Superpoint IDs must be consecutive from 0 to M-1.

Important: Do not store point2sp as a GEON field of the superpoint point cloud because its length N differs from the M superpoints.

Instead, store it under the additional HDF5 root group.
Also store metadata identifying:
- The corresponding original ABCParts object.
- Original point count N.
- Superpoint count M.
- Graph-cut configuration, if available.
- Original point-ordering convention.
- Preprocessing version.

The extra new group must not interfere with GEON's document loading.

Verify that GEON can ignore this additional root group when loading the document.

Also check whether resaving the file through GEON preserves this custom group. If not, document that limitation and provide a safe way to preserve the mapping.


Step 4: Implementation Requirements

- Use h5py and GEON's native serialization API where appropriate.
- Do not recompute graph-cut superpoints.
- Do not alter the HPNet architecture or training code.
- Support variable numbers of superpoints.
- Allow skipping already processed files.
- Add command-line arguments for input paths, output path, overwrite behavior, and a maximum number of objects.
- Store all numeric arrays with explicit dtypes.
- Ensure compatibility with the existing repository environment. If GEON requires a newer Python environment than HPNet, implement preprocessing in a separate compatible environment rather than upgrading HPNet's training environment.

Step 5: Validation
For every generated file, verify:
- points.shape == (M, 3)
- normals.shape == (M, 3)
- len(point2sp) == N
- sum(sp_count) == N
- All superpoints contain at least one point.
- All superpoint indices are valid.
- All purity values are between 0 and 1.
- All valid normals are unit vectors.
- All stored coordinates and feature values are finite.
- Primitive-instance and primitive-type labels preserve the original encoding.
- The original point ordering matches the stored mapping.

Step 6: Testing and deliverables
Before processing the complete dataset:
- Generate 10 example HDF5 files.
- Validate their geometry, labels, and superpoint mappings.
- Confirm that GEON successfully opens and displays the superpoint centroids.
- Confirm that primitive-instance and primitive-type fields are correctly recognized.
- Report superpoint counts, mean size, mean purity, and point-weighted purity.

also generate a new sh file, take inspiration from one of the sh files in the repo which can run the code on server by asking for appropriate number of resources. 

Deliver:
- A complete preprocessing Python script.
- A validation script.
- Example commands for processing 10 files and the entire dataset.
- A brief explanation of the output HDF5 structure.
- A README documenting every generated field and its meaning.

Do not begin full-dataset processing automatically.
Prioritize correctness, reproducibility, GEON compatibility, and preservation of the original point-to-superpoint mapping.