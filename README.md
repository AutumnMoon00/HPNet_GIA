# HPNet

This repository contains the PyTorch implementation of paper: [HPNet: Deep Primitive Segmentation Using Hybrid Representations](https://arxiv.org/abs/2105.10620).

<div align="center">
  <img width="100%" alt="HPNet Pipeline" src="imgs/architecture_2.jpg">
</div>

### Installation

The main experiments are implemented on pytorch 1.7.0, tensorflow 1.15.0. Please install the dependancy packages using `pip install -r requirements.txt`.

### Dataset

#### ABCParts Dataset
ABCParts Dataset is made by [ParseNet](https://arxiv.org/abs/2003.12181). Please download our preprocessed dataset [here](https://drive.google.com/file/d/1qH-1A8p3jDtTxS2i-423AZTjiBu9RGL1/view?usp=sharing)(69G) and put it under data/ABC folder. We add primitive parameters of each object in this dataset.

We also provide the preprocessing scripts under `utils` folder. To process by yourself, please run
```
cd utils
python process_abc.py --data_path=/path/to/parsenet-codebase/data/shapes --save_path=/path/to/saved/dir
```

### Usage

To train our model on ABC dataset: run
```
python train.py --data_path=./path/to/dataset`
```

To evaluate our model on ABC dataset: run 
```
python train.py --eval --checkpoint_path=./path/to/pretrained/model --val_skip=100
```
on the subset of test dataset. To test on the full dataset, simply set `val_skip=1`.

**pretrained models**

We provide pre-trained model on ABC Dataset [here](https://drive.google.com/file/d/1fj84kyD9CGT8j61IW-xSWZ5q4q5IpoYx/view?usp=sharing). This should generate the result reported in the paper.

### Acknowledgements

We would like to thank and acknowledge referenced codes from 

1. ParseNet: https://github.com/Hippogriff/parsenet-codebase.

2. DGCNN: https://github.com/WangYueFt/dgcnn.


### Citations

If you find this repository useful in your research, please cite:

```
@article{yan2021hpnet,
  title={HPNet: Deep Primitive Segmentation Using Hybrid Representations},
  author={Yan, Siming and Yang, Zhenpei and Ma, Chongyang and Huang, Haibin and Vouga, Etienne and Huang, Qixing},
  journal={arXiv preprint arXiv:2105.10620},
  year={2021}
}
```

### ABCParts evaluation on L40S

The original environment uses PyTorch 1.7 and CUDA 11.0. Use the L40S
specific environment and the validation script in this branch:

```bash
conda env create -f environment-l40s.yml
mkdir -p logs
sbatch validate_abc.sh
```

The script defaults to the pretrained ABC checkpoint in
`model_ABCParts/abc_normal/abc_normal`, the dataset at
`/data/users/wm-sharath/fib/ABC_final/`, and a validation stride of 100.
Each Slurm run writes one HDF5 file per evaluated shape to
`outputs/<experiment-name>_<job-id>/`. The optional fourth argument sets the
experiment name (default `abcparts_eval`). For a five-shape inspection run:

```bash
sbatch validate_abc.sh /path/to/checkpoint /path/to/ABC_final 1000 inspection
```

Each exported file contains the original `points`, `normals`, `labels`, `prim`,
and `T_param` fields for the 7,000 selected points, plus `labels_pred` and
`prim_pred`. The `source_index` field gives each selected point's row in the
original 10,000-point HDF5 file. Instance IDs in `labels_pred` are arbitrary
cluster IDs; validation compares them to `labels` after matching clusters.
`prim_pred` uses the same class mapping as the validation metric.

Pass a stride of 1 to evaluate the complete test split:

```bash
sbatch --time=24:00:00 validate_abc.sh /path/to/checkpoint /path/to/ABC_final 1
```

The full split is much slower than the default subset; the longer walltime overrides the script's four-hour default.

Metrics are printed in `logs/hpnet_eval_<job-id>.out`. The open and closed
spline checkpoints must be present in `log/pretrained_models/`. This branch
uses SciPy for assignment and a CPU eigensolver for 3-by-3 PCA, so numerical
scores may differ slightly from the original environment.
