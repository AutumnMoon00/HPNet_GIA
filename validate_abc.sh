#!/bin/bash -l

#SBATCH --job-name=HPNet_ABC_eval_skip_100
#SBATCH --output=logs/hpnet_eval_%j.out
#SBATCH --error=logs/hpnet_eval_%j.err
#SBATCH --time=04:00:00

#SBATCH --qos=research
#SBATCH --gres=gpu:1
#SBATCH -c 16

set -euo pipefail

CHECKPOINT_PATH="${1:-/home/wm-sharath/fib/HPNet_GIA/model_ABCParts/abc_normal/abc_normal}"
DATA_PATH="${2:-/data/users/wm-sharath/fib/ABC_final/}"

# Run from the directory from which sbatch was submitted, so relative paths work.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Initialize Conda in the batch shell and activate the project environment.
# source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hpnet_abc

srun python train.py \
    --eval \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --data_path="${DATA_PATH}" \
    --val_skip=100
