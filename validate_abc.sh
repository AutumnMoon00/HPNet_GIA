#!/bin/bash -l

#SBATCH --job-name=HPNet_ABC_eval
#SBATCH --output=logs/hpnet_eval_%j.out
#SBATCH --error=logs/hpnet_eval_%j.err
#SBATCH --time=04:00:00

#SBATCH --qos=research
#SBATCH --gres=gpu:1
#SBATCH -c 16

set -eo pipefail

CHECKPOINT_PATH="${1:-/home/wm-sharath/fib/HPNet_GIA/model_ABCParts/abc_normal/abc_normal}"
DATA_PATH="${2:-/data/users/wm-sharath/fib/ABC_final/}"
VAL_SKIP="${3:-100}"
EXPERIMENT_NAME="${4:-abcparts_eval}"
if [[ ! "$EXPERIMENT_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "Experiment name must contain only letters, digits, underscores and hyphens" >&2
    exit 2
fi

# Slurm executes a spooled copy of this script, so BASH_SOURCE is not the repo path.
cd "${PROJECT_DIR:-/home/wm-sharath/fib/HPNet_GIA}"

mkdir -p outputs
OUTPUT_DIR="outputs/${EXPERIMENT_NAME}_${SLURM_JOB_ID:?Run with sbatch}"
mkdir "$OUTPUT_DIR"
echo "Prediction files: ${OUTPUT_DIR}"

# Initialize Conda in the batch shell and activate the project environment.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hpnet_l40s

set -u

srun python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name(0)}")'

srun python train.py \
    --eval \
    --checkpoint_path="${CHECKPOINT_PATH}" \
    --data_path="${DATA_PATH}" \
    --val_skip="${VAL_SKIP}" \
    --output_dir="${OUTPUT_DIR}"
