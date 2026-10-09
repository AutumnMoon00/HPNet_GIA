#!/bin/bash -l
#SBATCH --job-name=HPNet_bridge_infer
#SBATCH --output=logs/hpnet_bridge_%j.out
#SBATCH --error=logs/hpnet_bridge_%j.err
#SBATCH --time=04:00:00
#SBATCH --qos=research
#SBATCH --gres=gpu:1
#SBATCH -c 16

set -eo pipefail
cd "${PROJECT_DIR:-/home/wm-sharath/fib/HPNet_GIA}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hpnet_l40s
set -u
mkdir -p outputs
OUTPUT_DIR="outputs/bridge_inference_${SLURM_JOB_ID:?Run with sbatch}"
mkdir "$OUTPUT_DIR"
srun python -u train_bridge.py --output_dir "$OUTPUT_DIR" "$@"
