#!/bin/bash -l

#SBATCH --job-name=bridge_rename_superpoints
#SBATCH --output=logs/bridge_rename_superpoints_%j.out
#SBATCH --error=logs/bridge_rename_superpoints_%j.err
#SBATCH --time=00:30:00
#SBATCH --qos=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Rename superpoints to superpoints_1; superpoints_2 is preserved.
# Submit from the project directory after creating logs/:
#   sbatch run_bridge_rename_superpoints.sh
# Preview only:
#   sbatch run_bridge_rename_superpoints.sh --dry-run
# Optional environment overrides: PROJECT_DIR, CONDA_ENV, INPUT_DIR.

set -eo pipefail
cd "${PROJECT_DIR:-/home/wm-sharath/fib/HPNet_GIA}"
: "${SLURM_JOB_ID:?Submit this script with sbatch}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-hpnet_l40s}"
set -u

INPUT_DIR="${INPUT_DIR:-/data/users/wm-sharath/fib/01-BridgePatchDataset.geon/org_copy_superpoints}"
SCRIPT_PATH="bridge_data_preparation/04_rename_superpoints_to_superpoints_1.py"
[[ -r "$SCRIPT_PATH" ]] || { echo "Cannot read script: $PWD/$SCRIPT_PATH" >&2; exit 1; }

srun python -u "$SCRIPT_PATH" --input-dir "$INPUT_DIR" "$@"
