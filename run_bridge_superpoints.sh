#!/bin/bash -l

#SBATCH --job-name=bridge_superpoints_2
#SBATCH --output=logs/bridge_superpoints_%j.out
#SBATCH --error=logs/bridge_superpoints_%j.err
#SBATCH --time=06:00:00
#SBATCH --qos=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Submit from the project directory; Slurm needs logs before the job starts:
#   cd /home/wm-sharath/fib/HPNet_GIA
#   mkdir -p logs
#   sbatch run_bridge_superpoints.sh
# Adds superpoints_2 to existing files; excludes classes -1, 9, and 22.
# Existing fields are preserved; completed clouds are skipped.
# Optional environment overrides: PROJECT_DIR, CONDA_ENV, INPUT_DIR.
# Extra Python arguments are forwarded, for example:
#   sbatch run_bridge_superpoints.sh --normal-field normals_0.100
# Each file is updated through a temporary copy in the same directory.

set -eo pipefail

# Slurm runs a spooled copy, so BASH_SOURCE does not identify the repo path.
cd "${PROJECT_DIR:-/home/wm-sharath/fib/HPNet_GIA}"
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SLURM_CPUS_PER_TASK:?Request CPUs with --cpus-per-task}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-hpnet_l40s}"
set -u

INPUT_DIR="${INPUT_DIR:-/data/users/wm-sharath/fib/01-BridgePatchDataset.geon/org_copy_superpoints}"
SCRIPT_PATH="bridge_data_preparation/03_update_new_superpoints_3_excluded_classes.py"
[[ -r "$SCRIPT_PATH" ]] || { echo "Cannot read script: $PWD/$SCRIPT_PATH" >&2; exit 1; }

# Files are processed sequentially; Geon uses the allocated CPUs internally.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

echo "Job: ${SLURM_JOB_ID}; CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Updating files in: ${INPUT_DIR}; adding superpoints_2 (excluded: -1, 9, 22)"

srun python -u "$SCRIPT_PATH" \
    --input-dir "$INPUT_DIR" \
    "$@"
