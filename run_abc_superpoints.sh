#!/bin/bash -l

#SBATCH --job-name=ABC_superpoints
#SBATCH --output=logs/abc_superpoints_%j.out
#SBATCH --error=logs/abc_superpoints_%j.err
#SBATCH --time=06:00:00
#SBATCH --qos=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Submit from the project directory after creating logs:
#   mkdir -p logs
#   sbatch run_abc_superpoints.sh
# Extra arguments are forwarded to abc_superpoints_batch.py:
#   sbatch run_abc_superpoints.sh --limit 10 --progress-every 1
# Use CONDA_ENV to select another environment with geon, numpy and h5py.

set -eo pipefail

# Slurm runs a spooled copy, so BASH_SOURCE does not identify the repo path.
cd "${PROJECT_DIR:-/home/wm-sharath/fib/HPNet_GIA}"
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${SLURM_CPUS_PER_TASK:?Request CPUs with --cpus-per-task}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-hpnet_l40s}"
set -u

INPUT_DIR="${INPUT_DIR:-/data/users/wm-sharath/fib/ABC_final/}"
ABC_OUTPUT_DIR="${ABC_OUTPUT_DIR:-/data/users/wm-sharath/fib/ABC_final_abc_superpoints/}"
GEON_OUTPUT_DIR="${GEON_OUTPUT_DIR:-/data/users/wm-sharath/fib/ABC_final_geon_superpoints/}"

# Each Python worker uses one native thread; parallelism is across files.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

echo "Job: ${SLURM_JOB_ID}; CPUs/workers: ${SLURM_CPUS_PER_TASK}"
echo "Conda environment: ${CONDA_DEFAULT_ENV}"

srun python -u ABC_superpoints_code/abc_superpoints_batch.py \
    --input-dir "$INPUT_DIR" \
    --abc-output-dir "$ABC_OUTPUT_DIR" \
    --geon-output-dir "$GEON_OUTPUT_DIR" \
    --workers "$SLURM_CPUS_PER_TASK" \
    "$@"
