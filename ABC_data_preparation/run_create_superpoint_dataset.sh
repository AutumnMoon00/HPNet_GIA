#!/bin/bash -l

#SBATCH --job-name=ABC_sp_level
#SBATCH --output=logs/abc_sp_level_%j.out
#SBATCH --error=logs/abc_sp_level_%j.err
#SBATCH --time=12:00:00
#SBATCH --qos=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Submit from the repository root after creating logs:
#   mkdir -p logs
#   sbatch ABC_data_preparation/run_create_superpoint_dataset.sh --max-objects 10
# Full dataset (run only when ready):
#   sbatch ABC_data_preparation/run_create_superpoint_dataset.sh
# Optional environment overrides: INPUT_DIR, OUTPUT_DIR, CONDA_ENV, PROJECT_DIR.

set -eo pipefail
cd "${PROJECT_DIR:-/home/wm-sharath/fib/HPNet_GIA}"
: "${SLURM_JOB_ID:?Submit this script with sbatch}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-hpnet_l40s}"
set -u

INPUT_DIR="${INPUT_DIR:-/data/users/wm-sharath/fib/ABC_final_geon_superpoints_combined}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/users/wm-sharath/fib/ABC_superpoint_level_reg_0.025}"

# One native thread is sufficient; the preprocessing runs sequentially.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

echo "Job: ${SLURM_JOB_ID}; CPUs: ${SLURM_CPUS_PER_TASK:-2}"
echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"

srun python -u ABC_data_preparation/create_superpoint_dataset.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
