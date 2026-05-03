#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J p2dynreduce
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_eval/logs/progen2_dynamics_reduce_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_eval/logs/progen2_dynamics_reduce_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:-configs/progen2_dynamics_20260504.yaml}
LOG_DIR=/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_eval/logs

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export PYTHONPATH=${REPO_DIR}/src:${REPO_DIR}/vis-code

mkdir -p "${LOG_DIR}"

cd "${REPO_DIR}"
python vis-code/progen2_dynamics_reduce.py --config "${CONFIG_PATH}"
