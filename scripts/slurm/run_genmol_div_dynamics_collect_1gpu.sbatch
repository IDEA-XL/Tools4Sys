#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J gmdivcol
#SBATCH -o /public/home/xinwuye/genmol_div_dynamics_collect_%A_%a.out
#SBATCH -e /public/home/xinwuye/genmol_div_dynamics_collect_%A_%a.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:-${REPO_DIR}/configs/genmol_div_dynamics_20260504.yaml}

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:$PATH
unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=${REPO_DIR}/src:${REPO_DIR}/vis-code
export TOKENIZERS_PARALLELISM=false

cd /public/home/xinwuye
python "${REPO_DIR}/vis-code/genmol_div_dynamics_collect.py" --config "${CONFIG_PATH}" --model-index "${SLURM_ARRAY_TASK_ID}"
