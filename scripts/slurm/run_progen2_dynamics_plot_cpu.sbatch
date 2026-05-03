#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH -J p2dynplot
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_eval/logs/progen2_dynamics_plot_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_eval/logs/progen2_dynamics_plot_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:-configs/progen2_dynamics_20260504.yaml}
LOG_DIR=/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_eval/logs

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${REPO_DIR}/vis-code

mkdir -p "${LOG_DIR}"

cd "${REPO_DIR}"
python vis-code/progen2_dynamics_plot.py --config "${CONFIG_PATH}"
