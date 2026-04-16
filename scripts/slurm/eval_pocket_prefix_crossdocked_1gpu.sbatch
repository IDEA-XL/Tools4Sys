#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J mmxdeckeval
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_eval_pocket_prefix_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_eval_pocket_prefix_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:-/public/home/xinwuye/ai4s-tool-joint-train/genmol/configs/eval_pocket_prefix_crossdocked_template.yaml}
EVAL_CONDA_PREFIX=${EVAL_CONDA_PREFIX:-/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/conda_env}

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
if [[ ! -d "${EVAL_CONDA_PREFIX}" ]]; then
  echo "Missing evaluation conda env: ${EVAL_CONDA_PREFIX}" >&2
  exit 1
fi
conda activate "${EVAL_CONDA_PREFIX}"
export PATH=/opt/gridview/slurm/bin:$PATH
export TOKENIZERS_PARALLELISM=false

cd "${REPO_DIR}"
python scripts/eval_pocket_prefix_crossdocked.py --config "${CONFIG_PATH}"
