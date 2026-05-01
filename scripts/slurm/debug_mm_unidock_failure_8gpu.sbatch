#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH -t 08:00:00
#SBATCH -J mmudbg8
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/unidock_official/logs/debug_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/unidock_official/logs/debug_%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/genmol}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-/public/home/xinwuye/ai4s-tool-joint-train/runs/unidock_official/conda_env}
SCRIPT_ARGS=${SCRIPT_ARGS:-}

if [[ -z "${SCRIPT_ARGS}" ]]; then
  echo "SCRIPT_ARGS is required" >&2
  exit 1
fi

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_PREFIX}"
export PATH=/opt/gridview/slurm/bin:$PATH

cd "${REPO_DIR}"

export HF_HOME=/tmp/xinwuye_hf_mmunidockdebug_${SLURM_JOB_ID}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GENMOL_REWARD_WORKERS=${GENMOL_REWARD_WORKERS:-7}

MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
accelerate launch \
  --config_file configs/accelerate_zero2.yaml \
  --num_processes 8 \
  --main_process_port "${MASTER_PORT}" \
  scripts/debug_mm_unidock_failure.py \
  ${SCRIPT_ARGS}
