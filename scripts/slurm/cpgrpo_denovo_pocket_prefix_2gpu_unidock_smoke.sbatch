#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -J mmunidock
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/unidock_official/logs/smoke_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/unidock_official/logs/smoke_%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/genmol}
CONFIG_PATH=${CONFIG_PATH:-configs/cpgrpo_denovo_pocket_prefix_ng192_bs384_lr5e-5_beta5e-3_ni1_q03_sa02_unidock05_smoke.yaml}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-genmol_unidock}
MIHOMO_DIR=/public/home/xinwuye/mihomo

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"
export PATH=/opt/gridview/slurm/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /public/home/xinwuye/ai4s-tool-joint-train/runs/unidock_official/logs

if ! pgrep -u "$USER" -f "${MIHOMO_DIR}/mihomo" >/dev/null 2>&1; then
  nohup "${MIHOMO_DIR}/mihomo" -d "${MIHOMO_DIR}" >/tmp/mihomo_${SLURM_JOB_ID}.log 2>&1 &
  sleep 5
fi

export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=${http_proxy}
export HTTPS_PROXY=${https_proxy}

cd "${REPO_DIR}"
export PYTHONPATH=${REPO_DIR}/src

MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
accelerate launch \
  --config_file configs/accelerate_zero2.yaml \
  --num_processes 2 \
  --main_process_port "${MASTER_PORT}" \
  scripts/train_cpgrpo_denovo_pocket_prefix.py \
  --config "${CONFIG_PATH}"
