#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH -t 06:00:00
#SBATCH -J mmdrug8p
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/drugclip_official/logs/probe8gpu_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/drugclip_official/logs/probe8gpu_%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/genmol}
CONFIG_PATH=${CONFIG_PATH:-configs/cpgrpo_denovo_pocket_prefix_ng192_bs384_lr5e-5_beta5e-3_ni1_q03_sa02_drugclip05_probe8gpu.yaml}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-genmol_drugclip4}
GENMOL_REWARD_WORKERS=${GENMOL_REWARD_WORKERS:-7}
LOG_DIR=/public/home/xinwuye/ai4s-tool-joint-train/runs/drugclip_official/logs
MIHOMO_DIR=/public/home/xinwuye/mihomo

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"
export PATH=/opt/gridview/slurm/bin:$PATH

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

export HF_HOME=/tmp/xinwuye_hf_mmdrugclip8_${SLURM_JOB_ID}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GENMOL_REWARD_WORKERS

if [[ "${http_proxy:-}${https_proxy:-}${all_proxy:-}" == *"127.0.0.1:7897"* ]]; then
  if ! pgrep -u "$USER" -f "${MIHOMO_DIR}/mihomo" >/dev/null 2>&1; then
    nohup "${MIHOMO_DIR}/mihomo" -d "${MIHOMO_DIR}" >/tmp/mihomo_${SLURM_JOB_ID}.log 2>&1 &
    sleep 5
  fi
fi

MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
accelerate launch \
  --config_file configs/accelerate_zero2.yaml \
  --num_processes 8 \
  --main_process_port "${MASTER_PORT}" \
  scripts/train_cpgrpo_denovo_pocket_prefix.py \
  --config "${CONFIG_PATH}"
