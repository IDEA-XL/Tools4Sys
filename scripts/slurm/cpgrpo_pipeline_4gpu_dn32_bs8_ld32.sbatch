#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH -t 12:00:00
#SBATCH -J cpgrpo_pipe4g
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_pipeline/logs/cpgrpo_pipe4g_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_pipeline/logs/cpgrpo_pipe4g_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:-configs/cpgrpo_pipeline_4gpu_dn32_bs8_ld32_ch64_lr5e-5_beta5e-3.yaml}
LOG_DIR=/public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_pipeline/logs
MIHOMO_DIR=/public/home/xinwuye/mihomo

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is required" >&2
  exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-tools4sys}"
export WANDB_NAME="${WANDB_NAME:-cpgrpo-pipe-4g-dn32-bs8-ld32-ch64-lr5e-5-b5e-3}"

source ~/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:$PATH

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

export HF_HOME=/tmp/xinwuye_hf_cpgrpo_pipeline_${SLURM_JOB_ID}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export TRITON_CACHE_DIR=${HF_HOME}/triton_cache
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GENMOL_REWARD_WORKERS="${GENMOL_REWARD_WORKERS:-1}"
mkdir -p "${TRITON_CACHE_DIR}"

if [[ "${http_proxy:-}${https_proxy:-}${all_proxy:-}" == *"127.0.0.1:7897"* ]]; then
  if ! pgrep -u "$USER" -f "${MIHOMO_DIR}/mihomo" >/dev/null 2>&1; then
    nohup "${MIHOMO_DIR}/mihomo" -d "${MIHOMO_DIR}" >/tmp/mihomo_${SLURM_JOB_ID}.log 2>&1 &
    sleep 5
  fi
fi

MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
accelerate launch \
  --config_file configs/accelerate_ddp_4gpu.yaml \
  --num_processes 4 \
  --main_process_port "${MASTER_PORT}" \
  scripts/train_cpgrpo_pipeline.py \
  --config "${CONFIG_PATH}"
