#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH -t 00:30:00
#SBATCH -J cpgrpo_p2lop
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_pipeline/logs/cpgrpo_p2lop_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_pipeline/logs/cpgrpo_p2lop_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:-configs/cpgrpo_pipeline_2gpu_dn16_bs8_ld16_ch32_lr5e-5_beta5e-3_nogc.yaml}
LOG_DIR=/public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_pipeline/logs

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
export GENMOL_DEBUG_ANOMALY="${GENMOL_DEBUG_ANOMALY:-1}"
mkdir -p "${TRITON_CACHE_DIR}"

MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
accelerate launch \
  --config_file configs/accelerate_ddp_2gpu.yaml \
  --num_processes 2 \
  --main_process_port "${MASTER_PORT}" \
  scripts/debug_pipeline_lead_only_prepare.py \
  --config "${CONFIG_PATH}"
