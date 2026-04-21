#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -t 48:00:00
#SBATCH -J mmsg8g32
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_denovo_pocket_prefix/logs/train_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_denovo_pocket_prefix/logs/train_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:-configs/cpgrpo_denovo_pocket_prefix_sgrpo_ng32_sg8_bs512_lr5e-5_beta5e-3_gw09.yaml}
LOG_DIR=/public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_denovo_pocket_prefix/logs
MIHOMO_DIR=/public/home/xinwuye/mihomo

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is required" >&2
  exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-tools4sys}"
export WANDB_NAME="${WANDB_NAME:-cpgrpo-denovo-pocket-prefix-8gpu-sgrpo-ng32-sg8-bs512-gw09}"

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:$PATH

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

export HF_HOME=/tmp/xinwuye_hf_mmcpgrpo_${SLURM_JOB_ID}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
