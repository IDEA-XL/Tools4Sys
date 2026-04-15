#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -J progen2smk
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_sgrpo/logs/train_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_sgrpo/logs/train_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:-/public/home/xinwuye/ai4s-tool-joint-train/genmol/configs/progen2_sgrpo_small_smoke.yaml}
MIHOMO_DIR=/public/home/xinwuye/mihomo

export WANDB_PROJECT="${WANDB_PROJECT:-genmol}"
export WANDB_NAME="${WANDB_NAME:-progen2-sgrpo-small-smoke}"

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:$PATH
export PYTHONPATH=${REPO_DIR}/src
export TOKENIZERS_PARALLELISM=false

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login --relogin "${WANDB_API_KEY}" >/dev/null 2>&1 || true
fi

mkdir -p /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_sgrpo/logs

if ! pgrep -u "$USER" -f "${MIHOMO_DIR}/mihomo" >/dev/null 2>&1; then
  nohup "${MIHOMO_DIR}/mihomo" -d "${MIHOMO_DIR}" >/tmp/mihomo_${SLURM_JOB_ID}.log 2>&1 &
  sleep 5
fi

export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=${http_proxy}
export HTTPS_PROXY=${https_proxy}

cd "${REPO_DIR}"

python scripts/train_progen2_sgrpo.py "${CONFIG_PATH}"
