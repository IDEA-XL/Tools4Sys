#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -J mmgentune
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_supervised_tune/logs/train_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_supervised_tune/logs/train_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_NAME=${CONFIG_NAME:-base_pocket_prefix_tune_1gpu}
MIHOMO_DIR=/public/home/xinwuye/mihomo

if [[ -z "${WANDB_API_KEY:-}" ]] && [[ ! -f "${HOME}/.netrc" ]]; then
  echo "WANDB_API_KEY or existing wandb login (~/.netrc) is required" >&2
  exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-genmol}"
export WANDB_NAME="${WANDB_NAME:-pocket-prefix-supervised-tune}"

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:$PATH
export PYTHONPATH=${REPO_DIR}/src
export TOKENIZERS_PARALLELISM=false

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login --relogin "${WANDB_API_KEY}" >/dev/null 2>&1 || true
fi

mkdir -p /public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_supervised_tune/logs

if ! pgrep -u "$USER" -f "${MIHOMO_DIR}/mihomo" >/dev/null 2>&1; then
  nohup "${MIHOMO_DIR}/mihomo" -d "${MIHOMO_DIR}" >/tmp/mihomo_${SLURM_JOB_ID}.log 2>&1 &
  sleep 5
fi

export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=${http_proxy}
export HTTPS_PROXY=${https_proxy}

cd "${REPO_DIR}"

python scripts/train_pocket_prefix.py --config-name "${CONFIG_NAME}"
