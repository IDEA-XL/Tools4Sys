#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH -t 48:00:00
#SBATCH -J mmsguni
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/unidock_official/logs/train_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/unidock_official/logs/train_%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/genmol}
CONFIG_PATH=${CONFIG_PATH:-configs/cpgrpo_denovo_pocket_prefix_sgrpo_ng24_sg8_bs384_lr5e-5_beta5e-3_gw09_q03_sa02_unidock05.yaml}
DEFAULT_CONDA_ENV_PREFIX=/public/home/xinwuye/ai4s-tool-joint-train/runs/unidock_official/conda_env
CONDA_ENV_NAME=${CONDA_ENV_NAME:-}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-}
GENMOL_REWARD_WORKERS=${GENMOL_REWARD_WORKERS:-7}
WANDB_NETRC_PATH=${WANDB_NETRC_PATH:-${HOME}/.netrc}
LOG_DIR=/public/home/xinwuye/ai4s-tool-joint-train/runs/unidock_official/logs
MIHOMO_DIR=/public/home/xinwuye/mihomo

if [[ -n "${CONDA_ENV_NAME}" && -n "${CONDA_ENV_PREFIX}" ]]; then
  echo "Set at most one of CONDA_ENV_NAME and CONDA_ENV_PREFIX" >&2
  exit 1
fi
if [[ -z "${CONDA_ENV_NAME}" && -z "${CONDA_ENV_PREFIX}" ]]; then
  CONDA_ENV_PREFIX=${DEFAULT_CONDA_ENV_PREFIX}
fi

if [[ -z "${WANDB_API_KEY:-}" && ! -f "${WANDB_NETRC_PATH}" ]]; then
  echo "Either WANDB_API_KEY or ~/.netrc is required for wandb auth" >&2
  exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-tools4sys}"
export WANDB_NAME="${WANDB_NAME:-cpgrpo-denovo-pocket-prefix-8gpu-sgrpo-ng24-sg8-bs384-unidock}"

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
if [[ -n "${CONDA_ENV_PREFIX}" ]]; then
  if [[ ! -d "${CONDA_ENV_PREFIX}" ]]; then
    echo "Conda env prefix not found: ${CONDA_ENV_PREFIX}" >&2
    exit 1
  fi
  conda activate "${CONDA_ENV_PREFIX}"
else
  conda activate "${CONDA_ENV_NAME}"
fi
export PATH=/opt/gridview/slurm/bin:$PATH

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

export HF_HOME=/tmp/xinwuye_hf_mmsgunidock8_${SLURM_JOB_ID}
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

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login --relogin "${WANDB_API_KEY}" >/dev/null 2>&1 || true
fi

MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
accelerate launch \
  --config_file configs/accelerate_zero2.yaml \
  --num_processes 8 \
  --main_process_port "${MASTER_PORT}" \
  scripts/train_cpgrpo_denovo_pocket_prefix.py \
  --config "${CONFIG_PATH}"
