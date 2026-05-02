#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -J p2grpo2s
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_sgrpo/logs/progen2_grpo_2gpu_smoke_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_sgrpo/logs/progen2_grpo_2gpu_smoke_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:-configs/progen2_grpo_ng96_bs2_len256_rbs16_hbd_smoke10.yaml}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
PYTHON_OVERLAY_DIR=${PYTHON_OVERLAY_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/python_overlay}
PROTBERT_DIR=${PROTBERT_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/prot_bert_bfd}
CUDA_TOOLCHAIN_DIR=${CUDA_TOOLCHAIN_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/cuda_toolchain_cu124}
MIHOMO_DIR=/public/home/xinwuye/mihomo
LOG_DIR=/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_sgrpo/logs

export WANDB_PROJECT="${WANDB_PROJECT:-progen2}"
export WANDB_NAME="${WANDB_NAME:-progen2-grpo-2gpu-hbd-smoke}"
export PYTHON_OVERLAY_DIR
export PROTBERT_DIR

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
TORCH_LIB_DIR=$(python - <<'PY'
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent / "lib")
PY
)
if [[ ! -d "${TORCH_LIB_DIR}" ]]; then
  echo "Missing torch library directory: ${TORCH_LIB_DIR}" >&2
  exit 1
fi
CUDA_LIB_DIR=${CUDA_TOOLCHAIN_DIR}/lib
if [[ ! -d "${CUDA_LIB_DIR}" ]]; then
  echo "Missing CUDA toolchain library directory: ${CUDA_LIB_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CUDA_LIB_DIR}/libcudart.so.12" ]]; then
  echo "Missing CUDA runtime library: ${CUDA_LIB_DIR}/libcudart.so.12" >&2
  exit 1
fi
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login --relogin "${WANDB_API_KEY}" >/dev/null 2>&1 || true
fi

mkdir -p "${LOG_DIR}"

if ! pgrep -u "$USER" -f "${MIHOMO_DIR}/mihomo" >/dev/null 2>&1; then
  nohup "${MIHOMO_DIR}/mihomo" -d "${MIHOMO_DIR}" >/tmp/mihomo_${SLURM_JOB_ID}.log 2>&1 &
  sleep 5
fi

export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=${http_proxy}
export HTTPS_PROXY=${https_proxy}

if [[ ! -d "${PYTHON_OVERLAY_DIR}" ]]; then
  echo "Missing ProGen2 python overlay: ${PYTHON_OVERLAY_DIR}" >&2
  exit 1
fi
if [[ ! -d "${PROTBERT_DIR}" ]]; then
  echo "Missing preprovisioned ProtBERT directory: ${PROTBERT_DIR}" >&2
  exit 1
fi
for required_file in config.json pytorch_model.bin vocab.txt tokenizer_config.json special_tokens_map.json; do
  if [[ ! -f "${PROTBERT_DIR}/${required_file}" ]]; then
    echo "Incomplete ProtBERT directory: missing ${PROTBERT_DIR}/${required_file}" >&2
    exit 1
  fi
done
if [[ ! -d "${PYTHON_OVERLAY_DIR}/openfold" ]]; then
  echo "Missing preprovisioned openfold package under overlay: ${PYTHON_OVERLAY_DIR}/openfold" >&2
  exit 1
fi
if ! compgen -G "${PYTHON_OVERLAY_DIR}/attn_core_inplace_cuda*.so" >/dev/null; then
  echo "Missing compiled OpenFold attention extension under overlay: ${PYTHON_OVERLAY_DIR}/attn_core_inplace_cuda*.so" >&2
  exit 1
fi

export PYTHONPATH=${PYTHON_OVERLAY_DIR}:${REPO_DIR}/src

cd "${REPO_DIR}"

python - <<'PY'
import importlib
import os
import sys

overlay_dir = os.environ['PYTHON_OVERLAY_DIR']
sys.path.insert(0, overlay_dir)
module = importlib.import_module('attn_core_inplace_cuda')
print(module.__file__)
PY

MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
TRAIN_ARGS=("${CONFIG_PATH}")
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=("--resume_from_checkpoint" "${RESUME_FROM_CHECKPOINT}")
fi
accelerate launch \
  --config_file configs/accelerate_ddp_2gpu.yaml \
  --num_processes 2 \
  --main_process_port "${MASTER_PORT}" \
  scripts/train_progen2_sgrpo.py \
  "${TRAIN_ARGS[@]}"
