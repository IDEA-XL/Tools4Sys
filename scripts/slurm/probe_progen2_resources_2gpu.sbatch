#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH -t 24:00:00
#SBATCH -J p2probe2g
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_progen2probe2g_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_progen2probe2g_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:-configs/progen2_sgrpo_2gpu_probe_base.yaml}
PROBE_MODE=${PROBE_MODE:-calibration_prompt_batch_size}
PYTHON_OVERLAY_DIR=${PYTHON_OVERLAY_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/python_overlay}
PROTBERT_DIR=${PROTBERT_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/prot_bert_bfd}
CUDA_TOOLCHAIN_DIR=${CUDA_TOOLCHAIN_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/cuda_toolchain_cu124}
MIHOMO_DIR=/public/home/xinwuye/mihomo
OUTPUT_ROOT=${OUTPUT_ROOT:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_batch_probe_2gpu/${PROBE_MODE}_${SLURM_JOB_ID}}
CALIBRATION_INITIAL=${CALIBRATION_INITIAL:-128}
CALIBRATION_MIN=${CALIBRATION_MIN:-1}
REWARD_INITIAL=${REWARD_INITIAL:-16}
REWARD_MAX=${REWARD_MAX:-1024}
FIXED_CALIBRATION_PROMPT_BATCH_SIZE=${FIXED_CALIBRATION_PROMPT_BATCH_SIZE:-}

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

cmd=(
  python scripts/probe_progen2_2gpu_resources.py
  --config "${CONFIG_PATH}"
  --output-root "${OUTPUT_ROOT}"
  --mode "${PROBE_MODE}"
  --accelerate-config configs/accelerate_ddp_2gpu.yaml
  --num-processes 2
  --main-process-port $((20000 + SLURM_JOB_ID % 10000))
  --calibration-initial "${CALIBRATION_INITIAL}"
  --calibration-min "${CALIBRATION_MIN}"
  --reward-initial "${REWARD_INITIAL}"
  --reward-max "${REWARD_MAX}"
)

if [[ -n "${FIXED_CALIBRATION_PROMPT_BATCH_SIZE}" ]]; then
  cmd+=(--fixed-calibration-prompt-batch-size "${FIXED_CALIBRATION_PROMPT_BATCH_SIZE}")
fi

"${cmd[@]}"
