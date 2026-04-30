#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH -t 24:00:00
#SBATCH -J p2sweepdevcpu
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_progen2sweepdevcpu_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_progen2sweepdevcpu_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:-configs/eval_progen2_temperature_sweep_20260430.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_sweep_probe_cpu/developability_${SLURM_JOB_ID}}
TOTAL_SEQUENCES=${TOTAL_SEQUENCES:-512}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-256}
REWARD_MAX=${REWARD_MAX:-4096}
SAFE_RESERVED_RATIO=${SAFE_RESERVED_RATIO:-0.90}

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

PYTHON_OVERLAY_DIR=${PYTHON_OVERLAY_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/python_overlay}
PROTBERT_DIR=${PROTBERT_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/prot_bert_bfd}
CUDA_TOOLCHAIN_DIR=${CUDA_TOOLCHAIN_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/cuda_toolchain_cu124}
MIHOMO_DIR=/public/home/xinwuye/mihomo

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

python scripts/probe_progen2_sweep_resources.py \
  single-reward-controller \
  --config "${CONFIG_PATH}" \
  --output-root "${OUTPUT_ROOT}" \
  --repo-root "${REPO_DIR}" \
  --reward-name developability \
  --total-sequences "${TOTAL_SEQUENCES}" \
  --sequence-length "${SEQUENCE_LENGTH}" \
  --reward-max "${REWARD_MAX}" \
  --safe-reserved-ratio "${SAFE_RESERVED_RATIO}"
