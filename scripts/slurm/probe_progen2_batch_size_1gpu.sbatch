#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J progen2probe
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_progen2probe_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_progen2probe_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:-/public/home/xinwuye/ai4s-tool-joint-train/genmol/configs/progen2_sgrpo_1gpu_memprobe.yaml}
PYTHON_OVERLAY_DIR=${PYTHON_OVERLAY_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/python_overlay}
PROTBERT_DIR=${PROTBERT_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/prot_bert_bfd}
MIHOMO_DIR=/public/home/xinwuye/mihomo
CANDIDATES=${CANDIDATES:-1,2,4,8,12,16,20,24}
TARGET_RESERVED_RATIO=${TARGET_RESERVED_RATIO:-0.95}
OUTPUT_ROOT=${OUTPUT_ROOT:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_batch_probe/probe_${SLURM_JOB_ID}}

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:$PATH
export TOKENIZERS_PARALLELISM=false

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

python scripts/probe_progen2_batch_size.py \
  --config "${CONFIG_PATH}" \
  --candidates "${CANDIDATES}" \
  --output-root "${OUTPUT_ROOT}" \
  --target-reserved-ratio "${TARGET_RESERVED_RATIO}" \
  --stop-after-oom
