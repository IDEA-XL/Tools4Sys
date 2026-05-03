#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J p2dyncollect
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_eval/logs/progen2_dynamics_collect_%A_%a.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_eval/logs/progen2_dynamics_collect_%A_%a.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:-configs/progen2_dynamics_20260504.yaml}
PYTHON_OVERLAY_DIR=${PYTHON_OVERLAY_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/python_overlay}
PROTBERT_DIR=${PROTBERT_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/prot_bert_bfd}
MIHOMO_DIR=/public/home/xinwuye/mihomo
LOG_DIR=/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_eval/logs
TASK_INDEX=${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-}}

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:$PATH
export TOKENIZERS_PARALLELISM=false

mkdir -p "${LOG_DIR}"

if [[ -z "${TASK_INDEX}" ]]; then
  echo "TASK_INDEX is required" >&2
  exit 1
fi

if ! pgrep -u "$USER" -f "${MIHOMO_DIR}/mihomo" >/dev/null 2>&1; then
  nohup "${MIHOMO_DIR}/mihomo" -d "${MIHOMO_DIR}" >/tmp/mihomo_${SLURM_JOB_ID}.log 2>&1 &
  sleep 5
fi

export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=${http_proxy}
export HTTPS_PROXY=${https_proxy}
export PYTHONPATH=${PYTHON_OVERLAY_DIR}:${REPO_DIR}/src:${REPO_DIR}/vis-code

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

cd "${REPO_DIR}"

MODEL_ID=$(python - <<'PY' "${CONFIG_PATH}" "${TASK_INDEX}"
import sys
import yaml

config_path = sys.argv[1]
task_index = int(sys.argv[2])
with open(config_path) as handle:
    raw = yaml.safe_load(handle)
models = raw.get("models", [])
if task_index < 0 or task_index >= len(models):
    raise SystemExit(f"Task index out of range: {task_index} vs {len(models)}")
print(models[task_index]["model_id"])
PY
)

python vis-code/progen2_dynamics_collect.py \
  --config "${CONFIG_PATH}" \
  --model-id "${MODEL_ID}"
