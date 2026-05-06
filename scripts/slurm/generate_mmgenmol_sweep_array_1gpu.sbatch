#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --array=0-31
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J mmgen
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_mmgenmol_gen_%A_%a.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_mmgenmol_gen_%A_%a.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
TASKS_PATH=${TASKS_PATH:-/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/mmgenmol/generation_sweep_tasks_20260423.tsv}
MANIFEST_PATH=${MANIFEST_PATH:-/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix/data/gdrive_exact_only/crossdocked_pocket10_pocket_prefix_manifest.pt}
MEMLOG_DIR=${MEMLOG_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/mmgenmol_sweep_generation_20260423/logs}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-genmol}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-}

if [[ -n "${CONDA_ENV_NAME}" && -n "${CONDA_ENV_PREFIX}" ]]; then
  echo "Set at most one of CONDA_ENV_NAME and CONDA_ENV_PREFIX" >&2
  exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is required" >&2
  exit 1
fi

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
export PATH=/opt/gridview/slurm/bin:${PATH}
export TOKENIZERS_PARALLELISM=false

cd "${REPO_DIR}"
if [[ ! -f "${TASKS_PATH}" ]]; then
  echo "Missing task manifest: ${TASKS_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Missing CrossDocked manifest: ${MANIFEST_PATH}" >&2
  exit 1
fi

task_line=$(awk -F '\t' -v task_id="${SLURM_ARRAY_TASK_ID}" 'NR > 1 && $1 == task_id {print; found=1} END {exit found ? 0 : 1}' "${TASKS_PATH}") || {
  echo "No task row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in ${TASKS_PATH}" >&2
  exit 1
}

IFS=$'\t' read -r task_id model_name sweep_type sweep_value randomness temperature checkpoint_path output_path <<< "${task_line}"
if [[ "${task_id}" != "${SLURM_ARRAY_TASK_ID}" ]]; then
  echo "Task ID mismatch: row=${task_id} array=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi
if [[ ! -f "${checkpoint_path}" ]]; then
  echo "Missing checkpoint file: ${checkpoint_path}" >&2
  exit 1
fi
if [[ -e "${output_path}" ]]; then
  echo "Output already exists: ${output_path}" >&2
  exit 1
fi

mkdir -p "$(dirname "${output_path}")" "${MEMLOG_DIR}"
memlog="${MEMLOG_DIR}/task_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${model_name}_${sweep_type}_${sweep_value}_gpu_mem.csv"

echo "task_id=${task_id}"
echo "model_name=${model_name}"
echo "sweep_type=${sweep_type}"
echo "sweep_value=${sweep_value}"
echo "randomness=${randomness}"
echo "temperature=${temperature}"
echo "checkpoint_path=${checkpoint_path}"
echo "output_path=${output_path}"
echo "memlog=${memlog}"

(while true; do
  printf "%s," "$(date +%s)"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
  sleep 1
done > "${memlog}") &
sampler=$!
trap 'kill "${sampler}" 2>/dev/null || true' EXIT

/usr/bin/time -p python scripts/sample_denovo_pocket_prefix.py \
  --checkpoint_path "${checkpoint_path}" \
  --manifest_path "${MANIFEST_PATH}" \
  --split test \
  --num_pockets 100 \
  --num_samples_per_pocket 16 \
  --generation_batch_size 1600 \
  --generation_temperature "${temperature}" \
  --randomness "${randomness}" \
  --min_add_len 60 \
  --seed 42 \
  --device cuda \
  --bf16 \
  --output_path "${output_path}"

row_count=$(wc -l < "${output_path}")
if [[ "${row_count}" != "1600" ]]; then
  echo "Generated row count mismatch for ${output_path}: ${row_count} vs 1600" >&2
  exit 1
fi

awk -F, '
  {
    used = $2 + 0
    total = $3 + 0
    if (total > 0) {
      last_total = total
    }
    if (used > max_used) {
      max_used = used
      max_total = total
    }
  }
  END {
    denom = max_total > 0 ? max_total : last_total
    if (NR > 0 && denom > 0) {
      printf("peak_used_mib=%d total_mib=%d ratio=%.4f rows=%d\n", max_used, denom, max_used / denom, NR)
    }
  }
' "${memlog}"
