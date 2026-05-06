#!/bin/bash
#SBATCH -p cpu
#SBATCH --array=0-31
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -J mmvdock
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_mmgenmol_vinadock_%A_%a.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_mmgenmol_vinadock_%A_%a.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
TASKS_PATH=${TASKS_PATH:-/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/mmgenmol/generation_sweep_tasks_20260423.tsv}
MANIFEST_PATH=${MANIFEST_PATH:-/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix/data/gdrive_exact_only/crossdocked_pocket10_pocket_prefix_manifest.pt}
CROSSDOCKED_ROOT=${CROSSDOCKED_ROOT:-/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/test_set/test_set}
QVINA_PATH=${QVINA_PATH:-/public/home/xinwuye/ai4s-tool-joint-train/genmol/scripts/exps/lead/docking/qvina02}
DOCKING_CACHE_DIR=${DOCKING_CACHE_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/pocket_prefix_crossdocked_5500ckpt/docking_cache}
OUTPUT_ROOT=${OUTPUT_ROOT:-/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/mmgenmol_sweep_vina_dock_20260423}
DEFAULT_EVAL_CONDA_PREFIX=/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/conda_env
CONDA_ENV_NAME=${CONDA_ENV_NAME:-}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-}

if [[ -n "${CONDA_ENV_NAME}" && -n "${CONDA_ENV_PREFIX}" ]]; then
  echo "Set at most one of CONDA_ENV_NAME and CONDA_ENV_PREFIX" >&2
  exit 1
fi
if [[ -z "${CONDA_ENV_NAME}" && -z "${CONDA_ENV_PREFIX}" ]]; then
  CONDA_ENV_PREFIX=${DEFAULT_EVAL_CONDA_PREFIX}
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
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${REPO_DIR}"
if [[ ! -f "${TASKS_PATH}" ]]; then
  echo "Missing task manifest: ${TASKS_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Missing CrossDocked manifest: ${MANIFEST_PATH}" >&2
  exit 1
fi
if [[ ! -d "${CROSSDOCKED_ROOT}" ]]; then
  echo "Missing CrossDocked root: ${CROSSDOCKED_ROOT}" >&2
  exit 1
fi

task_line=$(awk -F '\t' -v task_id="${SLURM_ARRAY_TASK_ID}" 'NR > 1 && $1 == task_id {print; found=1} END {exit found ? 0 : 1}' "${TASKS_PATH}") || {
  echo "No task row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} in ${TASKS_PATH}" >&2
  exit 1
}

IFS=$'\t' read -r task_id model_name sweep_type sweep_value randomness temperature checkpoint_path generated_rows_path <<< "${task_line}"
if [[ "${task_id}" != "${SLURM_ARRAY_TASK_ID}" ]]; then
  echo "Task ID mismatch: row=${task_id} array=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi
if [[ ! -f "${generated_rows_path}" ]]; then
  echo "Missing generated rows file: ${generated_rows_path}" >&2
  exit 1
fi
row_count=$(wc -l < "${generated_rows_path}")
if [[ "${row_count}" != "1600" ]]; then
  echo "Generated row count mismatch for ${generated_rows_path}: ${row_count} vs 1600" >&2
  exit 1
fi

output_dir="${OUTPUT_ROOT}/${model_name}/${sweep_type}_${sweep_value}"
records_path="${output_dir}/docking.records.jsonl"
summary_path="${output_dir}/docking.summary.json"
if [[ -e "${records_path}" || -e "${summary_path}" ]]; then
  echo "Docking output already exists under ${output_dir}" >&2
  exit 1
fi
mkdir -p "${output_dir}"

echo "task_id=${task_id}"
echo "model_name=${model_name}"
echo "sweep_type=${sweep_type}"
echo "sweep_value=${sweep_value}"
echo "generated_rows_path=${generated_rows_path}"
echo "records_path=${records_path}"
echo "summary_path=${summary_path}"
echo "num_workers=${SLURM_CPUS_PER_TASK}"

/usr/bin/time -p python scripts/dock_pocket_prefix_generated_rows.py \
  --generated_rows_path "${generated_rows_path}" \
  --manifest_path "${MANIFEST_PATH}" \
  --split test \
  --crossdocked_root "${CROSSDOCKED_ROOT}" \
  --qvina_path "${QVINA_PATH}" \
  --docking_cache_dir "${DOCKING_CACHE_DIR}" \
  --output_rows_path "${records_path}" \
  --output_summary_path "${summary_path}" \
  --num_workers "${SLURM_CPUS_PER_TASK}" \
  --docking_modes vina_dock \
  --docking_exhaustiveness 8 \
  --docking_num_cpu 1 \
  --docking_num_modes 10 \
  --docking_timeout_gen3d 30 \
  --docking_timeout_dock 100 \
  --docking_box_size 20.0 20.0 20.0 \
  --progress_every 100
