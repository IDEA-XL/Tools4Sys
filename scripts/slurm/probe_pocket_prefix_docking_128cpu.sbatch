#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -J mmpdock128
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_mmpdock128_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_mmpdock128_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
EVAL_CONDA_PREFIX=${EVAL_CONDA_PREFIX:-/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/conda_env}
GENERATED_ROWS_PATH=${GENERATED_ROWS_PATH:-/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/gen_batch_probe/n16_bs1600/generated.rows.jsonl}
MANIFEST_PATH=${MANIFEST_PATH:-/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix/data/gdrive_exact_only/crossdocked_pocket10_pocket_prefix_manifest.pt}
CROSSDOCKED_ROOT=${CROSSDOCKED_ROOT:-/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/test_set/test_set}
QVINA_PATH=${QVINA_PATH:-/public/home/xinwuye/ai4s-tool-joint-train/genmol/scripts/exps/lead/docking/qvina02}
DOCKING_CACHE_DIR=${DOCKING_CACHE_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/pocket_prefix_crossdocked_5500ckpt/docking_cache}
OUTPUT_DIR=${OUTPUT_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/docking_parallel_probe/n16_bs1600_128cpu_${SLURM_JOB_ID}}
NUM_WORKERS=${NUM_WORKERS:-128}
MAX_ROWS=${MAX_ROWS:-}

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
if [[ ! -d "${EVAL_CONDA_PREFIX}" ]]; then
  echo "Missing evaluation conda env: ${EVAL_CONDA_PREFIX}" >&2
  exit 1
fi
conda activate "${EVAL_CONDA_PREFIX}"
export PATH=/opt/gridview/slurm/bin:${PATH}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}"

args=(
  --generated_rows_path "${GENERATED_ROWS_PATH}"
  --manifest_path "${MANIFEST_PATH}"
  --split test
  --crossdocked_root "${CROSSDOCKED_ROOT}"
  --qvina_path "${QVINA_PATH}"
  --docking_cache_dir "${DOCKING_CACHE_DIR}"
  --output_rows_path "${OUTPUT_DIR}/docking.records.jsonl"
  --output_summary_path "${OUTPUT_DIR}/docking.summary.json"
  --num_workers "${NUM_WORKERS}"
  --docking_modes vina_dock qvina
  --docking_exhaustiveness 8
  --docking_num_cpu 1
  --docking_num_modes 10
  --docking_timeout_gen3d 30
  --docking_timeout_dock 100
  --docking_box_size 20.0 20.0 20.0
  --progress_every 100
)

if [[ -n "${MAX_ROWS}" ]]; then
  args+=(--max_rows "${MAX_ROWS}")
fi

/usr/bin/time -p python scripts/dock_pocket_prefix_generated_rows.py "${args[@]}"
