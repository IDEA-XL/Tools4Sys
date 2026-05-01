#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J p2sweepdiv
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_progen2sweepdiv_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_progen2sweepdiv_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:-configs/progen2_temperature_sweep_pipeline_20260430.yaml}
TASK_ID=${TASK_ID:-${SLURM_ARRAY_TASK_ID:-}}

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ -z "${TASK_ID}" ]]; then
  echo "TASK_ID is required" >&2
  exit 1
fi

cd "${REPO_DIR}"
python scripts/progen2_sweep_pipeline.py score-point-diversity-task \
  --config "${CONFIG_PATH}" \
  --task-id "${TASK_ID}"
