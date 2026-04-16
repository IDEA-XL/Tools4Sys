#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -J ofext
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/logs/setup_openfold_extension_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/logs/setup_openfold_extension_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
OPENFOLD_SOURCE_DIR=${OPENFOLD_SOURCE_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/openfold_official}
PYTHON_OVERLAY_DIR=${PYTHON_OVERLAY_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/python_overlay}

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:$PATH
export TOKENIZERS_PARALLELISM=false

mkdir -p /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/logs

if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
  fi
fi

if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA_HOME is not configured to a valid nvcc installation" >&2
  exit 1
fi

cd "${REPO_DIR}"
python scripts/setup_openfold_extension.py \
  --source-dir "${OPENFOLD_SOURCE_DIR}" \
  --overlay-dir "${PYTHON_OVERLAY_DIR}"
