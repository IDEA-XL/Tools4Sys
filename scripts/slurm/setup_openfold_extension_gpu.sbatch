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
CUDA_TOOLCHAIN_DIR=${CUDA_TOOLCHAIN_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/cuda_toolchain_cu124}
MIHOMO_DIR=/public/home/xinwuye/mihomo
CUDA_CUSPARSE_HEADER=${CUDA_TOOLCHAIN_DIR}/targets/x86_64-linux/include/cusparse.h

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:$PATH
export TOKENIZERS_PARALLELISM=false

mkdir -p /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/logs

if ! pgrep -u "$USER" -f "${MIHOMO_DIR}/mihomo" >/dev/null 2>&1; then
  nohup "${MIHOMO_DIR}/mihomo" -d "${MIHOMO_DIR}" >/tmp/mihomo_${SLURM_JOB_ID}.log 2>&1 &
  sleep 5
fi

export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=${http_proxy}
export HTTPS_PROXY=${https_proxy}

if [[ ! -x "${CUDA_TOOLCHAIN_DIR}/bin/nvcc" || ! -f "${CUDA_CUSPARSE_HEADER}" ]]; then
  rm -rf "${CUDA_TOOLCHAIN_DIR}"
  conda create -y -p "${CUDA_TOOLCHAIN_DIR}" -c nvidia/label/cuda-12.4.1 cuda-nvcc cuda-cudart-dev cuda-libraries-dev
fi

export CUDA_HOME="${CUDA_TOOLCHAIN_DIR}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" || ! -f "${CUDA_CUSPARSE_HEADER}" ]]; then
  echo "CUDA toolchain prefix is incomplete: nvcc=${CUDA_HOME}/bin/nvcc cusparse=${CUDA_CUSPARSE_HEADER}" >&2
  exit 1
fi

cd "${REPO_DIR}"
python scripts/setup_openfold_extension.py \
  --source-dir "${OPENFOLD_SOURCE_DIR}" \
  --overlay-dir "${PYTHON_OVERLAY_DIR}"
