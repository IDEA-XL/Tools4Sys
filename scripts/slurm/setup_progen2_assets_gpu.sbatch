#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -J progen2ast
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/logs/setup_assets_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/logs/setup_assets_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
PROGEN2_OUT_DIR=${PROGEN2_OUT_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_official}
TEMBERTURE_OUT_DIR=${TEMBERTURE_OUT_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/temberture_official}
PROTEINSOL_OUT_DIR=${PROTEINSOL_OUT_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/proteinsol_official}
PYTHON_OVERLAY_DIR=${PYTHON_OVERLAY_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/python_overlay}
OPENFOLD_SOURCE_DIR=${OPENFOLD_SOURCE_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/openfold_official}
CUDA_TOOLCHAIN_DIR=${CUDA_TOOLCHAIN_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/cuda_toolchain_cu124}
MIHOMO_DIR=/public/home/xinwuye/mihomo
CUDA_CUSPARSE_HEADER=${CUDA_TOOLCHAIN_DIR}/include/cusparse.h
CONDA_ENV_NAME=${CONDA_ENV_NAME:-genmol}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-}

if [[ -n "${CONDA_ENV_NAME}" && -n "${CONDA_ENV_PREFIX}" ]]; then
  echo "Set at most one of CONDA_ENV_NAME and CONDA_ENV_PREFIX" >&2
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
python scripts/setup_progen2_python_overlay.py --overlay-dir "${PYTHON_OVERLAY_DIR}"
python scripts/setup_openfold_extension.py --source-dir "${OPENFOLD_SOURCE_DIR}" --overlay-dir "${PYTHON_OVERLAY_DIR}"
export PYTHONPATH=${PYTHON_OVERLAY_DIR}:${REPO_DIR}/src
python scripts/setup_progen2_official.py --output-dir "${PROGEN2_OUT_DIR}" --model progen2-small
python scripts/setup_temberture_official.py --output-dir "${TEMBERTURE_OUT_DIR}"
python scripts/setup_proteinsol_official.py --output-dir "${PROTEINSOL_OUT_DIR}"
