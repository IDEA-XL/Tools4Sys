#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J mmdrugsetup
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/drugclip_official/logs/setup_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/drugclip_official/logs/setup_%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/genmol}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-genmol_drugclip}
CONDA_BASE=/public/home/xinwuye/miniconda
SOURCE_ENV_DIR=${CONDA_BASE}/envs/genmol
TARGET_ENV_DIR=${CONDA_BASE}/envs/${CONDA_ENV_NAME}
DRUGCLIP_RUN_DIR=${DRUGCLIP_RUN_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/drugclip_official}
DRUGCLIP_CHECKPOINT_PATH=${DRUGCLIP_CHECKPOINT_PATH:-${DRUGCLIP_RUN_DIR}/checkpoint_best.pt}
DRUGCLIP_CHECKPOINT_ID=${DRUGCLIP_CHECKPOINT_ID:-1i87thnbNk8qeLF_tLx_BzelTukWbHaTR}
UNICORE_SOURCE_DIR=${UNICORE_SOURCE_DIR:-${DRUGCLIP_RUN_DIR}/Uni-Core}
MIHOMO_DIR=/public/home/xinwuye/mihomo

source ${CONDA_BASE}/etc/profile.d/conda.sh
export PATH=/opt/gridview/slurm/bin:$PATH
export CONDA_NOTICES_DISABLE=1

mkdir -p "${DRUGCLIP_RUN_DIR}/logs"

if ! pgrep -u "$USER" -f "${MIHOMO_DIR}/mihomo" >/dev/null 2>&1; then
  nohup "${MIHOMO_DIR}/mihomo" -d "${MIHOMO_DIR}" >/tmp/mihomo_${SLURM_JOB_ID}.log 2>&1 &
  sleep 5
fi

export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=${http_proxy}
export HTTPS_PROXY=${https_proxy}

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "Runtime repo dir does not exist: ${REPO_DIR}" >&2
  exit 1
fi

if [[ -d "${TARGET_ENV_DIR}" && ! -x "${TARGET_ENV_DIR}/bin/python" ]]; then
  rm -rf "${TARGET_ENV_DIR}"
fi

if [[ -x "${TARGET_ENV_DIR}/bin/python" ]]; then
  if ! "${TARGET_ENV_DIR}/bin/python" -c "import torch, rdkit, tdc, rd_filters" >/dev/null 2>&1; then
    rm -rf "${TARGET_ENV_DIR}"
  fi
fi

if [[ ! -x "${TARGET_ENV_DIR}/bin/python" ]]; then
  conda create --offline -y -p "${TARGET_ENV_DIR}" --clone "${SOURCE_ENV_DIR}"
fi

conda activate "${TARGET_ENV_DIR}"

if [[ ! -x "${TARGET_ENV_DIR}/bin/python" ]]; then
  echo "Cloned env is missing python executable: ${TARGET_ENV_DIR}/bin/python" >&2
  exit 1
fi

python -c "import torch, rdkit, tdc, rd_filters"

if ! python -c "import gdown" >/dev/null 2>&1; then
  python -m pip install --ignore-installed --no-user gdown
fi

if [[ ! -d "${UNICORE_SOURCE_DIR}" ]]; then
  git clone https://github.com/dptech-corp/Uni-Core.git "${UNICORE_SOURCE_DIR}"
else
  git -C "${UNICORE_SOURCE_DIR}" fetch origin
  git -C "${UNICORE_SOURCE_DIR}" pull --ff-only
fi

(
  cd "${UNICORE_SOURCE_DIR}"
  python setup.py install
)

if [[ ! -s "${DRUGCLIP_CHECKPOINT_PATH}" ]]; then
  mkdir -p "$(dirname "${DRUGCLIP_CHECKPOINT_PATH}")"
  python -m gdown --continue "${DRUGCLIP_CHECKPOINT_ID}" -O "${DRUGCLIP_CHECKPOINT_PATH}"
fi

cd "${REPO_DIR}"
export PYTHONPATH=${REPO_DIR}/src
python - <<'PY'
import os

import unicore
from genmol.mm.utils import DrugCLIPConfig, DrugCLIPScorer

checkpoint_path = os.environ['DRUGCLIP_CHECKPOINT_PATH']
print(f'unicore_import_ok={unicore.__file__}')
print(f'drugclip_checkpoint_exists={os.path.exists(checkpoint_path)} size={os.path.getsize(checkpoint_path)}')
print(f'drugclip_scorer_import_ok={DrugCLIPScorer.__name__}')
print(f'drugclip_config_import_ok={DrugCLIPConfig.__name__}')
PY
