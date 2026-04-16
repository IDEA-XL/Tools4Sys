#!/bin/bash
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -J mmdevalset
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_setup_pocket_prefix_eval_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_setup_pocket_prefix_eval_%j.err

set -euo pipefail

RUN_ROOT=/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval
EVAL_CONDA_PREFIX=${EVAL_CONDA_PREFIX:-${RUN_ROOT}/conda_env}
DOWNLOAD_DIR=${DOWNLOAD_DIR:-${RUN_ROOT}/downloads}
TEST_SET_ROOT=${TEST_SET_ROOT:-${RUN_ROOT}/test_set}
TEST_SET_FOLDER_URL=${TEST_SET_FOLDER_URL:-https://drive.google.com/drive/folders/1j21cc7-97TedKh_El5E34yI8o5ckI7eK}
BASE_ENV_PREFIX=/public/home/xinwuye/miniconda/envs/genmol
MIHOMO_DIR=/public/home/xinwuye/mihomo

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
export PATH=/opt/gridview/slurm/bin:$PATH

mkdir -p "${RUN_ROOT}" "${DOWNLOAD_DIR}" "${TEST_SET_ROOT}"

if ! pgrep -u "$USER" -f "${MIHOMO_DIR}/mihomo" >/dev/null 2>&1; then
  nohup "${MIHOMO_DIR}/mihomo" -d "${MIHOMO_DIR}" >/tmp/mihomo_${SLURM_JOB_ID}.log 2>&1 &
  sleep 5
fi

export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=${http_proxy}
export HTTPS_PROXY=${https_proxy}

if [[ ! -d "${BASE_ENV_PREFIX}" ]]; then
  echo "Missing base genmol env: ${BASE_ENV_PREFIX}" >&2
  exit 1
fi

if [[ ! -d "${EVAL_CONDA_PREFIX}" ]]; then
  conda create -y -p "${EVAL_CONDA_PREFIX}" --clone "${BASE_ENV_PREFIX}"
fi

conda activate "${EVAL_CONDA_PREFIX}"
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip
python -m pip install vina meeko pdb2pqr git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
python -m pip install --ignore-installed --no-user gdown

if ! python -c "import gdown; print(gdown.__file__)" >/dev/null 2>&1; then
  echo "gdown import failed inside ${EVAL_CONDA_PREFIX}" >&2
  exit 1
fi

if [[ ! -f "${DOWNLOAD_DIR}/test_set.zip" ]]; then
  python -m gdown --folder "${TEST_SET_FOLDER_URL}" -O "${DOWNLOAD_DIR}"
fi

if [[ ! -f "${DOWNLOAD_DIR}/test_set.zip" ]]; then
  echo "Missing downloaded test_set.zip under ${DOWNLOAD_DIR}" >&2
  find "${DOWNLOAD_DIR}" -maxdepth 2 -type f | sort >&2 || true
  exit 1
fi

if [[ ! -f "${TEST_SET_ROOT}/.unpacked.ok" ]]; then
  unzip -q -o "${DOWNLOAD_DIR}/test_set.zip" -d "${TEST_SET_ROOT}"
  touch "${TEST_SET_ROOT}/.unpacked.ok"
fi

if ! find "${TEST_SET_ROOT}" -name '*.pdb' -print -quit | grep -q .; then
  echo "No PDB files found under ${TEST_SET_ROOT} after unpacking test_set.zip" >&2
  exit 1
fi

which pdb2pqr30

python - <<'PY'
import glob

import AutoDockTools
import meeko
import vina

print(vina.__file__)
print(meeko.__file__)
print(AutoDockTools.__file__)
print(glob.glob('/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/test_set/**/*.pdb', recursive=True)[:5])
PY
