#!/bin/bash

set -euo pipefail

CONDA_ENV_NAME=${CONDA_ENV_NAME:-}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-}
INSTALL_MM_PROGEN2_DEPS=${INSTALL_MM_PROGEN2_DEPS:-1}

if [[ -n "${CONDA_ENV_NAME}" && -n "${CONDA_ENV_PREFIX}" ]]; then
  echo "Set at most one of CONDA_ENV_NAME and CONDA_ENV_PREFIX" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found on PATH" >&2
  exit 1
fi

CONDA_BASE=$(conda info --base)
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "Missing conda activation script: ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ -z "${CONDA_ENV_NAME}" && -z "${CONDA_ENV_PREFIX}" ]]; then
  CONDA_ENV_NAME=genmol
fi

if [[ -n "${CONDA_ENV_PREFIX}" ]]; then
  conda create -y -p "${CONDA_ENV_PREFIX}" python==3.10 pip==23.3.1
  conda activate "${CONDA_ENV_PREFIX}"
else
  conda create -y -n "${CONDA_ENV_NAME}" python==3.10 pip==23.3.1
  conda activate "${CONDA_ENV_NAME}"
fi

pip install -r env/requirements.txt
pip install -e .
pip install scikit-learn==1.2.2

if [[ "${INSTALL_MM_PROGEN2_DEPS}" == "1" ]]; then
  pip install fair-esm==2.0.0 lmdb biotite scipy requests
fi
