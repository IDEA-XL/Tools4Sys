#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -J gmdivred
#SBATCH -o /public/home/xinwuye/genmol_div_dynamics_reduce_%j.out
#SBATCH -e /public/home/xinwuye/genmol_div_dynamics_reduce_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:-${REPO_DIR}/configs/genmol_div_dynamics_20260504.yaml}

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate latentchem_dev
export PATH=/opt/gridview/slurm/bin:$PATH
unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=${REPO_DIR}/src:${REPO_DIR}/vis-code

cd /public/home/xinwuye
python "${REPO_DIR}/vis-code/genmol_div_dynamics_reduce.py" --config "${CONFIG_PATH}"
