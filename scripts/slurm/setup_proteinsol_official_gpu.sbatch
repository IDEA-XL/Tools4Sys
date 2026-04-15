#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -J proteinset
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/logs/setup_proteinsol_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/logs/setup_proteinsol_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
OUT_DIR=${OUT_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/proteinsol_official}
MIHOMO_DIR=/public/home/xinwuye/mihomo

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:$PATH
export PYTHONPATH=${REPO_DIR}/src
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

cd "${REPO_DIR}"
python scripts/setup_proteinsol_official.py --output-dir "${OUT_DIR}"
