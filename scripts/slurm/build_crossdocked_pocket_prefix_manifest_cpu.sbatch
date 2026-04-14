#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J xdockmmmanifest
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix/logs/manifest_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix/logs/manifest_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
DATA_DIR=/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix/data/gdrive_exact_only
LMDB_PATH=${LMDB_PATH:-${DATA_DIR}/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb}
SPLIT_PATH=${SPLIT_PATH:-${DATA_DIR}/crossdocked_pocket10_pose_split.pt}
OUTPUT_PATH=${OUTPUT_PATH:-${DATA_DIR}/crossdocked_pocket10_pocket_prefix_manifest.pt}
MAX_TOTAL_POSITIONS=${MAX_TOTAL_POSITIONS:-256}

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:$PATH
export PYTHONPATH=${REPO_DIR}/src

cd "${REPO_DIR}"
mkdir -p /public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix/logs

python scripts/build_crossdocked_pocket_prefix_manifest.py \
  --crossdocked_lmdb_path "${LMDB_PATH}" \
  --crossdocked_split_path "${SPLIT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --max_total_positions "${MAX_TOTAL_POSITIONS}"
