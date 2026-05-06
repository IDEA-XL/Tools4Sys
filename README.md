# GenMol SGRPO Paper Reproduction

This repository contains the training, evaluation, and plotting code used for the SGRPO experiments in our NeurIPS submission across three domains:

- GenMol de novo molecule design
- mmGenMol pocket-conditioned molecule design
- ProGen2 de novo protein design

This README is intentionally narrow. It only covers the canonical SGRPO experiments that appear in `figs/main-pareto.pdf`:

- `GenMol De Novo SGRPO RewardSum LOO 2000`
- `SGRPO + UniDock RewardSum LOO 1000`
- `ProGen2 SGRPO gw0.8 RewardSum LOO 100`

## Scope and assumptions

- The commands below assume the repo is cloned on Pudong at `/public/home/xinwuye/ai4s-tool-joint-train/genmol`.
- Many configs in this repo use absolute Pudong paths. If you clone elsewhere, update those paths before running.
- The environment setup below is verified for the three paper-critical blocks above. It is not intended to be a universal environment for every historical script in this repo.
- For mmGenMol, the reported docking metric in the paper result tables and JSON summaries is `vina_dock_mean`. The legacy field name `unidock_score_mean` is only a Vina-derived reward proxy retained for compatibility with training-time reward accounting.
- Assumption: the OpenFold source tree required by `scripts/setup_openfold_extension.py` is available at `/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/openfold_official`. This repo currently ships the extension builder, but not a downloader for that source tree.

## 1. Create a fresh environment

On Pudong:

```bash
cd /public/home/xinwuye/ai4s-tool-joint-train/genmol

export CONDA_ENV_NAME=genmol-paper-20260506
bash env/setup.sh
```

What `env/setup.sh` does:

- creates a fresh Python 3.10 conda environment
- installs the base GenMol dependencies from `env/requirements.txt`
- installs the repo in editable mode
- installs the extra packages required by mmGenMol and ProGen2 reproduction:
  - `fair-esm`
  - `lmdb`
  - `biotite`
  - `scipy`
  - `requests`

If you prefer a conda prefix instead of a named env:

```bash
export CONDA_ENV_PREFIX=/public/home/xinwuye/conda_envs/genmol-paper-20260506
bash env/setup.sh
```

The Slurm launchers used below now accept either `CONDA_ENV_NAME` or `CONDA_ENV_PREFIX`.

## 2. One-time assets

### GenMol de novo

The de novo SGRPO training config initializes from the canonical GenMol v2 checkpoint:

- `/public/home/xinwuye/ai4s-tool-joint-train/genmol/checkpoints/genmol_v2_v1.0/model_v2.ckpt`

### mmGenMol

The mmGenMol SGRPO training config expects:

- the CrossDocked LMDB
- the CrossDocked split file
- the pocket-prefix manifest
- Uni-Dock available on `PATH`

These paths are already pinned inside:

- `configs/cpgrpo_denovo_pocket_prefix_sgrpo_ng24_sg8_bs384_lr5e-5_beta5e-3_gw09_q03_sa02_unidock05_rewardsum_loo.yaml`

### ProGen2

The ProGen2 training and sweep pipeline expect the following assets:

- ProGen2 official tokenizer/checkpoint tree
- a Python overlay containing pinned `transformers` and `adapters`
- an OpenFold extension compiled into that overlay
- ProtBERT
- TemBERTure
- Protein-Sol

The repo provides a bundled setup launcher for the assets that it knows how to build or download:

```bash
sbatch --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME} \
  scripts/slurm/setup_progen2_assets_gpu.sbatch
```

That launcher covers:

- `scripts/setup_progen2_python_overlay.py`
- `scripts/setup_openfold_extension.py`
- `scripts/setup_progen2_official.py`
- `scripts/setup_temberture_official.py`
- `scripts/setup_proteinsol_official.py`

You still need the OpenFold source tree itself at:

- `/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/openfold_official`

## 3. Train the canonical SGRPO models

### 3.1 GenMol de novo

Training config:

- `configs/cpgrpo_denovo_sgrpo_ng64_sg8_bs1024_lr5e-5_beta5e-3_gw09_rewardsum_loo_ms2000.yaml`

Launch:

```bash
cd /public/home/xinwuye/ai4s-tool-joint-train/genmol
sbatch --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},CONFIG_PATH=configs/cpgrpo_denovo_sgrpo_ng64_sg8_bs1024_lr5e-5_beta5e-3_gw09_rewardsum_loo_ms2000.yaml \
  scripts/slurm/cpgrpo_denovo_8gpu_ng512_bs1024_ni1.sbatch
```

### 3.2 mmGenMol

Training config:

- `configs/cpgrpo_denovo_pocket_prefix_sgrpo_ng24_sg8_bs384_lr5e-5_beta5e-3_gw09_q03_sa02_unidock05_rewardsum_loo.yaml`

Launch:

```bash
cd /public/home/xinwuye/ai4s-tool-joint-train/genmol
sbatch --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},CONFIG_PATH=configs/cpgrpo_denovo_pocket_prefix_sgrpo_ng24_sg8_bs384_lr5e-5_beta5e-3_gw09_q03_sa02_unidock05_rewardsum_loo.yaml \
  scripts/slurm/cpgrpo_denovo_pocket_prefix_8gpu_sgrpo_ng24_sg8_bs384_unidock_train.sbatch
```

### 3.3 ProGen2

Training config:

- `configs/progen2_sgrpo_ng12_sg8_bs2_len256_rbs16_gw08_rewardsum_loo.yaml`

Launch:

```bash
cd /public/home/xinwuye/ai4s-tool-joint-train/genmol
sbatch --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},CONFIG_PATH=configs/progen2_sgrpo_ng12_sg8_bs2_len256_rbs16_gw08_rewardsum_loo.yaml \
  scripts/slurm/train_progen2_sgrpo_8gpu.sbatch
```

The paper figure uses `checkpoint-000100` for ProGen2, not the final checkpoint.

## 4. Reproduce the paper sweeps

### 4.1 GenMol de novo paired sweep

This minimal config evaluates only the paper SGRPO 2000 checkpoint:

- `configs/readme_denovo_sgrpo_rewardsum_loo_2000_paired.yaml`

Run it on a GPU node:

```bash
cd /public/home/xinwuye/ai4s-tool-joint-train/genmol
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=08:00:00 \
  python scripts/eval_denovo_sgrpo.py \
    --config configs/readme_denovo_sgrpo_rewardsum_loo_2000_paired.yaml
```

### 4.2 mmGenMol paired sweep

The committed task manifest below contains exactly the 6 paired sweep points for the paper SGRPO + UniDock RewardSum LOO 1000 checkpoint:

- `sgrpo-main-results/mmgenmol/readme_generation_paired_sgrpo_unidock_rewardsum_loo_1000.tsv`

Step 1: generate molecules.

```bash
cd /public/home/xinwuye/ai4s-tool-joint-train/genmol
sbatch --array=0-5 \
  --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},TASKS_PATH=/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/mmgenmol/readme_generation_paired_sgrpo_unidock_rewardsum_loo_1000.tsv \
  scripts/slurm/generate_mmgenmol_sweep_array_1gpu.sbatch
```

Step 2: dock with Vina on CPU.

```bash
cd /public/home/xinwuye/ai4s-tool-joint-train/genmol
sbatch --array=0-5 \
  --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},TASKS_PATH=/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/mmgenmol/readme_generation_paired_sgrpo_unidock_rewardsum_loo_1000.tsv,OUTPUT_ROOT=/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/readme_mmgenmol_paired_vina_dock \
  scripts/slurm/dock_mmgenmol_sweep_vina_array_64cpu.sbatch
```

Step 3: aggregate the 6 sweep points into one JSON/Markdown report.

```bash
cd /public/home/xinwuye/ai4s-tool-joint-train/genmol
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

python scripts/aggregate_mmgenmol_sweep_results.py \
  --tasks_path sgrpo-main-results/mmgenmol/readme_generation_paired_sgrpo_unidock_rewardsum_loo_1000.tsv \
  --docking_root /public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/readme_mmgenmol_paired_vina_dock \
  --output_dir /public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/readme_mmgenmol_paired_aggregate \
  --output_prefix readme_mmgenmol_paired_sgrpo_unidock_rewardsum_loo_1000 \
  --expected_num_tasks 6 \
  --expected_rows_per_task 1600 \
  --expected_num_pockets 100 \
  --expected_samples_per_pocket 16 \
  --docking_mode vina_dock \
  --plot_name_prefix readme_mmgenmol \
  --plot_title_prefix "mmGenMol" \
  --qed_weight 0.3 \
  --sa_score_weight 0.2 \
  --drugclip_score_weight 0.0
```

### 4.3 ProGen2 temperature sweep

This minimal config evaluates only the paper SGRPO gw0.8 RewardSum LOO step-100 checkpoint:

- `configs/readme_progen2_temperature_sweep_sgrpo_gw08_rewardsum_loo_step100.yaml`

Pipeline:

```bash
cd /public/home/xinwuye/ai4s-tool-joint-train/genmol
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

python scripts/progen2_sweep_pipeline.py build-tasks \
  --config configs/readme_progen2_temperature_sweep_sgrpo_gw08_rewardsum_loo_step100.yaml
```

Then run the stages:

```bash
sbatch --export=ALL,CONDA_ENV_NAME=${CONDA_ENV_NAME},CONFIG_PATH=configs/readme_progen2_temperature_sweep_sgrpo_gw08_rewardsum_loo_step100.yaml,MODE=generate-task,TASK_ID=0 \
  scripts/slurm/run_progen2_sweep_gpu.sbatch
```

Use the same launcher for other task IDs and reward modes:

- `MODE=generate-task`
- `MODE=score-packed-gpu-reward` with `REWARD_NAME=naturalness` or `stability`
- `MODE=score-point-reward-task` with `REWARD_NAME=foldability` or `developability`

Finally aggregate:

```bash
cd /public/home/xinwuye/ai4s-tool-joint-train/genmol
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

python scripts/progen2_sweep_pipeline.py aggregate \
  --config configs/readme_progen2_temperature_sweep_sgrpo_gw08_rewardsum_loo_step100.yaml
```

## 5. Paper figure inputs

The committed JSON files used by `vis-code/plot_main_pareto.py` are:

- `sgrpo-main-results/genmol-denovo/denovo_main_results_paired_sweep_20260504.json`
- `sgrpo-main-results/mmgenmol/mmgenmol_paired_main_results_20260504.json`
- `sgrpo-main-results/progen2/progen2_temperature_sweep_20260503.json`

To regenerate the PDF after those JSON files are in place:

```bash
cd /public/home/xinwuye/ai4s-tool-joint-train/genmol
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

python vis-code/plot_main_pareto.py
```

This writes:

- `figs/main-pareto.pdf`

## 6. Canonical output checkpoints

The three paper-critical checkpoints are:

- de novo: `.../cpgrpo_denovo_sgrpo_ng64_sg8_bs1024_lr5e-5_beta5e-3_gw09_rewardsum_loo_ms2000_20260426_115639/checkpoint-002000/model.ckpt`
- mmGenMol: `.../cpgrpo_denovo_pocket_prefix_sgrpo_ng24_sg8_bs384_lr5e-5_beta5e-3_gw09_q03_sa02_unidock05_rewardsum_loo_20260501_160306/checkpoint-001000/model.ckpt`
- ProGen2: `.../progen2_sgrpo_ng12_sg8_bs2_len256_rbs16_gw08_rewardsum_loo_slurm53602/checkpoint-000100`
