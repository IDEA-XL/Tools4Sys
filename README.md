# SGRPO

This repository contains the code for the NeurIPS paper on **Supergroup Relative Policy Optimization (SGRPO)**. It covers three biomolecular post-training settings:

- GenMol for de novo small-molecule design
- GenMol-P for pocket-based small-molecule design
- ProGen2 for de novo protein design

This README is the paper-facing reproduction guide. It is intentionally scoped to the **canonical SGRPO experiments** that produce the SGRPO curves in the paper's main Pareto figure:

- `GenMol De Novo SGRPO RewardSum LOO 2000`
- `SGRPO + UniDock RewardSum LOO 1000`
- `ProGen2 SGRPO gw0.8 RewardSum LOO 100`

Historical scripts, configs, and cluster launchers are still present in the repo, but this README only documents the paper-critical entry points.

## Terminology Map

The paper and the codebase use slightly different names in a few places. Use the following mapping when reading configs or logs:

| Paper term | Code term / location |
| --- | --- |
| SGRPO | `sgrpo` for ProGen2, `coupled_sgrpo` for the two GenMol tasks |
| GRPO | `grpo` for ProGen2, `coupled_grpo` for the two GenMol tasks |
| Memory-Assisted GRPO | `hbd: true` on a GRPO config |
| Utility | `soft_reward` or `soft_reward_mean` |
| Diversity | `diversity` |
| GenMol-P | `pocket_prefix_mm` in configs and training code; `mmGenMol` in some analysis scripts |
| Group size \(K\) | `num_generations` |
| Number of groups per same-condition supergroup \(M\) | `supergroup_num_groups` |
| Reward-Sum Hierarchy | `hierarchy: reward_sum` |
| Leave-one-out group credit | `group_rewrad_credit: loo` |

Two naming details matter for correct paper reproduction:

- For the two small-molecule settings, the paper labels the RL baselines as **GRPO** and **SGRPO**, but the implemented algorithms are **coupled-GRPO** and **coupled-SGRPO** because the generator is a discrete diffusion model.
- For pocket-based design, the paper reports **AutoDock Vina docking metrics** such as `vina_dock_mean`. The training-time field `unidock_score` is a **Vina-derived high-is-better reward proxy**, not the paper's reported docking metric.

## Recommended Repo-Relative Layout

The commands below assume you clone the repository as `SGRPO` and keep all paper assets inside the repo tree:

```text
SGRPO/
  checkpoints/
    genmol_v2_v1.0/
      model_v2.ckpt
  configs/
    ...
  data/
    crossdocked/
      processed/
        crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb
        crossdocked_pocket10_pose_split.pt
        crossdocked_pocket10_pocket_prefix_manifest.pt
      raw/
        test_set/
          ...
  runs/
    progen2_official/
    progen2_models/
      openfold_official/
      python_overlay/
      prot_bert_bfd/
      temberture_official/
      proteinsol_official/
  sgrpo-main-results/
    ...
```

`sgrpo-main-results/` is intentionally git-ignored. It is a local results workspace for sweep outputs, aggregated JSON files, and paper plotting inputs.

## Environment

The supported environment is Python 3.10 with the dependencies installed by `env/setup.sh`.

```bash
git clone <YOUR_GITHUB_URL>/SGRPO.git
cd SGRPO

source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_ENV_NAME=sgrpo
bash env/setup.sh
conda activate "${CONDA_ENV_NAME}"
```

If you prefer an explicit prefix instead of a named conda environment:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_ENV_PREFIX="$(pwd)/.conda/sgrpo"
bash env/setup.sh
conda activate "${CONDA_ENV_PREFIX}"
```

`env/setup.sh` performs the following steps:

- creates a fresh Python 3.10 environment
- installs the base repo requirements from `env/requirements.txt`
- installs the repo in editable mode
- installs the extra packages needed by the paper's mmGenMol and ProGen2 pipelines
- pins `setuptools<81` so that the historical `wandb==0.13.5` dependency remains importable

## One-Time Assets

### 1. GenMol de novo checkpoint

Place the pretrained GenMol v2 checkpoint at:

```text
checkpoints/genmol_v2_v1.0/model_v2.ckpt
```

This is the initialization checkpoint for the paper's de novo small-molecule SGRPO run.

### 2. GenMol-P / pocket-based assets

The pocket-based experiment needs three kinds of data:

- a processed CrossDocked LMDB for training
- the corresponding split file
- the raw CrossDocked test-set structure files for Vina docking evaluation

Place the processed files under `data/crossdocked/processed/`, then build the pocket-prefix manifest:

```bash
python scripts/build_crossdocked_pocket_prefix_manifest.py \
  --crossdocked_lmdb_path data/crossdocked/processed/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb \
  --crossdocked_split_path data/crossdocked/processed/crossdocked_pocket10_pose_split.pt \
  --output_path data/crossdocked/processed/crossdocked_pocket10_pocket_prefix_manifest.pt
```

For evaluation, place the raw CrossDocked test-set tree under:

```text
data/crossdocked/raw/test_set/
```

The pocket-based training run also expects a working `unidock` executable on `PATH`.

### 3. ProGen2 assets

The ProGen2 pipeline needs:

- the official ProGen2 tokenizer and checkpoint tree
- a Python overlay containing pinned `transformers` and `adapters`
- a compiled OpenFold extension installed into that overlay
- ProtBERT
- TemBERTure
- Protein-Sol

Create them under repo-relative paths:

```bash
python scripts/setup_progen2_official.py \
  --output-dir runs/progen2_official \
  --model progen2-small

python scripts/setup_progen2_python_overlay.py \
  --overlay-dir runs/progen2_models/python_overlay

git clone https://github.com/aqlaboratory/openfold.git \
  runs/progen2_models/openfold_official

python scripts/setup_openfold_extension.py \
  --source-dir runs/progen2_models/openfold_official \
  --overlay-dir runs/progen2_models/python_overlay

python scripts/setup_temberture_official.py \
  --output-dir runs/progen2_models/temberture_official

python scripts/setup_proteinsol_official.py \
  --output-dir runs/progen2_models/proteinsol_official

huggingface-cli download Rostlab/prot_bert_bfd \
  --local-dir runs/progen2_models/prot_bert_bfd
```

For ProGen2 training and evaluation, the OpenFold overlay must be importable:

```bash
export PYTHONPATH="$(pwd)/runs/progen2_models/python_overlay:$(pwd)/src:${PYTHONPATH:-}"
```

## Prepare Local Paper Configs

The repo still contains historical configs with site-specific asset paths. Do not edit those in place. Copy the paper configs to `configs/local/` and replace only the path-bearing fields with your repo-relative layout.

```bash
mkdir -p configs/local

cp configs/cpgrpo_denovo_sgrpo_ng64_sg8_bs1024_lr5e-5_beta5e-3_gw09_rewardsum_loo_ms2000.yaml \
  configs/local/denovo_sgrpo_main.yaml

cp configs/cpgrpo_denovo_pocket_prefix_sgrpo_ng24_sg8_bs384_lr5e-5_beta5e-3_gw09_q03_sa02_unidock05_rewardsum_loo.yaml \
  configs/local/mmgenmol_sgrpo_main.yaml

cp configs/progen2_sgrpo_ng12_sg8_bs2_len256_rbs16_gw08_rewardsum_loo.yaml \
  configs/local/progen2_sgrpo_main.yaml
```

Edit the following fields:

- `configs/local/denovo_sgrpo_main.yaml`
  - `init_ckpt_path`
- `configs/local/mmgenmol_sgrpo_main.yaml`
  - `init_ckpt_path`
  - `manifest_path`
  - `crossdocked_lmdb_path`
  - `crossdocked_split_path`
  - optionally `unidock_binary_path` if `unidock` is not already on `PATH`
- `configs/local/progen2_sgrpo_main.yaml`
  - `official_code_dir`
  - `tokenizer_path`
  - `init_checkpoint_dir`
  - `ref_checkpoint_dir`
  - `prompt_path`
  - `rewards.stability.model_name_or_path`
  - `rewards.stability.base_model_name_or_path`
  - `rewards.developability.model_name_or_path`

You can keep `output_dir: null` if you want the training scripts to use their default repo-relative `runs/...` output locations. If you prefer an explicit path, set `output_dir` to a relative path under `runs/`.

## Train the Canonical SGRPO Models

### 1. GenMol de novo

This run corresponds to the paper's **SGRPO** curve for de novo small-molecule design. In code, this is `coupled_sgrpo`.

```bash
accelerate launch \
  --config_file configs/accelerate_zero2.yaml \
  --num_processes 8 \
  --main_process_port 29501 \
  scripts/train_cpgrpo_denovo.py \
  --config configs/local/denovo_sgrpo_main.yaml
```

### 2. GenMol-P / pocket-based design

This run corresponds to the paper's **SGRPO** curve for pocket-based small-molecule design. In code, this is also `coupled_sgrpo`.

```bash
accelerate launch \
  --config_file configs/accelerate_zero2.yaml \
  --num_processes 8 \
  --main_process_port 29502 \
  scripts/train_cpgrpo_denovo_pocket_prefix.py \
  --config configs/local/mmgenmol_sgrpo_main.yaml
```

### 3. ProGen2

This run corresponds to the paper's **SGRPO** curve for de novo protein design.

```bash
export PYTHONPATH="$(pwd)/runs/progen2_models/python_overlay:$(pwd)/src:${PYTHONPATH:-}"

accelerate launch \
  --config_file configs/accelerate_ddp_8gpu.yaml \
  --num_processes 8 \
  --main_process_port 29503 \
  scripts/train_progen2_sgrpo.py \
  configs/local/progen2_sgrpo_main.yaml
```

For the paper, ProGen2 is evaluated at `checkpoint-000100`, not the final checkpoint.

## Reproduce the Paper SGRPO Sweeps

### 1. GenMol de novo paired \((\rho,\tau)\) sweep

Copy the evaluation config and rewrite the path-bearing fields to repo-relative outputs and your trained checkpoint:

```bash
cp configs/readme_denovo_sgrpo_rewardsum_loo_2000_paired.yaml \
  configs/local/eval_denovo_sgrpo_paired.yaml
```

Edit:

- `output_markdown_path`
- `output_json_path`
- `output_qed_diversity_plot_path`
- `output_sa_score_diversity_plot_path`
- `output_soft_reward_diversity_plot_path`
- `output_rows_path`
- `experiments[0].checkpoint_path`

Then run:

```bash
python scripts/eval_denovo_sgrpo.py \
  --config configs/local/eval_denovo_sgrpo_paired.yaml
```

This evaluates the six paper operating points:

- `(randomness, temperature) = (0.1, 0.5)`
- `(0.3, 0.8)`
- `(0.5, 1.1)`
- `(0.7, 1.4)`
- `(0.9, 1.7)`
- `(1.0, 2.0)`

### 2. GenMol-P paired \((\rho,\tau)\) sweep with Vina docking

The pocket-based SGRPO curve is reproduced by:

1. generating molecules at the six paired operating points
2. docking those generations with AutoDock Vina
3. aggregating the six points into one JSON summary

Create a local task manifest:

```bash
mkdir -p sgrpo-main-results/mmgenmol/generated
cat > sgrpo-main-results/mmgenmol/paired_tasks.tsv <<'EOF'
task_id	model_name	sweep_type	sweep_value	randomness	temperature	checkpoint_path	output_path
0	sgrpo_unidock_rewardsum_loo_1000	paired	1	0.1	0.5	runs/cpgrpo_denovo_pocket_prefix/<YOUR_RUN>/checkpoint-001000/model.ckpt	sgrpo-main-results/mmgenmol/generated/paired_1.rows.jsonl
1	sgrpo_unidock_rewardsum_loo_1000	paired	2	0.3	0.8	runs/cpgrpo_denovo_pocket_prefix/<YOUR_RUN>/checkpoint-001000/model.ckpt	sgrpo-main-results/mmgenmol/generated/paired_2.rows.jsonl
2	sgrpo_unidock_rewardsum_loo_1000	paired	3	0.5	1.1	runs/cpgrpo_denovo_pocket_prefix/<YOUR_RUN>/checkpoint-001000/model.ckpt	sgrpo-main-results/mmgenmol/generated/paired_3.rows.jsonl
3	sgrpo_unidock_rewardsum_loo_1000	paired	4	0.7	1.4	runs/cpgrpo_denovo_pocket_prefix/<YOUR_RUN>/checkpoint-001000/model.ckpt	sgrpo-main-results/mmgenmol/generated/paired_4.rows.jsonl
4	sgrpo_unidock_rewardsum_loo_1000	paired	5	0.9	1.7	runs/cpgrpo_denovo_pocket_prefix/<YOUR_RUN>/checkpoint-001000/model.ckpt	sgrpo-main-results/mmgenmol/generated/paired_5.rows.jsonl
5	sgrpo_unidock_rewardsum_loo_1000	paired	6	1.0	2.0	runs/cpgrpo_denovo_pocket_prefix/<YOUR_RUN>/checkpoint-001000/model.ckpt	sgrpo-main-results/mmgenmol/generated/paired_6.rows.jsonl
EOF
```

Run generation:

```bash
tail -n +2 sgrpo-main-results/mmgenmol/paired_tasks.tsv | while IFS=$'\t' read -r task_id model_name sweep_type sweep_value randomness temperature checkpoint_path output_path; do
  python scripts/sample_denovo_pocket_prefix.py \
    --checkpoint_path "${checkpoint_path}" \
    --manifest_path data/crossdocked/processed/crossdocked_pocket10_pocket_prefix_manifest.pt \
    --split test \
    --num_pockets 100 \
    --num_samples_per_pocket 16 \
    --generation_batch_size 1600 \
    --generation_temperature "${temperature}" \
    --randomness "${randomness}" \
    --min_add_len 60 \
    --seed 42 \
    --device cuda \
    --bf16 \
    --output_path "${output_path}"
done
```

Run docking:

```bash
tail -n +2 sgrpo-main-results/mmgenmol/paired_tasks.tsv | while IFS=$'\t' read -r task_id model_name sweep_type sweep_value randomness temperature checkpoint_path generated_rows_path; do
  output_dir="sgrpo-main-results/mmgenmol/docking/${model_name}/${sweep_type}_${sweep_value}"
  mkdir -p "${output_dir}"
  python scripts/dock_pocket_prefix_generated_rows.py \
    --generated_rows_path "${generated_rows_path}" \
    --manifest_path data/crossdocked/processed/crossdocked_pocket10_pocket_prefix_manifest.pt \
    --split test \
    --crossdocked_root data/crossdocked/raw/test_set \
    --qvina_path scripts/exps/lead/docking/qvina02 \
    --docking_cache_dir sgrpo-main-results/mmgenmol/docking_cache \
    --output_rows_path "${output_dir}/docking.records.jsonl" \
    --output_summary_path "${output_dir}/docking.summary.json" \
    --num_workers 64 \
    --docking_modes vina_dock \
    --docking_exhaustiveness 8 \
    --docking_num_cpu 1 \
    --docking_num_modes 10 \
    --docking_timeout_gen3d 30 \
    --docking_timeout_dock 100 \
    --docking_box_size 20.0 20.0 20.0 \
    --progress_every 100
done
```

Aggregate the six points:

```bash
python scripts/aggregate_mmgenmol_sweep_results.py \
  --tasks_path sgrpo-main-results/mmgenmol/paired_tasks.tsv \
  --docking_root sgrpo-main-results/mmgenmol/docking \
  --output_dir sgrpo-main-results/mmgenmol/aggregate \
  --output_prefix mmgenmol_paired_main_results \
  --expected_num_tasks 6 \
  --expected_rows_per_task 1600 \
  --expected_num_pockets 100 \
  --expected_samples_per_pocket 16 \
  --docking_mode vina_dock \
  --plot_name_prefix mmgenmol \
  --plot_title_prefix "GenMol-P" \
  --qed_weight 0.3 \
  --sa_score_weight 0.2 \
  --drugclip_score_weight 0.0
```

The paper-reported docking metric is `vina_dock_mean`.

### 3. ProGen2 temperature sweep

Copy the paper config and rewrite all path-bearing fields to repo-relative locations:

```bash
cp configs/readme_progen2_temperature_sweep_sgrpo_gw08_rewardsum_loo_step100.yaml \
  configs/local/eval_progen2_sgrpo_step100.yaml
```

At minimum, edit:

- all output paths under `tasks_path`, `*_output_root`, `packed_*`, and `output_*`
- `official_code_dir`
- `tokenizer_path`
- `prompt_path`
- `rewards.stability.model_name_or_path`
- `rewards.stability.base_model_name_or_path`
- `rewards.developability.model_name_or_path`
- `experiments[0].checkpoint_dir`

Build the sweep tasks:

```bash
export PYTHONPATH="$(pwd)/runs/progen2_models/python_overlay:$(pwd)/src:${PYTHONPATH:-}"
export PROGEN2_TASKS_PATH=<the same value as tasks_path in configs/local/eval_progen2_sgrpo_step100.yaml>

python scripts/progen2_sweep_pipeline.py build-tasks \
  --config configs/local/eval_progen2_sgrpo_step100.yaml
```

Run generation for each task id listed in the generated TSV:

```bash
for task_id in $(tail -n +2 "${PROGEN2_TASKS_PATH}" | cut -f1); do
  python scripts/progen2_sweep_pipeline.py generate-task \
    --config configs/local/eval_progen2_sgrpo_step100.yaml \
    --task-id "${task_id}"
done
```

Run the packed GPU rewards:

```bash
python scripts/progen2_sweep_pipeline.py score-packed-gpu-reward \
  --config configs/local/eval_progen2_sgrpo_step100.yaml \
  --reward-name naturalness

python scripts/progen2_sweep_pipeline.py score-packed-gpu-reward \
  --config configs/local/eval_progen2_sgrpo_step100.yaml \
  --reward-name stability
```

Run the point-wise rewards and diversity jobs:

```bash
for task_id in $(tail -n +2 "${PROGEN2_TASKS_PATH}" | cut -f1); do
  python scripts/progen2_sweep_pipeline.py score-point-reward-task \
    --config configs/local/eval_progen2_sgrpo_step100.yaml \
    --reward-name foldability \
    --task-id "${task_id}"

  python scripts/progen2_sweep_pipeline.py score-point-reward-task \
    --config configs/local/eval_progen2_sgrpo_step100.yaml \
    --reward-name developability \
    --task-id "${task_id}"

  python scripts/progen2_sweep_pipeline.py score-point-diversity-task \
    --config configs/local/eval_progen2_sgrpo_step100.yaml \
    --task-id "${task_id}"
done
```

Aggregate the final sweep:

```bash
python scripts/progen2_sweep_pipeline.py aggregate \
  --config configs/local/eval_progen2_sgrpo_step100.yaml
```

## Outputs and Paper Plotting

The three sweeps above produce the SGRPO evaluation artifacts used by the paper:

- de novo small molecules: aggregated paired-sweep JSON from `scripts/eval_denovo_sgrpo.py`
- pocket-based design: aggregated paired-sweep JSON from `scripts/aggregate_mmgenmol_sweep_results.py`
- de novo proteins: aggregated temperature-sweep JSON from `scripts/progen2_sweep_pipeline.py aggregate`

The figure code under `vis-code/` consumes these aggregated JSON files. In particular:

- `vis-code/plot_main_pareto.py` renders the main paper Pareto figure
- `vis-code/plot_denovo_ablation.py` renders the GenMol de novo ablation figure

`plot_main_pareto.py` combines the SGRPO curves with baseline curves. This README only reproduces the three **SGRPO** experiment blocks above. If you want to rerun the full paper figure exactly, place the corresponding aggregated JSON files under your local `sgrpo-main-results/` tree and adapt `vis-code/main_pareto_common.py` if you choose different filenames.

## Notes

- This README intentionally avoids absolute paths and cluster-specific assumptions.
- The Slurm launchers under `scripts/slurm/` are historical cluster templates, not the canonical open-source interface.
- The repo is fail-fast by design. If a required asset path or dependency is missing, the training or evaluation script should raise immediately rather than silently degrade.
