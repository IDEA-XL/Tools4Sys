# mmGenMol Sweep Results

- `summary_json`: `sgrpo-main-results/mmgenmol/mmgenmol_temperature_sweep_results_20260425.json`
- `raw_rows_jsonl`: `sgrpo-main-results/mmgenmol/mmgenmol_temperature_sweep_results_20260425.rows.jsonl`
- `num_pockets`: 100
- `samples_per_pocket`: 16
- `docking_mode`: `vina_dock`
- `diversity`: per sweep point, compute internal diversity separately within each pocket group, then average over pockets.
- `qed_mean` and `sa_score_mean`: means over valid generated molecules in the sweep point.
- `vina_dock_mean`: mean Vina dock affinity over successful dockings; lower is better.

| Model | Sweep | Value | Diversity | QED | SA Score | Vina Dock Mean | Dock Success | Valid Fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original 5500 | temperature | 0.500000 | 0.628460 | 0.458480 | 0.522132 | -7.024656 | 0.968125 | 0.990625 |
| Original 5500 | temperature | 1.000000 | 0.622208 | 0.478838 | 0.525759 | -7.058283 | 0.975000 | 0.994375 |
| Original 5500 | temperature | 5.000000 | 0.930619 | 0.464653 | 0.465808 | -3.756607 | 0.805625 | 0.919375 |
| Original 5500 | temperature | 10.000000 | 0.816960 | 0.370030 | 0.082051 | -1.657150 | 0.116875 | 0.186250 |
| GRPO 1000 | temperature | 0.500000 | 0.449927 | 0.798413 | 0.793173 | -6.434161 | 0.990625 | 0.993750 |
| GRPO 1000 | temperature | 1.000000 | 0.404945 | 0.801514 | 0.796646 | -6.383834 | 0.993750 | 0.996250 |
| GRPO 1000 | temperature | 5.000000 | 0.772108 | 0.650837 | 0.698516 | -5.323516 | 0.865625 | 0.963750 |
| GRPO 1000 | temperature | 10.000000 | 0.704705 | 0.375636 | 0.078080 | -1.740626 | 0.076875 | 0.152500 |
| SGRPO 1000 | temperature | 0.500000 | 0.778147 | 0.682971 | 0.736226 | -6.388259 | 0.951875 | 0.993750 |
| SGRPO 1000 | temperature | 1.000000 | 0.750277 | 0.691767 | 0.735042 | -6.232247 | 0.943125 | 0.990625 |
| SGRPO 1000 | temperature | 5.000000 | 0.904360 | 0.398137 | 0.483673 | -2.545893 | 0.820000 | 0.853750 |
| SGRPO 1000 | temperature | 10.000000 | 0.685451 | 0.366213 | 0.102358 | -1.687628 | 0.102500 | 0.149375 |
| GRPO DivReg 0.05 1000 | temperature | 0.500000 | 0.302039 | 0.841217 | 0.812937 | -7.176831 | 0.994375 | 0.996250 |
| GRPO DivReg 0.05 1000 | temperature | 1.000000 | 0.275579 | 0.846507 | 0.813896 | -7.201010 | 0.993125 | 0.996875 |
| GRPO DivReg 0.05 1000 | temperature | 5.000000 | 0.811014 | 0.623008 | 0.698080 | -5.570175 | 0.805625 | 0.950000 |
| GRPO DivReg 0.05 1000 | temperature | 10.000000 | 0.754439 | 0.375764 | 0.068807 | -1.801828 | 0.094375 | 0.166250 |

## temperature QED vs diversity

![temperature QED vs diversity](mmgenmol_temperature_diversity_vs_qed_mean_20260425.png)

## temperature SA Score vs diversity

![temperature SA Score vs diversity](mmgenmol_temperature_diversity_vs_sa_score_mean_20260425.png)

## temperature Vina Dock Mean vs diversity

![temperature Vina Dock Mean vs diversity](mmgenmol_temperature_diversity_vs_vina_dock_mean_20260425.png)
