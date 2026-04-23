# mmGenMol Sweep Results

- `summary_json`: `/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/mmgenmol/mmgenmol_sweep_results_20260423.json`
- `raw_rows_jsonl`: `/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/mmgenmol/mmgenmol_sweep_results_20260423.rows.jsonl`
- `num_pockets`: 100
- `samples_per_pocket`: 16
- `docking_mode`: `vina_dock`
- `diversity`: per sweep point, compute internal diversity separately within each pocket group, then average over pockets.
- `qed_mean` and `sa_score_mean`: means over valid generated molecules in the sweep point.
- `vina_dock_mean`: mean Vina dock affinity over successful dockings; lower is better.

| Model | Sweep | Value | Diversity | QED | SA Score | Vina Dock Mean | Dock Success | Valid Fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original 5500 | randomness | 0.100000 | 0.544726 | 0.479652 | 0.525897 | -7.321590 | 0.977500 | 0.996250 |
| Original 5500 | randomness | 0.300000 | 0.622208 | 0.478838 | 0.525759 | -7.243106 | 0.975000 | 0.994375 |
| Original 5500 | randomness | 0.600000 | 0.670246 | 0.473894 | 0.529055 | -7.175157 | 0.974375 | 0.991250 |
| Original 5500 | randomness | 1.000000 | 0.695035 | 0.484793 | 0.521087 | -7.121937 | 0.965625 | 0.986875 |
| Original 5500 | temperature | 0.100000 | 0.667775 | 0.469309 | 0.531659 | -7.096307 | 0.975625 | 0.993750 |
| Original 5500 | temperature | 0.300000 | 0.646161 | 0.465230 | 0.530596 | -7.209853 | 0.970625 | 0.988750 |
| Original 5500 | temperature | 0.600000 | 0.621510 | 0.467674 | 0.526196 | -7.127156 | 0.980625 | 0.995625 |
| Original 5500 | temperature | 1.000000 | 0.622208 | 0.478838 | 0.525759 | -7.056172 | 0.975000 | 0.994375 |
| GRPO 1000 | randomness | 0.100000 | 0.343151 | 0.805986 | 0.799975 | -6.300665 | 0.996250 | 0.998125 |
| GRPO 1000 | randomness | 0.300000 | 0.404945 | 0.801514 | 0.796646 | -6.432403 | 0.993750 | 0.996250 |
| GRPO 1000 | randomness | 0.600000 | 0.432903 | 0.801160 | 0.793475 | -6.375395 | 0.993750 | 0.995625 |
| GRPO 1000 | randomness | 1.000000 | 0.465324 | 0.795972 | 0.791575 | -6.390044 | 0.987500 | 0.990625 |
| GRPO 1000 | temperature | 0.100000 | 0.492474 | 0.794161 | 0.790090 | -6.455097 | 0.984375 | 0.991875 |
| GRPO 1000 | temperature | 0.300000 | 0.460905 | 0.798105 | 0.791639 | -6.384767 | 0.991875 | 0.994375 |
| GRPO 1000 | temperature | 0.600000 | 0.437596 | 0.800533 | 0.793329 | -6.367213 | 0.992500 | 0.996250 |
| GRPO 1000 | temperature | 1.000000 | 0.404945 | 0.801514 | 0.796646 | -6.409319 | 0.993750 | 0.996250 |
| SGRPO 1000 | randomness | 0.100000 | 0.661608 | 0.696176 | 0.735191 | -6.609278 | 0.958125 | 0.997500 |
| SGRPO 1000 | randomness | 0.300000 | 0.750277 | 0.691767 | 0.735042 | -6.418187 | 0.943125 | 0.990625 |
| SGRPO 1000 | randomness | 0.600000 | 0.806452 | 0.671060 | 0.732456 | -6.196509 | 0.941250 | 0.985625 |
| SGRPO 1000 | randomness | 1.000000 | 0.833511 | 0.660797 | 0.735957 | -6.146106 | 0.943750 | 0.986250 |
| SGRPO 1000 | temperature | 0.100000 | 0.822739 | 0.664581 | 0.736986 | -6.241247 | 0.948125 | 0.978750 |
| SGRPO 1000 | temperature | 0.300000 | 0.803842 | 0.670665 | 0.741770 | -6.115202 | 0.955000 | 0.990000 |
| SGRPO 1000 | temperature | 0.600000 | 0.769966 | 0.685704 | 0.732245 | -6.358646 | 0.946250 | 0.992500 |
| SGRPO 1000 | temperature | 1.000000 | 0.750277 | 0.691767 | 0.735042 | -6.409583 | 0.943125 | 0.990625 |
| GRPO DivReg 0.05 1000 | randomness | 0.100000 | 0.231363 | 0.855863 | 0.814055 | -7.284363 | 0.993750 | 0.998750 |
| GRPO DivReg 0.05 1000 | randomness | 0.300000 | 0.275579 | 0.846507 | 0.813896 | -7.154357 | 0.993125 | 0.996875 |
| GRPO DivReg 0.05 1000 | randomness | 0.600000 | 0.302271 | 0.841103 | 0.813651 | -7.174169 | 0.991875 | 0.996250 |
| GRPO DivReg 0.05 1000 | randomness | 1.000000 | 0.333871 | 0.833222 | 0.811154 | -7.110293 | 0.991250 | 0.997500 |
| GRPO DivReg 0.05 1000 | temperature | 0.100000 | 0.345071 | 0.832589 | 0.817787 | -7.100962 | 0.991250 | 0.995625 |
| GRPO DivReg 0.05 1000 | temperature | 0.300000 | 0.324592 | 0.837555 | 0.813652 | -7.068612 | 0.995000 | 0.999375 |
| GRPO DivReg 0.05 1000 | temperature | 0.600000 | 0.294551 | 0.842423 | 0.812717 | -7.160457 | 0.989375 | 0.996250 |
| GRPO DivReg 0.05 1000 | temperature | 1.000000 | 0.275579 | 0.846507 | 0.813896 | -7.192632 | 0.993125 | 0.996875 |

## randomness diversity vs QED

![randomness diversity vs QED](mmgenmol_randomness_diversity_vs_qed_mean_20260423.png)

## randomness diversity vs SA Score

![randomness diversity vs SA Score](mmgenmol_randomness_diversity_vs_sa_score_mean_20260423.png)

## randomness diversity vs Vina Dock Mean

![randomness diversity vs Vina Dock Mean](mmgenmol_randomness_diversity_vs_vina_dock_mean_20260423.png)

## temperature diversity vs QED

![temperature diversity vs QED](mmgenmol_temperature_diversity_vs_qed_mean_20260423.png)

## temperature diversity vs SA Score

![temperature diversity vs SA Score](mmgenmol_temperature_diversity_vs_sa_score_mean_20260423.png)

## temperature diversity vs Vina Dock Mean

![temperature diversity vs Vina Dock Mean](mmgenmol_temperature_diversity_vs_vina_dock_mean_20260423.png)
