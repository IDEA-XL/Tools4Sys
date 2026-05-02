# De Novo Evaluation

- `num_samples`: 1000
- `generation_batch_size`: 2048
- `min_add_len`: 60
- `max_completion_length`: None
- `sweep_axis`: generation_temperature
- `generation_temperature`: 1.0
- `randomness`: 0.3
- `sweep_values`: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

- `QED vs Diversity plot`: `/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/genmol-denovo/qed_vs_diversity_temperature_20260422.png`
- `SA Score vs Diversity plot`: `/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/genmol-denovo/sa_score_vs_diversity_temperature_20260422.png`

| Model | Sweep Axis | Sweep Value | Generation Temperature | Randomness | Overall De Novo Score | QED | SA Score | Soft Quality Score | Internal Diversity | Valid Molecule Rate | Alert Hit Rate | Invalid Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original GenMol v2 | generation_temperature | 0.100000 | 0.100000 | 0.300000 | 0.580626 | 0.786905 | 0.577388 | 0.703098 | 0.836594 | 1.000000 | 0.258000 | 0.000000 |
| Original GenMol v2 | generation_temperature | 0.200000 | 0.200000 | 0.300000 | 0.624695 | 0.839701 | 0.626454 | 0.754403 | 0.818409 | 1.000000 | 0.240000 | 0.000000 |
| Original GenMol v2 | generation_temperature | 0.300000 | 0.300000 | 0.300000 | 0.655261 | 0.857589 | 0.662194 | 0.779431 | 0.811151 | 1.000000 | 0.221000 | 0.000000 |
| Original GenMol v2 | generation_temperature | 0.400000 | 0.400000 | 0.300000 | 0.698777 | 0.864234 | 0.697799 | 0.797660 | 0.806157 | 1.000000 | 0.171000 | 0.000000 |
| Original GenMol v2 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.693429 | 0.867821 | 0.713369 | 0.806040 | 0.801846 | 1.000000 | 0.189000 | 0.000000 |
| Original GenMol v2 | generation_temperature | 0.600000 | 0.600000 | 0.300000 | 0.712149 | 0.868184 | 0.736451 | 0.815491 | 0.802722 | 1.000000 | 0.170000 | 0.000000 |
| Original GenMol v2 | generation_temperature | 0.700000 | 0.700000 | 0.300000 | 0.694096 | 0.860899 | 0.733470 | 0.809928 | 0.806885 | 1.000000 | 0.191000 | 0.000000 |
| Original GenMol v2 | generation_temperature | 0.800000 | 0.800000 | 0.300000 | 0.687302 | 0.856521 | 0.720292 | 0.802029 | 0.816576 | 1.000000 | 0.190000 | 0.000000 |
| Original GenMol v2 | generation_temperature | 0.900000 | 0.900000 | 0.300000 | 0.671741 | 0.847058 | 0.713547 | 0.793654 | 0.824007 | 1.000000 | 0.206000 | 0.000000 |
| Original GenMol v2 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.673101 | 0.843882 | 0.711123 | 0.790779 | 0.829922 | 1.000000 | 0.198000 | 0.000000 |
| GenMol De Novo GRPO | generation_temperature | 0.100000 | 0.100000 | 0.300000 | 0.705709 | 0.861510 | 0.678810 | 0.788430 | 0.805219 | 1.000000 | 0.146000 | 0.000000 |
| GenMol De Novo GRPO | generation_temperature | 0.200000 | 0.200000 | 0.300000 | 0.735653 | 0.873225 | 0.710623 | 0.808184 | 0.790612 | 1.000000 | 0.128000 | 0.000000 |
| GenMol De Novo GRPO | generation_temperature | 0.300000 | 0.300000 | 0.300000 | 0.782896 | 0.882876 | 0.744719 | 0.827613 | 0.776493 | 1.000000 | 0.078000 | 0.000000 |
| GenMol De Novo GRPO | generation_temperature | 0.400000 | 0.400000 | 0.300000 | 0.792128 | 0.885153 | 0.766363 | 0.837637 | 0.770153 | 1.000000 | 0.076000 | 0.000000 |
| GenMol De Novo GRPO | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.801563 | 0.883724 | 0.776823 | 0.840963 | 0.769721 | 1.000000 | 0.065000 | 0.000000 |
| GenMol De Novo GRPO | generation_temperature | 0.600000 | 0.600000 | 0.300000 | 0.798700 | 0.883738 | 0.776520 | 0.840851 | 0.776427 | 1.000000 | 0.067000 | 0.000000 |
| GenMol De Novo GRPO | generation_temperature | 0.700000 | 0.700000 | 0.300000 | 0.782002 | 0.875359 | 0.769655 | 0.833078 | 0.786045 | 1.000000 | 0.082000 | 0.000000 |
| GenMol De Novo GRPO | generation_temperature | 0.800000 | 0.800000 | 0.300000 | 0.761879 | 0.873133 | 0.757682 | 0.826953 | 0.792439 | 1.000000 | 0.106000 | 0.000000 |
| GenMol De Novo GRPO | generation_temperature | 0.900000 | 0.900000 | 0.300000 | 0.761061 | 0.870710 | 0.761584 | 0.827059 | 0.797845 | 1.000000 | 0.106000 | 0.000000 |
| GenMol De Novo GRPO | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.744680 | 0.864880 | 0.742290 | 0.815844 | 0.805996 | 1.000000 | 0.117000 | 0.000000 |
| GenMol De Novo SGRPO | generation_temperature | 0.100000 | 0.100000 | 0.300000 | 0.687721 | 0.833610 | 0.653657 | 0.761629 | 0.823080 | 1.000000 | 0.145000 | 0.000000 |
| GenMol De Novo SGRPO | generation_temperature | 0.200000 | 0.200000 | 0.300000 | 0.723566 | 0.854673 | 0.683162 | 0.786069 | 0.810100 | 1.000000 | 0.117000 | 0.000000 |
| GenMol De Novo SGRPO | generation_temperature | 0.300000 | 0.300000 | 0.300000 | 0.754870 | 0.862478 | 0.717425 | 0.804457 | 0.801382 | 1.000000 | 0.092000 | 0.000000 |
| GenMol De Novo SGRPO | generation_temperature | 0.400000 | 0.400000 | 0.300000 | 0.779360 | 0.871081 | 0.751198 | 0.823128 | 0.791542 | 1.000000 | 0.075000 | 0.000000 |
| GenMol De Novo SGRPO | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.784886 | 0.871231 | 0.761872 | 0.827487 | 0.789698 | 1.000000 | 0.071000 | 0.000000 |
| GenMol De Novo SGRPO | generation_temperature | 0.600000 | 0.600000 | 0.300000 | 0.777833 | 0.868229 | 0.763725 | 0.826427 | 0.794713 | 1.000000 | 0.081000 | 0.000000 |
| GenMol De Novo SGRPO | generation_temperature | 0.700000 | 0.700000 | 0.300000 | 0.767664 | 0.867297 | 0.761642 | 0.825035 | 0.800068 | 1.000000 | 0.094000 | 0.000000 |
| GenMol De Novo SGRPO | generation_temperature | 0.800000 | 0.800000 | 0.300000 | 0.754514 | 0.863073 | 0.756149 | 0.820304 | 0.804675 | 1.000000 | 0.108000 | 0.000000 |
| GenMol De Novo SGRPO | generation_temperature | 0.900000 | 0.900000 | 0.300000 | 0.729458 | 0.853677 | 0.741025 | 0.808616 | 0.816962 | 1.000000 | 0.131000 | 0.000000 |
| GenMol De Novo SGRPO | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.716995 | 0.849317 | 0.730097 | 0.801629 | 0.823235 | 0.999000 | 0.140000 | 0.001000 |

Column notes:
- `Overall De Novo Score`: mean final molecule-level reward after invalid handling and alert gating.
- `QED`: mean QED over valid generated molecules.
- `SA Score`: mean bounded SA-derived score used by training. Higher is better.
- `Soft Quality Score`: mean of `0.6 * QED + 0.4 * SA Score` over valid molecules.
- `Internal Diversity`: `1 - mean(pairwise Tanimoto similarity)` computed over all generated valid molecules for that run.
- `Valid Molecule Rate`: fraction of generated outputs that decode to valid molecules.
- `Alert Hit Rate`: fraction of generated outputs that hit the alert rule set.
- `Invalid Rate`: fraction of generated outputs that are invalid.

Row notes:
- Each row is one model evaluated at one sweep value.
- The line plots connect rows for the same model in increasing sweep order.
