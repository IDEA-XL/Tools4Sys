# De Novo Evaluation

- `num_samples`: 1000
- `generation_batch_size`: 2048
- `min_add_len`: 60
- `max_completion_length`: None
- `sweep_axis`: generation_temperature
- `generation_temperature`: 1.0
- `randomness`: 0.3
- `sweep_values`: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

- `QED vs Diversity plot`: `/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/genmol-denovo/qed_vs_diversity_temperature_20260425.png`
- `SA Score vs Diversity plot`: `/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/genmol-denovo/sa_score_vs_diversity_temperature_20260425.png`

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
| GenMol De Novo GRPO 1000 | generation_temperature | 0.100000 | 0.100000 | 0.300000 | 0.705709 | 0.861510 | 0.678810 | 0.788430 | 0.805219 | 1.000000 | 0.146000 | 0.000000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 0.200000 | 0.200000 | 0.300000 | 0.735653 | 0.873225 | 0.710623 | 0.808184 | 0.790612 | 1.000000 | 0.128000 | 0.000000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 0.300000 | 0.300000 | 0.300000 | 0.782896 | 0.882876 | 0.744719 | 0.827613 | 0.776493 | 1.000000 | 0.078000 | 0.000000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 0.400000 | 0.400000 | 0.300000 | 0.792128 | 0.885153 | 0.766363 | 0.837637 | 0.770153 | 1.000000 | 0.076000 | 0.000000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.801563 | 0.883724 | 0.776823 | 0.840963 | 0.769721 | 1.000000 | 0.065000 | 0.000000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 0.600000 | 0.600000 | 0.300000 | 0.798700 | 0.883738 | 0.776520 | 0.840851 | 0.776427 | 1.000000 | 0.067000 | 0.000000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 0.700000 | 0.700000 | 0.300000 | 0.782002 | 0.875359 | 0.769655 | 0.833078 | 0.786045 | 1.000000 | 0.082000 | 0.000000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 0.800000 | 0.800000 | 0.300000 | 0.761879 | 0.873133 | 0.757682 | 0.826953 | 0.792439 | 1.000000 | 0.106000 | 0.000000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 0.900000 | 0.900000 | 0.300000 | 0.761061 | 0.870710 | 0.761584 | 0.827059 | 0.797845 | 1.000000 | 0.106000 | 0.000000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.744680 | 0.864880 | 0.742290 | 0.815844 | 0.805996 | 1.000000 | 0.117000 | 0.000000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 0.100000 | 0.100000 | 0.300000 | 0.687721 | 0.833610 | 0.653657 | 0.761629 | 0.823080 | 1.000000 | 0.145000 | 0.000000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 0.200000 | 0.200000 | 0.300000 | 0.723566 | 0.854673 | 0.683162 | 0.786069 | 0.810100 | 1.000000 | 0.117000 | 0.000000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 0.300000 | 0.300000 | 0.300000 | 0.754870 | 0.862478 | 0.717425 | 0.804457 | 0.801382 | 1.000000 | 0.092000 | 0.000000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 0.400000 | 0.400000 | 0.300000 | 0.779360 | 0.871081 | 0.751198 | 0.823128 | 0.791542 | 1.000000 | 0.075000 | 0.000000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.784886 | 0.871231 | 0.761872 | 0.827487 | 0.789698 | 1.000000 | 0.071000 | 0.000000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 0.600000 | 0.600000 | 0.300000 | 0.777833 | 0.868229 | 0.763725 | 0.826427 | 0.794713 | 1.000000 | 0.081000 | 0.000000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 0.700000 | 0.700000 | 0.300000 | 0.767664 | 0.867297 | 0.761642 | 0.825035 | 0.800068 | 1.000000 | 0.094000 | 0.000000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 0.800000 | 0.800000 | 0.300000 | 0.754514 | 0.863073 | 0.756149 | 0.820304 | 0.804675 | 1.000000 | 0.108000 | 0.000000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 0.900000 | 0.900000 | 0.300000 | 0.729458 | 0.853677 | 0.741025 | 0.808616 | 0.816962 | 1.000000 | 0.131000 | 0.000000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.716995 | 0.849317 | 0.730097 | 0.801629 | 0.823235 | 0.999000 | 0.140000 | 0.001000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 0.100000 | 0.100000 | 0.300000 | 0.764880 | 0.872799 | 0.733943 | 0.817257 | 0.790905 | 1.000000 | 0.089000 | 0.000000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 0.200000 | 0.200000 | 0.300000 | 0.796018 | 0.884701 | 0.746499 | 0.829420 | 0.775145 | 1.000000 | 0.056000 | 0.000000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 0.300000 | 0.300000 | 0.300000 | 0.808908 | 0.889686 | 0.763874 | 0.839361 | 0.747163 | 1.000000 | 0.052000 | 0.000000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 0.400000 | 0.400000 | 0.300000 | 0.825913 | 0.895696 | 0.772914 | 0.846583 | 0.735487 | 1.000000 | 0.035000 | 0.000000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.829865 | 0.900434 | 0.775730 | 0.850553 | 0.724616 | 0.999000 | 0.031000 | 0.001000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 0.600000 | 0.600000 | 0.300000 | 0.828714 | 0.901412 | 0.780129 | 0.852899 | 0.720557 | 0.998000 | 0.034000 | 0.002000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 0.700000 | 0.700000 | 0.300000 | 0.822356 | 0.898065 | 0.777327 | 0.849770 | 0.728766 | 0.999000 | 0.040000 | 0.001000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 0.800000 | 0.800000 | 0.300000 | 0.817503 | 0.896165 | 0.769948 | 0.845678 | 0.736908 | 1.000000 | 0.045000 | 0.000000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 0.900000 | 0.900000 | 0.300000 | 0.813391 | 0.889556 | 0.770468 | 0.841921 | 0.752572 | 1.000000 | 0.046000 | 0.000000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.801329 | 0.886614 | 0.759618 | 0.835815 | 0.761776 | 1.000000 | 0.057000 | 0.000000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 0.100000 | 0.100000 | 0.300000 | 0.790304 | 0.872586 | 0.723002 | 0.812752 | 0.796137 | 1.000000 | 0.041000 | 0.000000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 0.200000 | 0.200000 | 0.300000 | 0.798168 | 0.879227 | 0.744764 | 0.825442 | 0.783197 | 1.000000 | 0.048000 | 0.000000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 0.300000 | 0.300000 | 0.300000 | 0.807078 | 0.881430 | 0.758349 | 0.832198 | 0.774948 | 1.000000 | 0.044000 | 0.000000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 0.400000 | 0.400000 | 0.300000 | 0.817360 | 0.887946 | 0.769843 | 0.840705 | 0.768314 | 0.999000 | 0.036000 | 0.001000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.823839 | 0.885329 | 0.773563 | 0.840623 | 0.769146 | 1.000000 | 0.027000 | 0.000000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 0.600000 | 0.600000 | 0.300000 | 0.825127 | 0.887866 | 0.777239 | 0.843616 | 0.768579 | 1.000000 | 0.030000 | 0.000000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 0.700000 | 0.700000 | 0.300000 | 0.807770 | 0.883491 | 0.775024 | 0.840105 | 0.772449 | 1.000000 | 0.054000 | 0.000000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 0.800000 | 0.800000 | 0.300000 | 0.802018 | 0.881670 | 0.760360 | 0.833146 | 0.780764 | 1.000000 | 0.052000 | 0.000000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 0.900000 | 0.900000 | 0.300000 | 0.792648 | 0.874711 | 0.757024 | 0.827636 | 0.790438 | 0.999000 | 0.053000 | 0.001000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.778939 | 0.870511 | 0.738037 | 0.817522 | 0.798605 | 1.000000 | 0.066000 | 0.000000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 0.100000 | 0.100000 | 0.300000 | 0.799861 | 0.873000 | 0.777468 | 0.834787 | 0.772820 | 0.999000 | 0.056000 | 0.001000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 0.200000 | 0.200000 | 0.300000 | 0.812718 | 0.887986 | 0.783517 | 0.846198 | 0.745769 | 0.999000 | 0.052000 | 0.001000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 0.300000 | 0.300000 | 0.300000 | 0.825418 | 0.890166 | 0.794430 | 0.851872 | 0.730370 | 1.000000 | 0.043000 | 0.000000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 0.400000 | 0.400000 | 0.300000 | 0.849157 | 0.897845 | 0.810759 | 0.863010 | 0.712006 | 0.998000 | 0.017000 | 0.002000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.848131 | 0.896093 | 0.810225 | 0.861746 | 0.713734 | 1.000000 | 0.022000 | 0.000000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 0.600000 | 0.600000 | 0.300000 | 0.842775 | 0.897172 | 0.806018 | 0.860710 | 0.706446 | 0.999000 | 0.026000 | 0.001000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 0.700000 | 0.700000 | 0.300000 | 0.840152 | 0.893121 | 0.803527 | 0.857284 | 0.718657 | 0.999000 | 0.025000 | 0.001000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 0.800000 | 0.800000 | 0.300000 | 0.837089 | 0.891307 | 0.802314 | 0.855710 | 0.721957 | 1.000000 | 0.030000 | 0.000000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 0.900000 | 0.900000 | 0.300000 | 0.824545 | 0.883641 | 0.793214 | 0.847470 | 0.748526 | 1.000000 | 0.037000 | 0.000000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.816059 | 0.880985 | 0.790981 | 0.844983 | 0.751374 | 1.000000 | 0.048000 | 0.000000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 0.100000 | 0.100000 | 0.300000 | 0.674082 | 0.837207 | 0.643064 | 0.759550 | 0.818640 | 1.000000 | 0.164000 | 0.000000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 0.200000 | 0.200000 | 0.300000 | 0.716133 | 0.860260 | 0.686980 | 0.790948 | 0.804709 | 1.000000 | 0.135000 | 0.000000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 0.300000 | 0.300000 | 0.300000 | 0.750900 | 0.867546 | 0.721859 | 0.809271 | 0.791934 | 1.000000 | 0.105000 | 0.000000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 0.400000 | 0.400000 | 0.300000 | 0.773262 | 0.874424 | 0.751291 | 0.825171 | 0.787256 | 1.000000 | 0.090000 | 0.000000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.775241 | 0.876640 | 0.758059 | 0.829208 | 0.782210 | 1.000000 | 0.091000 | 0.000000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 0.600000 | 0.600000 | 0.300000 | 0.767751 | 0.872140 | 0.763790 | 0.828800 | 0.792396 | 1.000000 | 0.100000 | 0.000000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 0.700000 | 0.700000 | 0.300000 | 0.755638 | 0.869881 | 0.759149 | 0.825589 | 0.795068 | 1.000000 | 0.114000 | 0.000000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 0.800000 | 0.800000 | 0.300000 | 0.761438 | 0.871078 | 0.758241 | 0.825943 | 0.799839 | 1.000000 | 0.105000 | 0.000000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 0.900000 | 0.900000 | 0.300000 | 0.733414 | 0.856575 | 0.745599 | 0.812185 | 0.813518 | 1.000000 | 0.129000 | 0.000000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.716405 | 0.849877 | 0.731992 | 0.802723 | 0.820869 | 1.000000 | 0.144000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 0.100000 | 0.100000 | 0.300000 | 0.685201 | 0.830895 | 0.653024 | 0.759747 | 0.822154 | 1.000000 | 0.149000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 0.200000 | 0.200000 | 0.300000 | 0.739007 | 0.857880 | 0.693510 | 0.792132 | 0.805019 | 1.000000 | 0.100000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 0.300000 | 0.300000 | 0.300000 | 0.765740 | 0.865143 | 0.724042 | 0.808702 | 0.798273 | 1.000000 | 0.079000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 0.400000 | 0.400000 | 0.300000 | 0.782193 | 0.872814 | 0.749791 | 0.823605 | 0.789217 | 1.000000 | 0.072000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.788285 | 0.869712 | 0.760797 | 0.826146 | 0.791573 | 1.000000 | 0.064000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 0.600000 | 0.600000 | 0.300000 | 0.782454 | 0.871973 | 0.764927 | 0.829154 | 0.792290 | 1.000000 | 0.077000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 0.700000 | 0.700000 | 0.300000 | 0.774808 | 0.869577 | 0.768914 | 0.829312 | 0.796202 | 0.999000 | 0.085000 | 0.001000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 0.800000 | 0.800000 | 0.300000 | 0.757537 | 0.864986 | 0.753729 | 0.820483 | 0.804792 | 1.000000 | 0.103000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 0.900000 | 0.900000 | 0.300000 | 0.740780 | 0.858124 | 0.745306 | 0.812997 | 0.812252 | 1.000000 | 0.119000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.731536 | 0.848269 | 0.735823 | 0.803291 | 0.824581 | 1.000000 | 0.121000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 0.100000 | 0.100000 | 0.300000 | 0.675575 | 0.834482 | 0.637723 | 0.755779 | 0.818881 | 0.999000 | 0.153000 | 0.001000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 0.200000 | 0.200000 | 0.300000 | 0.710027 | 0.855939 | 0.679553 | 0.785385 | 0.804640 | 1.000000 | 0.136000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 0.300000 | 0.300000 | 0.300000 | 0.733835 | 0.866209 | 0.715323 | 0.805855 | 0.793901 | 1.000000 | 0.129000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 0.400000 | 0.400000 | 0.300000 | 0.779724 | 0.875478 | 0.744792 | 0.823203 | 0.782629 | 1.000000 | 0.075000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.772862 | 0.873176 | 0.760706 | 0.828188 | 0.782634 | 1.000000 | 0.092000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 0.600000 | 0.600000 | 0.300000 | 0.777308 | 0.874792 | 0.766766 | 0.831582 | 0.782821 | 1.000000 | 0.088000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 0.700000 | 0.700000 | 0.300000 | 0.759867 | 0.868577 | 0.761644 | 0.825804 | 0.792598 | 1.000000 | 0.108000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 0.800000 | 0.800000 | 0.300000 | 0.758132 | 0.866664 | 0.755306 | 0.822121 | 0.797902 | 1.000000 | 0.106000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 0.900000 | 0.900000 | 0.300000 | 0.725639 | 0.853142 | 0.742662 | 0.808950 | 0.811955 | 1.000000 | 0.138000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.728024 | 0.852735 | 0.734503 | 0.805443 | 0.820162 | 0.999000 | 0.126000 | 0.001000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 0.100000 | 0.100000 | 0.300000 | 0.785050 | 0.869204 | 0.740455 | 0.817705 | 0.796032 | 1.000000 | 0.058000 | 0.000000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 0.200000 | 0.200000 | 0.300000 | 0.807098 | 0.879171 | 0.761492 | 0.832099 | 0.781387 | 1.000000 | 0.044000 | 0.000000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 0.300000 | 0.300000 | 0.300000 | 0.815704 | 0.883122 | 0.769879 | 0.837825 | 0.768653 | 1.000000 | 0.038000 | 0.000000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 0.400000 | 0.400000 | 0.300000 | 0.825493 | 0.885125 | 0.783658 | 0.844538 | 0.764216 | 1.000000 | 0.032000 | 0.000000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.832398 | 0.885754 | 0.793703 | 0.848933 | 0.758690 | 1.000000 | 0.027000 | 0.000000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 0.600000 | 0.600000 | 0.300000 | 0.828776 | 0.884854 | 0.785435 | 0.845087 | 0.766354 | 1.000000 | 0.027000 | 0.000000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 0.700000 | 0.700000 | 0.300000 | 0.810866 | 0.879085 | 0.782944 | 0.840629 | 0.771054 | 1.000000 | 0.049000 | 0.000000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 0.800000 | 0.800000 | 0.300000 | 0.799640 | 0.879699 | 0.769583 | 0.835653 | 0.778943 | 1.000000 | 0.059000 | 0.000000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 0.900000 | 0.900000 | 0.300000 | 0.792275 | 0.876403 | 0.765615 | 0.832087 | 0.784859 | 1.000000 | 0.064000 | 0.000000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.780772 | 0.872703 | 0.760037 | 0.827637 | 0.797460 | 1.000000 | 0.075000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 0.100000 | 0.100000 | 0.300000 | 0.780769 | 0.864783 | 0.739158 | 0.814533 | 0.800416 | 1.000000 | 0.060000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 0.200000 | 0.200000 | 0.300000 | 0.799358 | 0.877451 | 0.753686 | 0.827945 | 0.782338 | 1.000000 | 0.050000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 0.300000 | 0.300000 | 0.300000 | 0.815971 | 0.884262 | 0.773913 | 0.840123 | 0.771141 | 1.000000 | 0.041000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 0.400000 | 0.400000 | 0.300000 | 0.827094 | 0.887482 | 0.785922 | 0.846858 | 0.759635 | 0.999000 | 0.031000 | 0.001000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.830787 | 0.887273 | 0.789172 | 0.848033 | 0.757807 | 1.000000 | 0.028000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 0.600000 | 0.600000 | 0.300000 | 0.830742 | 0.888045 | 0.791419 | 0.849395 | 0.756130 | 1.000000 | 0.030000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 0.700000 | 0.700000 | 0.300000 | 0.815893 | 0.884746 | 0.785511 | 0.845052 | 0.767242 | 1.000000 | 0.047000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 0.800000 | 0.800000 | 0.300000 | 0.798483 | 0.876381 | 0.771816 | 0.834555 | 0.783465 | 1.000000 | 0.058000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 0.900000 | 0.900000 | 0.300000 | 0.796051 | 0.871330 | 0.763926 | 0.828368 | 0.792009 | 1.000000 | 0.052000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.775629 | 0.868284 | 0.753113 | 0.822215 | 0.802744 | 1.000000 | 0.077000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 0.100000 | 0.100000 | 0.300000 | 0.765944 | 0.866318 | 0.723517 | 0.809198 | 0.799640 | 0.999000 | 0.074000 | 0.001000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 0.200000 | 0.200000 | 0.300000 | 0.798662 | 0.881360 | 0.744857 | 0.826759 | 0.780989 | 1.000000 | 0.049000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 0.300000 | 0.300000 | 0.300000 | 0.806094 | 0.883293 | 0.762002 | 0.834777 | 0.771744 | 1.000000 | 0.048000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 0.400000 | 0.400000 | 0.300000 | 0.822671 | 0.889806 | 0.782712 | 0.846968 | 0.764553 | 1.000000 | 0.041000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.825874 | 0.888409 | 0.790786 | 0.849360 | 0.765010 | 0.999000 | 0.035000 | 0.001000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 0.600000 | 0.600000 | 0.300000 | 0.816446 | 0.890039 | 0.788056 | 0.849246 | 0.767494 | 0.999000 | 0.050000 | 0.001000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 0.700000 | 0.700000 | 0.300000 | 0.809679 | 0.886076 | 0.784217 | 0.845333 | 0.776417 | 1.000000 | 0.056000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 0.800000 | 0.800000 | 0.300000 | 0.800907 | 0.877751 | 0.773323 | 0.835980 | 0.785662 | 1.000000 | 0.057000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 0.900000 | 0.900000 | 0.300000 | 0.776581 | 0.873747 | 0.764940 | 0.830224 | 0.791854 | 0.999000 | 0.084000 | 0.001000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.764663 | 0.866353 | 0.751772 | 0.820520 | 0.808639 | 0.999000 | 0.088000 | 0.001000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 0.100000 | 0.100000 | 0.300000 | 0.677661 | 0.833414 | 0.651565 | 0.760674 | 0.820720 | 1.000000 | 0.164000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 0.200000 | 0.200000 | 0.300000 | 0.712672 | 0.854532 | 0.682761 | 0.785824 | 0.807260 | 1.000000 | 0.133000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 0.300000 | 0.300000 | 0.300000 | 0.747442 | 0.864960 | 0.723116 | 0.808223 | 0.799683 | 1.000000 | 0.111000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 0.400000 | 0.400000 | 0.300000 | 0.779855 | 0.871117 | 0.751571 | 0.823299 | 0.792338 | 1.000000 | 0.075000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.773570 | 0.870074 | 0.757241 | 0.824940 | 0.792090 | 1.000000 | 0.085000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 0.600000 | 0.600000 | 0.300000 | 0.781049 | 0.874626 | 0.766332 | 0.831308 | 0.791675 | 1.000000 | 0.083000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 0.700000 | 0.700000 | 0.300000 | 0.761732 | 0.867498 | 0.758637 | 0.823954 | 0.799016 | 1.000000 | 0.102000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 0.800000 | 0.800000 | 0.300000 | 0.757025 | 0.864869 | 0.757377 | 0.821872 | 0.805799 | 1.000000 | 0.106000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 0.900000 | 0.900000 | 0.300000 | 0.724438 | 0.853901 | 0.739680 | 0.808212 | 0.816075 | 1.000000 | 0.139000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.719713 | 0.851960 | 0.731678 | 0.803847 | 0.821835 | 1.000000 | 0.141000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 0.100000 | 0.100000 | 0.300000 | 0.779544 | 0.865107 | 0.724190 | 0.808740 | 0.804344 | 0.999000 | 0.050000 | 0.001000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 0.200000 | 0.200000 | 0.300000 | 0.795873 | 0.877753 | 0.745615 | 0.824897 | 0.789862 | 1.000000 | 0.051000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 0.300000 | 0.300000 | 0.300000 | 0.804130 | 0.884316 | 0.756989 | 0.833385 | 0.783629 | 1.000000 | 0.050000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 0.400000 | 0.400000 | 0.300000 | 0.825133 | 0.884359 | 0.777407 | 0.841578 | 0.775956 | 0.999000 | 0.025000 | 0.001000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.819746 | 0.879590 | 0.779287 | 0.839469 | 0.781882 | 0.999000 | 0.029000 | 0.001000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 0.600000 | 0.600000 | 0.300000 | 0.820125 | 0.884277 | 0.780508 | 0.842769 | 0.780634 | 1.000000 | 0.037000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 0.700000 | 0.700000 | 0.300000 | 0.805352 | 0.878509 | 0.775497 | 0.837304 | 0.784670 | 1.000000 | 0.053000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 0.800000 | 0.800000 | 0.300000 | 0.791392 | 0.870680 | 0.761532 | 0.827021 | 0.796792 | 1.000000 | 0.059000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 0.900000 | 0.900000 | 0.300000 | 0.776156 | 0.868755 | 0.752211 | 0.822137 | 0.803229 | 1.000000 | 0.074000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.770835 | 0.864247 | 0.745594 | 0.816786 | 0.810833 | 1.000000 | 0.076000 | 0.000000 |

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
