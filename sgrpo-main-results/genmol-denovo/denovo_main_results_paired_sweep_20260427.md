# De Novo Evaluation

- `num_samples`: 1000
- `generation_batch_size`: 2048
- `min_add_len`: 60
- `max_completion_length`: None
- `sweep_axis`: randomness_temperature_pair
- `generation_temperature`: paired
- `randomness`: paired
- `randomness_temperature_pairs`: (0.1, 0.5), (0.3, 0.8), (0.5, 1.1), (0.7, 1.4), (0.9, 1.7), (1.0, 2.0)

- `QED vs Diversity plot`: `/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/genmol-denovo/qed_vs_diversity_paired_20260427.png`
- `SA Score vs Diversity plot`: `/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/genmol-denovo/sa_score_vs_diversity_paired_20260427.png`
- `Soft Quality Score vs Diversity plot`: `/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/genmol-denovo/soft_reward_vs_diversity_paired_20260427.png`

| Model | Sweep Axis | Sweep Value | Generation Temperature | Randomness | Overall De Novo Score | QED | SA Score | Soft Quality Score | Internal Diversity | Valid Molecule Rate | Alert Hit Rate | Invalid Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original GenMol v2 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.831476 | 0.909307 | 0.825485 | 0.875778 | 0.601590 | 1.000000 | 0.068000 | 0.000000 |
| Original GenMol v2 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.687302 | 0.856521 | 0.720292 | 0.802029 | 0.816576 | 1.000000 | 0.190000 | 0.000000 |
| Original GenMol v2 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.646333 | 0.832564 | 0.665332 | 0.765671 | 0.845972 | 0.999000 | 0.208000 | 0.001000 |
| Original GenMol v2 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.635406 | 0.811930 | 0.619179 | 0.734829 | 0.862780 | 0.999000 | 0.180000 | 0.001000 |
| Original GenMol v2 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.576100 | 0.772299 | 0.575239 | 0.693736 | 0.878104 | 0.996000 | 0.218000 | 0.004000 |
| Original GenMol v2 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.539570 | 0.731035 | 0.549315 | 0.658584 | 0.893775 | 0.993000 | 0.227000 | 0.007000 |
| GenMol De Novo GRPO 1000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.866941 | 0.916374 | 0.840436 | 0.885999 | 0.552708 | 1.000000 | 0.029000 | 0.000000 |
| GenMol De Novo GRPO 1000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.761879 | 0.873133 | 0.757682 | 0.826953 | 0.792439 | 1.000000 | 0.106000 | 0.000000 |
| GenMol De Novo GRPO 1000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.688787 | 0.845062 | 0.694756 | 0.784939 | 0.837802 | 0.999000 | 0.162000 | 0.001000 |
| GenMol De Novo GRPO 1000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.659250 | 0.818586 | 0.647827 | 0.750283 | 0.858284 | 0.999000 | 0.164000 | 0.001000 |
| GenMol De Novo GRPO 1000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.590732 | 0.778796 | 0.596571 | 0.705906 | 0.873801 | 0.996000 | 0.211000 | 0.004000 |
| GenMol De Novo GRPO 1000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.560071 | 0.743761 | 0.563669 | 0.671968 | 0.887565 | 0.997000 | 0.220000 | 0.003000 |
| GenMol De Novo GRPO Q0.8 SA0.2 1000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.876601 | 0.915931 | 0.839300 | 0.900604 | 0.536305 | 1.000000 | 0.036000 | 0.000000 |
| GenMol De Novo GRPO Q0.8 SA0.2 1000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.791040 | 0.873498 | 0.754317 | 0.849662 | 0.788379 | 0.999000 | 0.087000 | 0.001000 |
| GenMol De Novo GRPO Q0.8 SA0.2 1000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.722281 | 0.844770 | 0.689290 | 0.813674 | 0.837077 | 1.000000 | 0.148000 | 0.000000 |
| GenMol De Novo GRPO Q0.8 SA0.2 1000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.686180 | 0.817361 | 0.641601 | 0.782209 | 0.859302 | 0.999000 | 0.159000 | 0.001000 |
| GenMol De Novo GRPO Q0.8 SA0.2 1000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.625313 | 0.781177 | 0.588300 | 0.742958 | 0.874466 | 0.993000 | 0.191000 | 0.007000 |
| GenMol De Novo GRPO Q0.8 SA0.2 1000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.584878 | 0.745972 | 0.563691 | 0.709842 | 0.887667 | 0.996000 | 0.224000 | 0.004000 |
| GenMol De Novo SGRPO 1000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.865939 | 0.913297 | 0.831886 | 0.880732 | 0.600247 | 1.000000 | 0.023000 | 0.000000 |
| GenMol De Novo SGRPO 1000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.754514 | 0.863073 | 0.756149 | 0.820304 | 0.804675 | 1.000000 | 0.108000 | 0.000000 |
| GenMol De Novo SGRPO 1000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.685061 | 0.836065 | 0.694287 | 0.779354 | 0.842729 | 1.000000 | 0.163000 | 0.000000 |
| GenMol De Novo SGRPO 1000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.640503 | 0.812546 | 0.629429 | 0.739299 | 0.862905 | 0.998000 | 0.174000 | 0.002000 |
| GenMol De Novo SGRPO 1000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.582199 | 0.765078 | 0.591030 | 0.695459 | 0.876716 | 0.996000 | 0.211000 | 0.004000 |
| GenMol De Novo SGRPO 1000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.530974 | 0.728765 | 0.559351 | 0.661945 | 0.891402 | 0.988000 | 0.230000 | 0.012000 |
| GenMol De Novo SGRPO Thr 1000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.858990 | 0.912781 | 0.828908 | 0.879232 | 0.595924 | 1.000000 | 0.031000 | 0.000000 |
| GenMol De Novo SGRPO Thr 1000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.761438 | 0.871078 | 0.758241 | 0.825943 | 0.799839 | 1.000000 | 0.105000 | 0.000000 |
| GenMol De Novo SGRPO Thr 1000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.675045 | 0.837499 | 0.687539 | 0.777515 | 0.841876 | 1.000000 | 0.178000 | 0.000000 |
| GenMol De Novo SGRPO Thr 1000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.642722 | 0.814809 | 0.630553 | 0.741107 | 0.861802 | 0.996000 | 0.169000 | 0.004000 |
| GenMol De Novo SGRPO Thr 1000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.577878 | 0.766416 | 0.589270 | 0.695557 | 0.876502 | 0.991000 | 0.203000 | 0.009000 |
| GenMol De Novo SGRPO Thr 1000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.546605 | 0.735324 | 0.560612 | 0.666158 | 0.892657 | 0.992000 | 0.217000 | 0.008000 |
| GenMol De Novo SGRPO RewardSum 1000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.865923 | 0.913005 | 0.835783 | 0.882116 | 0.588997 | 1.000000 | 0.025000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 1000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.757537 | 0.864986 | 0.753729 | 0.820483 | 0.804792 | 1.000000 | 0.103000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 1000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.688865 | 0.836411 | 0.692113 | 0.778692 | 0.842399 | 0.999000 | 0.156000 | 0.001000 |
| GenMol De Novo SGRPO RewardSum 1000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.649947 | 0.814377 | 0.634109 | 0.742270 | 0.862096 | 0.998000 | 0.165000 | 0.002000 |
| GenMol De Novo SGRPO RewardSum 1000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.588042 | 0.767559 | 0.591859 | 0.697537 | 0.876519 | 0.996000 | 0.202000 | 0.004000 |
| GenMol De Novo SGRPO RewardSum 1000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.532171 | 0.731698 | 0.559541 | 0.663313 | 0.890937 | 0.985000 | 0.217000 | 0.015000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.858323 | 0.912463 | 0.834037 | 0.881093 | 0.587061 | 1.000000 | 0.035000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.758132 | 0.866664 | 0.755306 | 0.822121 | 0.797902 | 1.000000 | 0.106000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.685885 | 0.840264 | 0.685221 | 0.778246 | 0.841144 | 1.000000 | 0.162000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.641833 | 0.814990 | 0.626030 | 0.739406 | 0.862116 | 0.998000 | 0.173000 | 0.002000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.583057 | 0.773064 | 0.582131 | 0.696691 | 0.877123 | 0.995000 | 0.205000 | 0.005000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.520699 | 0.725632 | 0.554244 | 0.657547 | 0.893469 | 0.986000 | 0.239000 | 0.014000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.861827 | 0.913091 | 0.832700 | 0.880935 | 0.600416 | 1.000000 | 0.029000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.757025 | 0.864869 | 0.757377 | 0.821872 | 0.805799 | 1.000000 | 0.106000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.675251 | 0.834490 | 0.687252 | 0.775595 | 0.842167 | 0.999000 | 0.172000 | 0.001000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.639245 | 0.812324 | 0.629057 | 0.739017 | 0.862452 | 0.997000 | 0.174000 | 0.003000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.587243 | 0.768125 | 0.590467 | 0.697062 | 0.876558 | 0.998000 | 0.212000 | 0.002000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.534291 | 0.725830 | 0.557289 | 0.658883 | 0.892139 | 0.988000 | 0.216000 | 0.012000 |
| GenMol De Novo SGRPO RewardSum LOO 1000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.862279 | 0.913427 | 0.834962 | 0.882041 | 0.576457 | 1.000000 | 0.030000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO 1000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.744044 | 0.858991 | 0.747609 | 0.814438 | 0.816294 | 1.000000 | 0.116000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO 1000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.690156 | 0.838671 | 0.690331 | 0.779335 | 0.845341 | 1.000000 | 0.156000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO 1000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.651424 | 0.813891 | 0.638002 | 0.743535 | 0.862535 | 0.997000 | 0.160000 | 0.003000 |
| GenMol De Novo SGRPO RewardSum LOO 1000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.592925 | 0.770078 | 0.587910 | 0.697211 | 0.878449 | 0.997000 | 0.196000 | 0.003000 |
| GenMol De Novo SGRPO RewardSum LOO 1000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.523546 | 0.726473 | 0.555007 | 0.658356 | 0.893775 | 0.990000 | 0.245000 | 0.010000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 1000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.863464 | 0.912974 | 0.831192 | 0.880261 | 0.622891 | 1.000000 | 0.026000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 1000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.767530 | 0.869880 | 0.763983 | 0.827521 | 0.797095 | 1.000000 | 0.099000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 1000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.686130 | 0.838779 | 0.687803 | 0.778388 | 0.840586 | 1.000000 | 0.161000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 1000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.659348 | 0.815020 | 0.645151 | 0.747072 | 0.860781 | 0.999000 | 0.155000 | 0.001000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 1000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.587666 | 0.767139 | 0.602320 | 0.701211 | 0.876352 | 0.995000 | 0.206000 | 0.005000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 1000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.549843 | 0.734786 | 0.567181 | 0.667983 | 0.889475 | 0.994000 | 0.222000 | 0.006000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 1000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.839359 | 0.909421 | 0.830770 | 0.877961 | 0.598964 | 1.000000 | 0.059000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 1000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.713993 | 0.857872 | 0.740864 | 0.811069 | 0.813515 | 1.000000 | 0.160000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 1000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.660417 | 0.831699 | 0.681818 | 0.771746 | 0.846134 | 1.000000 | 0.194000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 1000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.639307 | 0.811051 | 0.630340 | 0.738767 | 0.864501 | 0.997000 | 0.171000 | 0.003000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 1000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.575546 | 0.768568 | 0.584924 | 0.695110 | 0.878462 | 0.993000 | 0.210000 | 0.007000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 1000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.513422 | 0.723952 | 0.551421 | 0.655644 | 0.892854 | 0.982000 | 0.235000 | 0.018000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 1000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.845358 | 0.905842 | 0.831106 | 0.875947 | 0.609033 | 1.000000 | 0.047000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 1000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.708164 | 0.854493 | 0.736994 | 0.807493 | 0.816848 | 1.000000 | 0.164000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 1000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.664839 | 0.835021 | 0.680419 | 0.773180 | 0.845768 | 1.000000 | 0.190000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 1000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.632990 | 0.808667 | 0.626868 | 0.735948 | 0.863595 | 0.994000 | 0.171000 | 0.006000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 1000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.573006 | 0.764140 | 0.579587 | 0.690319 | 0.879516 | 0.996000 | 0.218000 | 0.004000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 1000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.512871 | 0.718686 | 0.551242 | 0.652171 | 0.894186 | 0.985000 | 0.242000 | 0.015000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 1000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.874356 | 0.912981 | 0.825632 | 0.895511 | 0.600938 | 1.000000 | 0.032000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 1000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.785362 | 0.870939 | 0.759390 | 0.848629 | 0.795837 | 1.000000 | 0.098000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 1000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.728814 | 0.846251 | 0.692702 | 0.815541 | 0.839175 | 1.000000 | 0.140000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 1000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.686659 | 0.814901 | 0.639012 | 0.779723 | 0.861381 | 0.999000 | 0.155000 | 0.001000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 1000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.613329 | 0.775690 | 0.594724 | 0.739849 | 0.875326 | 0.991000 | 0.202000 | 0.009000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 1000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.583407 | 0.737497 | 0.563545 | 0.703669 | 0.890487 | 0.993000 | 0.208000 | 0.007000 |
| GenMol De Novo GRPO 2000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.855756 | 0.900265 | 0.826814 | 0.870884 | 0.530927 | 1.000000 | 0.024000 | 0.000000 |
| GenMol De Novo GRPO 2000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.817503 | 0.896165 | 0.769948 | 0.845678 | 0.736908 | 1.000000 | 0.045000 | 0.000000 |
| GenMol De Novo GRPO 2000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.738930 | 0.860047 | 0.722103 | 0.804870 | 0.817734 | 0.999000 | 0.108000 | 0.001000 |
| GenMol De Novo GRPO 2000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.695756 | 0.834209 | 0.670031 | 0.768538 | 0.849942 | 1.000000 | 0.128000 | 0.000000 |
| GenMol De Novo GRPO 2000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.631906 | 0.799819 | 0.625996 | 0.730290 | 0.867667 | 0.993000 | 0.160000 | 0.007000 |
| GenMol De Novo GRPO 2000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.571261 | 0.753196 | 0.583905 | 0.685981 | 0.885856 | 0.990000 | 0.199000 | 0.010000 |
| GenMol De Novo GRPO Q0.8 SA0.2 2000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.883623 | 0.919093 | 0.812268 | 0.897728 | 0.610971 | 1.000000 | 0.022000 | 0.000000 |
| GenMol De Novo GRPO Q0.8 SA0.2 2000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.825887 | 0.892610 | 0.750694 | 0.864227 | 0.755567 | 1.000000 | 0.060000 | 0.000000 |
| GenMol De Novo GRPO Q0.8 SA0.2 2000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.764498 | 0.864534 | 0.698620 | 0.831351 | 0.820805 | 0.999000 | 0.104000 | 0.001000 |
| GenMol De Novo GRPO Q0.8 SA0.2 2000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.724375 | 0.840898 | 0.652047 | 0.803128 | 0.849436 | 0.999000 | 0.126000 | 0.001000 |
| GenMol De Novo GRPO Q0.8 SA0.2 2000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.652296 | 0.793552 | 0.602640 | 0.755369 | 0.871441 | 0.996000 | 0.176000 | 0.004000 |
| GenMol De Novo GRPO Q0.8 SA0.2 2000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.578535 | 0.746529 | 0.569519 | 0.711456 | 0.887100 | 0.990000 | 0.225000 | 0.010000 |
| GenMol De Novo SGRPO 2000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.871909 | 0.907599 | 0.835861 | 0.878904 | 0.644995 | 1.000000 | 0.011000 | 0.000000 |
| GenMol De Novo SGRPO 2000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.802018 | 0.881670 | 0.760360 | 0.833146 | 0.780764 | 1.000000 | 0.052000 | 0.000000 |
| GenMol De Novo SGRPO 2000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.733414 | 0.849890 | 0.699721 | 0.789822 | 0.835456 | 0.998000 | 0.090000 | 0.002000 |
| GenMol De Novo SGRPO 2000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.692180 | 0.825380 | 0.649107 | 0.754871 | 0.858110 | 0.999000 | 0.112000 | 0.001000 |
| GenMol De Novo SGRPO 2000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.606978 | 0.777391 | 0.606350 | 0.709239 | 0.875909 | 0.993000 | 0.175000 | 0.007000 |
| GenMol De Novo SGRPO 2000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.540616 | 0.733869 | 0.562375 | 0.665749 | 0.894650 | 0.992000 | 0.231000 | 0.008000 |
| GenMol De Novo GRPO DivReg0.05 2000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.876772 | 0.913784 | 0.837579 | 0.883302 | 0.616345 | 1.000000 | 0.010000 | 0.000000 |
| GenMol De Novo GRPO DivReg0.05 2000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.837089 | 0.891307 | 0.802314 | 0.855710 | 0.721957 | 1.000000 | 0.030000 | 0.000000 |
| GenMol De Novo GRPO DivReg0.05 2000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.756688 | 0.857122 | 0.736234 | 0.808767 | 0.814117 | 1.000000 | 0.088000 | 0.000000 |
| GenMol De Novo GRPO DivReg0.05 2000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.707457 | 0.828250 | 0.701536 | 0.777564 | 0.844052 | 0.997000 | 0.113000 | 0.003000 |
| GenMol De Novo GRPO DivReg0.05 2000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.645548 | 0.797822 | 0.648303 | 0.738014 | 0.865086 | 0.997000 | 0.161000 | 0.003000 |
| GenMol De Novo GRPO DivReg0.05 2000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.571184 | 0.759352 | 0.601315 | 0.696901 | 0.881661 | 0.989000 | 0.210000 | 0.011000 |
| GenMol De Novo SGRPO Thr 2000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.876311 | 0.911295 | 0.836710 | 0.881461 | 0.632784 | 1.000000 | 0.008000 | 0.000000 |
| GenMol De Novo SGRPO Thr 2000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.799640 | 0.879699 | 0.769583 | 0.835653 | 0.778943 | 1.000000 | 0.059000 | 0.000000 |
| GenMol De Novo SGRPO Thr 2000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.717156 | 0.847387 | 0.709637 | 0.792287 | 0.833523 | 0.996000 | 0.114000 | 0.004000 |
| GenMol De Novo SGRPO Thr 2000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.683483 | 0.825488 | 0.657966 | 0.758479 | 0.859090 | 0.999000 | 0.129000 | 0.001000 |
| GenMol De Novo SGRPO Thr 2000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.619058 | 0.780988 | 0.615733 | 0.714886 | 0.874383 | 0.996000 | 0.170000 | 0.004000 |
| GenMol De Novo SGRPO Thr 2000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.539139 | 0.736262 | 0.575294 | 0.672599 | 0.890737 | 0.986000 | 0.227000 | 0.014000 |
| GenMol De Novo SGRPO RewardSum 2000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.871035 | 0.912032 | 0.838651 | 0.882680 | 0.604055 | 1.000000 | 0.018000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 2000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.798483 | 0.876381 | 0.771816 | 0.834555 | 0.783465 | 1.000000 | 0.058000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 2000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.734075 | 0.846152 | 0.710181 | 0.791763 | 0.836415 | 0.999000 | 0.094000 | 0.001000 |
| GenMol De Novo SGRPO RewardSum 2000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.688178 | 0.820638 | 0.658064 | 0.755608 | 0.862347 | 1.000000 | 0.123000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 2000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.597472 | 0.779820 | 0.609953 | 0.711873 | 0.874836 | 0.991000 | 0.193000 | 0.009000 |
| GenMol De Novo SGRPO RewardSum 2000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.538212 | 0.732241 | 0.562790 | 0.664699 | 0.893667 | 0.990000 | 0.229000 | 0.010000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.878291 | 0.918979 | 0.844952 | 0.889369 | 0.553071 | 1.000000 | 0.017000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.800907 | 0.877751 | 0.773323 | 0.835980 | 0.785662 | 1.000000 | 0.057000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.713920 | 0.849899 | 0.703810 | 0.791464 | 0.836986 | 0.999000 | 0.129000 | 0.001000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.679725 | 0.826390 | 0.654730 | 0.757726 | 0.861618 | 0.997000 | 0.131000 | 0.003000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.597758 | 0.779419 | 0.604594 | 0.709489 | 0.876879 | 0.996000 | 0.204000 | 0.004000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.533583 | 0.734003 | 0.562760 | 0.666227 | 0.893102 | 0.985000 | 0.224000 | 0.015000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.871894 | 0.904962 | 0.846973 | 0.881766 | 0.666988 | 1.000000 | 0.015000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.791392 | 0.870680 | 0.761532 | 0.827021 | 0.796792 | 1.000000 | 0.059000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.718649 | 0.844494 | 0.700869 | 0.787044 | 0.838830 | 0.998000 | 0.112000 | 0.002000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.668223 | 0.820180 | 0.646174 | 0.750577 | 0.860676 | 0.999000 | 0.145000 | 0.001000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.597650 | 0.771268 | 0.603038 | 0.704237 | 0.877639 | 0.993000 | 0.185000 | 0.007000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.518095 | 0.727963 | 0.559297 | 0.661209 | 0.895055 | 0.982000 | 0.235000 | 0.018000 |
| GenMol De Novo SGRPO RewardSum LOO 2000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.880433 | 0.917115 | 0.838577 | 0.885700 | 0.619742 | 1.000000 | 0.008000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO 2000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.788807 | 0.865114 | 0.781052 | 0.831489 | 0.809722 | 1.000000 | 0.069000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO 2000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.739285 | 0.843860 | 0.723917 | 0.795883 | 0.842084 | 1.000000 | 0.096000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO 2000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.688064 | 0.819230 | 0.664461 | 0.757323 | 0.861702 | 1.000000 | 0.126000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO 2000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.600240 | 0.779126 | 0.615417 | 0.713909 | 0.877057 | 0.989000 | 0.184000 | 0.011000 |
| GenMol De Novo SGRPO RewardSum LOO 2000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.554824 | 0.739315 | 0.586627 | 0.678483 | 0.889994 | 0.986000 | 0.201000 | 0.014000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 2000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.887663 | 0.924135 | 0.844727 | 0.892372 | 0.640542 | 1.000000 | 0.007000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 2000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.827638 | 0.887503 | 0.789619 | 0.848350 | 0.776282 | 1.000000 | 0.033000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 2000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.750415 | 0.854214 | 0.732506 | 0.805531 | 0.831725 | 1.000000 | 0.094000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 2000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.689279 | 0.829211 | 0.679743 | 0.769424 | 0.856451 | 1.000000 | 0.140000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 2000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.630712 | 0.787272 | 0.635116 | 0.726410 | 0.871356 | 0.994000 | 0.163000 | 0.006000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 2000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.553294 | 0.741457 | 0.592298 | 0.682526 | 0.889231 | 0.988000 | 0.220000 | 0.012000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 2000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.866618 | 0.899566 | 0.842137 | 0.876594 | 0.649974 | 1.000000 | 0.015000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 2000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.791852 | 0.849471 | 0.796462 | 0.828267 | 0.807395 | 1.000000 | 0.059000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 2000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.721964 | 0.829770 | 0.732498 | 0.790861 | 0.843553 | 0.999000 | 0.114000 | 0.001000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 2000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.676384 | 0.816592 | 0.673346 | 0.759294 | 0.860619 | 0.998000 | 0.143000 | 0.002000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 2000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.623299 | 0.773759 | 0.618550 | 0.711675 | 0.876741 | 0.995000 | 0.154000 | 0.005000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 2000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.550243 | 0.725352 | 0.584750 | 0.669345 | 0.891861 | 0.989000 | 0.214000 | 0.011000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 2000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.866422 | 0.901165 | 0.840339 | 0.876834 | 0.648996 | 1.000000 | 0.016000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 2000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.780213 | 0.850330 | 0.789951 | 0.826178 | 0.813877 | 1.000000 | 0.073000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 2000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.719144 | 0.828701 | 0.726822 | 0.787949 | 0.845494 | 1.000000 | 0.117000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 2000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.675783 | 0.814015 | 0.668808 | 0.755932 | 0.863703 | 0.997000 | 0.135000 | 0.003000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 2000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.617127 | 0.773541 | 0.612496 | 0.709123 | 0.877660 | 0.997000 | 0.167000 | 0.003000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 2000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.548964 | 0.727541 | 0.582979 | 0.670423 | 0.892604 | 0.989000 | 0.216000 | 0.011000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 2000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.885689 | 0.911180 | 0.836860 | 0.896316 | 0.626937 | 1.000000 | 0.016000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 2000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.839751 | 0.889807 | 0.775908 | 0.867027 | 0.768189 | 1.000000 | 0.041000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 2000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.761359 | 0.855087 | 0.713414 | 0.826752 | 0.829111 | 1.000000 | 0.105000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 2000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.711245 | 0.823634 | 0.662389 | 0.791773 | 0.854800 | 0.999000 | 0.135000 | 0.001000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 2000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.649723 | 0.782419 | 0.618063 | 0.749905 | 0.871087 | 0.994000 | 0.164000 | 0.006000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 2000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.608783 | 0.753325 | 0.579758 | 0.718612 | 0.886216 | 0.996000 | 0.198000 | 0.004000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.7 2000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.864753 | 0.896804 | 0.832704 | 0.871164 | 0.699451 | 1.000000 | 0.010000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.7 2000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.811222 | 0.878394 | 0.773514 | 0.836442 | 0.792014 | 1.000000 | 0.040000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.7 2000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.748588 | 0.847691 | 0.728148 | 0.799874 | 0.836801 | 0.999000 | 0.083000 | 0.001000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.7 2000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.692916 | 0.826562 | 0.675360 | 0.766081 | 0.857736 | 0.998000 | 0.124000 | 0.002000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.7 2000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.618624 | 0.783493 | 0.632230 | 0.722988 | 0.876205 | 0.993000 | 0.174000 | 0.007000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.7 2000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.547864 | 0.737957 | 0.579980 | 0.675008 | 0.891195 | 0.989000 | 0.223000 | 0.011000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.3 2000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.869279 | 0.915797 | 0.830997 | 0.881877 | 0.656074 | 1.000000 | 0.019000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.3 2000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.817364 | 0.887269 | 0.788680 | 0.847833 | 0.774325 | 1.000000 | 0.048000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.3 2000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.748875 | 0.854297 | 0.737413 | 0.807543 | 0.828012 | 0.999000 | 0.095000 | 0.001000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.3 2000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.707466 | 0.822250 | 0.686110 | 0.767794 | 0.853985 | 1.000000 | 0.108000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.3 2000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.633332 | 0.790450 | 0.641024 | 0.730680 | 0.871400 | 0.997000 | 0.173000 | 0.003000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.3 2000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.563447 | 0.755339 | 0.591005 | 0.689605 | 0.884283 | 0.989000 | 0.213000 | 0.011000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.1 2000 | randomness_temperature_pair | 1.000000 | 0.500000 | 0.100000 | 0.888563 | 0.924499 | 0.845564 | 0.892925 | 0.525607 | 1.000000 | 0.007000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.1 2000 | randomness_temperature_pair | 2.000000 | 0.800000 | 0.300000 | 0.827443 | 0.894487 | 0.796277 | 0.855203 | 0.749917 | 1.000000 | 0.044000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.1 2000 | randomness_temperature_pair | 3.000000 | 1.100000 | 0.500000 | 0.764425 | 0.863812 | 0.738644 | 0.813745 | 0.818011 | 1.000000 | 0.083000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.1 2000 | randomness_temperature_pair | 4.000000 | 1.400000 | 0.700000 | 0.698358 | 0.830633 | 0.693368 | 0.775727 | 0.850807 | 0.999000 | 0.132000 | 0.001000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.1 2000 | randomness_temperature_pair | 5.000000 | 1.700000 | 0.900000 | 0.642482 | 0.799644 | 0.639179 | 0.735458 | 0.867146 | 0.997000 | 0.162000 | 0.003000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.1 2000 | randomness_temperature_pair | 6.000000 | 2.000000 | 1.000000 | 0.582066 | 0.759905 | 0.592135 | 0.693051 | 0.883919 | 0.993000 | 0.196000 | 0.007000 |
