# De Novo SGRPO Evaluation

- `num_samples`: 1000
- `generation_batch_size`: 2048
- `generation_temperature`: 1.0
- `min_add_len`: 60
- `max_completion_length`: None
- `randomness_values`: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

- `QED vs Diversity plot`: `/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/genmol-denovo/qed_vs_diversity_20260422.png`
- `SA Score vs Diversity plot`: `/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/genmol-denovo/sa_score_vs_diversity_20260422.png`

| Model | Randomness | Overall De Novo Score | QED | SA Score | Soft Quality Score | Internal Diversity | Valid Molecule Rate | Alert Hit Rate | Invalid Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original GenMol v2 | 0.100000 | 0.684465 | 0.850988 | 0.744781 | 0.808506 | 0.804835 | 0.999000 | 0.203000 | 0.001000 |
| Original GenMol v2 | 0.200000 | 0.670966 | 0.847219 | 0.733580 | 0.801763 | 0.816675 | 1.000000 | 0.216000 | 0.000000 |
| Original GenMol v2 | 0.300000 | 0.673101 | 0.843882 | 0.711123 | 0.790779 | 0.829922 | 1.000000 | 0.198000 | 0.000000 |
| Original GenMol v2 | 0.400000 | 0.641282 | 0.838496 | 0.678128 | 0.774349 | 0.835523 | 1.000000 | 0.227000 | 0.000000 |
| Original GenMol v2 | 0.500000 | 0.643318 | 0.829871 | 0.664530 | 0.763735 | 0.843746 | 1.000000 | 0.211000 | 0.000000 |
| Original GenMol v2 | 0.600000 | 0.635634 | 0.830702 | 0.652402 | 0.759382 | 0.846483 | 1.000000 | 0.217000 | 0.000000 |
| Original GenMol v2 | 0.700000 | 0.644257 | 0.825774 | 0.642789 | 0.752580 | 0.851215 | 0.999000 | 0.193000 | 0.001000 |
| Original GenMol v2 | 0.800000 | 0.638358 | 0.820289 | 0.634388 | 0.745929 | 0.853885 | 0.999000 | 0.192000 | 0.001000 |
| Original GenMol v2 | 0.900000 | 0.643764 | 0.816843 | 0.630175 | 0.742176 | 0.855750 | 0.999000 | 0.179000 | 0.001000 |
| Original GenMol v2 | 1.000000 | 0.648919 | 0.817685 | 0.623763 | 0.740116 | 0.857612 | 0.999000 | 0.166000 | 0.001000 |
| GenMol De Novo GRPO | 0.100000 | 0.779583 | 0.871043 | 0.780255 | 0.834728 | 0.769068 | 1.000000 | 0.088000 | 0.000000 |
| GenMol De Novo GRPO | 0.200000 | 0.755757 | 0.864539 | 0.762653 | 0.823785 | 0.785869 | 1.000000 | 0.110000 | 0.000000 |
| GenMol De Novo GRPO | 0.300000 | 0.744680 | 0.864880 | 0.742290 | 0.815844 | 0.805996 | 1.000000 | 0.117000 | 0.000000 |
| GenMol De Novo GRPO | 0.400000 | 0.718589 | 0.859556 | 0.716929 | 0.802505 | 0.822052 | 0.999000 | 0.138000 | 0.001000 |
| GenMol De Novo GRPO | 0.500000 | 0.706752 | 0.857466 | 0.700263 | 0.794585 | 0.831392 | 1.000000 | 0.149000 | 0.000000 |
| GenMol De Novo GRPO | 0.600000 | 0.690373 | 0.840308 | 0.684667 | 0.778051 | 0.837626 | 0.999000 | 0.148000 | 0.001000 |
| GenMol De Novo GRPO | 0.700000 | 0.681858 | 0.840112 | 0.673903 | 0.773629 | 0.844668 | 0.999000 | 0.157000 | 0.001000 |
| GenMol De Novo GRPO | 0.800000 | 0.672502 | 0.836115 | 0.657865 | 0.764815 | 0.848148 | 1.000000 | 0.163000 | 0.000000 |
| GenMol De Novo GRPO | 0.900000 | 0.678824 | 0.828685 | 0.653863 | 0.758756 | 0.853731 | 1.000000 | 0.142000 | 0.000000 |
| GenMol De Novo GRPO | 1.000000 | 0.673519 | 0.828547 | 0.654242 | 0.758825 | 0.853478 | 0.999000 | 0.147000 | 0.001000 |
| GenMol De Novo SGRPO | 0.100000 | 0.754926 | 0.857052 | 0.771059 | 0.822655 | 0.792719 | 0.999000 | 0.108000 | 0.001000 |
| GenMol De Novo SGRPO | 0.200000 | 0.749539 | 0.851155 | 0.760623 | 0.814942 | 0.805529 | 1.000000 | 0.107000 | 0.000000 |
| GenMol De Novo SGRPO | 0.300000 | 0.716995 | 0.849317 | 0.730097 | 0.801629 | 0.823235 | 0.999000 | 0.140000 | 0.001000 |
| GenMol De Novo SGRPO | 0.400000 | 0.691953 | 0.841468 | 0.707303 | 0.787802 | 0.832206 | 1.000000 | 0.164000 | 0.000000 |
| GenMol De Novo SGRPO | 0.500000 | 0.695378 | 0.837793 | 0.699599 | 0.782515 | 0.840483 | 0.999000 | 0.146000 | 0.001000 |
| GenMol De Novo SGRPO | 0.600000 | 0.675772 | 0.834141 | 0.681161 | 0.772949 | 0.844221 | 0.999000 | 0.166000 | 0.001000 |
| GenMol De Novo SGRPO | 0.700000 | 0.666297 | 0.831825 | 0.660934 | 0.763468 | 0.849847 | 0.998000 | 0.168000 | 0.002000 |
| GenMol De Novo SGRPO | 0.800000 | 0.664660 | 0.822564 | 0.645087 | 0.751573 | 0.852852 | 0.999000 | 0.156000 | 0.001000 |
| GenMol De Novo SGRPO | 0.900000 | 0.660311 | 0.814707 | 0.647907 | 0.747987 | 0.855645 | 1.000000 | 0.161000 | 0.000000 |
| GenMol De Novo SGRPO | 1.000000 | 0.652298 | 0.821014 | 0.638816 | 0.748135 | 0.858445 | 0.997000 | 0.165000 | 0.003000 |

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
- Each row is one model evaluated at one randomness value.
- The line plots connect rows for the same model in increasing randomness order.
