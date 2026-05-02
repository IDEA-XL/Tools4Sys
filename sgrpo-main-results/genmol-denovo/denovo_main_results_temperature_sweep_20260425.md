# De Novo Evaluation

- `num_samples`: 1000
- `generation_batch_size`: 2048
- `min_add_len`: 60
- `max_completion_length`: None
- `sweep_axis`: generation_temperature
- `generation_temperature`: 1.0
- `randomness`: 0.3
- `sweep_values`: 0.5, 1.0, 2.0, 3.0

- `QED vs Diversity plot`: `/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/genmol-denovo/qed_vs_diversity_temperature_20260425.png`
- `SA Score vs Diversity plot`: `/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/genmol-denovo/sa_score_vs_diversity_temperature_20260425.png`
- `Soft Quality Score vs Diversity plot`: `/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/genmol-denovo/soft_reward_vs_diversity_temperature_20260425.png`

| Model | Sweep Axis | Sweep Value | Generation Temperature | Randomness | Overall De Novo Score | QED | SA Score | Soft Quality Score | Internal Diversity | Valid Molecule Rate | Alert Hit Rate | Invalid Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GenMol De Novo GRPO 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.801563 | 0.883724 | 0.776823 | 0.840963 | 0.769721 | 1.000000 | 0.065000 | 0.000000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.744680 | 0.864880 | 0.742290 | 0.815844 | 0.805996 | 1.000000 | 0.117000 | 0.000000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.622164 | 0.806822 | 0.647610 | 0.743138 | 0.857103 | 0.997000 | 0.212000 | 0.003000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.435343 | 0.692070 | 0.574826 | 0.648458 | 0.899899 | 0.966000 | 0.328000 | 0.034000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.829865 | 0.900434 | 0.775730 | 0.850553 | 0.724616 | 0.999000 | 0.031000 | 0.001000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.801329 | 0.886614 | 0.759618 | 0.835815 | 0.761776 | 1.000000 | 0.057000 | 0.000000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.665596 | 0.823582 | 0.683086 | 0.767675 | 0.845677 | 0.998000 | 0.173000 | 0.002000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.461281 | 0.699674 | 0.607481 | 0.666589 | 0.897561 | 0.969000 | 0.319000 | 0.031000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.848131 | 0.896093 | 0.810225 | 0.861746 | 0.713734 | 1.000000 | 0.022000 | 0.000000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.816059 | 0.880985 | 0.790981 | 0.844983 | 0.751374 | 1.000000 | 0.048000 | 0.000000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.666174 | 0.818165 | 0.702103 | 0.772029 | 0.843195 | 0.995000 | 0.173000 | 0.005000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.456581 | 0.706839 | 0.609753 | 0.670963 | 0.891618 | 0.969000 | 0.332000 | 0.031000 |
| GenMol De Novo GRPO Q0.8 SA0.2 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.826814 | 0.888193 | 0.775984 | 0.865751 | 0.762120 | 1.000000 | 0.059000 | 0.000000 |
| GenMol De Novo GRPO Q0.8 SA0.2 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.770244 | 0.862870 | 0.739377 | 0.838171 | 0.804596 | 1.000000 | 0.107000 | 0.000000 |
| GenMol De Novo GRPO Q0.8 SA0.2 1000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.647293 | 0.805354 | 0.649376 | 0.774158 | 0.858129 | 0.996000 | 0.207000 | 0.004000 |
| GenMol De Novo GRPO Q0.8 SA0.2 1000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.464248 | 0.689400 | 0.558569 | 0.666105 | 0.898934 | 0.975000 | 0.326000 | 0.025000 |
| GenMol De Novo GRPO Q0.8 SA0.2 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.840368 | 0.902137 | 0.761735 | 0.874056 | 0.730919 | 1.000000 | 0.053000 | 0.000000 |
| GenMol De Novo GRPO Q0.8 SA0.2 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.818368 | 0.885584 | 0.736844 | 0.855836 | 0.772311 | 1.000000 | 0.058000 | 0.000000 |
| GenMol De Novo GRPO Q0.8 SA0.2 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.680925 | 0.823014 | 0.657650 | 0.790330 | 0.848188 | 0.994000 | 0.169000 | 0.006000 |
| GenMol De Novo GRPO Q0.8 SA0.2 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.470086 | 0.699207 | 0.575141 | 0.679171 | 0.897222 | 0.964000 | 0.299000 | 0.036000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.784886 | 0.871231 | 0.761872 | 0.827487 | 0.789698 | 1.000000 | 0.071000 | 0.000000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.716995 | 0.849317 | 0.730097 | 0.801629 | 0.823235 | 0.999000 | 0.140000 | 0.001000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.605159 | 0.794987 | 0.643894 | 0.735101 | 0.863735 | 0.991000 | 0.213000 | 0.009000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.410715 | 0.679953 | 0.562122 | 0.636020 | 0.904021 | 0.958000 | 0.341000 | 0.042000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.823839 | 0.885329 | 0.773563 | 0.840623 | 0.769146 | 1.000000 | 0.027000 | 0.000000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.778939 | 0.870511 | 0.738037 | 0.817522 | 0.798605 | 1.000000 | 0.066000 | 0.000000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.633072 | 0.806556 | 0.639199 | 0.740177 | 0.862286 | 0.995000 | 0.182000 | 0.005000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.425077 | 0.688899 | 0.560257 | 0.639827 | 0.902211 | 0.967000 | 0.340000 | 0.033000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.796096 | 0.878087 | 0.771871 | 0.835601 | 0.781855 | 1.000000 | 0.067000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.748380 | 0.854211 | 0.743454 | 0.809908 | 0.815537 | 1.000000 | 0.102000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 1000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.616882 | 0.805593 | 0.653030 | 0.744850 | 0.858625 | 0.993000 | 0.211000 | 0.007000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 1000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.427488 | 0.681972 | 0.564363 | 0.636626 | 0.902019 | 0.969000 | 0.349000 | 0.031000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.854110 | 0.893580 | 0.813939 | 0.861724 | 0.757937 | 1.000000 | 0.012000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.793410 | 0.872545 | 0.771328 | 0.832058 | 0.794669 | 1.000000 | 0.063000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.661953 | 0.815252 | 0.695061 | 0.768039 | 0.855081 | 0.992000 | 0.164000 | 0.008000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.434604 | 0.684327 | 0.597051 | 0.652397 | 0.899679 | 0.972000 | 0.361000 | 0.028000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.825906 | 0.881732 | 0.769719 | 0.859329 | 0.777609 | 1.000000 | 0.053000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.757884 | 0.856346 | 0.736848 | 0.832447 | 0.818247 | 1.000000 | 0.118000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 1000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.644427 | 0.806839 | 0.645727 | 0.775370 | 0.861271 | 0.993000 | 0.206000 | 0.007000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 1000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.453208 | 0.685611 | 0.557864 | 0.662628 | 0.901126 | 0.971000 | 0.338000 | 0.029000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.859087 | 0.896011 | 0.788568 | 0.874523 | 0.749809 | 1.000000 | 0.024000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.819294 | 0.881087 | 0.752013 | 0.855273 | 0.786138 | 0.999000 | 0.052000 | 0.001000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.661060 | 0.820675 | 0.665530 | 0.790425 | 0.852681 | 0.989000 | 0.185000 | 0.011000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.5 Q0.8 SA0.2 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.468884 | 0.698461 | 0.577307 | 0.678100 | 0.899482 | 0.965000 | 0.305000 | 0.035000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.773570 | 0.870074 | 0.757241 | 0.824940 | 0.792090 | 1.000000 | 0.085000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.719713 | 0.851960 | 0.731678 | 0.803847 | 0.821835 | 1.000000 | 0.141000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.602041 | 0.795718 | 0.639985 | 0.733976 | 0.862452 | 0.993000 | 0.227000 | 0.007000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.415547 | 0.681616 | 0.561697 | 0.635988 | 0.903482 | 0.965000 | 0.353000 | 0.035000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.819746 | 0.879590 | 0.779287 | 0.839469 | 0.781882 | 0.999000 | 0.029000 | 0.001000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.770835 | 0.864247 | 0.745594 | 0.816786 | 0.810833 | 1.000000 | 0.076000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.606153 | 0.792332 | 0.633598 | 0.729111 | 0.864659 | 0.994000 | 0.210000 | 0.006000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.423772 | 0.673898 | 0.567309 | 0.633541 | 0.905147 | 0.969000 | 0.342000 | 0.031000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.788285 | 0.869712 | 0.760797 | 0.826146 | 0.791573 | 1.000000 | 0.064000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.731536 | 0.848269 | 0.735823 | 0.803291 | 0.824581 | 1.000000 | 0.121000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.610374 | 0.798173 | 0.645629 | 0.737710 | 0.862472 | 0.993000 | 0.212000 | 0.007000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.420386 | 0.681827 | 0.557300 | 0.634147 | 0.901791 | 0.964000 | 0.343000 | 0.036000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.830787 | 0.887273 | 0.789172 | 0.848033 | 0.757807 | 1.000000 | 0.028000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.775629 | 0.868284 | 0.753113 | 0.822215 | 0.802744 | 1.000000 | 0.077000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.642785 | 0.807218 | 0.660159 | 0.748958 | 0.860802 | 0.996000 | 0.180000 | 0.004000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.425805 | 0.684087 | 0.561990 | 0.639516 | 0.903811 | 0.969000 | 0.353000 | 0.031000 |
| GenMol De Novo SGRPO RewardSum LOO 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.777323 | 0.866549 | 0.760762 | 0.824234 | 0.799955 | 1.000000 | 0.079000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.703413 | 0.840329 | 0.728022 | 0.795406 | 0.830189 | 1.000000 | 0.157000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO 1000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.609453 | 0.794874 | 0.640866 | 0.733821 | 0.866714 | 0.994000 | 0.213000 | 0.006000 |
| GenMol De Novo SGRPO RewardSum LOO 1000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.437886 | 0.688971 | 0.561110 | 0.639991 | 0.901067 | 0.969000 | 0.331000 | 0.031000 |
| GenMol De Novo SGRPO RewardSum LOO 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.806898 | 0.870471 | 0.787110 | 0.837127 | 0.794699 | 1.000000 | 0.050000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.764331 | 0.849696 | 0.751817 | 0.810545 | 0.824339 | 1.000000 | 0.078000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.636593 | 0.802189 | 0.668016 | 0.748800 | 0.864495 | 0.992000 | 0.180000 | 0.008000 |
| GenMol De Novo SGRPO RewardSum LOO 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.462852 | 0.689128 | 0.567787 | 0.643810 | 0.905033 | 0.978000 | 0.316000 | 0.022000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.734094 | 0.861551 | 0.743680 | 0.814402 | 0.804484 | 1.000000 | 0.134000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.681352 | 0.840552 | 0.718775 | 0.791842 | 0.829367 | 0.999000 | 0.183000 | 0.001000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 1000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.590983 | 0.790518 | 0.629725 | 0.726472 | 0.865080 | 0.997000 | 0.248000 | 0.003000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 1000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.413494 | 0.674722 | 0.559067 | 0.630946 | 0.904379 | 0.971000 | 0.372000 | 0.029000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.808586 | 0.861946 | 0.803723 | 0.838657 | 0.794943 | 1.000000 | 0.048000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.738632 | 0.837401 | 0.757625 | 0.805491 | 0.827870 | 1.000000 | 0.112000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.621818 | 0.788608 | 0.666639 | 0.739821 | 0.865815 | 0.991000 | 0.188000 | 0.009000 |
| GenMol De Novo SGRPO RewardSum LOO+Temp/Rand 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.418954 | 0.669570 | 0.592203 | 0.642750 | 0.907049 | 0.960000 | 0.339000 | 0.040000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.738350 | 0.861163 | 0.751576 | 0.817329 | 0.803430 | 1.000000 | 0.132000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.683339 | 0.839407 | 0.726125 | 0.794094 | 0.829313 | 1.000000 | 0.186000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 1000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.590040 | 0.791613 | 0.631049 | 0.727661 | 0.864574 | 0.994000 | 0.241000 | 0.006000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 1000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.412284 | 0.676598 | 0.555407 | 0.631257 | 0.904598 | 0.968000 | 0.358000 | 0.032000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.812296 | 0.862374 | 0.809277 | 0.841135 | 0.788743 | 1.000000 | 0.046000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.745533 | 0.842941 | 0.773432 | 0.815137 | 0.821603 | 1.000000 | 0.113000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.632848 | 0.794695 | 0.670172 | 0.744885 | 0.864653 | 0.995000 | 0.189000 | 0.005000 |
| GenMol De Novo SGRPO RewardSum Temp/Rand 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.455347 | 0.687352 | 0.599427 | 0.654988 | 0.904060 | 0.967000 | 0.307000 | 0.033000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.775241 | 0.876640 | 0.758059 | 0.829208 | 0.782210 | 1.000000 | 0.091000 | 0.000000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.716405 | 0.849877 | 0.731992 | 0.802723 | 0.820869 | 1.000000 | 0.144000 | 0.000000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.601756 | 0.798559 | 0.636727 | 0.734103 | 0.862705 | 0.995000 | 0.230000 | 0.005000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.420741 | 0.681705 | 0.560247 | 0.635468 | 0.902695 | 0.963000 | 0.337000 | 0.037000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.832398 | 0.885754 | 0.793703 | 0.848933 | 0.758690 | 1.000000 | 0.027000 | 0.000000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.780772 | 0.872703 | 0.760037 | 0.827637 | 0.797460 | 1.000000 | 0.075000 | 0.000000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.634657 | 0.802341 | 0.666007 | 0.748368 | 0.860423 | 0.992000 | 0.179000 | 0.008000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.426980 | 0.688217 | 0.577946 | 0.647145 | 0.900925 | 0.965000 | 0.345000 | 0.035000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.772862 | 0.873176 | 0.760706 | 0.828188 | 0.782634 | 1.000000 | 0.092000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.728024 | 0.852735 | 0.734503 | 0.805443 | 0.820162 | 0.999000 | 0.126000 | 0.001000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.601899 | 0.795746 | 0.634445 | 0.731776 | 0.863203 | 0.994000 | 0.225000 | 0.006000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.428845 | 0.681969 | 0.559486 | 0.635301 | 0.903004 | 0.972000 | 0.348000 | 0.028000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.825874 | 0.888409 | 0.790786 | 0.849360 | 0.765010 | 0.999000 | 0.035000 | 0.001000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.764663 | 0.866353 | 0.751772 | 0.820520 | 0.808639 | 0.999000 | 0.088000 | 0.001000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.633882 | 0.807554 | 0.655511 | 0.746737 | 0.861626 | 0.995000 | 0.188000 | 0.005000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.436725 | 0.686017 | 0.567403 | 0.641788 | 0.902991 | 0.970000 | 0.334000 | 0.030000 |
| Original GenMol v2 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.693429 | 0.867821 | 0.713369 | 0.806040 | 0.801846 | 1.000000 | 0.189000 | 0.000000 |
| Original GenMol v2 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.673101 | 0.843882 | 0.711123 | 0.790779 | 0.829922 | 1.000000 | 0.198000 | 0.000000 |
| Original GenMol v2 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.582950 | 0.791163 | 0.621924 | 0.724562 | 0.866391 | 0.990000 | 0.236000 | 0.010000 |
| Original GenMol v2 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.436720 | 0.681555 | 0.561753 | 0.635521 | 0.902412 | 0.979000 | 0.355000 | 0.021000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.7 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.835796 | 0.883021 | 0.798198 | 0.849092 | 0.777233 | 1.000000 | 0.021000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.7 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.795235 | 0.874994 | 0.762594 | 0.830034 | 0.801480 | 1.000000 | 0.058000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.7 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.664097 | 0.811188 | 0.670606 | 0.755239 | 0.860764 | 0.998000 | 0.156000 | 0.002000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.7 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.435211 | 0.691891 | 0.584869 | 0.652588 | 0.901373 | 0.965000 | 0.333000 | 0.035000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.3 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.840049 | 0.895925 | 0.798398 | 0.856914 | 0.759864 | 1.000000 | 0.027000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.3 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.806786 | 0.882052 | 0.765014 | 0.835237 | 0.789105 | 1.000000 | 0.046000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.3 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.666306 | 0.822308 | 0.692031 | 0.770197 | 0.851837 | 0.995000 | 0.170000 | 0.005000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.3 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.463665 | 0.704657 | 0.601427 | 0.666750 | 0.895013 | 0.971000 | 0.318000 | 0.029000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.1 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.847126 | 0.901923 | 0.806995 | 0.863952 | 0.731283 | 0.998000 | 0.021000 | 0.002000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.1 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.802483 | 0.882440 | 0.783875 | 0.843014 | 0.777841 | 0.999000 | 0.062000 | 0.001000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.1 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.672348 | 0.823663 | 0.688608 | 0.769641 | 0.848171 | 0.997000 | 0.161000 | 0.003000 |
| GenMol De Novo SGRPO RewardSum LOO GW0.1 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.457504 | 0.703114 | 0.596416 | 0.662662 | 0.895319 | 0.980000 | 0.356000 | 0.020000 |
