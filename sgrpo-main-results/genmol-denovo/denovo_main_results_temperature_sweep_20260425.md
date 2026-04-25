# De Novo Evaluation

- `num_samples`: 1000
- `generation_batch_size`: 2048
- `min_add_len`: 60
- `max_completion_length`: None
- `sweep_axis`: generation_temperature
- `generation_temperature`: 1.0
- `randomness`: 0.3
- `sweep_values`: 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0

- `QED vs Diversity plot`: `/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/genmol-denovo/qed_vs_diversity_temperature_20260425.png`
- `SA Score vs Diversity plot`: `/public/home/xinwuye/ai4s-tool-joint-train/genmol/sgrpo-main-results/genmol-denovo/sa_score_vs_diversity_temperature_20260425.png`

| Model | Sweep Axis | Sweep Value | Generation Temperature | Randomness | Overall De Novo Score | QED | SA Score | Soft Quality Score | Internal Diversity | Valid Molecule Rate | Alert Hit Rate | Invalid Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original GenMol v2 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.693429 | 0.867821 | 0.713369 | 0.806040 | 0.801846 | 1.000000 | 0.189000 | 0.000000 |
| Original GenMol v2 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.673101 | 0.843882 | 0.711123 | 0.790779 | 0.829922 | 1.000000 | 0.198000 | 0.000000 |
| Original GenMol v2 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.582950 | 0.791163 | 0.621924 | 0.724562 | 0.866391 | 0.990000 | 0.236000 | 0.010000 |
| Original GenMol v2 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.436720 | 0.681555 | 0.561753 | 0.635521 | 0.902412 | 0.979000 | 0.355000 | 0.021000 |
| Original GenMol v2 | generation_temperature | 5.000000 | 5.000000 | 0.300000 | -0.091636 | 0.387916 | 0.341465 | 0.380881 | 0.980245 | 0.693000 | 0.189000 | 0.307000 |
| Original GenMol v2 | generation_temperature | 8.000000 | 8.000000 | 0.300000 | -0.693616 | 0.341978 | 0.054756 | 0.231244 | 0.999169 | 0.266000 | 0.114000 | 0.734000 |
| Original GenMol v2 | generation_temperature | 10.000000 | 10.000000 | 0.300000 | -0.750320 | 0.342174 | 0.020430 | 0.219162 | 0.999785 | 0.222000 | 0.116000 | 0.778000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.801563 | 0.883724 | 0.776823 | 0.840963 | 0.769721 | 1.000000 | 0.065000 | 0.000000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.744680 | 0.864880 | 0.742290 | 0.815844 | 0.805996 | 1.000000 | 0.117000 | 0.000000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.622164 | 0.806822 | 0.647610 | 0.743138 | 0.857103 | 0.997000 | 0.212000 | 0.003000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.435343 | 0.692070 | 0.574826 | 0.648458 | 0.899899 | 0.966000 | 0.328000 | 0.034000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 5.000000 | 5.000000 | 0.300000 | -0.101264 | 0.386204 | 0.353584 | 0.384637 | 0.980368 | 0.687000 | 0.204000 | 0.313000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 8.000000 | 8.000000 | 0.300000 | -0.693696 | 0.342145 | 0.049868 | 0.229656 | 0.999256 | 0.267000 | 0.119000 | 0.733000 |
| GenMol De Novo GRPO 1000 | generation_temperature | 10.000000 | 10.000000 | 0.300000 | -0.753410 | 0.342376 | 0.023495 | 0.221007 | 0.999810 | 0.220000 | 0.123000 | 0.780000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.784886 | 0.871231 | 0.761872 | 0.827487 | 0.789698 | 1.000000 | 0.071000 | 0.000000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.716995 | 0.849317 | 0.730097 | 0.801629 | 0.823235 | 0.999000 | 0.140000 | 0.001000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.605159 | 0.794987 | 0.643894 | 0.735101 | 0.863735 | 0.991000 | 0.213000 | 0.009000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.410715 | 0.679953 | 0.562122 | 0.636020 | 0.904021 | 0.958000 | 0.341000 | 0.042000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 5.000000 | 5.000000 | 0.300000 | -0.137416 | 0.383561 | 0.323978 | 0.371458 | 0.982837 | 0.666000 | 0.188000 | 0.334000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 8.000000 | 8.000000 | 0.300000 | -0.722678 | 0.340876 | 0.051850 | 0.227962 | 0.999389 | 0.241000 | 0.104000 | 0.759000 |
| GenMol De Novo SGRPO 1000 | generation_temperature | 10.000000 | 10.000000 | 0.300000 | -0.730430 | 0.342981 | 0.022121 | 0.221353 | 0.999734 | 0.239000 | 0.124000 | 0.761000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.829865 | 0.900434 | 0.775730 | 0.850553 | 0.724616 | 0.999000 | 0.031000 | 0.001000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.801329 | 0.886614 | 0.759618 | 0.835815 | 0.761776 | 1.000000 | 0.057000 | 0.000000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.665596 | 0.823582 | 0.683086 | 0.767675 | 0.845677 | 0.998000 | 0.173000 | 0.002000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.461281 | 0.699674 | 0.607481 | 0.666589 | 0.897561 | 0.969000 | 0.319000 | 0.031000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 5.000000 | 5.000000 | 0.300000 | -0.057946 | 0.389138 | 0.348042 | 0.382629 | 0.979339 | 0.724000 | 0.221000 | 0.276000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 8.000000 | 8.000000 | 0.300000 | -0.704136 | 0.344935 | 0.051806 | 0.237189 | 0.999381 | 0.255000 | 0.106000 | 0.745000 |
| GenMol De Novo GRPO 2000 | generation_temperature | 10.000000 | 10.000000 | 0.300000 | -0.737867 | 0.339722 | 0.027860 | 0.215564 | 0.999743 | 0.231000 | 0.108000 | 0.769000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.823839 | 0.885329 | 0.773563 | 0.840623 | 0.769146 | 1.000000 | 0.027000 | 0.000000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.778939 | 0.870511 | 0.738037 | 0.817522 | 0.798605 | 1.000000 | 0.066000 | 0.000000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.633072 | 0.806556 | 0.639199 | 0.740177 | 0.862286 | 0.995000 | 0.182000 | 0.005000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.425077 | 0.688899 | 0.560257 | 0.639827 | 0.902211 | 0.967000 | 0.340000 | 0.033000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 5.000000 | 5.000000 | 0.300000 | -0.116876 | 0.384333 | 0.352204 | 0.383166 | 0.980057 | 0.678000 | 0.199000 | 0.322000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 8.000000 | 8.000000 | 0.300000 | -0.714775 | 0.343631 | 0.053410 | 0.235251 | 0.999408 | 0.245000 | 0.092000 | 0.755000 |
| GenMol De Novo SGRPO 2000 | generation_temperature | 10.000000 | 10.000000 | 0.300000 | -0.756204 | 0.341188 | 0.028862 | 0.220026 | 0.999746 | 0.216000 | 0.110000 | 0.784000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.848131 | 0.896093 | 0.810225 | 0.861746 | 0.713734 | 1.000000 | 0.022000 | 0.000000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.816059 | 0.880985 | 0.790981 | 0.844983 | 0.751374 | 1.000000 | 0.048000 | 0.000000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.666174 | 0.818165 | 0.702103 | 0.772029 | 0.843195 | 0.995000 | 0.173000 | 0.005000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.456581 | 0.706839 | 0.609753 | 0.670963 | 0.891618 | 0.969000 | 0.332000 | 0.031000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 5.000000 | 5.000000 | 0.300000 | -0.046531 | 0.388949 | 0.378543 | 0.394501 | 0.978491 | 0.725000 | 0.217000 | 0.275000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 8.000000 | 8.000000 | 0.300000 | -0.701685 | 0.340893 | 0.056122 | 0.229448 | 0.999277 | 0.259000 | 0.113000 | 0.741000 |
| GenMol De Novo GRPO DivReg0.05 2000 | generation_temperature | 10.000000 | 10.000000 | 0.300000 | -0.739029 | 0.341537 | 0.024458 | 0.218851 | 0.999761 | 0.230000 | 0.108000 | 0.770000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.775241 | 0.876640 | 0.758059 | 0.829208 | 0.782210 | 1.000000 | 0.091000 | 0.000000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.716405 | 0.849877 | 0.731992 | 0.802723 | 0.820869 | 1.000000 | 0.144000 | 0.000000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.601756 | 0.798559 | 0.636727 | 0.734103 | 0.862705 | 0.995000 | 0.230000 | 0.005000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.420741 | 0.681705 | 0.560247 | 0.635468 | 0.902695 | 0.963000 | 0.337000 | 0.037000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 5.000000 | 5.000000 | 0.300000 | -0.124850 | 0.386654 | 0.335778 | 0.378418 | 0.982657 | 0.676000 | 0.209000 | 0.324000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 8.000000 | 8.000000 | 0.300000 | -0.716300 | 0.341938 | 0.038300 | 0.225033 | 0.999291 | 0.246000 | 0.098000 | 0.754000 |
| GenMol De Novo SGRPO Thr 1000 | generation_temperature | 10.000000 | 10.000000 | 0.300000 | -0.738935 | 0.342290 | 0.019924 | 0.218962 | 0.999737 | 0.232000 | 0.120000 | 0.768000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.788285 | 0.869712 | 0.760797 | 0.826146 | 0.791573 | 1.000000 | 0.064000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.731536 | 0.848269 | 0.735823 | 0.803291 | 0.824581 | 1.000000 | 0.121000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.610374 | 0.798173 | 0.645629 | 0.737710 | 0.862472 | 0.993000 | 0.212000 | 0.007000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.420386 | 0.681827 | 0.557300 | 0.634147 | 0.901791 | 0.964000 | 0.343000 | 0.036000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 5.000000 | 5.000000 | 0.300000 | -0.093501 | 0.387700 | 0.321606 | 0.372161 | 0.980725 | 0.699000 | 0.199000 | 0.301000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 8.000000 | 8.000000 | 0.300000 | -0.710300 | 0.342063 | 0.051292 | 0.230392 | 0.999291 | 0.251000 | 0.106000 | 0.749000 |
| GenMol De Novo SGRPO RewardSum 1000 | generation_temperature | 10.000000 | 10.000000 | 0.300000 | -0.738543 | 0.340752 | 0.016369 | 0.213610 | 0.999741 | 0.232000 | 0.116000 | 0.768000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.772862 | 0.873176 | 0.760706 | 0.828188 | 0.782634 | 1.000000 | 0.092000 | 0.000000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.728024 | 0.852735 | 0.734503 | 0.805443 | 0.820162 | 0.999000 | 0.126000 | 0.001000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.601899 | 0.795746 | 0.634445 | 0.731776 | 0.863203 | 0.994000 | 0.225000 | 0.006000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.428845 | 0.681969 | 0.559486 | 0.635301 | 0.903004 | 0.972000 | 0.348000 | 0.028000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 5.000000 | 5.000000 | 0.300000 | -0.126902 | 0.385995 | 0.334756 | 0.377868 | 0.981573 | 0.671000 | 0.192000 | 0.329000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 8.000000 | 8.000000 | 0.300000 | -0.710416 | 0.341844 | 0.054500 | 0.231182 | 0.999299 | 0.250000 | 0.099000 | 0.750000 |
| GenMol De Novo SGRPO Thr+RewardSum 1000 | generation_temperature | 10.000000 | 10.000000 | 0.300000 | -0.750607 | 0.342183 | 0.018688 | 0.218520 | 0.999798 | 0.222000 | 0.117000 | 0.778000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.832398 | 0.885754 | 0.793703 | 0.848933 | 0.758690 | 1.000000 | 0.027000 | 0.000000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.780772 | 0.872703 | 0.760037 | 0.827637 | 0.797460 | 1.000000 | 0.075000 | 0.000000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.634657 | 0.802341 | 0.666007 | 0.748368 | 0.860423 | 0.992000 | 0.179000 | 0.008000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.426980 | 0.688217 | 0.577946 | 0.647145 | 0.900925 | 0.965000 | 0.345000 | 0.035000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 5.000000 | 5.000000 | 0.300000 | -0.132756 | 0.383984 | 0.335729 | 0.376729 | 0.982352 | 0.668000 | 0.191000 | 0.332000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 8.000000 | 8.000000 | 0.300000 | -0.717024 | 0.343200 | 0.070556 | 0.241042 | 0.999315 | 0.243000 | 0.100000 | 0.757000 |
| GenMol De Novo SGRPO Thr 2000 | generation_temperature | 10.000000 | 10.000000 | 0.300000 | -0.725217 | 0.341055 | 0.026420 | 0.218203 | 0.999717 | 0.242000 | 0.115000 | 0.758000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.830787 | 0.887273 | 0.789172 | 0.848033 | 0.757807 | 1.000000 | 0.028000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.775629 | 0.868284 | 0.753113 | 0.822215 | 0.802744 | 1.000000 | 0.077000 | 0.000000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.642785 | 0.807218 | 0.660159 | 0.748958 | 0.860802 | 0.996000 | 0.180000 | 0.004000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.425805 | 0.684087 | 0.561990 | 0.639516 | 0.903811 | 0.969000 | 0.353000 | 0.031000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 5.000000 | 5.000000 | 0.300000 | -0.159955 | 0.380374 | 0.318286 | 0.367466 | 0.984518 | 0.653000 | 0.203000 | 0.347000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 8.000000 | 8.000000 | 0.300000 | -0.688346 | 0.344869 | 0.082111 | 0.248707 | 0.999200 | 0.266000 | 0.105000 | 0.734000 |
| GenMol De Novo SGRPO RewardSum 2000 | generation_temperature | 10.000000 | 10.000000 | 0.300000 | -0.743455 | 0.341770 | 0.030879 | 0.222161 | 0.999738 | 0.226000 | 0.109000 | 0.774000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.825874 | 0.888409 | 0.790786 | 0.849360 | 0.765010 | 0.999000 | 0.035000 | 0.001000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.764663 | 0.866353 | 0.751772 | 0.820520 | 0.808639 | 0.999000 | 0.088000 | 0.001000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.633882 | 0.807554 | 0.655511 | 0.746737 | 0.861626 | 0.995000 | 0.188000 | 0.005000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.436725 | 0.686017 | 0.567403 | 0.641788 | 0.902991 | 0.970000 | 0.334000 | 0.030000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 5.000000 | 5.000000 | 0.300000 | -0.126720 | 0.383304 | 0.328615 | 0.372862 | 0.982324 | 0.677000 | 0.206000 | 0.323000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 8.000000 | 8.000000 | 0.300000 | -0.686266 | 0.344955 | 0.050162 | 0.235766 | 0.999323 | 0.273000 | 0.127000 | 0.727000 |
| GenMol De Novo SGRPO Thr+RewardSum 2000 | generation_temperature | 10.000000 | 10.000000 | 0.300000 | -0.748574 | 0.343015 | 0.029524 | 0.225034 | 0.999738 | 0.222000 | 0.112000 | 0.778000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.773570 | 0.870074 | 0.757241 | 0.824940 | 0.792090 | 1.000000 | 0.085000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.719713 | 0.851960 | 0.731678 | 0.803847 | 0.821835 | 1.000000 | 0.141000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.602041 | 0.795718 | 0.639985 | 0.733976 | 0.862452 | 0.993000 | 0.227000 | 0.007000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.415547 | 0.681616 | 0.561697 | 0.635988 | 0.903482 | 0.965000 | 0.353000 | 0.035000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 5.000000 | 5.000000 | 0.300000 | -0.093589 | 0.388533 | 0.330974 | 0.376907 | 0.981301 | 0.698000 | 0.205000 | 0.302000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 8.000000 | 8.000000 | 0.300000 | -0.714372 | 0.342644 | 0.055358 | 0.233565 | 0.999232 | 0.246000 | 0.097000 | 0.754000 |
| GenMol De Novo SGRPO HierarchicalSum 1000 | generation_temperature | 10.000000 | 10.000000 | 0.300000 | -0.738515 | 0.343194 | 0.024156 | 0.223032 | 0.999756 | 0.231000 | 0.115000 | 0.769000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 0.500000 | 0.500000 | 0.300000 | 0.819746 | 0.879590 | 0.779287 | 0.839469 | 0.781882 | 0.999000 | 0.029000 | 0.001000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 1.000000 | 1.000000 | 0.300000 | 0.770835 | 0.864247 | 0.745594 | 0.816786 | 0.810833 | 1.000000 | 0.076000 | 0.000000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 2.000000 | 2.000000 | 0.300000 | 0.606153 | 0.792332 | 0.633598 | 0.729111 | 0.864659 | 0.994000 | 0.210000 | 0.006000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 3.000000 | 3.000000 | 0.300000 | 0.423772 | 0.673898 | 0.567309 | 0.633541 | 0.905147 | 0.969000 | 0.342000 | 0.031000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 5.000000 | 5.000000 | 0.300000 | -0.106626 | 0.382410 | 0.329867 | 0.372253 | 0.981932 | 0.684000 | 0.175000 | 0.316000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 8.000000 | 8.000000 | 0.300000 | -0.691133 | 0.344037 | 0.080047 | 0.246041 | 0.999151 | 0.263000 | 0.099000 | 0.737000 |
| GenMol De Novo SGRPO HierarchicalSum 2000 | generation_temperature | 10.000000 | 10.000000 | 0.300000 | -0.743943 | 0.341060 | 0.018315 | 0.215254 | 0.999778 | 0.226000 | 0.105000 | 0.774000 |

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
