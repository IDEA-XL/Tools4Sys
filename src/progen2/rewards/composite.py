import torch

from progen2.rewards.developability import ProteinSolScorer, developability_reward
from progen2.rewards.foldability import ESMFoldFoldabilityScorer
from progen2.rewards.naturalness import ESM2NaturalnessScorer
from progen2.rewards.stability import TemBERTureTmScorer


def _quantiles(values):
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.numel() == 0:
        raise ValueError('quantile calibration requires at least one value')
    return (
        torch.quantile(tensor, 0.10).item(),
        torch.quantile(tensor, 0.90).item(),
    )


def _scale_quantile(raw_values, q10, q90):
    denom = (q90 - q10) + 1e-8
    outputs = []
    for value in raw_values:
        normalized = (float(value) - q10) / denom
        outputs.append(max(0.0, min(1.0, normalized)))
    return outputs


class CompositeProteinReward:
    def __init__(self, config, device='cpu'):
        self.device = torch.device(device)
        self.naturalness = ESM2NaturalnessScorer(
            model_name=str(config['naturalness']['model_name']),
            device=self.device,
        )
        self.foldability = ESMFoldFoldabilityScorer(device=self.device)
        self.stability = TemBERTureTmScorer(
            model_name_or_path=str(config['stability']['model_name_or_path']),
            tokenizer_name_or_path=config['stability'].get('tokenizer_name_or_path'),
            device=self.device,
        )
        self.developability = ProteinSolScorer(
            model_name_or_path=str(config['developability']['model_name_or_path']),
            tokenizer_name_or_path=config['developability'].get('tokenizer_name_or_path'),
            device=self.device,
        )
        self.calibration = None

    def calibrate(self, sequences):
        nat_raw = self.naturalness.score_raw(sequences)
        stab_raw = self.stability.score_raw(sequences)
        nat_q10, nat_q90 = _quantiles(nat_raw)
        stab_q10, stab_q90 = _quantiles(stab_raw)
        self.calibration = {
            'naturalness_q10': nat_q10,
            'naturalness_q90': nat_q90,
            'stability_q10': stab_q10,
            'stability_q90': stab_q90,
        }
        return dict(self.calibration)

    def score(self, sequences):
        if self.calibration is None:
            raise RuntimeError('CompositeProteinReward.calibrate must be called before score')
        nat_raw = self.naturalness.score_raw(sequences)
        fold = self.foldability.score_raw(sequences)
        stab_raw = self.stability.score_raw(sequences)
        dev_raw = self.developability.score_raw(sequences)

        nat = _scale_quantile(
            nat_raw,
            self.calibration['naturalness_q10'],
            self.calibration['naturalness_q90'],
        )
        stab = _scale_quantile(
            stab_raw,
            self.calibration['stability_q10'],
            self.calibration['stability_q90'],
        )
        dev = developability_reward(dev_raw, sequences)

        total = []
        for idx in range(len(sequences)):
            total.append(
                0.25 * nat[idx]
                + 0.30 * fold[idx]
                + 0.20 * stab[idx]
                + 0.25 * dev[idx]
            )

        metrics = {
            'reward_nat_mean': float(torch.tensor(nat, dtype=torch.float32).mean().item()),
            'reward_fold_mean': float(torch.tensor(fold, dtype=torch.float32).mean().item()),
            'reward_stab_mean': float(torch.tensor(stab, dtype=torch.float32).mean().item()),
            'reward_dev_mean': float(torch.tensor(dev, dtype=torch.float32).mean().item()),
            'reward_total_mean': float(torch.tensor(total, dtype=torch.float32).mean().item()),
        }
        return total, metrics
