import time

import torch

from progen2.rewards.common import synchronize_device
from progen2.rewards.developability import ProteinSolScorer, score_developability_components
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
        naturalness_cfg = dict(config['naturalness'])
        foldability_cfg = dict(config.get('foldability', {}))
        stability_cfg = dict(config['stability'])
        developability_cfg = dict(config['developability'])
        self.naturalness = ESM2NaturalnessScorer(
            model_name=str(naturalness_cfg['model_name']),
            device=naturalness_cfg.get('device', self.device),
            batch_size=naturalness_cfg.get('batch_size', 8),
        )
        self.foldability = ESMFoldFoldabilityScorer(
            device=foldability_cfg.get('device', self.device),
            batch_size=foldability_cfg.get('batch_size', 1),
            num_recycles=foldability_cfg.get('num_recycles'),
        )
        self.stability = TemBERTureTmScorer(
            model_name_or_path=str(stability_cfg['model_name_or_path']),
            tokenizer_name_or_path=stability_cfg.get('tokenizer_name_or_path'),
            device=stability_cfg.get('device', self.device),
            batch_size=stability_cfg.get('batch_size', 16),
            base_model_name_or_path=stability_cfg.get('base_model_name_or_path'),
        )
        self.developability = ProteinSolScorer(
            model_name_or_path=str(developability_cfg['model_name_or_path']),
            tokenizer_name_or_path=developability_cfg.get('tokenizer_name_or_path'),
            device=developability_cfg.get('device', self.device),
            batch_size=developability_cfg.get('batch_size', 16),
        )
        self.calibration = None

    def _timed_score_raw(self, scorer, sequences):
        if hasattr(scorer, 'device'):
            synchronize_device(scorer.device)
        start = time.perf_counter()
        values = scorer.score_raw(sequences)
        if hasattr(scorer, 'device'):
            synchronize_device(scorer.device)
        elapsed = time.perf_counter() - start
        transfer_in = float(getattr(scorer, 'last_move_to_device_sec', 0.0))
        return values, elapsed, transfer_in

    def _timed_release(self, scorer):
        if hasattr(scorer, 'device'):
            synchronize_device(scorer.device)
        start = time.perf_counter()
        scorer.release()
        if hasattr(scorer, 'device'):
            synchronize_device(scorer.device)
        elapsed = time.perf_counter() - start
        transfer_out = float(getattr(scorer, 'last_release_to_cpu_sec', 0.0))
        return elapsed, transfer_out

    def calibrate(self, sequences):
        nat_raw, _, _ = self._timed_score_raw(self.naturalness, sequences)
        stab_raw, _, _ = self._timed_score_raw(self.stability, sequences)
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
        details, metrics = self.score_details(sequences)
        return details['total'], metrics

    def score_details(self, sequences):
        if self.calibration is None:
            raise RuntimeError('CompositeProteinReward.calibrate must be called before score')
        nat_raw, nat_score_sec, nat_cpu_to_gpu_sec = self._timed_score_raw(self.naturalness, sequences)
        fold, fold_score_sec, fold_cpu_to_gpu_sec = self._timed_score_raw(self.foldability, sequences)
        stab_raw, stab_score_sec, stab_cpu_to_gpu_sec = self._timed_score_raw(self.stability, sequences)
        dev_raw, dev_score_sec, dev_cpu_to_gpu_sec = self._timed_score_raw(self.developability, sequences)
        nat_release_sec = 0.0
        nat_gpu_to_cpu_sec = 0.0
        fold_release_sec = 0.0
        fold_gpu_to_cpu_sec = 0.0
        stab_release_sec = 0.0
        stab_gpu_to_cpu_sec = 0.0
        dev_release_sec = 0.0
        dev_gpu_to_cpu_sec = 0.0

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
        developability_components = score_developability_components(dev_raw, sequences)
        dev = developability_components['developability']

        total = []
        for idx in range(len(sequences)):
            total.append(
                0.25 * nat[idx]
                + 0.30 * fold[idx]
                + 0.20 * stab[idx]
                + 0.25 * dev[idx]
            )

        details = {
            'naturalness_raw': nat_raw,
            'naturalness': nat,
            'foldability': fold,
            'stability_raw': stab_raw,
            'stability': stab,
            'solubility': developability_components['solubility'],
            'liability_reward': developability_components['liability_reward'],
            'developability': dev,
            'total': total,
        }
        metrics = {
            'reward_nat_mean': float(torch.tensor(nat, dtype=torch.float32).mean().item()),
            'reward_fold_mean': float(torch.tensor(fold, dtype=torch.float32).mean().item()),
            'reward_stab_mean': float(torch.tensor(stab, dtype=torch.float32).mean().item()),
            'reward_dev_mean': float(torch.tensor(dev, dtype=torch.float32).mean().item()),
            'reward_sol_mean': float(torch.tensor(developability_components['solubility'], dtype=torch.float32).mean().item()),
            'reward_liability_mean': float(torch.tensor(developability_components['liability_reward'], dtype=torch.float32).mean().item()),
            'reward_total_mean': float(torch.tensor(total, dtype=torch.float32).mean().item()),
            'reward_nat_score_sec': nat_score_sec,
            'reward_fold_score_sec': fold_score_sec,
            'reward_stab_score_sec': stab_score_sec,
            'reward_dev_score_sec': dev_score_sec,
            'reward_nat_cpu_to_gpu_sec': nat_cpu_to_gpu_sec,
            'reward_fold_cpu_to_gpu_sec': fold_cpu_to_gpu_sec,
            'reward_stab_cpu_to_gpu_sec': stab_cpu_to_gpu_sec,
            'reward_dev_cpu_to_gpu_sec': dev_cpu_to_gpu_sec,
            'reward_nat_gpu_to_cpu_sec': nat_gpu_to_cpu_sec,
            'reward_fold_gpu_to_cpu_sec': fold_gpu_to_cpu_sec,
            'reward_stab_gpu_to_cpu_sec': stab_gpu_to_cpu_sec,
            'reward_dev_gpu_to_cpu_sec': dev_gpu_to_cpu_sec,
            'reward_nat_release_sec': nat_release_sec,
            'reward_fold_release_sec': fold_release_sec,
            'reward_stab_release_sec': stab_release_sec,
            'reward_dev_release_sec': dev_release_sec,
            'reward_score_sec_total': nat_score_sec + fold_score_sec + stab_score_sec + dev_score_sec,
            'reward_cpu_to_gpu_sec_total': nat_cpu_to_gpu_sec + fold_cpu_to_gpu_sec + stab_cpu_to_gpu_sec + dev_cpu_to_gpu_sec,
            'reward_gpu_to_cpu_sec_total': nat_gpu_to_cpu_sec + fold_gpu_to_cpu_sec + stab_gpu_to_cpu_sec + dev_gpu_to_cpu_sec,
        }
        return details, metrics
