import time

import torch

from progen2.rewards.common import synchronize_device
from progen2.rewards.developability import ProteinSolScorer, score_developability_components
from progen2.rewards.foldability import ESMFoldFoldabilityScorer
from progen2.rewards.naturalness import ESM2NaturalnessScorer
from progen2.rewards.stability import TemBERTureTmScorer

REWARD_NAME_ORDER = (
    'naturalness',
    'foldability',
    'stability',
    'developability',
)
DEFAULT_PROTEIN_REWARD_WEIGHTS = {
    'naturalness': 0.25,
    'foldability': 0.30,
    'stability': 0.20,
    'developability': 0.25,
}


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


def _resolve_reward_batch_size(config, default_batch_size, field_name):
    if config is None:
        raise ValueError(f'{field_name} config is required')
    if 'batch_size' in config and config['batch_size'] is not None:
        return config['batch_size']
    if default_batch_size is None:
        raise ValueError(
            f'{field_name} must be set explicitly when no default_reward_batch_size is provided'
        )
    return default_batch_size


def normalize_protein_reward_weights(config):
    if config is None:
        raw = dict(DEFAULT_PROTEIN_REWARD_WEIGHTS)
    else:
        raw = dict(DEFAULT_PROTEIN_REWARD_WEIGHTS)
        for reward_name in REWARD_NAME_ORDER:
            if reward_name in config and config[reward_name] is not None:
                raw[reward_name] = config[reward_name]
    normalized = {}
    for reward_name in REWARD_NAME_ORDER:
        value = raw[reward_name]
        try:
            weight = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'protein reward weight for {reward_name!r} must be numeric, got {value!r}') from exc
        if weight < 0.0:
            raise ValueError(f'protein reward weight for {reward_name!r} must be non-negative, got {weight}')
        normalized[reward_name] = weight
    return normalized


def normalize_reward_compute_every_n_steps(config):
    if config is None:
        return {name: 1 for name in REWARD_NAME_ORDER}
    if isinstance(config, (list, tuple)):
        if len(config) != len(REWARD_NAME_ORDER):
            raise ValueError(
                'reward_compute_every_n_steps list must have exactly '
                f'{len(REWARD_NAME_ORDER)} entries in order {REWARD_NAME_ORDER}, got {len(config)}'
            )
        raw = dict(zip(REWARD_NAME_ORDER, config))
    elif isinstance(config, dict):
        raw = dict(config)
        missing = [name for name in REWARD_NAME_ORDER if name not in raw]
        extra = sorted(name for name in raw if name not in REWARD_NAME_ORDER)
        if missing or extra:
            raise ValueError(
                'reward_compute_every_n_steps dict must have exactly these keys: '
                f'{REWARD_NAME_ORDER}; missing={missing}, extra={extra}'
            )
    else:
        raise ValueError(
            'reward_compute_every_n_steps must be null, a dict keyed by '
            f'{REWARD_NAME_ORDER}, or a list in that order; got {type(config).__name__}'
        )

    normalized = {}
    for name in REWARD_NAME_ORDER:
        value = raw[name]
        try:
            interval = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'reward_compute_every_n_steps[{name!r}] must be a positive integer, got {value!r}'
            ) from exc
        if interval <= 0:
            raise ValueError(
                f'reward_compute_every_n_steps[{name!r}] must be a positive integer, got {value!r}'
            )
        normalized[name] = interval
    return normalized


def _normalize_step_number(step_number):
    if step_number is None:
        return None
    try:
        normalized = int(step_number)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'step_number must be a positive integer or null, got {step_number!r}') from exc
    if normalized <= 0:
        raise ValueError(f'step_number must be a positive integer or null, got {step_number!r}')
    return normalized


class CompositeProteinReward:
    def __init__(
        self,
        config,
        device='cpu',
        default_reward_batch_size=None,
        reward_compute_every_n_steps=None,
        reward_weights=None,
        always_compute_metrics=False,
    ):
        self.device = torch.device(device)
        self.reward_weights = normalize_protein_reward_weights(reward_weights)
        self.always_compute_metrics = bool(always_compute_metrics)
        self.reward_compute_every_n_steps = normalize_reward_compute_every_n_steps(
            reward_compute_every_n_steps
        )
        self._compute_flags = {
            reward_name: self.always_compute_metrics or self.reward_weights[reward_name] > 0.0
            for reward_name in REWARD_NAME_ORDER
        }
        naturalness_cfg = dict(config['naturalness']) if self._compute_flags['naturalness'] else None
        foldability_cfg = dict(config.get('foldability', {})) if self._compute_flags['foldability'] else None
        stability_cfg = dict(config['stability']) if self._compute_flags['stability'] else None
        developability_cfg = dict(config['developability']) if self._compute_flags['developability'] else None
        self.naturalness = None
        if self._compute_flags['naturalness']:
            self.naturalness = ESM2NaturalnessScorer(
                model_name=str(naturalness_cfg['model_name']),
                device=naturalness_cfg.get('device', self.device),
                batch_size=_resolve_reward_batch_size(
                    naturalness_cfg,
                    default_batch_size=default_reward_batch_size,
                    field_name='naturalness.batch_size',
                ),
            )
        self.foldability = None
        if self._compute_flags['foldability']:
            self.foldability = ESMFoldFoldabilityScorer(
                device=foldability_cfg.get('device', self.device),
                batch_size=_resolve_reward_batch_size(
                    foldability_cfg,
                    default_batch_size=default_reward_batch_size,
                    field_name='foldability.batch_size',
                ),
                num_recycles=foldability_cfg.get('num_recycles'),
            )
        self.stability = None
        if self._compute_flags['stability']:
            self.stability = TemBERTureTmScorer(
                model_name_or_path=str(stability_cfg['model_name_or_path']),
                tokenizer_name_or_path=stability_cfg.get('tokenizer_name_or_path'),
                device=stability_cfg.get('device', self.device),
                batch_size=_resolve_reward_batch_size(
                    stability_cfg,
                    default_batch_size=default_reward_batch_size,
                    field_name='stability.batch_size',
                ),
                base_model_name_or_path=stability_cfg.get('base_model_name_or_path'),
            )
        self.developability = None
        if self._compute_flags['developability']:
            self.developability = ProteinSolScorer(
                model_name_or_path=str(developability_cfg['model_name_or_path']),
                tokenizer_name_or_path=developability_cfg.get('tokenizer_name_or_path'),
                device=developability_cfg.get('device', self.device),
                batch_size=_resolve_reward_batch_size(
                    developability_cfg,
                    default_batch_size=default_reward_batch_size,
                    field_name='developability.batch_size',
                ),
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
        calibration = {}
        if self._compute_flags['naturalness']:
            nat_raw, _, _ = self._timed_score_raw(self.naturalness, sequences)
            nat_q10, nat_q90 = _quantiles(nat_raw)
            calibration['naturalness_q10'] = nat_q10
            calibration['naturalness_q90'] = nat_q90
        if self._compute_flags['stability']:
            stab_raw, _, _ = self._timed_score_raw(self.stability, sequences)
            stab_q10, stab_q90 = _quantiles(stab_raw)
            calibration['stability_q10'] = stab_q10
            calibration['stability_q90'] = stab_q90
        self.calibration = calibration
        return dict(self.calibration)

    def _should_compute_reward(self, reward_name, step_number):
        if step_number is None:
            return True
        interval = self.reward_compute_every_n_steps[reward_name]
        return (step_number - 1) % interval == 0

    def score(self, sequences, step_number=None):
        details, metrics = self.score_details(sequences, step_number=step_number)
        return details['total'], metrics

    def score_details(self, sequences, step_number=None):
        if self.calibration is None:
            raise RuntimeError('CompositeProteinReward.calibrate must be called before score')
        step_number = _normalize_step_number(step_number)
        num_sequences = len(sequences)

        if self._compute_flags['naturalness'] and self._should_compute_reward('naturalness', step_number):
            nat_raw, nat_score_sec, nat_cpu_to_gpu_sec = self._timed_score_raw(self.naturalness, sequences)
            nat = _scale_quantile(
                nat_raw,
                self.calibration['naturalness_q10'],
                self.calibration['naturalness_q90'],
            )
            nat_skipped = 0.0
        else:
            nat_raw = [0.0] * num_sequences
            nat_score_sec = 0.0
            nat_cpu_to_gpu_sec = 0.0
            nat = [0.0] * num_sequences
            nat_skipped = 1.0

        if self._compute_flags['foldability'] and self._should_compute_reward('foldability', step_number):
            fold, fold_score_sec, fold_cpu_to_gpu_sec = self._timed_score_raw(self.foldability, sequences)
            fold_skipped = 0.0
        else:
            fold = [0.0] * num_sequences
            fold_score_sec = 0.0
            fold_cpu_to_gpu_sec = 0.0
            fold_skipped = 1.0

        if self._compute_flags['stability'] and self._should_compute_reward('stability', step_number):
            stab_raw, stab_score_sec, stab_cpu_to_gpu_sec = self._timed_score_raw(self.stability, sequences)
            stab = _scale_quantile(
                stab_raw,
                self.calibration['stability_q10'],
                self.calibration['stability_q90'],
            )
            stab_skipped = 0.0
        else:
            stab_raw = [0.0] * num_sequences
            stab_score_sec = 0.0
            stab_cpu_to_gpu_sec = 0.0
            stab = [0.0] * num_sequences
            stab_skipped = 1.0

        if self._compute_flags['developability'] and self._should_compute_reward('developability', step_number):
            dev_raw, dev_score_sec, dev_cpu_to_gpu_sec = self._timed_score_raw(self.developability, sequences)
            developability_components = score_developability_components(dev_raw, sequences)
            dev = developability_components['developability']
            dev_skipped = 0.0
        else:
            dev_raw = [0.0] * num_sequences
            dev_score_sec = 0.0
            dev_cpu_to_gpu_sec = 0.0
            developability_components = {
                'solubility': [0.0] * num_sequences,
                'liability_reward': [0.0] * num_sequences,
                'developability': [0.0] * num_sequences,
            }
            dev = developability_components['developability']
            dev_skipped = 1.0

        nat_release_sec = 0.0
        nat_gpu_to_cpu_sec = 0.0
        fold_release_sec = 0.0
        fold_gpu_to_cpu_sec = 0.0
        stab_release_sec = 0.0
        stab_gpu_to_cpu_sec = 0.0
        dev_release_sec = 0.0
        dev_gpu_to_cpu_sec = 0.0

        total = []
        for idx in range(len(sequences)):
            total.append(
                self.reward_weights['naturalness'] * nat[idx]
                + self.reward_weights['foldability'] * fold[idx]
                + self.reward_weights['stability'] * stab[idx]
                + self.reward_weights['developability'] * dev[idx]
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
            'reward_nat_skipped': nat_skipped,
            'reward_fold_skipped': fold_skipped,
            'reward_stab_skipped': stab_skipped,
            'reward_dev_skipped': dev_skipped,
            'reward_nat_every_n_steps': float(self.reward_compute_every_n_steps['naturalness']),
            'reward_fold_every_n_steps': float(self.reward_compute_every_n_steps['foldability']),
            'reward_stab_every_n_steps': float(self.reward_compute_every_n_steps['stability']),
            'reward_dev_every_n_steps': float(self.reward_compute_every_n_steps['developability']),
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
            'reward_nat_weight': float(self.reward_weights['naturalness']),
            'reward_fold_weight': float(self.reward_weights['foldability']),
            'reward_stab_weight': float(self.reward_weights['stability']),
            'reward_dev_weight': float(self.reward_weights['developability']),
            'reward_score_sec_total': nat_score_sec + fold_score_sec + stab_score_sec + dev_score_sec,
            'reward_cpu_to_gpu_sec_total': nat_cpu_to_gpu_sec + fold_cpu_to_gpu_sec + stab_cpu_to_gpu_sec + dev_cpu_to_gpu_sec,
            'reward_gpu_to_cpu_sec_total': nat_gpu_to_cpu_sec + fold_gpu_to_cpu_sec + stab_gpu_to_cpu_sec + dev_gpu_to_cpu_sec,
        }
        return details, metrics
