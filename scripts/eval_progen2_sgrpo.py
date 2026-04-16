import argparse
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass

import torch
import yaml

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))

from progen2.data.prompts import load_prompt_texts
from progen2.evaluation import classify_protein_sequence, compute_group_diversity_rewards, nanmean
from progen2.modeling.wrapper import OfficialProGen2CausalLM
from progen2.rewards import CompositeProteinReward
from progen2.rl.policy import ProGen2Policy


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalExperimentConfig:
    name: str
    checkpoint_dir: str
    display_name: str | None = None
    checkpoint_subdir: str | None = None


@dataclass(frozen=True)
class EvalConfig:
    output_markdown_path: str
    output_json_path: str
    official_code_dir: str
    tokenizer_path: str
    prompt_path: str
    rewards: dict
    output_rows_path: str | None = None
    seed: int = 42
    bf16: bool = False
    device: str = 'cuda'
    num_samples: int = 1000
    generation_prompt_batch_size: int = 8
    group_size: int = 4
    max_new_tokens: int = 64
    top_p: float = 0.95
    temperature: float = 0.8
    reward_calibration_size: int = 256
    reward_calibration_prompt_batch_size: int = 8
    experiments: list[EvalExperimentConfig] | None = None


def configure_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_jsonl(path, payload):
    with open(path, 'a') as handle:
        handle.write(json.dumps(payload, sort_keys=True) + '\n')


def _format_metric(value):
    if value is None:
        return 'nan'
    value = float(value)
    if math.isnan(value):
        return 'nan'
    return f'{value:.6f}'


def _display_name(experiment):
    return experiment.display_name or experiment.name


def load_config(path):
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    experiments = [EvalExperimentConfig(**item) for item in raw.pop('experiments')]
    config = EvalConfig(experiments=experiments, **raw)

    if not os.path.isdir(config.official_code_dir):
        raise FileNotFoundError(f'official_code_dir not found: {config.official_code_dir}')
    if not os.path.isfile(config.tokenizer_path):
        raise FileNotFoundError(f'tokenizer_path not found: {config.tokenizer_path}')
    if not os.path.isfile(config.prompt_path):
        raise FileNotFoundError(f'prompt_path not found: {config.prompt_path}')
    if config.num_samples <= 0:
        raise ValueError(f'num_samples must be positive, got {config.num_samples}')
    if config.generation_prompt_batch_size <= 0:
        raise ValueError('generation_prompt_batch_size must be positive')
    if config.group_size <= 1:
        raise ValueError('group_size must be greater than 1')
    if config.num_samples % config.group_size != 0:
        raise ValueError(
            'num_samples must be divisible by group_size: '
            f'{config.num_samples} vs {config.group_size}'
        )
    if config.max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be positive')
    if not 0.0 < config.top_p <= 1.0:
        raise ValueError('top_p must be in (0, 1]')
    if config.temperature <= 0.0:
        raise ValueError('temperature must be positive')
    if config.reward_calibration_size <= 0:
        raise ValueError('reward_calibration_size must be positive')
    if config.reward_calibration_prompt_batch_size <= 0:
        raise ValueError('reward_calibration_prompt_batch_size must be positive')
    if not config.rewards:
        raise ValueError('rewards config must be non-empty')
    if not config.experiments:
        raise ValueError('experiments must be non-empty')
    for experiment in config.experiments:
        if not experiment.name:
            raise ValueError('experiment name must be non-empty')
        if not os.path.isdir(experiment.checkpoint_dir):
            raise FileNotFoundError(f'checkpoint_dir not found: {experiment.checkpoint_dir}')
    return config


def resolve_device(device_name):
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('device=cuda requested but CUDA is not available')
        return torch.device('cuda')
    return torch.device(device_name)


def _cycle_prompt_batch(prompts, batch_size, start_index):
    batch = []
    for offset in range(batch_size):
        batch.append(prompts[(start_index + offset) % len(prompts)])
    return batch


def _collect_calibration_sequences(policy, prompts, config, seed):
    collected = []
    prompt_cursor = 0
    attempts = 0
    max_batches = max(
        8,
        math.ceil(config.reward_calibration_size / config.reward_calibration_prompt_batch_size) * 20,
    )
    while len(collected) < config.reward_calibration_size:
        if attempts >= max_batches:
            raise RuntimeError(
                'Failed to collect enough valid calibration sequences before hitting the '
                f'maximum attempt budget: collected={len(collected)} '
                f'target={config.reward_calibration_size} max_batches={max_batches}'
            )
        remaining = config.reward_calibration_size - len(collected)
        prompt_batch_size = min(config.reward_calibration_prompt_batch_size, remaining)
        prompt_batch = _cycle_prompt_batch(prompts, prompt_batch_size, prompt_cursor)
        prompt_cursor = (prompt_cursor + len(prompt_batch)) % len(prompts)
        rollout = policy.generate_rollouts(
            prompt_batch,
            num_return_sequences=1,
            max_new_tokens=config.max_new_tokens,
            top_p=config.top_p,
            temperature=config.temperature,
            seed=seed + attempts,
        )
        for sequence in rollout.protein_sequences:
            classification = classify_protein_sequence(sequence)
            if classification['is_valid']:
                collected.append(classification['sequence'])
        attempts += 1
    return collected[: config.reward_calibration_size]


def _generate_eval_rows(policy, prompts, config, seed):
    rows = []
    prompt_cursor = 0
    prompts_remaining = config.num_samples // config.group_size
    batch_index = 0
    sample_index = 0
    while prompts_remaining > 0:
        prompt_batch_size = min(config.generation_prompt_batch_size, prompts_remaining)
        prompt_batch = _cycle_prompt_batch(prompts, prompt_batch_size, prompt_cursor)
        prompt_cursor = (prompt_cursor + len(prompt_batch)) % len(prompts)
        rollout = policy.generate_rollouts(
            prompt_batch,
            num_return_sequences=config.group_size,
            max_new_tokens=config.max_new_tokens,
            top_p=config.top_p,
            temperature=config.temperature,
            seed=seed + batch_index,
        )
        for prompt_text, decoded_text, raw_sequence in zip(
            rollout.prompt_texts,
            rollout.decoded_texts,
            rollout.protein_sequences,
        ):
            classification = classify_protein_sequence(raw_sequence)
            rows.append(
                {
                    'sample_index': sample_index,
                    'prompt_text': prompt_text,
                    'decoded_text': decoded_text,
                    'raw_sequence': raw_sequence,
                    'sequence': classification['sequence'],
                    'is_valid': classification['is_valid'],
                    'invalid_reason': classification['invalid_reason'],
                }
            )
            sample_index += 1
        prompts_remaining -= prompt_batch_size
        batch_index += 1
    if len(rows) != config.num_samples:
        raise RuntimeError(f'Generated {len(rows)} rows, expected {config.num_samples}')
    return rows


def _initialize_row_metrics(rows):
    for row in rows:
        row.update(
            {
                'naturalness_raw': None,
                'naturalness': 0.0,
                'foldability': 0.0,
                'stability_raw': None,
                'stability': 0.0,
                'solubility': 0.0,
                'liability_reward': 0.0,
                'developability': 0.0,
                'total_reward': 0.0,
                'group_diversity_reward': 0.0,
            }
        )


def _score_rows(rows, reward_model, group_size):
    _initialize_row_metrics(rows)
    valid_indices = [idx for idx, row in enumerate(rows) if row['is_valid']]
    valid_sequences = [rows[idx]['sequence'] for idx in valid_indices]
    reward_metrics = {
        'reward_nat_mean': 0.0,
        'reward_fold_mean': 0.0,
        'reward_stab_mean': 0.0,
        'reward_dev_mean': 0.0,
        'reward_sol_mean': 0.0,
        'reward_liability_mean': 0.0,
        'reward_total_mean': 0.0,
    }
    if valid_sequences:
        details, reward_metrics = reward_model.score_details(valid_sequences)
        for detail_index, row_index in enumerate(valid_indices):
            rows[row_index]['naturalness_raw'] = float(details['naturalness_raw'][detail_index])
            rows[row_index]['naturalness'] = float(details['naturalness'][detail_index])
            rows[row_index]['foldability'] = float(details['foldability'][detail_index])
            rows[row_index]['stability_raw'] = float(details['stability_raw'][detail_index])
            rows[row_index]['stability'] = float(details['stability'][detail_index])
            rows[row_index]['solubility'] = float(details['solubility'][detail_index])
            rows[row_index]['liability_reward'] = float(details['liability_reward'][detail_index])
            rows[row_index]['developability'] = float(details['developability'][detail_index])
            rows[row_index]['total_reward'] = float(details['total'][detail_index])

    group_rewards = compute_group_diversity_rewards([row['sequence'] for row in rows], group_size=group_size)
    if len(group_rewards) * group_size != len(rows):
        raise RuntimeError(
            f'group reward count does not match rows: groups={len(group_rewards)} '
            f'group_size={group_size} rows={len(rows)}'
        )
    for group_index, group_reward in enumerate(group_rewards):
        start = group_index * group_size
        end = start + group_size
        for row_index in range(start, end):
            rows[row_index]['group_diversity_reward'] = float(group_reward)

    valid_sequences_set = set(valid_sequences)
    valid_only_metrics = {f'{key}_valid_only': float(value) for key, value in reward_metrics.items()}
    return {
        'reward_mean': float(sum(row['total_reward'] for row in rows) / len(rows)),
        'reward_nat_mean': float(sum(row['naturalness'] for row in rows) / len(rows)),
        'reward_fold_mean': float(sum(row['foldability'] for row in rows) / len(rows)),
        'reward_stab_mean': float(sum(row['stability'] for row in rows) / len(rows)),
        'reward_dev_mean': float(sum(row['developability'] for row in rows) / len(rows)),
        'reward_sol_mean': float(sum(row['solubility'] for row in rows) / len(rows)),
        'reward_liability_mean': float(sum(row['liability_reward'] for row in rows) / len(rows)),
        'group_diversity_mean': float(sum(group_rewards) / len(group_rewards)),
        'valid_fraction': float(len(valid_indices) / len(rows)),
        'invalid_fraction': float(1.0 - (len(valid_indices) / len(rows))),
        'unique_valid_fraction': 0.0 if not valid_sequences else float(len(valid_sequences_set) / len(valid_sequences)),
        'unique_overall_fraction': float(len(valid_sequences_set) / len(rows)),
        'mean_valid_length': nanmean([len(sequence) for sequence in valid_sequences]),
        'naturalness_raw_mean_valid': nanmean([row['naturalness_raw'] for row in rows]),
        'stability_raw_mean_valid': nanmean([row['stability_raw'] for row in rows]),
        **valid_only_metrics,
    }


def _build_markdown(config, results):
    lines = [
        '# ProGen2 SGRPO Evaluation',
        '',
        f'- `num_samples`: {config.num_samples}',
        f'- `generation_prompt_batch_size`: {config.generation_prompt_batch_size}',
        f'- `group_size`: {config.group_size}',
        f'- `max_new_tokens`: {config.max_new_tokens}',
        f'- `top_p`: {config.top_p}',
        f'- `temperature`: {config.temperature}',
        f'- `reward_calibration_size`: {config.reward_calibration_size}',
        '',
        '| Model | Total Reward | Nat Reward | Fold Reward | Stab Reward | Dev Reward | Solubility | Liability Reward | Group Diversity | Valid Fraction | Unique Valid Fraction | Mean Valid Length |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in results:
        lines.append(
            '| '
            + ' | '.join(
                [
                    row['display_name'],
                    _format_metric(row['reward_mean']),
                    _format_metric(row['reward_nat_mean']),
                    _format_metric(row['reward_fold_mean']),
                    _format_metric(row['reward_stab_mean']),
                    _format_metric(row['reward_dev_mean']),
                    _format_metric(row['reward_sol_mean']),
                    _format_metric(row['reward_liability_mean']),
                    _format_metric(row['group_diversity_mean']),
                    _format_metric(row['valid_fraction']),
                    _format_metric(row['unique_valid_fraction']),
                    _format_metric(row['mean_valid_length']),
                ]
            )
            + ' |'
        )
    lines.extend(
        [
            '',
            'Column notes:',
            '- `Total Reward` is the mean final rollout reward with invalid sequences contributing zero, matching training-time invalid handling.',
            '- `Nat/Fold/Stab/Dev Reward` are the normalized reward components used by SGRPO.',
            '- `Solubility` and `Liability Reward` are the two developability subcomponents.',
            '- `Group Diversity` is the mean contiguous-group diversity reward using the configured `group_size`.',
            '- `Valid Fraction` is the fraction of generated sequences containing only the supported 20-residue alphabet.',
            '- `Unique Valid Fraction` is the fraction of unique sequences among valid generated sequences.',
            '',
        ]
    )
    return '\n'.join(lines)


def evaluate_experiment(config, experiment, prompts, device):
    logger.info('Evaluating %s', experiment.name)
    policy = ProGen2Policy(
        OfficialProGen2CausalLM(
            official_code_dir=config.official_code_dir,
            checkpoint_dir=experiment.checkpoint_dir,
            tokenizer_path=config.tokenizer_path,
            checkpoint_subdir=experiment.checkpoint_subdir,
            device=device,
            use_fp16=False,
            autocast_dtype=torch.bfloat16 if config.bf16 and device.type == 'cuda' else None,
        ),
        trainable=False,
    )
    reward_model = CompositeProteinReward(config.rewards, device=device)
    calibration_sequences = _collect_calibration_sequences(
        policy,
        prompts,
        config,
        seed=config.seed + 100000,
    )
    calibration = reward_model.calibrate(calibration_sequences)
    rows = _generate_eval_rows(
        policy,
        prompts,
        config,
        seed=config.seed,
    )
    metrics = _score_rows(rows, reward_model, group_size=config.group_size)
    metrics.update(
        {
            'experiment': experiment.name,
            'display_name': _display_name(experiment),
            'calibration': calibration,
        }
    )
    return rows, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    device = resolve_device(config.device)
    prompts = load_prompt_texts(config.prompt_path)

    all_rows = []
    results = []
    for experiment in config.experiments:
        rows, metrics = evaluate_experiment(config, experiment, prompts, device)
        results.append(metrics)
        for row in rows:
            all_rows.append({'experiment': experiment.name, **row})

    markdown = _build_markdown(config, results)
    payload = {
        'config': asdict(config),
        'results': results,
    }

    _ensure_parent_dir(config.output_markdown_path)
    with open(config.output_markdown_path, 'w') as handle:
        handle.write(markdown)
    _ensure_parent_dir(config.output_json_path)
    with open(config.output_json_path, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    if config.output_rows_path:
        _ensure_parent_dir(config.output_rows_path)
        if os.path.exists(config.output_rows_path):
            os.remove(config.output_rows_path)
        for row in all_rows:
            _write_jsonl(config.output_rows_path, row)

    logger.info('Wrote markdown report to %s', config.output_markdown_path)
    logger.info('Wrote JSON report to %s', config.output_json_path)
    if config.output_rows_path:
        logger.info('Wrote row-level report to %s', config.output_rows_path)


if __name__ == '__main__':
    main()
