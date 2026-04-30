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
from progen2.evaluation import classify_protein_sequence, global_edit_diversity, nanmean
from progen2.modeling.wrapper import OfficialProGen2CausalLM
from progen2.rewards import CompositeProteinReward
from progen2.rewards.composite import normalize_protein_reward_weights
from progen2.rl.policy import ProGen2Policy


logger = logging.getLogger(__name__)
_PYPLOT = None
_MARKER_CYCLE = ('o', '^', 's', 'D', 'P', 'X', 'v', '<', '>')


def _get_pyplot():
    global _PYPLOT
    if _PYPLOT is None:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        _PYPLOT = plt
    return _PYPLOT


def _series_style(series_index):
    plt = _get_pyplot()
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color')
    if not color_cycle:
        raise ValueError('Matplotlib color cycle is empty')
    color_index = series_index % len(color_cycle)
    marker_index = (series_index // len(color_cycle)) % len(_MARKER_CYCLE)
    return {
        'color': color_cycle[color_index],
        'marker': _MARKER_CYCLE[marker_index],
    }


@dataclass(frozen=True)
class EvalExperimentConfig:
    name: str
    checkpoint_dir: str
    display_name: str | None = None
    checkpoint_subdir: str | None = None
    naturalness: float | None = None
    foldability: float | None = None
    stability: float | None = None
    developability: float | None = None


@dataclass(frozen=True)
class EvalConfig:
    output_markdown_path: str
    output_json_path: str
    output_naturalness_diversity_plot_path: str
    output_foldability_diversity_plot_path: str
    output_stability_diversity_plot_path: str
    output_developability_diversity_plot_path: str
    output_soft_reward_diversity_plot_path: str
    official_code_dir: str
    tokenizer_path: str
    prompt_path: str
    rewards: dict
    output_rows_path: str | None = None
    seed: int = 42
    bf16: bool = False
    device: str = 'cuda'
    num_samples: int = 960
    generation_prompt_batch_size: int = 8
    num_return_sequences: int = 12
    max_new_tokens: int = 64
    top_p: float = 0.95
    temperature: float = 0.8
    temperature_values: list[float] | None = None
    calibration_temperature: float | None = None
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


def _resolve_temperature_values(raw):
    values = raw.get('temperature_values')
    if values is None:
        return [float(raw.get('temperature', 0.8))]
    if not isinstance(values, list) or not values:
        raise ValueError('temperature_values must be a non-empty list when provided')
    return [float(value) for value in values]


def load_config(path):
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    experiments = [EvalExperimentConfig(**item) for item in raw.pop('experiments')]
    raw['temperature_values'] = _resolve_temperature_values(raw)
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
    if config.num_return_sequences <= 0:
        raise ValueError('num_return_sequences must be positive')
    if config.num_samples % config.num_return_sequences != 0:
        raise ValueError(
            'num_samples must be divisible by num_return_sequences: '
            f'{config.num_samples} vs {config.num_return_sequences}'
        )
    if config.max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be positive')
    if not 0.0 < config.top_p <= 1.0:
        raise ValueError('top_p must be in (0, 1]')
    if config.temperature <= 0.0:
        raise ValueError('temperature must be positive')
    if config.calibration_temperature is not None and config.calibration_temperature <= 0.0:
        raise ValueError('calibration_temperature must be positive when provided')
    if not config.temperature_values:
        raise ValueError('temperature_values must be non-empty')
    for temperature in config.temperature_values:
        if temperature <= 0.0:
            raise ValueError(f'all temperature_values must be positive, got {temperature}')
    if len(set(config.temperature_values)) != len(config.temperature_values):
        raise ValueError(f'temperature_values must be unique, got {config.temperature_values}')
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
        normalize_protein_reward_weights(
            {
                'naturalness': experiment.naturalness,
                'foldability': experiment.foldability,
                'stability': experiment.stability,
                'developability': experiment.developability,
            }
        )
    return config


def resolve_device(device_name):
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('device=cuda requested but CUDA is not available')
        return torch.device('cuda')
    return torch.device(device_name)


def _cycle_prompt_batch(prompts, batch_size, start_index):
    return [prompts[(start_index + offset) % len(prompts)] for offset in range(batch_size)]


def _default_reward_batch_size(config):
    return int(config.generation_prompt_batch_size * config.num_return_sequences)


def _collect_calibration_sequences(policy, prompts, config, seed, temperature):
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
            temperature=temperature,
            seed=seed + attempts,
        )
        for sequence in rollout.protein_sequences:
            classification = classify_protein_sequence(sequence)
            if classification['is_valid']:
                collected.append(classification['sequence'])
        attempts += 1
    return collected[: config.reward_calibration_size]


def _generate_eval_rows(policy, prompts, config, seed, temperature):
    rows = []
    prompt_cursor = 0
    prompts_remaining = config.num_samples // config.num_return_sequences
    batch_index = 0
    sample_index = 0
    while prompts_remaining > 0:
        prompt_batch_size = min(config.generation_prompt_batch_size, prompts_remaining)
        prompt_batch = _cycle_prompt_batch(prompts, prompt_batch_size, prompt_cursor)
        prompt_cursor = (prompt_cursor + len(prompt_batch)) % len(prompts)
        rollout = policy.generate_rollouts(
            prompt_batch,
            num_return_sequences=config.num_return_sequences,
            max_new_tokens=config.max_new_tokens,
            top_p=config.top_p,
            temperature=temperature,
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
                'soft_reward': 0.0,
                'diversity': 0.0,
            }
        )


def _score_rows(rows, reward_model):
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
            rows[row_index]['soft_reward'] = float(details['total'][detail_index])

    valid_sequences_set = set(valid_sequences)
    valid_only_metrics = {f'{key}_valid_only': float(value) for key, value in reward_metrics.items()}
    soft_reward_mean = float(sum(row['soft_reward'] for row in rows) / len(rows))
    diversity = float(global_edit_diversity(valid_sequences))
    return {
        'soft_reward_mean': soft_reward_mean,
        'reward_mean': soft_reward_mean,
        'reward_nat_mean': float(sum(row['naturalness'] for row in rows) / len(rows)),
        'reward_fold_mean': float(sum(row['foldability'] for row in rows) / len(rows)),
        'reward_stab_mean': float(sum(row['stability'] for row in rows) / len(rows)),
        'reward_dev_mean': float(sum(row['developability'] for row in rows) / len(rows)),
        'reward_sol_mean': float(sum(row['solubility'] for row in rows) / len(rows)),
        'reward_liability_mean': float(sum(row['liability_reward'] for row in rows) / len(rows)),
        'diversity': diversity,
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
        '# ProGen2 Temperature Sweep Evaluation',
        '',
        f'- `num_samples`: {config.num_samples}',
        f'- `generation_prompt_batch_size`: {config.generation_prompt_batch_size}',
        f'- `num_return_sequences`: {config.num_return_sequences}',
        f'- `max_new_tokens`: {config.max_new_tokens}',
        f'- `top_p`: {config.top_p}',
        f'- `calibration_temperature`: {config.calibration_temperature if config.calibration_temperature is not None else config.temperature}',
        f'- `temperature_values`: {", ".join(f"{value:.1f}" for value in config.temperature_values or [])}',
        f'- `reward_calibration_size`: {config.reward_calibration_size}',
        '',
        '| Model | Temperature | Soft Reward | Naturalness | Foldability | Stability | Developability | Diversity | Valid Fraction | Unique Valid Fraction | Mean Valid Length |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in results:
        lines.append(
            '| '
            + ' | '.join(
                [
                    row['display_name'],
                    _format_metric(row['temperature']),
                    _format_metric(row['soft_reward_mean']),
                    _format_metric(row['reward_nat_mean']),
                    _format_metric(row['reward_fold_mean']),
                    _format_metric(row['reward_stab_mean']),
                    _format_metric(row['reward_dev_mean']),
                    _format_metric(row['diversity']),
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
            '- `Soft Reward` is the rollout-level weighted sum of normalized reward components.',
            '- Calibration is performed once per experiment at `calibration_temperature`, then reused across the full temperature sweep to keep reward scales comparable.',
            '- `Diversity` is the global edit-distance diversity over all valid sequences at that sweep point.',
            '- `Valid Fraction` uses the supported 20-residue alphabet check.',
            '',
        ]
    )
    return '\n'.join(lines)


def _plot_metric_tradeoff(results, experiments, x_key, x_label, y_key, y_label, title, output_path):
    plt = _get_pyplot()
    _ensure_parent_dir(output_path)
    figure, axis = plt.subplots(figsize=(7.5, 5.75), dpi=160)
    finite_rows_found = False
    for series_index, experiment in enumerate(experiments):
        experiment_rows = [row for row in results if row['experiment'] == experiment.name]
        if not experiment_rows:
            raise ValueError(f'No rows found for experiment {experiment.name}')
        x_values = [float(row[x_key]) for row in experiment_rows]
        y_values = [float(row[y_key]) for row in experiment_rows]
        if not all(math.isfinite(value) for value in x_values + y_values):
            raise ValueError(f'Non-finite values encountered while plotting {experiment.name} for {x_key} vs {y_key}')
        finite_rows_found = True
        style = _series_style(series_index)
        axis.plot(
            x_values,
            y_values,
            label=_display_name(experiment),
            color=style['color'],
            marker=style['marker'],
            linewidth=1.8,
            markersize=6.5,
        )
    if not finite_rows_found:
        raise ValueError(f'No finite rows available for plot {title}')
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


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
    reward_model = CompositeProteinReward(
        config.rewards,
        device=device,
        default_reward_batch_size=_default_reward_batch_size(config),
        reward_weights={
            'naturalness': experiment.naturalness,
            'foldability': experiment.foldability,
            'stability': experiment.stability,
            'developability': experiment.developability,
        },
        always_compute_metrics=True,
    )
    calibration_temperature = (
        float(config.calibration_temperature)
        if config.calibration_temperature is not None
        else float(config.temperature)
    )
    calibration_sequences = _collect_calibration_sequences(
        policy,
        prompts,
        config,
        seed=config.seed + 100000,
        temperature=calibration_temperature,
    )
    calibration = reward_model.calibrate(calibration_sequences)

    all_rows = []
    results = []
    for sweep_index, temperature in enumerate(config.temperature_values or []):
        logger.info('Evaluating %s at temperature=%.3f', experiment.name, temperature)
        rows = _generate_eval_rows(
            policy,
            prompts,
            config,
            seed=config.seed + (1000 * sweep_index),
            temperature=float(temperature),
        )
        metrics = _score_rows(rows, reward_model)
        metrics.update(
            {
                'experiment': experiment.name,
                'display_name': _display_name(experiment),
                'reward_weights': reward_model.reward_weights,
                'calibration': calibration,
                'temperature': float(temperature),
                'calibration_temperature': calibration_temperature,
            }
        )
        results.append(metrics)
        for row in rows:
            all_rows.append(
                {
                    'experiment': experiment.name,
                    'display_name': _display_name(experiment),
                    'temperature': float(temperature),
                    **row,
                }
            )
    return all_rows, results


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
        experiment_rows, experiment_results = evaluate_experiment(config, experiment, prompts, device)
        all_rows.extend(experiment_rows)
        results.extend(experiment_results)

    experiment_order = {experiment.name: idx for idx, experiment in enumerate(config.experiments)}
    temperature_order = {float(value): idx for idx, value in enumerate(config.temperature_values or [])}
    results.sort(key=lambda row: (experiment_order[row['experiment']], temperature_order[float(row['temperature'])]))

    _plot_metric_tradeoff(
        results=results,
        experiments=config.experiments,
        x_key='reward_nat_mean',
        x_label='Naturalness',
        y_key='diversity',
        y_label='Diversity',
        title='Naturalness vs Diversity',
        output_path=config.output_naturalness_diversity_plot_path,
    )
    _plot_metric_tradeoff(
        results=results,
        experiments=config.experiments,
        x_key='reward_fold_mean',
        x_label='Foldability',
        y_key='diversity',
        y_label='Diversity',
        title='Foldability vs Diversity',
        output_path=config.output_foldability_diversity_plot_path,
    )
    _plot_metric_tradeoff(
        results=results,
        experiments=config.experiments,
        x_key='reward_stab_mean',
        x_label='Stability',
        y_key='diversity',
        y_label='Diversity',
        title='Stability vs Diversity',
        output_path=config.output_stability_diversity_plot_path,
    )
    _plot_metric_tradeoff(
        results=results,
        experiments=config.experiments,
        x_key='reward_dev_mean',
        x_label='Developability',
        y_key='diversity',
        y_label='Diversity',
        title='Developability vs Diversity',
        output_path=config.output_developability_diversity_plot_path,
    )
    _plot_metric_tradeoff(
        results=results,
        experiments=config.experiments,
        x_key='soft_reward_mean',
        x_label='Soft Reward',
        y_key='diversity',
        y_label='Diversity',
        title='Soft Reward vs Diversity',
        output_path=config.output_soft_reward_diversity_plot_path,
    )

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
    logger.info('Wrote naturalness-vs-diversity plot to %s', config.output_naturalness_diversity_plot_path)
    logger.info('Wrote foldability-vs-diversity plot to %s', config.output_foldability_diversity_plot_path)
    logger.info('Wrote stability-vs-diversity plot to %s', config.output_stability_diversity_plot_path)
    logger.info('Wrote developability-vs-diversity plot to %s', config.output_developability_diversity_plot_path)
    logger.info('Wrote soft-reward-vs-diversity plot to %s', config.output_soft_reward_diversity_plot_path)
    if config.output_rows_path:
        logger.info('Wrote row-level report to %s', config.output_rows_path)


if __name__ == '__main__':
    main()
