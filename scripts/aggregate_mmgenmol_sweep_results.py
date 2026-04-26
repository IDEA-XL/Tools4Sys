import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))

from genmol.rl.reward import sa_to_score


def _read_jsonl(path):
    rows = []
    with open(path) as handle:
        for line_idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Failed to parse JSONL line {line_idx} in {path}') from exc
    if not rows:
        raise ValueError(f'No rows found in {path}')
    return rows


def _write_json(path, payload):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_jsonl(path, rows):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')


def _format_metric(value):
    if value is None:
        return 'nan'
    value = float(value)
    if math.isnan(value):
        return 'nan'
    return f'{value:.6f}'


def _parse_task_manifest(path):
    with open(path, newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        rows = list(reader)
    required = {
        'task_id',
        'model_name',
        'sweep_type',
        'sweep_value',
        'randomness',
        'temperature',
        'checkpoint_path',
        'output_path',
    }
    if reader.fieldnames is None or set(reader.fieldnames) != required:
        raise ValueError(f'Unexpected task manifest header in {path}: {reader.fieldnames}')
    if not rows:
        raise ValueError(f'Task manifest is empty: {path}')
    seen = set()
    for row in rows:
        task_id = int(row['task_id'])
        if task_id in seen:
            raise ValueError(f'Duplicate task_id in {path}: {task_id}')
        seen.add(task_id)
    return sorted(rows, key=lambda item: int(item['task_id']))


def _canonicalize_smiles(smiles, chem):
    if smiles is None:
        return None
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = chem.MolFromSmiles(smiles, sanitize=True)
    except Exception:
        return None
    if mol is None:
        return None
    return chem.MolToSmiles(mol)


def _mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return float('nan')
    return float(sum(values) / len(values))


def _compute_internal_diversity_for_group(canonical_smiles, fingerprint_cache, data_structs):
    valid = [smiles for smiles in canonical_smiles if smiles is not None]
    if len(valid) < 2:
        return 0.0
    similarity_sum = 0.0
    pair_count = 0
    for left_idx in range(len(valid)):
        left_fp = fingerprint_cache[valid[left_idx]]
        for right_idx in range(left_idx + 1, len(valid)):
            right_fp = fingerprint_cache[valid[right_idx]]
            similarity_sum += float(data_structs.TanimotoSimilarity(left_fp, right_fp))
            pair_count += 1
    if pair_count == 0:
        return 0.0
    return float(1.0 - similarity_sum / pair_count)


def _load_docking_summary(path, docking_mode):
    with open(path) as handle:
        payload = json.load(handle)
    summaries = payload.get('summaries')
    if not isinstance(summaries, dict) or docking_mode not in summaries:
        raise ValueError(f'Missing docking summary for mode {docking_mode!r}: {path}')
    summary = summaries[docking_mode]
    success_fraction = float(summary['docking_success_fraction'])
    if success_fraction <= 0.0:
        raise ValueError(f'Docking success fraction is zero for {path}')
    return payload, summary


def _compute_task_metrics(task, args, chem, qed, sa_oracle, fingerprint_generator, data_structs):
    generated_rows_path = task['output_path']
    if not os.path.exists(generated_rows_path):
        raise FileNotFoundError(f'Missing generated rows file: {generated_rows_path}')
    generated_rows = _read_jsonl(generated_rows_path)
    if len(generated_rows) != args.expected_rows_per_task:
        raise ValueError(
            f'Generated row count mismatch for {generated_rows_path}: '
            f'{len(generated_rows)} vs {args.expected_rows_per_task}'
        )

    output_dir = os.path.join(
        args.docking_root,
        task['model_name'],
        f"{task['sweep_type']}_{task['sweep_value']}",
    )
    docking_records_path = os.path.join(output_dir, 'docking.records.jsonl')
    docking_summary_path = os.path.join(output_dir, 'docking.summary.json')
    if not os.path.exists(docking_records_path):
        raise FileNotFoundError(f'Missing docking records file: {docking_records_path}')
    if not os.path.exists(docking_summary_path):
        raise FileNotFoundError(f'Missing docking summary file: {docking_summary_path}')
    docking_records = _read_jsonl(docking_records_path)
    if len(docking_records) != args.expected_rows_per_task:
        raise ValueError(
            f'Docking record count mismatch for {docking_records_path}: '
            f'{len(docking_records)} vs {args.expected_rows_per_task}'
        )
    docking_payload, docking_summary = _load_docking_summary(docking_summary_path, args.docking_mode)

    rows_by_source_index = defaultdict(list)
    canonical_smiles_by_row = []
    for row_idx, row in enumerate(generated_rows):
        if 'source_index' not in row:
            raise ValueError(f'Generated row {row_idx} missing source_index in {generated_rows_path}')
        source_index = int(row['source_index'])
        canonical_smiles = _canonicalize_smiles(row.get('smiles'), chem)
        canonical_smiles_by_row.append(canonical_smiles)
        rows_by_source_index[source_index].append(canonical_smiles)

    if len(rows_by_source_index) != args.expected_num_pockets:
        raise ValueError(
            f'Expected {args.expected_num_pockets} pockets for {generated_rows_path}, '
            f'found {len(rows_by_source_index)}'
        )
    bad_group_sizes = {
        source_index: len(rows)
        for source_index, rows in rows_by_source_index.items()
        if len(rows) != args.expected_samples_per_pocket
    }
    if bad_group_sizes:
        raise ValueError(f'Unexpected per-pocket sample counts in {generated_rows_path}: {bad_group_sizes}')

    unique_valid_smiles = sorted({smiles for smiles in canonical_smiles_by_row if smiles is not None})
    qed_by_smiles = {}
    sa_by_smiles = {}
    sa_score_by_smiles = {}
    fp_by_smiles = {}
    if unique_valid_smiles:
        sa_values = sa_oracle(unique_valid_smiles)
        if len(sa_values) != len(unique_valid_smiles):
            raise RuntimeError(
                f'SA oracle returned {len(sa_values)} values for {len(unique_valid_smiles)} SMILES'
            )
        for smiles, sa_value in zip(unique_valid_smiles, sa_values):
            mol = chem.MolFromSmiles(smiles, sanitize=True)
            if mol is None:
                raise RuntimeError(f'Canonical valid SMILES failed to parse on second pass: {smiles}')
            qed_by_smiles[smiles] = float(qed.qed(mol))
            sa_by_smiles[smiles] = float(sa_value)
            sa_score_by_smiles[smiles] = float(sa_to_score(sa_value))
            fp_by_smiles[smiles] = fingerprint_generator.GetFingerprint(mol)

    pocket_diversities = []
    for source_index in sorted(rows_by_source_index):
        diversity = _compute_internal_diversity_for_group(
            rows_by_source_index[source_index],
            fingerprint_cache=fp_by_smiles,
            data_structs=data_structs,
        )
        pocket_diversities.append(diversity)

    qeds = [qed_by_smiles[smiles] for smiles in canonical_smiles_by_row if smiles is not None]
    sa_values = [sa_by_smiles[smiles] for smiles in canonical_smiles_by_row if smiles is not None]
    sa_scores = [sa_score_by_smiles[smiles] for smiles in canonical_smiles_by_row if smiles is not None]
    soft_rewards = [
        0.6 * qed_by_smiles[smiles] + 0.4 * sa_score_by_smiles[smiles]
        for smiles in canonical_smiles_by_row
        if smiles is not None
    ]

    valid_count = len(qeds)
    valid_fraction = float(valid_count / len(generated_rows))
    duplicate_fraction = 1.0 - float(len(unique_valid_smiles) / valid_count) if valid_count else float('nan')
    row = {
        'task_id': int(task['task_id']),
        'model_name': task['model_name'],
        'sweep_type': task['sweep_type'],
        'sweep_value': float(task['sweep_value']),
        'randomness': float(task['randomness']),
        'temperature': float(task['temperature']),
        'checkpoint_path': task['checkpoint_path'],
        'generated_rows_path': generated_rows_path,
        'docking_records_path': docking_records_path,
        'docking_summary_path': docking_summary_path,
        'num_rows': len(generated_rows),
        'num_pockets': len(rows_by_source_index),
        'samples_per_pocket': args.expected_samples_per_pocket,
        'valid_count': valid_count,
        'valid_fraction': valid_fraction,
        'unique_valid_count': len(unique_valid_smiles),
        'duplicate_fraction': duplicate_fraction,
        'qed_mean': _mean(qeds),
        'sa_mean': _mean(sa_values),
        'sa_score_mean': _mean(sa_scores),
        'soft_reward_mean': _mean(soft_rewards),
        'diversity': _mean(pocket_diversities),
        'diversity_definition': (
            'mean over pockets of 1 - mean pairwise Morgan-fingerprint Tanimoto similarity '
            'within the generated molecules for that pocket'
        ),
        'vina_dock_success_fraction': float(docking_summary['docking_success_fraction']),
        'vina_dock_num_docked': int(docking_summary['num_docked']),
        'vina_score_mean': float(docking_summary.get('vina_score_mean', float('nan'))),
        'vina_min_mean': float(docking_summary.get('vina_min_mean', float('nan'))),
        'vina_dock_mean': float(docking_summary['vina_dock_mean']),
        'vina_dock_median': float(docking_summary['vina_dock_median']),
        'docking_elapsed_sec': float(docking_payload['elapsed_sec']),
    }
    for key in ['qed_mean', 'sa_score_mean', 'soft_reward_mean', 'diversity', 'vina_dock_mean']:
        if not math.isfinite(float(row[key])):
            raise ValueError(f'Non-finite {key} for task {task["task_id"]}: {row[key]}')
    return row


def _display_name(name):
    return {
        'original_5500': 'Original 5500',
        'grpo_1000': 'GRPO 1000',
        'sgrpo_1000': 'SGRPO 1000',
        'grpo_divreg005_1000': 'GRPO DivReg 0.05 1000',
    }.get(name, name)


def _model_order(name):
    order = {
        'original_5500': 0,
        'grpo_1000': 1,
        'sgrpo_1000': 2,
        'grpo_divreg005_1000': 3,
    }
    return order.get(name, len(order))


def _plot_metric(rows, sweep_type, metric_key, metric_label, output_path):
    sweep_rows = [row for row in rows if row['sweep_type'] == sweep_type]
    if not sweep_rows:
        raise ValueError(f'No rows for sweep_type={sweep_type!r}')
    model_names = []
    for row in sorted(sweep_rows, key=lambda item: (_model_order(item['model_name']), item['model_name'])):
        if row['model_name'] not in model_names:
            model_names.append(row['model_name'])
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name in model_names:
        model_rows = [row for row in sweep_rows if row['model_name'] == model_name]
        model_rows.sort(key=lambda row: float(row['sweep_value']))
        x_values = [float(row[metric_key]) for row in model_rows]
        y_values = [float(row['diversity']) for row in model_rows]
        ax.plot(x_values, y_values, marker='o', linewidth=2, label=_display_name(model_name))
        for row, x_value, y_value in zip(model_rows, x_values, y_values):
            ax.annotate(
                f"{float(row['sweep_value']):.1f}",
                (x_value, y_value),
                textcoords='offset points',
                xytext=(4, 4),
                fontsize=8,
            )
    ax.set_title(f'mmGenMol {sweep_type}: {metric_label} vs Diversity')
    ax.set_xlabel(metric_label)
    ax.set_ylabel('Mean Per-Pocket Internal Diversity')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_markdown(rows, plot_paths, json_path, rows_path):
    lines = [
        '# mmGenMol Sweep Results',
        '',
        f'- `summary_json`: `{json_path}`',
        f'- `raw_rows_jsonl`: `{rows_path}`',
        '- `num_pockets`: 100',
        '- `samples_per_pocket`: 16',
        '- `docking_mode`: `vina_dock`',
        '- `diversity`: per sweep point, compute internal diversity separately within each pocket group, then average over pockets.',
        '- `qed_mean` and `sa_score_mean`: means over valid generated molecules in the sweep point.',
        '- `soft_reward_mean`: `0.6 * qed_mean + 0.4 * sa_score_mean`, matching the rollout-level training reward before invalid and alert gating.',
        '- `vina_dock_mean`: mean Vina dock affinity over successful dockings; lower is better.',
        '',
        '| Model | Sweep | Value | Diversity | QED | SA Score | Soft Quality Score | Vina Dock Mean | Dock Success | Valid Fraction |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in rows:
        lines.append(
            '| '
            + ' | '.join(
                [
                    _display_name(row['model_name']),
                    row['sweep_type'],
                    _format_metric(row['sweep_value']),
                    _format_metric(row['diversity']),
                    _format_metric(row['qed_mean']),
                    _format_metric(row['sa_score_mean']),
                    _format_metric(row['soft_reward_mean']),
                    _format_metric(row['vina_dock_mean']),
                    _format_metric(row['vina_dock_success_fraction']),
                    _format_metric(row['valid_fraction']),
                ]
            )
            + ' |'
        )
    lines.append('')
    for title, path in plot_paths:
        lines.extend([f'## {title}', '', f'![{title}]({os.path.basename(path)})', ''])
    return '\n'.join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks_path', required=True)
    parser.add_argument('--docking_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--output_prefix', default='mmgenmol_sweep_results_20260423')
    parser.add_argument('--expected_num_tasks', type=int, default=32)
    parser.add_argument('--expected_rows_per_task', type=int, default=1600)
    parser.add_argument('--expected_num_pockets', type=int, default=100)
    parser.add_argument('--expected_samples_per_pocket', type=int, default=16)
    parser.add_argument('--docking_mode', default='vina_dock')
    return parser.parse_args()


def _resolve_plot_suffix(output_prefix):
    parts = output_prefix.rsplit('_', 1)
    if len(parts) == 2 and len(parts[1]) == 8 and parts[1].isdigit():
        return parts[1]
    return output_prefix


def main():
    args = parse_args()
    tasks = _parse_task_manifest(args.tasks_path)
    if len(tasks) != args.expected_num_tasks:
        raise ValueError(f'Expected {args.expected_num_tasks} tasks, found {len(tasks)}')

    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED, rdFingerprintGenerator
    from tdc import Oracle

    sa_oracle = Oracle('sa')
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    rows = [
        _compute_task_metrics(
            task=task,
            args=args,
            chem=Chem,
            qed=QED,
            sa_oracle=sa_oracle,
            fingerprint_generator=fingerprint_generator,
            data_structs=DataStructs,
        )
        for task in tasks
    ]
    rows.sort(key=lambda row: (_model_order(row['model_name']), row['model_name'], row['sweep_type'], row['sweep_value']))

    os.makedirs(args.output_dir, exist_ok=True)
    output_json_path = os.path.join(args.output_dir, f'{args.output_prefix}.json')
    output_rows_path = os.path.join(args.output_dir, f'{args.output_prefix}.rows.jsonl')
    output_markdown_path = os.path.join(args.output_dir, f'{args.output_prefix}.md')

    _write_json(output_json_path, rows)
    _write_jsonl(output_rows_path, rows)

    plot_paths = []
    present_sweep_types = {row['sweep_type'] for row in rows}
    sweep_types = [sweep_type for sweep_type in ['randomness', 'temperature'] if sweep_type in present_sweep_types]
    if not sweep_types:
        raise ValueError('No supported sweep types found; expected at least one of randomness or temperature')
    plot_suffix = _resolve_plot_suffix(args.output_prefix)
    for sweep_type in sweep_types:
        for metric_key, metric_label in [
            ('qed_mean', 'QED'),
            ('sa_score_mean', 'SA Score'),
            ('soft_reward_mean', 'Soft Quality Score'),
            ('vina_dock_mean', 'Vina Dock Mean'),
        ]:
            plot_path = os.path.join(
                args.output_dir,
                f'mmgenmol_{sweep_type}_diversity_vs_{metric_key}_{plot_suffix}.png',
            )
            _plot_metric(rows, sweep_type, metric_key, metric_label, plot_path)
            plot_paths.append((f'{sweep_type} {metric_label} vs diversity', plot_path))

    markdown = _build_markdown(rows, plot_paths, output_json_path, output_rows_path)
    with open(output_markdown_path, 'w') as handle:
        handle.write(markdown)

    print(json.dumps({
        'output_json_path': output_json_path,
        'output_rows_path': output_rows_path,
        'output_markdown_path': output_markdown_path,
        'plot_paths': [path for _, path in plot_paths],
        'num_rows': len(rows),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
