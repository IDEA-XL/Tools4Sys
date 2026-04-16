import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))


logger = logging.getLogger(__name__)


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _parse_candidates(raw):
    values = []
    for item in str(raw).split(','):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f'candidate batch sizes must be positive, got {value}')
        values.append(value)
    if not values:
        raise ValueError('candidate batch sizes must be non-empty')
    if values != sorted(values):
        raise ValueError(f'candidate batch sizes must be sorted ascending, got {values}')
    if len(set(values)) != len(values):
        raise ValueError(f'candidate batch sizes must be unique, got {values}')
    return values


def _load_yaml(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def _read_jsonl(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _looks_like_oom(stderr_text):
    lowered = stderr_text.lower()
    return (
        'cuda out of memory' in lowered
        or 'out of memory' in lowered
        or 'cublas_status_alloc_failed' in lowered
    )


def _build_markdown(summary):
    lines = [
        '# ProGen2 1-GPU Batch Probe',
        '',
        f"- `base_config_path`: `{summary['base_config_path']}`",
        f"- `target_reserved_ratio`: `{summary['target_reserved_ratio']}`",
        f"- `candidates`: `{', '.join(str(value) for value in summary['candidates'])}`",
        '',
        f"- `recommended_batch_size`: `{summary['recommended_batch_size']}`",
        f"- `recommendation_reason`: `{summary['recommendation_reason']}`",
        '',
        '| Batch Size | Status | Steps | Peak Reserved GiB | Peak Reserved Ratio | Peak Allocated GiB | Peak Allocated Ratio | Output Dir |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for result in summary['results']:
        lines.append(
            '| '
            + ' | '.join(
                [
                    str(result['batch_size']),
                    result['status'],
                    str(result['steps_completed']),
                    'nan' if result['max_reserved_gib'] is None else f"{result['max_reserved_gib']:.6f}",
                    'nan' if result['max_reserved_ratio'] is None else f"{result['max_reserved_ratio']:.6f}",
                    'nan' if result['max_allocated_gib'] is None else f"{result['max_allocated_gib']:.6f}",
                    'nan' if result['max_allocated_ratio'] is None else f"{result['max_allocated_ratio']:.6f}",
                    result['output_dir'],
                ]
            )
            + ' |'
        )
    lines.extend(
        [
            '',
            'Column notes:',
            '- `Batch Size` is `per_device_prompt_batch_size`.',
            '- `Peak Reserved/Allocated` are the maximum CUDA memory metrics emitted by the trainer during the probe run.',
            '- `Status=oom` means the subprocess failed with an explicit CUDA allocation error.',
            '- `recommended_batch_size` chooses the largest successful batch whose reserved ratio stays within the configured target. If none satisfy the target, it falls back to the largest successful batch and marks that explicitly.',
            '',
        ]
    )
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--target-reserved-ratio', type=float, default=0.95)
    parser.add_argument('--stop-after-oom', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    )

    candidates = _parse_candidates(args.candidates)
    if not 0.0 < args.target_reserved_ratio <= 1.0:
        raise ValueError(
            f'target_reserved_ratio must be in (0, 1], got {args.target_reserved_ratio}'
        )

    repo_root = os.path.realpath('.')
    output_root = os.path.abspath(args.output_root)
    probe_runs_dir = os.path.join(output_root, 'runs')
    os.makedirs(probe_runs_dir, exist_ok=True)

    base_config = _load_yaml(args.config)
    results = []

    for candidate in candidates:
        run_dir = os.path.join(probe_runs_dir, f'bs{candidate}')
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        temp_config_path = os.path.join(probe_runs_dir, f'bs{candidate}.yaml')
        stdout_path = os.path.join(probe_runs_dir, f'bs{candidate}.stdout.log')
        stderr_path = os.path.join(probe_runs_dir, f'bs{candidate}.stderr.log')

        config_payload = dict(base_config)
        config_payload['per_device_prompt_batch_size'] = int(candidate)
        config_payload['output_dir'] = run_dir
        config_payload['overwrite_output_dir'] = True

        with open(temp_config_path, 'w') as handle:
            yaml.safe_dump(config_payload, handle, sort_keys=False)

        logger.info('Probing batch size %d', candidate)
        with open(stdout_path, 'w') as stdout_handle, open(stderr_path, 'w') as stderr_handle:
            completed = subprocess.run(
                [sys.executable, 'scripts/train_progen2_sgrpo.py', temp_config_path],
                cwd=repo_root,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
                text=True,
            )

        metrics_path = os.path.join(run_dir, 'metrics.jsonl')
        state = {
            'batch_size': int(candidate),
            'return_code': int(completed.returncode),
            'status': 'failed',
            'steps_completed': 0,
            'max_reserved_gib': None,
            'max_reserved_ratio': None,
            'max_allocated_gib': None,
            'max_allocated_ratio': None,
            'output_dir': run_dir,
            'stdout_path': stdout_path,
            'stderr_path': stderr_path,
        }

        if os.path.exists(metrics_path):
            rows = _read_jsonl(metrics_path)
            state['steps_completed'] = len(rows)
            if rows:
                state['max_reserved_gib'] = max(
                    float(row['cuda_run_max_reserved_gib'])
                    for row in rows
                    if 'cuda_run_max_reserved_gib' in row
                )
                state['max_reserved_ratio'] = max(
                    float(row['cuda_run_max_reserved_ratio'])
                    for row in rows
                    if 'cuda_run_max_reserved_ratio' in row
                )
                state['max_allocated_gib'] = max(
                    float(row['cuda_run_max_allocated_gib'])
                    for row in rows
                    if 'cuda_run_max_allocated_gib' in row
                )
                state['max_allocated_ratio'] = max(
                    float(row['cuda_run_max_allocated_ratio'])
                    for row in rows
                    if 'cuda_run_max_allocated_ratio' in row
                )

        with open(stderr_path) as handle:
            stderr_text = handle.read()

        if completed.returncode == 0:
            state['status'] = 'success'
        elif _looks_like_oom(stderr_text):
            state['status'] = 'oom'
        results.append(state)

        if state['status'] == 'failed':
            raise RuntimeError(
                'Batch probe failed with a non-OOM error at '
                f'batch_size={candidate}. Inspect {stderr_path}.'
            )
        if state['status'] == 'oom' and args.stop_after_oom:
            break

    within_target = [
        item for item in results
        if item['status'] == 'success'
        and item['max_reserved_ratio'] is not None
        and item['max_reserved_ratio'] <= args.target_reserved_ratio
    ]
    successes = [item for item in results if item['status'] == 'success']
    if within_target:
        recommended = max(within_target, key=lambda item: item['batch_size'])
        recommendation_reason = 'largest_success_within_target_reserved_ratio'
    elif successes:
        recommended = max(successes, key=lambda item: item['batch_size'])
        recommendation_reason = 'largest_success_exceeds_target_reserved_ratio'
    else:
        recommended = None
        recommendation_reason = 'no_successful_candidate'

    summary = {
        'base_config_path': os.path.abspath(args.config),
        'target_reserved_ratio': float(args.target_reserved_ratio),
        'candidates': candidates,
        'results': results,
        'recommended_batch_size': None if recommended is None else int(recommended['batch_size']),
        'recommendation_reason': recommendation_reason,
    }

    summary_json_path = os.path.join(output_root, 'summary.json')
    summary_markdown_path = os.path.join(output_root, 'summary.md')
    _ensure_parent_dir(summary_json_path)
    with open(summary_json_path, 'w') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with open(summary_markdown_path, 'w') as handle:
        handle.write(_build_markdown(summary))

    logger.info('Wrote batch probe summary to %s', summary_json_path)
    logger.info('Wrote batch probe markdown to %s', summary_markdown_path)
    if recommended is None:
        raise RuntimeError('No successful batch-size candidate found')
    logger.info(
        'Recommended per_device_prompt_batch_size=%d (%s)',
        recommended['batch_size'],
        recommendation_reason,
    )


if __name__ == '__main__':
    main()
