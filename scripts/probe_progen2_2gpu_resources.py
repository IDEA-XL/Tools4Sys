import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


logger = logging.getLogger(__name__)

GPU_REWARD_NAMES = (
    'naturalness',
    'foldability',
    'stability',
)


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_yaml(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def _write_yaml(path, payload):
    _ensure_parent_dir(path)
    with open(path, 'w') as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _read_jsonl(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _max_float_metric(rows, key):
    values = [float(row[key]) for row in rows if key in row]
    if not values:
        return None
    return max(values)


def _looks_like_oom(text):
    lowered = text.lower()
    return (
        'cuda out of memory' in lowered
        or 'out of memory' in lowered
        or 'cublas_status_alloc_failed' in lowered
    )


def _oom_reward_name(stderr_text):
    lowered = stderr_text.lower()
    if not _looks_like_oom(lowered):
        return None
    for reward_name in GPU_REWARD_NAMES + ('developability',):
        if f'{reward_name} reward scoring failed' in lowered:
            return reward_name
    return None


def _calibration_prompt_candidates(initial_value, min_value):
    if initial_value <= 0:
        raise ValueError(f'initial calibration prompt batch size must be positive, got {initial_value}')
    if min_value <= 0:
        raise ValueError(f'min calibration prompt batch size must be positive, got {min_value}')
    if initial_value < min_value:
        raise ValueError(
            f'initial calibration prompt batch size must be >= min value, got {initial_value} < {min_value}'
        )
    values = []
    current = int(initial_value)
    while current >= min_value:
        values.append(current)
        if current == 1:
            break
        current //= 2
    if values[-1] != min_value and min_value not in values:
        values.append(int(min_value))
    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _format_reward_batch_label(reward_batch_sizes):
    return '_'.join(f'{name[:4]}{reward_batch_sizes[name]}' for name in GPU_REWARD_NAMES)


def _build_launch_command(args, config_path, port_offset):
    return [
        'accelerate',
        'launch',
        '--config_file',
        args.accelerate_config,
        '--num_processes',
        str(args.num_processes),
        '--main_process_port',
        str(args.main_process_port + port_offset),
        'scripts/train_progen2_sgrpo.py',
        config_path,
    ]


def _run_probe_attempt(base_config, args, run_name, *, calibration_prompt_batch_size=None, reward_batch_sizes=None, port_offset=0):
    run_dir = os.path.join(args.output_root, 'runs', run_name)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    temp_config_path = os.path.join(args.output_root, 'configs', f'{run_name}.yaml')
    stdout_path = os.path.join(args.output_root, 'logs', f'{run_name}.stdout.log')
    stderr_path = os.path.join(args.output_root, 'logs', f'{run_name}.stderr.log')

    payload = dict(base_config)
    payload['output_dir'] = run_dir
    payload['overwrite_output_dir'] = True
    if calibration_prompt_batch_size is not None:
        payload['reward_calibration_prompt_batch_size'] = int(calibration_prompt_batch_size)
    if reward_batch_sizes is not None:
        rewards_cfg = dict(payload['rewards'])
        for reward_name in GPU_REWARD_NAMES:
            reward_cfg = dict(rewards_cfg[reward_name])
            reward_cfg['batch_size'] = int(reward_batch_sizes[reward_name])
            rewards_cfg[reward_name] = reward_cfg
        payload['rewards'] = rewards_cfg

    _write_yaml(temp_config_path, payload)
    os.makedirs(os.path.dirname(stdout_path), exist_ok=True)

    command = _build_launch_command(args, temp_config_path, port_offset)
    logger.info('Running probe attempt %s', run_name)
    with open(stdout_path, 'w') as stdout_handle, open(stderr_path, 'w') as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=args.repo_root,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
            text=True,
        )

    metrics_path = os.path.join(run_dir, 'metrics.jsonl')
    state = {
        'run_name': run_name,
        'return_code': int(completed.returncode),
        'status': 'failed',
        'output_dir': run_dir,
        'stdout_path': stdout_path,
        'stderr_path': stderr_path,
        'steps_completed': 0,
        'max_reserved_ratio': None,
        'max_allocated_ratio': None,
        'calibration_prompt_batch_size': None if calibration_prompt_batch_size is None else int(calibration_prompt_batch_size),
        'reward_batch_sizes': None if reward_batch_sizes is None else {name: int(reward_batch_sizes[name]) for name in GPU_REWARD_NAMES},
        'oom_reward': None,
    }

    if os.path.exists(metrics_path):
        rows = _read_jsonl(metrics_path)
        state['steps_completed'] = len(rows)
        if rows:
            state['max_reserved_ratio'] = _max_float_metric(rows, 'cuda_run_max_reserved_ratio')
            state['max_allocated_ratio'] = _max_float_metric(rows, 'cuda_run_max_allocated_ratio')

    with open(stderr_path) as handle:
        stderr_text = handle.read()

    if completed.returncode == 0:
        state['status'] = 'success'
    elif _looks_like_oom(stderr_text):
        state['status'] = 'oom'
        state['oom_reward'] = _oom_reward_name(stderr_text)
    return state


def _probe_calibration_prompt_batch_size(base_config, args):
    results = []
    candidates = _calibration_prompt_candidates(args.calibration_initial, args.calibration_min)
    selected = None
    for offset, candidate in enumerate(candidates):
        state = _run_probe_attempt(
            base_config,
            args,
            f'calibpbs{candidate}',
            calibration_prompt_batch_size=candidate,
            port_offset=offset,
        )
        results.append(state)
        if state['status'] == 'success':
            selected = candidate
            break
        if state['status'] != 'oom':
            raise RuntimeError(
                'Calibration prompt batch-size probe failed with a non-OOM error at '
                f'{candidate}. Inspect {state["stderr_path"]}.'
            )
    if selected is None:
        raise RuntimeError('No calibration prompt batch-size candidate succeeded')
    return {
        'mode': 'calibration_prompt_batch_size',
        'selected_reward_calibration_prompt_batch_size': int(selected),
        'results': results,
    }


def _probe_gpu_reward_batch_sizes(base_config, args):
    current = {name: int(args.reward_initial) for name in GPU_REWARD_NAMES}
    safe = None
    active = set(GPU_REWARD_NAMES)
    current_target = int(args.reward_initial)
    results = []
    port_offset = 0

    while True:
        run_name = f'rewardbs_{_format_reward_batch_label(current)}'
        state = _run_probe_attempt(
            base_config,
            args,
            run_name,
            calibration_prompt_batch_size=args.fixed_calibration_prompt_batch_size,
            reward_batch_sizes=current,
            port_offset=port_offset,
        )
        port_offset += 1
        results.append(state)

        if state['status'] == 'success':
            safe = dict(current)
            if not active:
                break
            next_target = current_target * 2
            if next_target > args.reward_max:
                unresolved = sorted(active)
                if unresolved:
                    raise RuntimeError(
                        'GPU reward batch-size probe hit reward_max before every GPU reward reached an OOM '
                        f'boundary. Unresolved rewards: {unresolved}; current safe config: {safe}'
                    )
                break
            current_target = next_target
            for reward_name in active:
                current[reward_name] = current_target
            continue

        if state['status'] != 'oom':
            raise RuntimeError(
                'GPU reward batch-size probe failed with a non-OOM error for config '
                f'{current}. Inspect {state["stderr_path"]}.'
            )

        culprit = state['oom_reward']
        if culprit not in GPU_REWARD_NAMES:
            raise RuntimeError(
                'GPU reward batch-size probe encountered an OOM but could not attribute it to one of '
                f'{GPU_REWARD_NAMES}. Culprit={culprit!r}. Inspect {state["stderr_path"]}.'
            )
        if culprit not in active:
            raise RuntimeError(
                f'GPU reward batch-size probe received a repeated or inconsistent culprit {culprit!r}. '
                f'Active rewards: {sorted(active)}'
            )
        if safe is None:
            raise RuntimeError(
                f'Initial GPU reward batch size {args.reward_initial} OOMed at {culprit!r}; '
                'no lower fallback is defined in this probe path.'
            )
        current[culprit] = safe[culprit]
        active.remove(culprit)
        if not active:
            break

    if safe is None:
        raise RuntimeError('GPU reward batch-size probe found no successful configuration')
    return {
        'mode': 'gpu_reward_batch_size',
        'selected_gpu_reward_batch_sizes': {name: int(current[name]) for name in GPU_REWARD_NAMES},
        'results': results,
    }


def _build_markdown(summary):
    lines = [
        '# ProGen2 2-GPU Resource Probe',
        '',
        f"- `mode`: `{summary['mode']}`",
    ]
    if 'selected_reward_calibration_prompt_batch_size' in summary:
        lines.append(
            f"- `selected_reward_calibration_prompt_batch_size`: `{summary['selected_reward_calibration_prompt_batch_size']}`"
        )
    if 'selected_gpu_reward_batch_sizes' in summary:
        lines.append(
            '- `selected_gpu_reward_batch_sizes`: `'
            + ', '.join(f'{name}={value}' for name, value in summary['selected_gpu_reward_batch_sizes'].items())
            + '`'
        )
    lines.extend(
        [
            '',
            '| Run | Status | Calibration Prompt BS | Naturalness BS | Foldability BS | Stability BS | OOM Reward | Steps | Max Reserved Ratio | stderr |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
        ]
    )
    for result in summary['results']:
        reward_batch_sizes = result.get('reward_batch_sizes') or {}
        lines.append(
            '| '
            + ' | '.join(
                [
                    result['run_name'],
                    result['status'],
                    'nan' if result['calibration_prompt_batch_size'] is None else str(result['calibration_prompt_batch_size']),
                    'nan' if 'naturalness' not in reward_batch_sizes else str(reward_batch_sizes['naturalness']),
                    'nan' if 'foldability' not in reward_batch_sizes else str(reward_batch_sizes['foldability']),
                    'nan' if 'stability' not in reward_batch_sizes else str(reward_batch_sizes['stability']),
                    str(result['oom_reward']),
                    str(result['steps_completed']),
                    'nan' if result['max_reserved_ratio'] is None else f"{result['max_reserved_ratio']:.6f}",
                    result['stderr_path'],
                ]
            )
            + ' |'
        )
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--mode', required=True, choices=('calibration_prompt_batch_size', 'gpu_reward_batch_size'))
    parser.add_argument('--accelerate-config', default='configs/accelerate_ddp_2gpu.yaml')
    parser.add_argument('--num-processes', type=int, default=2)
    parser.add_argument('--main-process-port', type=int, default=29000)
    parser.add_argument('--calibration-initial', type=int, default=128)
    parser.add_argument('--calibration-min', type=int, default=1)
    parser.add_argument('--reward-initial', type=int, default=16)
    parser.add_argument('--reward-max', type=int, default=1024)
    parser.add_argument('--fixed-calibration-prompt-batch-size', type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    )

    args.repo_root = os.path.realpath('.')
    args.output_root = os.path.abspath(args.output_root)
    os.makedirs(args.output_root, exist_ok=True)
    base_config = _load_yaml(args.config)

    if args.mode == 'calibration_prompt_batch_size':
        summary = _probe_calibration_prompt_batch_size(base_config, args)
    else:
        summary = _probe_gpu_reward_batch_sizes(base_config, args)

    summary['base_config_path'] = os.path.abspath(args.config)
    summary_json_path = os.path.join(args.output_root, 'summary.json')
    summary_md_path = os.path.join(args.output_root, 'summary.md')
    _ensure_parent_dir(summary_json_path)
    with open(summary_json_path, 'w') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with open(summary_md_path, 'w') as handle:
        handle.write(_build_markdown(summary))
    logger.info('Wrote probe summary to %s', summary_json_path)


if __name__ == '__main__':
    main()
