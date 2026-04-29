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


def _ordered_active_rewards(active):
    return [reward_name for reward_name in GPU_REWARD_NAMES if reward_name in active]


def _parse_reward_batch_sizes(text):
    pairs = [segment.strip() for segment in text.split(',') if segment.strip()]
    if not pairs:
        raise ValueError('reward start batch sizes string must not be empty')
    parsed = {}
    for pair in pairs:
        if '=' not in pair:
            raise ValueError(
                f"invalid reward batch-size override {pair!r}; expected format 'reward=value'"
            )
        reward_name, value_text = pair.split('=', 1)
        reward_name = reward_name.strip()
        if reward_name not in GPU_REWARD_NAMES:
            raise ValueError(
                f'unknown GPU reward name {reward_name!r}; expected one of {GPU_REWARD_NAMES}'
            )
        value = int(value_text.strip())
        if value <= 0:
            raise ValueError(f'batch size for {reward_name!r} must be positive, got {value}')
        parsed[reward_name] = value
    missing = [reward_name for reward_name in GPU_REWARD_NAMES if reward_name not in parsed]
    if missing:
        raise ValueError(
            f'reward start batch sizes must specify every GPU reward exactly once; missing {missing}'
        )
    return parsed


def _parse_active_rewards(text):
    names = [segment.strip() for segment in text.split(',') if segment.strip()]
    if not names:
        raise ValueError('reward active list must not be empty')
    parsed = []
    seen = set()
    for reward_name in names:
        if reward_name not in GPU_REWARD_NAMES:
            raise ValueError(
                f'unknown active GPU reward {reward_name!r}; expected one of {GPU_REWARD_NAMES}'
            )
        if reward_name in seen:
            raise ValueError(f'duplicate active GPU reward {reward_name!r}')
        parsed.append(reward_name)
        seen.add(reward_name)
    return set(parsed)


def _set_individual_probe_current(current, safe, active, probe_reward, target_value):
    for reward_name in active:
        current[reward_name] = safe[reward_name]
    current[probe_reward] = int(target_value)


def _advance_individual_probe(current, safe, active, reward_max):
    capped = []
    ordered = _ordered_active_rewards(active)
    while ordered:
        probe_reward = ordered[0]
        next_target = safe[probe_reward] * 2
        if next_target <= reward_max:
            _set_individual_probe_current(current, safe, active, probe_reward, next_target)
            return probe_reward, next_target, capped
        active.remove(probe_reward)
        capped.append(probe_reward)
        ordered = _ordered_active_rewards(active)
    return None, None, capped


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


def _run_probe_attempt(
    base_config,
    args,
    run_name,
    *,
    calibration_prompt_batch_size=None,
    reward_batch_sizes=None,
    port_offset=0,
    phase=None,
    probed_reward=None,
):
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
        'phase': phase,
        'probed_reward': probed_reward,
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
    if args.reward_start_batch_sizes is None:
        current = {name: int(args.reward_initial) for name in GPU_REWARD_NAMES}
        safe = None
        active = set(GPU_REWARD_NAMES)
        current_target = int(args.reward_initial)
        phase = 'joint'
        current_probe_reward = None
    else:
        current = _parse_reward_batch_sizes(args.reward_start_batch_sizes)
        safe = dict(current)
        active = set(GPU_REWARD_NAMES) if args.reward_active is None else _parse_active_rewards(args.reward_active)
        current_target = int(args.reward_current_target)
        if current_target <= 0:
            raise ValueError(f'reward_current_target must be positive, got {current_target}')
        phase = args.reward_phase
        if phase == 'joint':
            for reward_name in active:
                current[reward_name] = current_target
            current_probe_reward = None
        else:
            ordered = _ordered_active_rewards(active)
            if not ordered:
                raise ValueError('individual reward probe resume requires at least one active reward')
            current_probe_reward = ordered[0]
            _set_individual_probe_current(current, safe, active, current_probe_reward, current_target)
    results = []
    port_offset = 0
    capped_rewards = []

    while True:
        run_name = f'rewardbs_{_format_reward_batch_label(current)}'
        state = _run_probe_attempt(
            base_config,
            args,
            run_name,
            calibration_prompt_batch_size=args.fixed_calibration_prompt_batch_size,
            reward_batch_sizes=current,
            port_offset=port_offset,
            phase=phase,
            probed_reward=current_probe_reward,
        )
        port_offset += 1
        results.append(state)

        if state['status'] == 'success':
            safe = dict(current)
            if not active:
                break
            if phase == 'joint':
                next_target = current_target * 2
                if next_target > args.reward_max:
                    capped_rewards.extend(_ordered_active_rewards(active))
                    active.clear()
                    break
                current_target = next_target
                for reward_name in active:
                    current[reward_name] = current_target
                continue
            if current_probe_reward is None:
                raise RuntimeError('individual GPU reward probe succeeded without an active probed reward')
            next_target = current_target * 2
            if next_target <= args.reward_max:
                current_target = next_target
                _set_individual_probe_current(current, safe, active, current_probe_reward, current_target)
                continue
            capped_rewards.append(current_probe_reward)
            active.remove(current_probe_reward)
            current_probe_reward = None
            if not active:
                break
            next_probe_reward, next_target, newly_capped = _advance_individual_probe(
                current,
                safe,
                active,
                args.reward_max,
            )
            capped_rewards.extend(newly_capped)
            current_probe_reward = next_probe_reward
            current_target = next_target
            if current_probe_reward is None:
                break
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
        if safe is None:
            raise RuntimeError(
                f'Initial GPU reward batch size {args.reward_initial} OOMed at {culprit!r}; '
                'no lower fallback is defined in this probe path.'
            )
        if phase == 'joint':
            if culprit in active:
                current[culprit] = safe[culprit]
                active.remove(culprit)
                if not active:
                    break
                for reward_name in active:
                    current[reward_name] = current_target
                continue
            ordered = _ordered_active_rewards(active)
            if not ordered:
                raise RuntimeError(
                    f'GPU reward batch-size probe received an OOM attributed to frozen reward {culprit!r} '
                    'after all active rewards had already been resolved.'
                )
            phase = 'individual'
            current_probe_reward = ordered[0]
            _set_individual_probe_current(current, safe, active, current_probe_reward, current_target)
            continue
        if current_probe_reward is None:
            raise RuntimeError('individual GPU reward probe OOMed without an active probed reward')
        current[current_probe_reward] = safe[current_probe_reward]
        active.remove(current_probe_reward)
        if not active:
            break
        next_probe_reward, next_target, newly_capped = _advance_individual_probe(
            current,
            safe,
            active,
            args.reward_max,
        )
        capped_rewards.extend(newly_capped)
        current_probe_reward = next_probe_reward
        current_target = next_target
        if current_probe_reward is None:
            break

    if safe is None:
        raise RuntimeError('GPU reward batch-size probe found no successful configuration')
    return {
        'mode': 'gpu_reward_batch_size',
        'selected_gpu_reward_batch_sizes': {name: int(safe[name]) for name in GPU_REWARD_NAMES},
        'capped_gpu_rewards_without_oom': capped_rewards,
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
    if 'capped_gpu_rewards_without_oom' in summary:
        lines.append(
            '- `capped_gpu_rewards_without_oom`: `'
            + ', '.join(summary['capped_gpu_rewards_without_oom'])
            + '`'
        )
    lines.extend(
        [
            '',
            '| Run | Phase | Probed Reward | Status | Calibration Prompt BS | Naturalness BS | Foldability BS | Stability BS | OOM Reward | Steps | Max Reserved Ratio | stderr |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
        ]
    )
    for result in summary['results']:
        reward_batch_sizes = result.get('reward_batch_sizes') or {}
        lines.append(
            '| '
            + ' | '.join(
                [
                    result['run_name'],
                    str(result.get('phase')),
                    str(result.get('probed_reward')),
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
    parser.add_argument('--reward-start-batch-sizes', default=None)
    parser.add_argument('--reward-active', default=None)
    parser.add_argument('--reward-current-target', type=int, default=None)
    parser.add_argument('--reward-phase', choices=('joint', 'individual'), default='joint')
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
