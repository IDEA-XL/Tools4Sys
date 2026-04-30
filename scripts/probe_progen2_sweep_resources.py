import argparse
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))

from progen2.data.prompts import load_prompt_texts
from progen2.modeling.wrapper import OfficialProGen2CausalLM
from progen2.rewards.developability import ProteinSolScorer
from progen2.rewards.foldability import ESMFoldFoldabilityScorer
from progen2.rewards.naturalness import ESM2NaturalnessScorer
from progen2.rewards.stability import TemBERTureTmScorer
from progen2.rl.policy import ProGen2Policy


logger = logging.getLogger(__name__)
GPU_REWARD_NAMES = ('naturalness', 'foldability', 'stability')
REWARD_NAMES = GPU_REWARD_NAMES + ('developability',)
VALID_AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


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


def _write_json(path, payload):
    _ensure_parent_dir(path)
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _load_yaml(path):
    with open(path) as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f'Expected mapping config at {path}, got {type(payload).__name__}')
    return payload


def _safe_float(value):
    if value is None:
        return None
    return float(value)


def _successive_halves(initial_value, min_value):
    if initial_value <= 0:
        raise ValueError(f'initial_value must be positive, got {initial_value}')
    if min_value <= 0:
        raise ValueError(f'min_value must be positive, got {min_value}')
    if initial_value < min_value:
        raise ValueError(f'initial_value must be >= min_value, got {initial_value} < {min_value}')
    values = []
    current = int(initial_value)
    while current >= min_value:
        values.append(current)
        if current == 1:
            break
        current //= 2
    if min_value not in values:
        values.append(int(min_value))
    deduped = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _looks_like_oom(text):
    lowered = text.lower()
    return (
        'cuda out of memory' in lowered
        or 'outofmemoryerror' in lowered
        or 'out of memory' in lowered
        or 'cublas_status_alloc_failed' in lowered
    )


def _resolve_eval_probe_context(config_path):
    payload = _load_yaml(config_path)
    required_paths = (
        'official_code_dir',
        'tokenizer_path',
        'prompt_path',
        'rewards',
        'experiments',
    )
    for key in required_paths:
        if key not in payload:
            raise ValueError(f'Missing required key {key!r} in {config_path}')
    experiments = payload['experiments']
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(f'experiments must be a non-empty list in {config_path}')
    checkpoint_dir = experiments[0].get('checkpoint_dir')
    if not checkpoint_dir:
        raise ValueError(f'experiments[0].checkpoint_dir missing in {config_path}')
    for path_key in ('official_code_dir', 'tokenizer_path', 'prompt_path'):
        if not os.path.exists(payload[path_key]):
            raise FileNotFoundError(f'{path_key} not found: {payload[path_key]}')
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f'checkpoint_dir not found: {checkpoint_dir}')
    return {
        'official_code_dir': str(payload['official_code_dir']),
        'tokenizer_path': str(payload['tokenizer_path']),
        'prompt_path': str(payload['prompt_path']),
        'checkpoint_dir': str(checkpoint_dir),
        'bf16': bool(payload.get('bf16', False)),
        'top_p': float(payload.get('top_p', 0.95)),
        'temperature': float(payload.get('temperature', 0.8)),
        'max_new_tokens': int(payload.get('max_new_tokens', 256)),
        'reward_cfg': dict(payload['rewards']),
        'reward_batch_sizes': {
            reward_name: int(payload['rewards'][reward_name]['batch_size'])
            for reward_name in payload['rewards']
            if reward_name in GPU_REWARD_NAMES or reward_name == 'developability'
        },
    }


def _resolve_device(device_name):
    device = torch.device(device_name)
    if device.type != 'cuda':
        raise RuntimeError(f'Probe expects CUDA device, got {device_name!r}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    return device


def _peak_cuda_ratios(device):
    if device is None:
        return {
            'max_reserved_bytes': None,
            'max_allocated_bytes': None,
            'max_reserved_ratio': None,
            'max_allocated_ratio': None,
        }
    total_memory = torch.cuda.get_device_properties(device).total_memory
    max_reserved = int(torch.cuda.max_memory_reserved(device))
    max_allocated = int(torch.cuda.max_memory_allocated(device))
    return {
        'max_reserved_bytes': max_reserved,
        'max_allocated_bytes': max_allocated,
        'max_reserved_ratio': float(max_reserved / total_memory),
        'max_allocated_ratio': float(max_allocated / total_memory),
    }


def _reward_label(reward_name, batch_size):
    return f'{reward_name}_bs{batch_size}'


def _read_rank_results(run_dir, world_size):
    rank_results = []
    for rank in range(world_size):
        result_path = os.path.join(run_dir, f'rank{rank}.json')
        if not os.path.isfile(result_path):
            raise FileNotFoundError(f'Missing rank result file: {result_path}')
        with open(result_path) as handle:
            rank_results.append(json.load(handle))
    return rank_results


def _synthetic_sequences(total_sequences, sequence_length, seed):
    if total_sequences <= 0:
        raise ValueError(f'total_sequences must be positive, got {total_sequences}')
    if sequence_length <= 0:
        raise ValueError(f'sequence_length must be positive, got {sequence_length}')
    rng = random.Random(seed)
    outputs = []
    for _ in range(total_sequences):
        outputs.append(''.join(rng.choice(VALID_AMINO_ACIDS) for _ in range(sequence_length)))
    return outputs


def _instantiate_gpu_scorer(reward_name, reward_cfg, batch_size, device_name):
    reward_cfg = dict(reward_cfg)
    reward_cfg['batch_size'] = int(batch_size)
    if reward_name == 'naturalness':
        return ESM2NaturalnessScorer(
            model_name=str(reward_cfg['model_name']),
            device=device_name,
            batch_size=reward_cfg['batch_size'],
        )
    if reward_name == 'foldability':
        return ESMFoldFoldabilityScorer(
            device=device_name,
            batch_size=reward_cfg['batch_size'],
            num_recycles=reward_cfg.get('num_recycles'),
        )
    if reward_name == 'stability':
        return TemBERTureTmScorer(
            model_name_or_path=str(reward_cfg['model_name_or_path']),
            tokenizer_name_or_path=reward_cfg.get('tokenizer_name_or_path'),
            device=device_name,
            batch_size=reward_cfg['batch_size'],
            base_model_name_or_path=reward_cfg.get('base_model_name_or_path'),
        )
    raise ValueError(f'Unsupported GPU reward name {reward_name!r}')


def _instantiate_reward_scorer(reward_name, reward_cfg, batch_size, device_name):
    if reward_name in GPU_REWARD_NAMES:
        return _instantiate_gpu_scorer(reward_name, reward_cfg, batch_size, device_name)
    if reward_name == 'developability':
        return ProteinSolScorer(
            model_name_or_path=str(reward_cfg['model_name_or_path']),
            tokenizer_name_or_path=reward_cfg.get('tokenizer_name_or_path'),
            device=reward_cfg.get('device', device_name),
            batch_size=int(batch_size),
            num_workers=reward_cfg.get('num_workers'),
        )
    raise ValueError(f'Unsupported reward name {reward_name!r}')


def _run_generation_worker(args):
    context = _resolve_eval_probe_context(args.config)
    device_name = 'cuda:0' if args.device == 'cuda' else args.device
    device = _resolve_device(device_name)
    if args.generation_prompt_batch_size <= 0:
        raise ValueError('generation_prompt_batch_size must be positive')
    if args.num_return_sequences <= 0:
        raise ValueError('num_return_sequences must be positive')
    prompts = load_prompt_texts(context['prompt_path'])
    if not prompts:
        raise ValueError(f'Prompt file produced no prompts: {context["prompt_path"]}')
    prompt_batch = [prompts[idx % len(prompts)] for idx in range(args.generation_prompt_batch_size)]

    model = OfficialProGen2CausalLM(
        checkpoint_dir=context['checkpoint_dir'],
        official_code_dir=context['official_code_dir'],
        tokenizer_path=context['tokenizer_path'],
        device=device,
        use_fp16=False,
        autocast_dtype=torch.bfloat16 if context['bf16'] and device.type == 'cuda' else None,
    )
    policy = ProGen2Policy(model, trainable=False)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    rollout = policy.generate_rollouts(
        prompt_batch,
        num_return_sequences=args.num_return_sequences,
        max_new_tokens=context['max_new_tokens'],
        top_p=context['top_p'],
        temperature=context['temperature'],
        seed=args.seed,
    )
    elapsed = time.perf_counter() - start
    sequence_lengths = [len(sequence) for sequence in rollout.protein_sequences]
    payload = {
        'checkpoint_dir': context['checkpoint_dir'],
        'generation_prompt_batch_size': int(args.generation_prompt_batch_size),
        'num_return_sequences': int(args.num_return_sequences),
        'num_generated_sequences': int(len(rollout.protein_sequences)),
        'elapsed_sec': float(elapsed),
        'sequence_length_min': int(min(sequence_lengths)) if sequence_lengths else 0,
        'sequence_length_max': int(max(sequence_lengths)) if sequence_lengths else 0,
        'sequence_length_mean': float(sum(sequence_lengths) / len(sequence_lengths)) if sequence_lengths else 0.0,
        **_peak_cuda_ratios(device),
    }
    _write_json(args.output_path, payload)


def _run_reward_worker(args):
    if args.reward_name not in REWARD_NAMES:
        raise ValueError(f'reward_name must be one of {REWARD_NAMES}, got {args.reward_name!r}')
    context = _resolve_eval_probe_context(args.config)
    if args.total_sequences <= 0:
        raise ValueError(f'total_sequences must be positive, got {args.total_sequences}')
    local_sequences = _synthetic_sequences(
        total_sequences=args.total_sequences,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
    reward_cfg = context['reward_cfg'].get(args.reward_name)
    if reward_cfg is None:
        raise ValueError(f'Missing reward config for {args.reward_name!r}')
    device = None
    device_name = 'cpu' if args.reward_name == 'developability' else 'cuda:0'
    if device_name.startswith('cuda'):
        device = _resolve_device(device_name)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    scorer = _instantiate_reward_scorer(
        args.reward_name,
        reward_cfg=reward_cfg,
        batch_size=args.batch_size,
        device_name=device_name,
    )
    start = time.perf_counter()
    scores = scorer.score_raw(local_sequences)
    elapsed = time.perf_counter() - start
    if len(scores) != len(local_sequences):
        raise RuntimeError(
            f'{args.reward_name} scorer returned {len(scores)} scores for {len(local_sequences)} sequences'
        )
    scorer.release()
    result = {
        'reward_name': args.reward_name,
        'batch_size': int(args.batch_size),
        'num_local_sequences': int(len(local_sequences)),
        'elapsed_sec': float(elapsed),
        'device_name': device_name,
        'last_num_workers': int(getattr(scorer, 'last_num_workers', 0)),
        'last_effective_batch_size': int(getattr(scorer, 'last_effective_batch_size', 0)),
        'last_chunk_count': int(getattr(scorer, 'last_chunk_count', 0)),
        'last_short_sequence_count': int(getattr(scorer, 'last_short_sequence_count', 0)),
        **_peak_cuda_ratios(device),
    }
    _write_json(os.path.join(args.output_dir, 'result.json'), result)


def _run_generation_controller(args):
    context = _resolve_eval_probe_context(args.config)
    candidates = _successive_halves(args.generation_initial, args.generation_min)
    results = []
    safe_candidate = None
    first_success = None
    for candidate in candidates:
        run_name = f'numret{candidate}'
        run_dir = os.path.join(args.output_root, 'runs', run_name)
        result_path = os.path.join(run_dir, 'result.json')
        stdout_path = os.path.join(args.output_root, 'logs', f'{run_name}.stdout.log')
        stderr_path = os.path.join(args.output_root, 'logs', f'{run_name}.stderr.log')
        os.makedirs(run_dir, exist_ok=True)
        _ensure_parent_dir(stdout_path)
        _ensure_parent_dir(stderr_path)
        command = [
            sys.executable,
            'scripts/probe_progen2_sweep_resources.py',
            'generation-worker',
            '--config',
            args.config,
            '--device',
            'cuda',
            '--generation-prompt-batch-size',
            str(args.generation_prompt_batch_size),
            '--num-return-sequences',
            str(candidate),
            '--seed',
            str(args.seed),
            '--output-path',
            result_path,
        ]
        with open(stdout_path, 'w') as stdout_handle, open(stderr_path, 'w') as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=args.repo_root,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
                text=True,
            )
        record = {
            'run_name': run_name,
            'num_return_sequences': int(candidate),
            'stdout_path': stdout_path,
            'stderr_path': stderr_path,
            'return_code': int(completed.returncode),
            'status': 'failed',
            'max_reserved_ratio': None,
            'max_allocated_ratio': None,
            'elapsed_sec': None,
        }
        stderr_text = Path(stderr_path).read_text() if os.path.isfile(stderr_path) else ''
        if completed.returncode == 0:
            with open(result_path) as handle:
                payload = json.load(handle)
            record.update(
                {
                    'status': 'success',
                    'max_reserved_ratio': _safe_float(payload['max_reserved_ratio']),
                    'max_allocated_ratio': _safe_float(payload['max_allocated_ratio']),
                    'elapsed_sec': _safe_float(payload['elapsed_sec']),
                }
            )
            if first_success is None:
                first_success = int(candidate)
            results.append(record)
            if record['max_reserved_ratio'] < args.safe_reserved_ratio:
                safe_candidate = int(candidate)
                break
            continue
        if _looks_like_oom(stderr_text):
            record['status'] = 'oom'
            results.append(record)
            continue
        results.append(record)
        if record['status'] not in {'success', 'oom'}:
            raise RuntimeError(
                'Generation probe failed with a non-OOM error at '
                f'num_return_sequences={candidate}. Inspect {stderr_path}.'
            )
    if first_success is None:
        raise RuntimeError('No num_return_sequences candidate succeeded in generation probe')
    selected_candidate = safe_candidate if safe_candidate is not None else int(first_success)
    summary = {
        'mode': 'generation_num_return_sequences',
        'checkpoint_dir': context['checkpoint_dir'],
        'generation_prompt_batch_size': int(args.generation_prompt_batch_size),
        'total_samples_planned': int(args.total_samples),
        'safe_reserved_ratio': float(args.safe_reserved_ratio),
        'selected_num_return_sequences': int(selected_candidate),
        'results': results,
    }
    _write_json(os.path.join(args.output_root, 'summary.json'), summary)


def _run_single_reward_controller(args):
    context = _resolve_eval_probe_context(args.config)
    reward_name = args.reward_name
    initial = int(context['reward_batch_sizes'][reward_name])
    candidates = []
    current = initial
    while current <= args.reward_max:
        candidates.append(current)
        current *= 2
    if not candidates:
        raise RuntimeError(f'No reward batch-size candidates generated for {reward_name}')
    results = []
    first_success = None
    safe_candidate = None
    for candidate in candidates:
        run_name = _reward_label(reward_name, candidate)
        run_dir = os.path.join(args.output_root, 'runs', run_name)
        stdout_path = os.path.join(args.output_root, 'logs', f'{run_name}.stdout.log')
        stderr_path = os.path.join(args.output_root, 'logs', f'{run_name}.stderr.log')
        worker_total_sequences = int(candidate) if reward_name in GPU_REWARD_NAMES else int(args.total_sequences)
        os.makedirs(run_dir, exist_ok=True)
        _ensure_parent_dir(stdout_path)
        _ensure_parent_dir(stderr_path)
        command = [
            sys.executable,
            'scripts/probe_progen2_sweep_resources.py',
            'reward-worker',
            '--config',
            args.config,
            '--reward-name',
            reward_name,
            '--batch-size',
            str(candidate),
            '--total-sequences',
            str(worker_total_sequences),
            '--sequence-length',
            str(args.sequence_length),
            '--seed',
            str(args.seed),
            '--output-dir',
            run_dir,
        ]
        with open(stdout_path, 'w') as stdout_handle, open(stderr_path, 'w') as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=args.repo_root,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
                text=True,
            )
        record = {
            'reward_name': reward_name,
            'batch_size': int(candidate),
            'total_sequences': int(worker_total_sequences),
            'stdout_path': stdout_path,
            'stderr_path': stderr_path,
            'return_code': int(completed.returncode),
            'status': 'failed',
            'max_reserved_ratio': None,
            'max_allocated_ratio': None,
            'elapsed_sec': None,
            'last_num_workers': None,
            'last_effective_batch_size': None,
            'last_chunk_count': None,
            'last_short_sequence_count': None,
        }
        stderr_text = Path(stderr_path).read_text() if os.path.isfile(stderr_path) else ''
        if completed.returncode == 0:
            result_path = os.path.join(run_dir, 'result.json')
            if not os.path.isfile(result_path):
                raise FileNotFoundError(f'Missing reward probe result file: {result_path}')
            with open(result_path) as handle:
                payload = json.load(handle)
            record.update(
                {
                    'status': 'success',
                    'max_reserved_ratio': _safe_float(payload['max_reserved_ratio']),
                    'max_allocated_ratio': _safe_float(payload['max_allocated_ratio']),
                    'elapsed_sec': _safe_float(payload['elapsed_sec']),
                    'last_num_workers': int(payload.get('last_num_workers', 0)),
                    'last_effective_batch_size': int(payload.get('last_effective_batch_size', 0)),
                    'last_chunk_count': int(payload.get('last_chunk_count', 0)),
                    'last_short_sequence_count': int(payload.get('last_short_sequence_count', 0)),
                }
            )
            if first_success is None:
                first_success = int(candidate)
            if reward_name in GPU_REWARD_NAMES:
                if record['max_reserved_ratio'] < args.safe_reserved_ratio:
                    safe_candidate = int(candidate)
                results.append(record)
            else:
                if record['last_effective_batch_size'] == candidate:
                    safe_candidate = int(candidate)
                results.append(record)
            continue
        if _looks_like_oom(stderr_text):
            record['status'] = 'oom'
            results.append(record)
            break
        results.append(record)
        raise RuntimeError(
            f'{reward_name} reward probe failed with a non-OOM error at batch_size={candidate}. '
            f'Inspect {stderr_path}.'
        )
    if first_success is None:
        raise RuntimeError(f'No successful batch size found for {reward_name}')
    selected = int(safe_candidate if safe_candidate is not None else first_success)
    summary = {
        'mode': 'single_reward_batch_size',
        'reward_name': reward_name,
        'total_sequences': int(args.total_sequences),
        'sequence_length': int(args.sequence_length),
        'safe_reserved_ratio': float(args.safe_reserved_ratio),
        'selected_batch_size': selected,
        'results': results,
    }
    _write_json(os.path.join(args.output_root, 'summary.json'), summary)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)

    generation_controller = subparsers.add_parser('generation-controller')
    generation_controller.add_argument('--config', required=True)
    generation_controller.add_argument('--output-root', required=True)
    generation_controller.add_argument('--repo-root', default='.')
    generation_controller.add_argument('--generation-prompt-batch-size', type=int, default=1)
    generation_controller.add_argument('--total-samples', type=int, default=512)
    generation_controller.add_argument('--generation-initial', type=int, default=512)
    generation_controller.add_argument('--generation-min', type=int, default=1)
    generation_controller.add_argument('--safe-reserved-ratio', type=float, default=0.90)
    generation_controller.add_argument('--seed', type=int, default=42)

    generation_worker = subparsers.add_parser('generation-worker')
    generation_worker.add_argument('--config', required=True)
    generation_worker.add_argument('--device', default='cuda')
    generation_worker.add_argument('--generation-prompt-batch-size', type=int, required=True)
    generation_worker.add_argument('--num-return-sequences', type=int, required=True)
    generation_worker.add_argument('--seed', type=int, default=42)
    generation_worker.add_argument('--output-path', required=True)

    reward_controller = subparsers.add_parser('single-reward-controller')
    reward_controller.add_argument('--config', required=True)
    reward_controller.add_argument('--output-root', required=True)
    reward_controller.add_argument('--repo-root', default='.')
    reward_controller.add_argument('--reward-name', required=True, choices=REWARD_NAMES)
    reward_controller.add_argument('--total-sequences', type=int, default=512)
    reward_controller.add_argument('--sequence-length', type=int, default=256)
    reward_controller.add_argument('--reward-max', type=int, default=4096)
    reward_controller.add_argument('--safe-reserved-ratio', type=float, default=0.90)
    reward_controller.add_argument('--seed', type=int, default=42)

    reward_worker = subparsers.add_parser('reward-worker')
    reward_worker.add_argument('--config', required=True)
    reward_worker.add_argument('--reward-name', required=True, choices=REWARD_NAMES)
    reward_worker.add_argument('--batch-size', type=int, required=True)
    reward_worker.add_argument('--total-sequences', type=int, required=True)
    reward_worker.add_argument('--sequence-length', type=int, default=256)
    reward_worker.add_argument('--seed', type=int, default=42)
    reward_worker.add_argument('--output-dir', required=True)

    args = parser.parse_args()
    configure_logging()
    if args.mode == 'generation-controller':
        _run_generation_controller(args)
        return
    if args.mode == 'generation-worker':
        _run_generation_worker(args)
        return
    if args.mode == 'single-reward-controller':
        _run_single_reward_controller(args)
        return
    if args.mode == 'reward-worker':
        _run_reward_worker(args)
        return
    raise ValueError(f'Unsupported mode {args.mode!r}')


if __name__ == '__main__':
    main()
