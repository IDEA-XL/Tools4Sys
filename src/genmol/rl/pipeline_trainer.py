import json
import logging
import math
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime

import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from genmol.rl.cpgrpo import (
    compute_clipped_grpo_loss,
    compute_grouped_advantages,
    split_tensor_dict,
)
from genmol.rl.lead_policy import LeadOptCpGRPOPolicy
from genmol.rl.lead_reward import compute_similarity
from genmol.rl.lead_specs import LeadOptSpec
from genmol.rl.pipeline_reward import (
    aggregate_selected_seed_downstream_base_rewards,
    combine_seed_rewards,
    merge_selected_seed_downstream_base_rewards,
    partition_selected_seed_entries,
    select_full_denovo_groups_for_lead,
)
from genmol.rl.policy import GenMolCpGRPOPolicy
from genmol.rl.reward import MolecularReward
from genmol.rl.specs import (
    deserialize_specs as deserialize_denovo_specs,
    expand_group_specs as expand_denovo_group_specs,
    sample_group_specs as sample_denovo_group_specs,
    serialize_specs as serialize_denovo_specs,
)
from genmol.rl.trainer import (
    TrainResult,
    _aggregate_scalar_list,
    _nanmean,
    _nanreduce,
    build_scheduler,
    ensure_exists,
    find_last_checkpoint,
    maybe_trim_checkpoints,
    write_jsonl,
)


logger = logging.getLogger(__name__)

@dataclass
class JointTrainConfig:
    denovo_init_ckpt_path: str
    lead_init_ckpt_path: str
    denovo_ref_ckpt_path: str | None = None
    lead_ref_ckpt_path: str | None = None
    output_dir: str | None = None
    overwrite_output_dir: bool = False
    distributed_backend: str = 'accelerator'
    seed: int = 42
    bf16: bool = True
    log_level: str = 'info'
    report_to: list[str] = field(default_factory=list)
    log_completions: bool = True
    logging_first_step: bool = True
    logging_steps: int = 10
    logging_strategy: str = 'steps'
    max_steps: int = 500
    save_strategy: str = 'steps'
    save_steps: int = 50
    save_total_limit: int = 5
    do_eval: bool = False
    scale_rewards: bool = False
    gradient_accumulation_steps: int = 1
    num_iterations: int = 1
    random_masking: bool = True
    ddp_broadcast_buffers: bool = True

    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 0.2
    lr_scheduler_type: str = 'cosine_with_min_lr'
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {'min_lr_rate': 0.1})
    warmup_ratio: float = 0.0001

    denovo_num_generations: int = 32
    denovo_per_device_train_batch_size: int = 8
    denovo_generation_batch_size: int = 8
    denovo_generation_temperature: float = 1.0
    denovo_randomness: float = 0.3
    denovo_min_add_len: int = 60
    denovo_max_completion_length: int | None = None
    denovo_learning_rate: float = 5e-5
    denovo_beta: float = 0.005
    denovo_epsilon: float = 0.5
    denovo_sync_ref_model: bool = True
    denovo_ref_model_sync_steps: int = 64
    denovo_ref_model_mixup_alpha: float = 0.6
    denovo_gradient_checkpointing: bool = True
    denovo_gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {'use_reentrant': False})

    lead_num_generations: int = 32
    lead_per_device_train_batch_size: int | None = None
    lead_generation_batch_size: int = 64
    lead_generation_temperature: float = 1.0
    lead_randomness: float = 0.3
    lead_min_seed_len: int = 60
    lead_rescore_chunk_size: int = 64
    lead_learning_rate: float = 5e-5
    lead_beta: float = 0.005
    lead_epsilon: float = 0.5
    lead_sync_ref_model: bool = True
    lead_ref_model_sync_steps: int = 64
    lead_ref_model_mixup_alpha: float = 0.6
    lead_gradient_checkpointing: bool = True
    lead_gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {'use_reentrant': False})

    sim_weight: float = 1.0
    denovo_reward_alpha: float = 0.7
    downstream_topk: int = 2


def load_config(path):
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    config = JointTrainConfig(**raw)
    if config.distributed_backend not in {'accelerator', 'process_group_ddp'}:
        raise ValueError(
            "distributed_backend must be one of {'accelerator', 'process_group_ddp'}"
        )
    if config.denovo_ref_ckpt_path is None:
        config.denovo_ref_ckpt_path = config.denovo_init_ckpt_path
    if config.lead_ref_ckpt_path is None:
        config.lead_ref_ckpt_path = config.lead_init_ckpt_path
    if config.logging_strategy != 'steps':
        raise ValueError('Only logging_strategy=steps is supported')
    if config.save_strategy != 'steps':
        raise ValueError('Only save_strategy=steps is supported')
    if config.do_eval:
        raise ValueError('Joint pipeline trainer does not support do_eval yet')
    if config.gradient_accumulation_steps <= 0:
        raise ValueError('gradient_accumulation_steps must be positive')
    if config.num_iterations <= 0:
        raise ValueError('num_iterations must be positive')
    if config.denovo_num_generations <= 1:
        raise ValueError('denovo_num_generations must be greater than 1')
    if config.lead_num_generations <= 1:
        raise ValueError('lead_num_generations must be greater than 1')
    if config.denovo_per_device_train_batch_size <= 0:
        raise ValueError('denovo_per_device_train_batch_size must be positive')
    if config.denovo_generation_batch_size <= 0:
        raise ValueError('denovo_generation_batch_size must be positive')
    if config.lead_per_device_train_batch_size is None:
        config.lead_per_device_train_batch_size = config.denovo_per_device_train_batch_size * config.lead_num_generations
    if config.lead_per_device_train_batch_size <= 0:
        raise ValueError('lead_per_device_train_batch_size must be positive')
    if config.lead_generation_batch_size <= 0:
        raise ValueError('lead_generation_batch_size must be positive')
    if config.lead_rescore_chunk_size <= 0:
        raise ValueError('lead_rescore_chunk_size must be positive')
    if config.downstream_topk <= 0:
        raise ValueError('downstream_topk must be positive')
    if not 0.0 <= config.denovo_ref_model_mixup_alpha <= 1.0:
        raise ValueError('denovo_ref_model_mixup_alpha must be in [0, 1]')
    if not 0.0 <= config.lead_ref_model_mixup_alpha <= 1.0:
        raise ValueError('lead_ref_model_mixup_alpha must be in [0, 1]')
    if not 0.0 <= config.denovo_reward_alpha <= 1.0:
        raise ValueError('denovo_reward_alpha must be in [0, 1]')
    return config


def resolve_output_dir(config, config_path):
    if config.output_dir is not None:
        return config.output_dir

    cluster_root = '/public/home/xinwuye/ai4s-tool-joint-train'
    if os.path.isdir(cluster_root):
        base_dir = os.path.join(cluster_root, 'runs', 'cpgrpo_pipeline')
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        base_dir = os.path.join(repo_root, 'runs', 'cpgrpo_pipeline')

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(base_dir, f'{config_name}_{timestamp}')


class JointCpGRPOTrainer:
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=config.ddp_broadcast_buffers)
        self.accelerator = Accelerator(
            split_batches=True,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with=config.report_to if config.report_to else None,
            mixed_precision='bf16' if config.bf16 else 'no',
            kwargs_handlers=[ddp_kwargs],
        )
        deepspeed_plugin = getattr(self.accelerator.state, 'deepspeed_plugin', None)
        self.device = self.accelerator.device
        self.world_size = self.accelerator.num_processes
        self.denovo_micro_batch_size = config.denovo_per_device_train_batch_size
        self.denovo_local_sample_count = self.denovo_micro_batch_size * config.gradient_accumulation_steps
        self.denovo_global_sample_count = self.denovo_local_sample_count * self.world_size
        if self.denovo_global_sample_count % config.denovo_num_generations != 0:
            raise ValueError(
                'global de novo seed batch must be divisible by denovo_num_generations: '
                f'{self.denovo_global_sample_count} vs {config.denovo_num_generations}'
            )
        self.denovo_num_groups_global = self.denovo_global_sample_count // config.denovo_num_generations
        self.lead_micro_batch_size = int(config.lead_per_device_train_batch_size)
        self.lead_local_sample_count = self.lead_micro_batch_size * config.gradient_accumulation_steps
        if self.lead_local_sample_count % config.lead_num_generations != 0:
            raise ValueError(
                'lead local sample count must be divisible by lead_num_generations: '
                f'{self.lead_local_sample_count} vs {config.lead_num_generations}'
            )
        self.lead_num_groups_local = self.lead_local_sample_count // config.lead_num_generations
        self.lead_global_sample_count = self.lead_local_sample_count * self.world_size
        if self.lead_global_sample_count % config.lead_num_generations != 0:
            raise ValueError(
                'global lead train batch size must be divisible by lead_num_generations: '
                f'{self.lead_global_sample_count} vs {config.lead_num_generations}'
            )
        self.lead_num_groups_global = self.lead_global_sample_count // config.lead_num_generations
        if self.lead_num_groups_global % config.denovo_num_generations != 0:
            raise ValueError(
                'lead global group count must be divisible by denovo_num_generations so whole de novo groups can '
                f'be sampled for lead: {self.lead_num_groups_global} vs {config.denovo_num_generations}'
            )
        self.selected_denovo_group_count = self.lead_num_groups_global // config.denovo_num_generations

        ensure_exists(config.denovo_init_ckpt_path, 'de novo init checkpoint')
        ensure_exists(config.denovo_ref_ckpt_path, 'de novo reference checkpoint')
        ensure_exists(config.lead_init_ckpt_path, 'lead init checkpoint')
        ensure_exists(config.lead_ref_ckpt_path, 'lead reference checkpoint')

        set_seed(config.seed, device_specific=True)

        self.denovo_policy = GenMolCpGRPOPolicy(
            checkpoint_path=config.denovo_init_ckpt_path,
            device=self.device,
            bf16=config.bf16,
            trainable=True,
        )
        self.denovo_reference = GenMolCpGRPOPolicy(
            checkpoint_path=config.denovo_ref_ckpt_path,
            device=self.device,
            bf16=config.bf16,
            trainable=False,
        )
        self.lead_policy = LeadOptCpGRPOPolicy(
            checkpoint_path=config.lead_init_ckpt_path,
            device=self.device,
            bf16=config.bf16,
            trainable=True,
            score_chunk_size=config.lead_rescore_chunk_size,
        )
        self.lead_reference = LeadOptCpGRPOPolicy(
            checkpoint_path=config.lead_ref_ckpt_path,
            device=self.device,
            bf16=config.bf16,
            trainable=False,
            score_chunk_size=config.lead_rescore_chunk_size,
        )

        if config.denovo_gradient_checkpointing:
            self.denovo_policy.enable_gradient_checkpointing(config.denovo_gradient_checkpointing_kwargs)
        if config.lead_gradient_checkpointing:
            self.lead_policy.enable_gradient_checkpointing(config.lead_gradient_checkpointing_kwargs)
        self.denovo_policy.train()
        self.lead_policy.train()

        denovo_optimizer = torch.optim.AdamW(
            self.denovo_policy.trainable_parameters(),
            lr=config.denovo_learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
        lead_optimizer = torch.optim.AdamW(
            self.lead_policy.trainable_parameters(),
            lr=config.lead_learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
        denovo_scheduler = build_scheduler(denovo_optimizer, config)
        lead_scheduler = build_scheduler(lead_optimizer, config)

        (
            self.denovo_policy.model.backbone,
            self.denovo_optimizer,
            self.denovo_scheduler,
            self.lead_policy.model.backbone,
            self.lead_optimizer,
            self.lead_scheduler,
        ) = self.accelerator.prepare(
            self.denovo_policy.model.backbone,
            denovo_optimizer,
            denovo_scheduler,
            self.lead_policy.model.backbone,
            lead_optimizer,
            lead_scheduler,
        )

        self.base_reward_model = MolecularReward()
        self.metrics_path = os.path.join(output_dir, 'metrics.jsonl')
        self.text_logs_path = os.path.join(output_dir, 'completions.jsonl')
        self.state_path = os.path.join(output_dir, 'trainer_state.json')

        self.global_step = 0
        self.generation_cycle_idx = 0
        self._buffer_iteration = 0
        self._buffered_inputs = None
        self._buffer_metadata = None
        self._last_rollout_metrics = None
        self._metrics = defaultdict(list)
        self._textual_logs = []
        self._last_train_metrics = None

        if config.report_to:
            init_kwargs = {}
            if 'wandb' in config.report_to:
                init_kwargs['wandb'] = {'name': os.path.basename(output_dir)}
            self.accelerator.init_trackers(
                'genmol-cpgrpo-pipeline',
                config=asdict(config),
                init_kwargs=init_kwargs or None,
            )

        logger.info(
            'process_index=%s device=%s world_size=%s denovo_micro_batch_size=%s denovo_local_sample_count=%s '
            'lead_micro_batch_size=%s lead_local_sample_count=%s lead_num_generations=%s '
            'lead_num_groups_global=%s gradient_accumulation_steps=%s num_iterations=%s reward_workers=%s',
            self.accelerator.process_index,
            self.device,
            self.world_size,
            self.denovo_micro_batch_size,
            self.denovo_local_sample_count,
            self.lead_micro_batch_size,
            self.lead_local_sample_count,
            self.config.lead_num_generations,
            self.lead_num_groups_global,
            self.config.gradient_accumulation_steps,
            self.config.num_iterations,
            self.base_reward_model.num_workers,
        )

    def _broadcast_denovo_specs(self, group_specs):
        payload = [None]
        if self.accelerator.is_main_process:
            payload[0] = serialize_denovo_specs(expand_denovo_group_specs(group_specs, self.config.denovo_num_generations))
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(payload, src=0)
        return deserialize_denovo_specs(payload[0])

    def _all_gather_objects(self, payload):
        if not dist.is_available() or not dist.is_initialized():
            return [payload]
        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, payload)
        return gathered

    def _gather_variable_scalars(self, values):
        local_values = [float(item) for item in values]
        gathered = self._all_gather_objects(local_values)
        local_start = sum(len(item) for item in gathered[:self.accelerator.process_index])
        flat = [value for shard in gathered for value in shard]
        if not flat:
            return torch.empty((0,), device=self.device, dtype=torch.float32), local_start
        return torch.tensor(flat, device=self.device, dtype=torch.float32), local_start

    def _sample_mask_seeds(self):
        if self.config.random_masking:
            return torch.randint(0, 2**12, (self.config.num_iterations,), device=self.device).tolist()
        return [42] * self.config.num_iterations

    def _score_lead_records(self, seed_smiles_list, candidate_smiles_list):
        base_records = self.base_reward_model.score(candidate_smiles_list)
        combined = []
        for seed_smiles, base_record in zip(seed_smiles_list, base_records):
            if not base_record.is_valid or base_record.smiles is None:
                combined.append(
                    {
                        'reward': -1.0,
                        'base_reward': -1.0,
                        'sim': None,
                        'record': base_record,
                        'seed_smiles': seed_smiles,
                    }
                )
                continue
            sim = compute_similarity(seed_smiles, base_record.smiles)
            if sim is None:
                combined.append(
                    {
                        'reward': -1.0,
                        'base_reward': -1.0,
                        'sim': None,
                        'record': base_record,
                        'seed_smiles': seed_smiles,
                    }
                )
                continue
            combined.append(
                {
                    'reward': float(base_record.reward) + self.config.sim_weight * float(sim),
                    'base_reward': float(base_record.reward),
                    'sim': float(sim),
                    'record': base_record,
                    'seed_smiles': seed_smiles,
                }
            )
        return combined

    def _build_lead_specs(self, seed_smiles_list, cycle_seed):
        rng = random.Random(cycle_seed + 7919 + self.accelerator.process_index * 1000)
        lead_specs = []
        for seed_smiles in seed_smiles_list:
            for _ in range(self.config.lead_num_generations):
                lead_specs.append(
                    LeadOptSpec(
                        seed_smiles=seed_smiles,
                        mutation_seed=rng.randrange(2**31),
                        generation_temperature=self.config.lead_generation_temperature,
                        randomness=self.config.lead_randomness,
                        min_seed_len=self.config.lead_min_seed_len,
                    )
                )
        return lead_specs

    def _select_lead_seed_payload(self, denovo_reward_records, local_seed_base_rewards, cycle_seed, local_start):
        local_seed_entries = []
        for record, base_reward in zip(denovo_reward_records, local_seed_base_rewards.tolist()):
            local_seed_entries.append(
                {
                    'is_valid': bool(record.is_valid and record.smiles is not None),
                    'smiles': record.smiles,
                    'base_reward': float(base_reward),
                }
            )
        gathered_seed_entries = self._all_gather_objects(local_seed_entries)
        payload = [None]
        if self.accelerator.is_main_process:
            try:
                global_seed_entries = [item for shard in gathered_seed_entries for item in shard]
                if len(global_seed_entries) != self.denovo_global_sample_count:
                    raise ValueError(
                        'Global de novo seed entry count mismatch: '
                        f'{len(global_seed_entries)} vs {self.denovo_global_sample_count}'
                    )
                selected_group_indices, selected_seed_entries, selected_seed_mask = select_full_denovo_groups_for_lead(
                    seed_entries=global_seed_entries,
                    denovo_group_size=self.config.denovo_num_generations,
                    lead_num_seed_groups=self.lead_num_groups_global,
                    seed=cycle_seed + 15485863,
                )
                lead_seed_shards = partition_selected_seed_entries(selected_seed_entries, self.world_size)
                payload[0] = {
                    'error': None,
                    'selected_group_indices': selected_group_indices,
                    'selected_seed_mask': selected_seed_mask,
                    'lead_seed_shards': lead_seed_shards,
                    'default_downstream_base_rewards': [item['base_reward'] for item in global_seed_entries],
                }
            except Exception as exc:
                payload[0] = {'error': str(exc)}
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(payload, src=0)
        selection_payload = payload[0]
        if selection_payload is None:
            raise RuntimeError('Lead seed selection payload was not broadcast')
        if selection_payload.get('error') is not None:
            raise ValueError(selection_payload['error'])
        local_end = local_start + len(denovo_reward_records)
        local_selected_seed_mask = torch.tensor(
            selection_payload['selected_seed_mask'][local_start:local_end],
            device=self.device,
            dtype=torch.bool,
        )
        local_selected_seed_entries = selection_payload['lead_seed_shards'][self.accelerator.process_index]
        if len(local_selected_seed_entries) != self.lead_num_groups_local:
            raise ValueError(
                'Per-rank selected lead seed count mismatch: '
                f'{len(local_selected_seed_entries)} vs {self.lead_num_groups_local}'
            )
        return selection_payload, local_selected_seed_entries, local_selected_seed_mask

    def _record_text_logs(self, rows):
        if not self.config.log_completions or not rows:
            return
        gathered = self._all_gather_objects(rows)
        if self.accelerator.is_main_process:
            for shard in gathered:
                self._textual_logs.extend(shard)

    def _flush_text_logs(self):
        if not self.config.log_completions:
            self._textual_logs = []
            return
        if not self.accelerator.is_main_process:
            self._textual_logs = []
            return
        for row in self._textual_logs:
            write_jsonl(self.text_logs_path, row)
        self._textual_logs = []

    def _append_metric(self, key, value):
        self._metrics[key].append(float(value))

    def _record_rollout_metrics(self, metrics):
        self._last_rollout_metrics = dict(metrics)
        for key, value in metrics.items():
            self._append_metric(key, value)

    def _record_stage_loss_metrics(self, prefix, step_metrics):
        gathered_ratio = self.accelerator.gather_for_metrics(step_metrics['ratio_mean'].detach().reshape(1))
        self._append_metric(f'{prefix}/ratio_mean', torch.nanmean(gathered_ratio).item())

        gathered_low = self.accelerator.gather_for_metrics(step_metrics['clip_ratio_low_mean'].detach().reshape(1))
        self._append_metric(f'{prefix}/clip_ratio/low_mean', torch.nanmean(gathered_low).item())
        self._append_metric(f'{prefix}/clip_ratio/low_min', _nanreduce(gathered_low, mode='min'))

        gathered_high = self.accelerator.gather_for_metrics(step_metrics['clip_ratio_high_mean'].detach().reshape(1))
        self._append_metric(f'{prefix}/clip_ratio/high_mean', torch.nanmean(gathered_high).item())
        self._append_metric(f'{prefix}/clip_ratio/high_max', _nanreduce(gathered_high, mode='max'))

        gathered_region = self.accelerator.gather_for_metrics(step_metrics['clip_ratio_region_mean'].detach().reshape(1))
        self._append_metric(f'{prefix}/clip_ratio/region_mean', torch.nanmean(gathered_region).item())

        if 'kl_mean' in step_metrics:
            gathered_kl = self.accelerator.gather_for_metrics(step_metrics['kl_mean'].detach().reshape(1))
            self._append_metric(f'{prefix}/kl_mean', torch.nanmean(gathered_kl).item())

    def _consume_logged_metrics(self):
        if self._buffer_metadata is None:
            buffer_cycle = self.generation_cycle_idx - 1
        else:
            buffer_cycle = int(self._buffer_metadata['buffer_cycle'])
        metrics = {'step': self.global_step, 'buffer_cycle': buffer_cycle}
        rollout_metrics = self._last_rollout_metrics or {}
        for key, value in rollout_metrics.items():
            values = self._metrics.get(key, [])
            if values:
                metrics[key] = _aggregate_scalar_list(values)
            else:
                metrics[key] = float(value)
        for key, values in list(self._metrics.items()):
            if key in metrics:
                continue
            metrics[key] = _aggregate_scalar_list(values)
        self._metrics = defaultdict(list)
        return metrics

    def _empty_lead_inputs(self, mask_seeds):
        return {
            'input_ids': None,
            'completion_mask': None,
            'advantages': torch.empty((0,), device=self.device, dtype=torch.float32),
            'old_per_token_logps': None,
            'ref_per_token_logps': None,
            'mask_seeds': list(mask_seeds),
            'has_samples': False,
        }

    def _split_lead_inputs(self, inputs):
        if self.config.gradient_accumulation_steps == 1:
            return [inputs]

        if not inputs['has_samples']:
            return [self._empty_lead_inputs(inputs['mask_seeds']) for _ in range(self.config.gradient_accumulation_steps)]

        tensor_keys = {
            'input_ids': inputs['input_ids'],
            'completion_mask': inputs['completion_mask'],
            'advantages': inputs['advantages'],
            'old_per_token_logps': inputs['old_per_token_logps'],
            'ref_per_token_logps': inputs['ref_per_token_logps'],
        }
        lead_chunks = split_tensor_dict(tensor_keys, self.config.gradient_accumulation_steps)
        for chunk in lead_chunks:
            chunk['mask_seeds'] = list(inputs['mask_seeds'])
            chunk['has_samples'] = True
        return lead_chunks

    def _split_pipeline_inputs(self, accumulated_local_batch):
        denovo_chunks = split_tensor_dict(
            accumulated_local_batch['denovo'],
            self.config.gradient_accumulation_steps,
        )
        lead_chunks = self._split_lead_inputs(accumulated_local_batch['lead'])
        if len(denovo_chunks) != len(lead_chunks):
            raise ValueError(
                'denovo chunk count and lead chunk count must match: '
                f'{len(denovo_chunks)} vs {len(lead_chunks)}'
            )
        return {
            'denovo': denovo_chunks,
            'lead': lead_chunks,
        }

    def _prepare_inputs(self, mode='train'):
        if self._buffered_inputs is None or self._buffer_iteration >= self.config.num_iterations:
            accumulated_local_batch = self._generate_and_score_pipeline(mode=mode)
            self._record_rollout_metrics(accumulated_local_batch['metrics'])
            self._buffered_inputs = self._split_pipeline_inputs(accumulated_local_batch)
            self._buffer_metadata = {'buffer_cycle': int(accumulated_local_batch['buffer_cycle'])}
            self._buffer_iteration = 0

        return self._buffered_inputs, self._buffer_iteration

    def _clear_rollout_buffer(self):
        self._buffered_inputs = None
        self._buffer_iteration = 0

    def _log_metrics(self, split, metrics):
        if self.accelerator.is_main_process:
            logger.info(
                '%s step=%s denovo/reward_mean=%.6f denovo/base_reward_mean=%.6f '
                'denovo/downstream_base_mean=%.6f denovo/valid_fraction=%.6f '
                'lead/reward_mean=%.6f lead/base_reward_mean=%.6f lead/sim_mean=%.6f '
                'lead/valid_fraction=%.6f denovo/ratio_mean=%.6f lead/ratio_mean=%.6f '
                'denovo/grad_norm=%.6f lead/grad_norm=%.6f denovo/lr=%.8f lead/lr=%.8f%s%s',
                split,
                metrics['step'],
                metrics['denovo/reward_mean'],
                metrics['denovo/base_reward_mean'],
                metrics['denovo/downstream_base_mean'],
                metrics['denovo/valid_fraction'],
                metrics['lead/reward_mean'],
                metrics['lead/base_reward_mean'],
                metrics['lead/rewards/sim_mean'],
                metrics['lead/valid_fraction'],
                metrics['denovo/ratio_mean'],
                metrics['lead/ratio_mean'],
                metrics['denovo/grad_norm'],
                metrics['lead/grad_norm'],
                metrics['denovo/lr'],
                metrics['lead/lr'],
                '' if 'denovo/kl_mean' not in metrics else f" denovo/kl_mean={metrics['denovo/kl_mean']:.6f}",
                '' if 'lead/kl_mean' not in metrics else f" lead/kl_mean={metrics['lead/kl_mean']:.6f}",
            )
            write_jsonl(self.metrics_path, metrics)
            with open(self.state_path, 'w') as handle:
                json.dump(
                    {
                        'global_step': self.global_step,
                        'generation_cycle_idx': self.generation_cycle_idx,
                        'last_metrics': metrics,
                    },
                    handle,
                    sort_keys=True,
                    indent=2,
                )
        if self.config.report_to:
            self.accelerator.log(metrics, step=self.global_step)
        self._flush_text_logs()

    def _score_denovo_reference(self, rollout, mask_seeds):
        if self.config.denovo_beta == 0.0:
            return None
        prompt_completion_ids = torch.cat([rollout.prompt_ids, rollout.completion_ids], dim=1)
        expanded_ids = prompt_completion_ids.unsqueeze(0).expand(self.config.num_iterations, -1, -1)
        return self.denovo_reference.per_token_logps(
            input_ids=expanded_ids,
            logits_to_keep=rollout.completion_ids.size(1),
            completion_mask=rollout.completion_mask,
            mask_seeds=mask_seeds,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            requires_grad=False,
        )

    def _score_lead_reference(self, rollout, mask_seeds):
        if self.config.lead_beta == 0.0 or rollout is None:
            return None
        reference_input_ids = rollout.input_ids.detach().clone()
        reference_completion_mask = rollout.completion_mask.detach().clone()
        return self.lead_reference.per_token_logps(
            input_ids=reference_input_ids.unsqueeze(0).expand(self.config.num_iterations, -1, -1),
            completion_mask=reference_completion_mask,
            mask_seeds=mask_seeds,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            requires_grad=False,
        )

    def _generate_and_score_pipeline(self, mode):
        buffer_cycle = self.generation_cycle_idx
        cycle_seed = self.config.seed + buffer_cycle * 10000
        if self.accelerator.is_main_process:
            group_specs = sample_denovo_group_specs(
                num_groups=self.denovo_num_groups_global,
                generation_temperature=self.config.denovo_generation_temperature,
                randomness=self.config.denovo_randomness,
                min_add_len=self.config.denovo_min_add_len,
                seed=cycle_seed,
                max_completion_length=self.config.denovo_max_completion_length,
            )
        else:
            group_specs = []

        expanded_specs = self._broadcast_denovo_specs(group_specs)
        local_start = self.accelerator.process_index * self.denovo_local_sample_count
        local_end = (self.accelerator.process_index + 1) * self.denovo_local_sample_count
        local_denovo_specs = expanded_specs[local_start:local_end]
        denovo_rollout_seed = cycle_seed + self.accelerator.process_index * 1000
        denovo_rollout = self.denovo_policy.rollout_specs(
            specs=local_denovo_specs,
            generation_batch_size=self.config.denovo_generation_batch_size,
            seed=denovo_rollout_seed,
        )
        denovo_reward_records = self.base_reward_model.score(denovo_rollout.smiles)
        local_seed_base_rewards = torch.tensor(
            [record.reward for record in denovo_reward_records],
            device=self.device,
            dtype=torch.float32,
        )

        selection_payload, local_selected_seed_entries, local_selected_seed_mask = self._select_lead_seed_payload(
            denovo_reward_records=denovo_reward_records,
            local_seed_base_rewards=local_seed_base_rewards,
            cycle_seed=cycle_seed,
            local_start=local_start,
        )
        local_selected_seed_smiles = [item['smiles'] for item in local_selected_seed_entries]
        lead_specs = self._build_lead_specs(local_selected_seed_smiles, cycle_seed)
        if len(lead_specs) != self.lead_local_sample_count:
            raise ValueError(
                'Local lead rollout spec count mismatch: '
                f'{len(lead_specs)} vs {self.lead_local_sample_count}'
            )

        lead_rollout_seed = cycle_seed + 500000 + self.accelerator.process_index * 1000
        lead_rollout = self.lead_policy.rollout_specs(
            specs=lead_specs,
            generation_batch_size=self.config.lead_generation_batch_size,
            seed=lead_rollout_seed,
        )
        lead_records = self._score_lead_records(lead_rollout.seed_smiles, lead_rollout.smiles)
        local_lead_rewards = [item['reward'] for item in lead_records]
        if len(local_lead_rewards) != self.lead_local_sample_count:
            raise ValueError(
                'Local lead reward count mismatch: '
                f'{len(local_lead_rewards)} vs {self.lead_local_sample_count}'
            )
        local_lead_base_rewards_tensor = torch.tensor(
            [item['base_reward'] for item in lead_records],
            device=self.device,
            dtype=torch.float32,
        )
        local_selected_downstream_rewards = aggregate_selected_seed_downstream_base_rewards(
            selected_seed_entries=local_selected_seed_entries,
            lead_base_rewards=local_lead_base_rewards_tensor,
            lead_num_generations=self.config.lead_num_generations,
            downstream_topk=self.config.downstream_topk,
        )
        gathered_selected_downstream_rewards = self._all_gather_objects(local_selected_downstream_rewards)
        merged_downstream_base_rewards = merge_selected_seed_downstream_base_rewards(
            default_downstream_base_rewards=selection_payload['default_downstream_base_rewards'],
            selected_seed_rewards=[
                item for shard in gathered_selected_downstream_rewards for item in shard
            ],
            expected_selected_seed_count=self.lead_num_groups_global,
        )
        global_downstream_base_rewards = torch.tensor(
            merged_downstream_base_rewards,
            device=self.device,
            dtype=torch.float32,
        )
        local_downstream_base_rewards = global_downstream_base_rewards[local_start:local_end]
        local_denovo_rewards = combine_seed_rewards(
            seed_base_rewards=local_seed_base_rewards,
            downstream_base_rewards=local_downstream_base_rewards,
            alpha=self.config.denovo_reward_alpha,
        )

        global_denovo_rewards = self.accelerator.gather(local_denovo_rewards).detach()
        global_denovo_advantages, global_denovo_reward_std, denovo_zero_std_ratio = compute_grouped_advantages(
            rewards=global_denovo_rewards,
            num_generations=self.config.denovo_num_generations,
            scale_rewards=self.config.scale_rewards,
        )
        local_denovo_advantages = global_denovo_advantages[local_start:local_end].to(self.device)

        global_lead_rewards, local_lead_start = self._gather_variable_scalars(local_lead_rewards)
        if global_lead_rewards.numel() > 0:
            global_lead_advantages, global_lead_reward_std, lead_zero_std_ratio = compute_grouped_advantages(
                rewards=global_lead_rewards,
                num_generations=self.config.lead_num_generations,
                scale_rewards=self.config.scale_rewards,
            )
            local_lead_advantages = global_lead_advantages[
                local_lead_start:local_lead_start + len(local_lead_rewards)
            ].to(self.device)
        else:
            global_lead_reward_std = torch.empty((0,), device=self.device)
            local_lead_advantages = torch.empty((0,), device=self.device)
            lead_zero_std_ratio = float('nan')

        denovo_mask_seeds = self._sample_mask_seeds()
        lead_mask_seeds = self._sample_mask_seeds()

        prompt_completion_ids = torch.cat([denovo_rollout.prompt_ids, denovo_rollout.completion_ids], dim=1)
        denovo_expanded_ids = prompt_completion_ids.unsqueeze(0).expand(self.config.num_iterations, -1, -1)
        if self.config.num_iterations > 1:
            denovo_old_per_token_logps = self.denovo_policy.per_token_logps(
                input_ids=denovo_expanded_ids,
                logits_to_keep=denovo_rollout.completion_ids.size(1),
                completion_mask=denovo_rollout.completion_mask,
                mask_seeds=denovo_mask_seeds,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                requires_grad=False,
            )
        else:
            denovo_old_per_token_logps = None
        denovo_ref_per_token_logps = self._score_denovo_reference(denovo_rollout, denovo_mask_seeds)

        if lead_rollout is not None and self.config.num_iterations > 1:
            lead_old_per_token_logps = self.lead_policy.per_token_logps(
                input_ids=lead_rollout.input_ids.unsqueeze(0).expand(self.config.num_iterations, -1, -1),
                completion_mask=lead_rollout.completion_mask,
                mask_seeds=lead_mask_seeds,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                requires_grad=False,
            )
        else:
            lead_old_per_token_logps = None
        lead_ref_per_token_logps = self._score_lead_reference(lead_rollout, lead_mask_seeds)

        gathered_denovo_valid = self.accelerator.gather(
            torch.tensor([float(record.is_valid) for record in denovo_reward_records], device=self.device)
        )
        gathered_denovo_alert = self.accelerator.gather(
            torch.tensor([float(record.alert_hit) for record in denovo_reward_records], device=self.device)
        )
        gathered_denovo_qed = self.accelerator.gather(
            torch.tensor(
                [float('nan') if record.qed is None else float(record.qed) for record in denovo_reward_records],
                device=self.device,
            )
        )
        gathered_denovo_sa = self.accelerator.gather(
            torch.tensor(
                [float('nan') if record.sa is None else float(record.sa) for record in denovo_reward_records],
                device=self.device,
            )
        )
        gathered_denovo_sa_score = self.accelerator.gather(
            torch.tensor(
                [float('nan') if record.sa_score is None else float(record.sa_score) for record in denovo_reward_records],
                device=self.device,
            )
        )
        gathered_denovo_soft = self.accelerator.gather(
            torch.tensor(
                [float('nan') if record.soft_reward is None else float(record.soft_reward) for record in denovo_reward_records],
                device=self.device,
            )
        )
        gathered_denovo_completion_lengths = self.accelerator.gather(denovo_rollout.completion_mask.sum(dim=1).float())
        gathered_seed_base_rewards = self.accelerator.gather(local_seed_base_rewards.detach())
        gathered_downstream_base_rewards = self.accelerator.gather(local_downstream_base_rewards.detach())
        gathered_denovo_advantages = self.accelerator.gather(local_denovo_advantages.detach())
        gathered_selected_seed_mask = self.accelerator.gather(local_selected_seed_mask.float())

        all_lead_valid_lists = self._all_gather_objects([float(item['record'].is_valid) for item in lead_records])
        all_lead_alert_lists = self._all_gather_objects([float(item['record'].alert_hit) for item in lead_records])
        all_lead_qed_lists = self._all_gather_objects(
            [float('nan') if item['record'].qed is None else float(item['record'].qed) for item in lead_records]
        )
        all_lead_sa_lists = self._all_gather_objects(
            [float('nan') if item['record'].sa is None else float(item['record'].sa) for item in lead_records]
        )
        all_lead_sa_score_lists = self._all_gather_objects(
            [float('nan') if item['record'].sa_score is None else float(item['record'].sa_score) for item in lead_records]
        )
        all_lead_soft_lists = self._all_gather_objects(
            [float('nan') if item['record'].soft_reward is None else float(item['record'].soft_reward) for item in lead_records]
        )
        all_lead_sim_lists = self._all_gather_objects(
            [float('nan') if item['sim'] is None else float(item['sim']) for item in lead_records]
        )
        all_lead_base_lists = self._all_gather_objects([float(item['base_reward']) for item in lead_records])
        all_lead_lengths_lists = self._all_gather_objects(
            [] if lead_rollout is None else lead_rollout.completion_mask.sum(dim=1).detach().cpu().tolist()
        )
        all_lead_adv_lists = self._all_gather_objects(local_lead_advantages.detach().cpu().tolist())

        global_lead_valid = torch.tensor(
            [value for shard in all_lead_valid_lists for value in shard],
            device=self.device,
            dtype=torch.float32,
        )
        global_lead_alert = torch.tensor(
            [value for shard in all_lead_alert_lists for value in shard],
            device=self.device,
            dtype=torch.float32,
        )
        global_lead_qed = torch.tensor([value for shard in all_lead_qed_lists for value in shard], device=self.device)
        global_lead_sa = torch.tensor([value for shard in all_lead_sa_lists for value in shard], device=self.device)
        global_lead_sa_score = torch.tensor(
            [value for shard in all_lead_sa_score_lists for value in shard],
            device=self.device,
        )
        global_lead_soft = torch.tensor(
            [value for shard in all_lead_soft_lists for value in shard],
            device=self.device,
        )
        global_lead_sim = torch.tensor([value for shard in all_lead_sim_lists for value in shard], device=self.device)
        global_lead_base = torch.tensor(
            [value for shard in all_lead_base_lists for value in shard],
            device=self.device,
        )
        global_lead_lengths = torch.tensor(
            [value for shard in all_lead_lengths_lists for value in shard],
            device=self.device,
            dtype=torch.float32,
        )
        global_lead_advantages_logged = torch.tensor(
            [value for shard in all_lead_adv_lists for value in shard],
            device=self.device,
        )

        rollout_metrics = {
            'denovo/reward_mean': global_denovo_rewards.mean().item(),
            'denovo/reward_std': global_denovo_reward_std.mean().item(),
            'denovo/advantage_mean': gathered_denovo_advantages.mean().item(),
            'denovo/zero_std_ratio': denovo_zero_std_ratio,
            'denovo/completion_length': gathered_denovo_completion_lengths.mean().item(),
            'denovo/valid_fraction': gathered_denovo_valid.mean().item(),
            'denovo/alert_hit_fraction': gathered_denovo_alert.mean().item(),
            'denovo/invalid_fraction': 1.0 - gathered_denovo_valid.mean().item(),
            'denovo/base_reward_mean': gathered_seed_base_rewards.mean().item(),
            'denovo/downstream_base_mean': gathered_downstream_base_rewards.mean().item(),
            'denovo/selected_seed_fraction': gathered_selected_seed_mask.mean().item(),
            'denovo/rewards/qed_mean': _nanmean(gathered_denovo_qed),
            'denovo/rewards/sa_mean': _nanmean(gathered_denovo_sa),
            'denovo/rewards/sa_score_mean': _nanmean(gathered_denovo_sa_score),
            'denovo/rewards/soft_mean': _nanmean(gathered_denovo_soft),
        }
        if global_lead_rewards.numel() > 0:
            rollout_metrics.update(
                {
                    'lead/reward_mean': global_lead_rewards.mean().item(),
                    'lead/reward_std': global_lead_reward_std.mean().item(),
                    'lead/advantage_mean': global_lead_advantages_logged.mean().item(),
                    'lead/zero_std_ratio': lead_zero_std_ratio,
                    'lead/completion_length': global_lead_lengths.mean().item(),
                    'lead/valid_fraction': global_lead_valid.mean().item(),
                    'lead/alert_hit_fraction': global_lead_alert.mean().item(),
                    'lead/invalid_fraction': 1.0 - global_lead_valid.mean().item(),
                    'lead/base_reward_mean': global_lead_base.mean().item(),
                    'lead/rewards/qed_mean': _nanmean(global_lead_qed),
                    'lead/rewards/sa_mean': _nanmean(global_lead_sa),
                    'lead/rewards/sa_score_mean': _nanmean(global_lead_sa_score),
                    'lead/rewards/soft_mean': _nanmean(global_lead_soft),
                    'lead/rewards/sim_mean': _nanmean(global_lead_sim),
                }
            )
        else:
            rollout_metrics.update(
                {
                    'lead/reward_mean': float('nan'),
                    'lead/reward_std': float('nan'),
                    'lead/advantage_mean': float('nan'),
                    'lead/zero_std_ratio': float('nan'),
                    'lead/completion_length': float('nan'),
                    'lead/valid_fraction': float('nan'),
                    'lead/alert_hit_fraction': float('nan'),
                    'lead/invalid_fraction': float('nan'),
                    'lead/base_reward_mean': float('nan'),
                    'lead/rewards/qed_mean': float('nan'),
                    'lead/rewards/sa_mean': float('nan'),
                    'lead/rewards/sa_score_mean': float('nan'),
                    'lead/rewards/soft_mean': float('nan'),
                    'lead/rewards/sim_mean': float('nan'),
                }
            )

        log_rows = []
        for row_idx, (spec, record, downstream_reward) in enumerate(
            zip(local_denovo_specs, denovo_reward_records, local_downstream_base_rewards.tolist())
        ):
            log_rows.append(
                {
                    'mode': mode,
                    'stage': 'denovo',
                    'buffer_cycle': buffer_cycle,
                    'step': self.global_step,
                    'spec': asdict(spec),
                    'safe': denovo_rollout.safe_strings[row_idx],
                    'smiles': record.smiles,
                    'base_reward': record.reward,
                    'downstream_base_reward': downstream_reward,
                    'reward': (1.0 - self.config.denovo_reward_alpha) * record.reward
                    + self.config.denovo_reward_alpha * downstream_reward,
                    'qed': record.qed,
                    'sa': record.sa,
                    'sa_score': record.sa_score,
                    'soft_reward': record.soft_reward,
                    'is_valid': record.is_valid,
                    'alert_hit': record.alert_hit,
                    'selected_for_lead': bool(local_selected_seed_mask[row_idx].item()),
                }
            )
        for spec, reward_item, safe_string in zip(lead_specs, lead_records, lead_rollout.safe_strings):
            record = reward_item['record']
            log_rows.append(
                {
                    'mode': mode,
                    'stage': 'lead',
                    'buffer_cycle': buffer_cycle,
                    'step': self.global_step,
                    'spec': asdict(spec),
                    'seed_smiles': reward_item['seed_smiles'],
                    'safe': safe_string,
                    'smiles': record.smiles,
                    'base_reward': reward_item['base_reward'],
                    'reward': reward_item['reward'],
                    'sim': reward_item['sim'],
                    'qed': record.qed,
                    'sa': record.sa,
                    'sa_score': record.sa_score,
                    'soft_reward': record.soft_reward,
                    'is_valid': record.is_valid,
                    'alert_hit': record.alert_hit,
                }
            )
        self._record_text_logs(log_rows)

        self.generation_cycle_idx += 1
        return {
            'buffer_cycle': buffer_cycle,
            'denovo': {
                'prompt_ids': denovo_rollout.prompt_ids,
                'completion_ids': denovo_rollout.completion_ids,
                'completion_mask': denovo_rollout.completion_mask,
                'advantages': local_denovo_advantages,
                'old_per_token_logps': denovo_old_per_token_logps,
                'ref_per_token_logps': denovo_ref_per_token_logps,
                'mask_seeds': denovo_mask_seeds,
            },
            'lead': {
                'input_ids': lead_rollout.input_ids.detach().clone(),
                'completion_mask': lead_rollout.completion_mask.detach().clone(),
                'advantages': local_lead_advantages,
                'old_per_token_logps': lead_old_per_token_logps,
                'ref_per_token_logps': lead_ref_per_token_logps,
                'mask_seeds': lead_mask_seeds,
                'has_samples': True,
            },
            'metrics': rollout_metrics,
        }

    def _compute_denovo_loss(self, inputs, iteration_idx):
        prompt_completion_ids = torch.cat([inputs['prompt_ids'], inputs['completion_ids']], dim=1).unsqueeze(0)
        logits_to_keep = inputs['completion_ids'].size(1)
        per_token_logps = self.denovo_policy.per_token_logps(
            input_ids=prompt_completion_ids,
            logits_to_keep=logits_to_keep,
            completion_mask=inputs['completion_mask'],
            mask_seeds=[inputs['mask_seeds'][iteration_idx]],
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            requires_grad=True,
        )
        if inputs['old_per_token_logps'] is None:
            old_per_token_logps = per_token_logps.detach()
        else:
            old_per_token_logps = inputs['old_per_token_logps'][:, iteration_idx, :].unsqueeze(1)

        if inputs['ref_per_token_logps'] is None:
            ref_per_token_logps = None
        else:
            ref_per_token_logps = inputs['ref_per_token_logps'][:, iteration_idx, :].unsqueeze(1)
        loss, step_metrics = compute_clipped_grpo_loss(
            new_log_probs=per_token_logps,
            old_log_probs=old_per_token_logps,
            advantages=inputs['advantages'],
            completion_mask=inputs['completion_mask'],
            epsilon=self.config.denovo_epsilon,
            ref_log_probs=ref_per_token_logps,
            beta=self.config.denovo_beta,
        )
        return loss, step_metrics

    def _zero_policy_loss(self, policy):
        loss = None
        for parameter in policy.trainable_parameters():
            term = parameter.float().sum() * 0.0
            loss = term if loss is None else loss + term
        if loss is None:
            raise RuntimeError('Policy has no trainable parameters')
        return loss

    def _compute_lead_loss(self, inputs, iteration_idx):
        if not inputs['has_samples']:
            nan_tensor = torch.tensor(float('nan'), device=self.device)
            return self._zero_policy_loss(self.lead_policy), {
                'ratio_mean': nan_tensor,
                'clip_ratio_low_mean': nan_tensor,
                'clip_ratio_high_mean': nan_tensor,
                'clip_ratio_region_mean': nan_tensor,
                'kl_mean': nan_tensor,
            }

        lead_input_ids = inputs['input_ids'].detach().clone()
        lead_completion_mask = inputs['completion_mask'].detach().clone()
        per_token_logps = self.lead_policy.per_token_logps(
            input_ids=lead_input_ids.unsqueeze(0),
            completion_mask=lead_completion_mask,
            mask_seeds=[inputs['mask_seeds'][iteration_idx]],
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            requires_grad=True,
        )
        if inputs['old_per_token_logps'] is None:
            old_per_token_logps = per_token_logps.detach()
        else:
            old_per_token_logps = inputs['old_per_token_logps'][:, iteration_idx, :].unsqueeze(1)

        if inputs['ref_per_token_logps'] is None:
            ref_per_token_logps = None
        else:
            ref_per_token_logps = inputs['ref_per_token_logps'][:, iteration_idx, :].unsqueeze(1)
        loss, step_metrics = compute_clipped_grpo_loss(
            new_log_probs=per_token_logps,
            old_log_probs=old_per_token_logps,
            advantages=inputs['advantages'],
            completion_mask=lead_completion_mask,
            epsilon=self.config.lead_epsilon,
            ref_log_probs=ref_per_token_logps,
            beta=self.config.lead_beta,
        )
        return loss, step_metrics

    def _checkpoint_dir(self):
        return os.path.join(self.output_dir, f'checkpoint-{self.global_step:06d}')

    def _save_checkpoint(self):
        checkpoint_dir = self._checkpoint_dir()
        accelerator_state_dir = os.path.join(checkpoint_dir, 'accelerator_state')
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(accelerator_state_dir)
        self.accelerator.wait_for_everyone()

        if not self.accelerator.is_main_process:
            return

        self.denovo_policy.save_checkpoint(
            os.path.join(checkpoint_dir, 'denovo_model.ckpt'),
            step=self.global_step,
            accelerator=self.accelerator,
        )
        self.lead_policy.save_checkpoint(
            os.path.join(checkpoint_dir, 'lead_model.ckpt'),
            step=self.global_step,
            accelerator=self.accelerator,
        )
        torch.save(self.denovo_reference.get_backbone_state_dict(), os.path.join(checkpoint_dir, 'denovo_reference_backbone.pt'))
        torch.save(self.lead_reference.get_backbone_state_dict(), os.path.join(checkpoint_dir, 'lead_reference_backbone.pt'))
        with open(os.path.join(checkpoint_dir, 'trainer_state.json'), 'w') as handle:
            json.dump(
                {
                    'global_step': self.global_step,
                    'generation_cycle_idx': self.generation_cycle_idx,
                    'last_metrics': self._last_train_metrics,
                },
                handle,
                sort_keys=True,
                indent=2,
            )
        maybe_trim_checkpoints(self.output_dir, self.config.save_total_limit)

    def _load_checkpoint(self, checkpoint_dir):
        accelerator_state_dir = os.path.join(checkpoint_dir, 'accelerator_state')
        trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
        ensure_exists(accelerator_state_dir, 'accelerator state')
        ensure_exists(trainer_state_path, 'trainer state')
        ensure_exists(os.path.join(checkpoint_dir, 'denovo_model.ckpt'), 'de novo model checkpoint')
        ensure_exists(os.path.join(checkpoint_dir, 'lead_model.ckpt'), 'lead model checkpoint')
        ensure_exists(os.path.join(checkpoint_dir, 'denovo_reference_backbone.pt'), 'de novo reference state')
        ensure_exists(os.path.join(checkpoint_dir, 'lead_reference_backbone.pt'), 'lead reference state')

        with open(trainer_state_path) as handle:
            trainer_state = json.load(handle)
        self.accelerator.load_state(accelerator_state_dir)
        denovo_checkpoint = torch.load(os.path.join(checkpoint_dir, 'denovo_model.ckpt'), map_location='cpu', weights_only=False)
        lead_checkpoint = torch.load(os.path.join(checkpoint_dir, 'lead_model.ckpt'), map_location='cpu', weights_only=False)
        self.denovo_policy.load_ema_state(denovo_checkpoint.get('ema'))
        self.lead_policy.load_ema_state(lead_checkpoint.get('ema'))
        self.denovo_reference.load_backbone_state_dict(
            torch.load(os.path.join(checkpoint_dir, 'denovo_reference_backbone.pt'), map_location='cpu', weights_only=False)
        )
        self.lead_reference.load_backbone_state_dict(
            torch.load(os.path.join(checkpoint_dir, 'lead_reference_backbone.pt'), map_location='cpu', weights_only=False)
        )
        self.global_step = int(trainer_state['global_step'])
        self.generation_cycle_idx = int(trainer_state['generation_cycle_idx'])
        self._last_train_metrics = trainer_state.get('last_metrics')
        self._buffer_iteration = 0
        self._buffered_inputs = None
        self._buffer_metadata = None
        self._last_rollout_metrics = None

    def train(self, resume_from_checkpoint=None):
        if resume_from_checkpoint is not None:
            logger.info('Resuming from checkpoint: %s', resume_from_checkpoint)
            self._load_checkpoint(resume_from_checkpoint)

        while self.global_step < self.config.max_steps:
            buffered_inputs, iteration_idx = self._prepare_inputs(mode='train')

            self.lead_optimizer.zero_grad(set_to_none=True)
            self.denovo_optimizer.zero_grad(set_to_none=True)

            for chunk_idx in range(self.config.gradient_accumulation_steps):
                denovo_inputs = buffered_inputs['denovo'][chunk_idx]
                lead_inputs = buffered_inputs['lead'][chunk_idx]

                lead_loss, lead_step_metrics = self._compute_lead_loss(lead_inputs, iteration_idx)
                self.accelerator.backward(lead_loss / self.config.gradient_accumulation_steps)
                self._record_stage_loss_metrics('lead', lead_step_metrics)

                denovo_loss, denovo_step_metrics = self._compute_denovo_loss(denovo_inputs, iteration_idx)
                self.accelerator.backward(denovo_loss / self.config.gradient_accumulation_steps)
                self._record_stage_loss_metrics('denovo', denovo_step_metrics)

            lead_grad_norm = self.accelerator.clip_grad_norm_(
                self.lead_policy.model.backbone.parameters(),
                self.config.max_grad_norm,
            )
            self.lead_optimizer.step()
            self.lead_scheduler.step()
            self.lead_optimizer.zero_grad(set_to_none=True)
            self.lead_policy.update_ema()

            denovo_grad_norm = self.accelerator.clip_grad_norm_(
                self.denovo_policy.model.backbone.parameters(),
                self.config.max_grad_norm,
            )
            self.denovo_optimizer.step()
            self.denovo_scheduler.step()
            self.denovo_optimizer.zero_grad(set_to_none=True)
            self.denovo_policy.update_ema()

            self._append_metric('denovo/grad_norm', float(denovo_grad_norm))
            self._append_metric('lead/grad_norm', float(lead_grad_norm))
            self._append_metric('denovo/lr', float(self.denovo_scheduler.get_last_lr()[0]))
            self._append_metric('lead/lr', float(self.lead_scheduler.get_last_lr()[0]))

            self.global_step += 1
            self._buffer_iteration += 1
            if self._buffer_iteration >= self.config.num_iterations:
                self._clear_rollout_buffer()

            if (
                self.config.denovo_sync_ref_model
                and self.global_step % self.config.denovo_ref_model_sync_steps == 0
            ):
                self.denovo_reference.sync_from(
                    self.denovo_policy,
                    alpha=self.config.denovo_ref_model_mixup_alpha,
                )
            if (
                self.config.lead_sync_ref_model
                and self.global_step % self.config.lead_ref_model_sync_steps == 0
            ):
                self.lead_reference.sync_from(
                    self.lead_policy,
                    alpha=self.config.lead_ref_model_mixup_alpha,
                )

            should_log = self.global_step == 1 and self.config.logging_first_step
            should_log = should_log or (self.global_step % self.config.logging_steps == 0)
            if should_log:
                self._last_train_metrics = self._consume_logged_metrics()
                self._log_metrics('train', self._last_train_metrics)

            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint()

        if self._metrics:
            self._last_train_metrics = self._consume_logged_metrics()

        return TrainResult(metrics=self._last_train_metrics or {})

    def evaluate(self):
        raise NotImplementedError('Joint pipeline evaluation is not implemented')

    def log_metrics(self, split, metrics):
        self._log_metrics(split, metrics)

    def save_metrics(self, split, metrics):
        if not self.accelerator.is_main_process:
            return
        path = os.path.join(self.output_dir, f'{split}_results.json')
        with open(path, 'w') as handle:
            json.dump(metrics, handle, sort_keys=True, indent=2)

    def save_state(self):
        if not self.accelerator.is_main_process:
            return
        with open(self.state_path, 'w') as handle:
            json.dump(
                {
                    'global_step': self.global_step,
                    'generation_cycle_idx': self.generation_cycle_idx,
                    'last_metrics': self._last_train_metrics,
                },
                handle,
                sort_keys=True,
                indent=2,
            )

    def save_model(self, output_dir):
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return
        self.denovo_policy.save_checkpoint(
            os.path.join(output_dir, 'final_denovo_model.ckpt'),
            step=self.global_step,
            accelerator=self.accelerator,
        )
        self.lead_policy.save_checkpoint(
            os.path.join(output_dir, 'final_lead_model.ckpt'),
            step=self.global_step,
            accelerator=self.accelerator,
        )

    def close(self):
        self.base_reward_model.close()
