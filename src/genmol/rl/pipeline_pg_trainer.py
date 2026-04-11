import json
import logging
import os
import random
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

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
from genmol.rl.pipeline_trainer import JointTrainConfig
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
    maybe_trim_checkpoints,
    write_jsonl,
)


logger = logging.getLogger(__name__)


def _set_process_seed(seed, process_index):
    full_seed = int(seed) + int(process_index)
    random.seed(full_seed)
    try:
        import numpy as np

        np.random.seed(full_seed)
    except ImportError:
        pass
    torch.manual_seed(full_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(full_seed)


class ProcessGroupJointCpGRPOTrainer:
    def __init__(self, config: JointTrainConfig, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        self._wandb = None

        unsupported_trackers = sorted(set(config.report_to) - {'wandb'})
        if unsupported_trackers:
            raise ValueError(
                'process_group_ddp backend only supports report_to containing at most wandb, got '
                f'{unsupported_trackers}'
            )

        if not dist.is_available():
            raise RuntimeError('torch.distributed is required for process_group_ddp backend')
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        if self.world_size < 4 or self.world_size % 2 != 0:
            raise ValueError(
                'process_group_ddp backend requires an even world size of at least 4 so both stage groups '
                f'can run real DDP, got world_size={self.world_size}'
            )

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device('cuda', self.local_rank)

        self.stage_group_size = self.world_size // 2
        self.denovo_ranks = list(range(self.stage_group_size))
        self.lead_ranks = list(range(self.stage_group_size, self.world_size))
        self.denovo_group = dist.new_group(ranks=self.denovo_ranks)
        self.lead_group = dist.new_group(ranks=self.lead_ranks)
        self.is_denovo_rank = self.rank in self.denovo_ranks
        self.is_lead_rank = self.rank in self.lead_ranks
        if self.is_denovo_rank == self.is_lead_rank:
            raise RuntimeError('rank must belong to exactly one stage group')

        if self.is_denovo_rank:
            self.stage = 'denovo'
            self.stage_group = self.denovo_group
            self.stage_ranks = self.denovo_ranks
            self.stage_local_rank = self.rank
            self.partner_rank = self.rank + self.stage_group_size
        else:
            self.stage = 'lead'
            self.stage_group = self.lead_group
            self.stage_ranks = self.lead_ranks
            self.stage_local_rank = self.rank - self.stage_group_size
            self.partner_rank = self.stage_local_rank

        self.is_main_process = self.rank == 0
        self.is_stage_main = self.stage_local_rank == 0
        self.lead_main_rank = self.stage_group_size

        ensure_exists(config.denovo_init_ckpt_path, 'de novo init checkpoint')
        ensure_exists(config.denovo_ref_ckpt_path, 'de novo reference checkpoint')
        ensure_exists(config.lead_init_ckpt_path, 'lead init checkpoint')
        ensure_exists(config.lead_ref_ckpt_path, 'lead reference checkpoint')

        _set_process_seed(config.seed, self.rank)

        self.denovo_micro_batch_size = config.denovo_per_device_train_batch_size
        self.denovo_local_sample_count = self.denovo_micro_batch_size * config.gradient_accumulation_steps
        self.denovo_global_sample_count = self.denovo_local_sample_count * self.stage_group_size
        if self.denovo_global_sample_count % config.denovo_num_generations != 0:
            raise ValueError(
                'global de novo seed batch must be divisible by denovo_num_generations for process_group_ddp: '
                f'{self.denovo_global_sample_count} vs {config.denovo_num_generations}'
            )
        self.denovo_num_groups_global = self.denovo_global_sample_count // config.denovo_num_generations
        self.lead_micro_batch_size = int(config.lead_per_device_train_batch_size)
        self.lead_local_sample_count = self.lead_micro_batch_size * config.gradient_accumulation_steps
        if self.lead_local_sample_count % config.lead_num_generations != 0:
            raise ValueError(
                'lead local sample count must be divisible by lead_num_generations for process_group_ddp: '
                f'{self.lead_local_sample_count} vs {config.lead_num_generations}'
            )
        self.lead_num_groups_local = self.lead_local_sample_count // config.lead_num_generations
        self.lead_global_sample_count = self.lead_local_sample_count * self.stage_group_size
        if self.lead_global_sample_count % config.lead_num_generations != 0:
            raise ValueError(
                'global lead batch must be divisible by lead_num_generations for process_group_ddp: '
                f'{self.lead_global_sample_count} vs {config.lead_num_generations}'
            )
        self.lead_num_groups_global = self.lead_global_sample_count // config.lead_num_generations
        if self.lead_num_groups_global % config.denovo_num_generations != 0:
            raise ValueError(
                'lead global group count must be divisible by denovo_num_generations so full de novo groups can '
                f'be sampled for lead: {self.lead_num_groups_global} vs {config.denovo_num_generations}'
            )
        self.selected_denovo_group_count = self.lead_num_groups_global // config.denovo_num_generations

        self.denovo_policy = None
        self.denovo_reference = None
        self.denovo_optimizer = None
        self.denovo_scheduler = None
        self.lead_policy = None
        self.lead_reference = None
        self.lead_optimizer = None
        self.lead_scheduler = None

        if self.is_denovo_rank:
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
            if config.denovo_gradient_checkpointing:
                self.denovo_policy.enable_gradient_checkpointing(config.denovo_gradient_checkpointing_kwargs)
            self.denovo_policy.train()
            self.denovo_policy.model.backbone = DistributedDataParallel(
                self.denovo_policy.model.backbone,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                process_group=self.denovo_group,
                broadcast_buffers=config.ddp_broadcast_buffers,
            )
            self.denovo_optimizer = torch.optim.AdamW(
                self.denovo_policy.trainable_parameters(),
                lr=config.denovo_learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_eps,
                weight_decay=config.weight_decay,
            )
            self.denovo_scheduler = build_scheduler(self.denovo_optimizer, config)

        if self.is_lead_rank:
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
            if config.lead_gradient_checkpointing:
                self.lead_policy.enable_gradient_checkpointing(config.lead_gradient_checkpointing_kwargs)
            self.lead_policy.train()
            self.lead_policy.model.backbone = DistributedDataParallel(
                self.lead_policy.model.backbone,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                process_group=self.lead_group,
                broadcast_buffers=config.ddp_broadcast_buffers,
            )
            self.lead_optimizer = torch.optim.AdamW(
                self.lead_policy.trainable_parameters(),
                lr=config.lead_learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_eps,
                weight_decay=config.weight_decay,
            )
            self.lead_scheduler = build_scheduler(self.lead_optimizer, config)

        if self.is_denovo_rank:
            self.policy = self.denovo_policy
            self.reference_policy = self.denovo_reference
            self.optimizer = self.denovo_optimizer
            self.scheduler = self.denovo_scheduler
        else:
            self.policy = self.lead_policy
            self.reference_policy = self.lead_reference
            self.optimizer = self.lead_optimizer
            self.scheduler = self.lead_scheduler

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

        if config.report_to and self.is_main_process:
            import wandb

            self._wandb = wandb.init(
                project='genmol-cpgrpo-pipeline',
                config=asdict(config),
                name=os.path.basename(output_dir),
            )

        logger.info(
            'rank=%s local_rank=%s stage=%s stage_local_rank=%s stage_group_size=%s world_size=%s '
            'denovo_micro_batch_size=%s denovo_local_sample_count=%s denovo_stage_global_sample_count=%s '
            'lead_micro_batch_size=%s lead_local_sample_count=%s lead_stage_global_sample_count=%s '
            'lead_num_generations=%s lead_num_groups_global=%s '
            'gradient_accumulation_steps=%s num_iterations=%s reward_workers=%s',
            self.rank,
            self.local_rank,
            self.stage,
            self.stage_local_rank,
            self.stage_group_size,
            self.world_size,
            self.denovo_micro_batch_size,
            self.denovo_local_sample_count,
            self.denovo_global_sample_count,
            self.lead_micro_batch_size,
            self.lead_local_sample_count,
            self.lead_global_sample_count,
            self.config.lead_num_generations,
            self.lead_num_groups_global,
            self.config.gradient_accumulation_steps,
            self.config.num_iterations,
            self.base_reward_model.num_workers,
        )

    def _world_barrier(self):
        dist.barrier()

    def _stage_all_gather_objects(self, payload):
        if self.stage_group_size == 1:
            return [payload]
        gathered = [None for _ in range(self.stage_group_size)]
        dist.all_gather_object(gathered, payload, group=self.stage_group)
        return gathered

    def _stage_gather_tensor(self, tensor):
        if self.stage_group_size == 1:
            return tensor
        gathered = [torch.empty_like(tensor) for _ in range(self.stage_group_size)]
        dist.all_gather(gathered, tensor, group=self.stage_group)
        return torch.cat(gathered, dim=0)

    def _stage_gather_variable_scalars(self, values):
        local_values = [float(item) for item in values]
        gathered = self._stage_all_gather_objects(local_values)
        local_start = sum(len(item) for item in gathered[:self.stage_local_rank])
        flat = [value for shard in gathered for value in shard]
        if not flat:
            return torch.empty((0,), device=self.device, dtype=torch.float32), local_start
        return torch.tensor(flat, device=self.device, dtype=torch.float32), local_start

    def _broadcast_main_object(self, payload):
        box = [payload if self.is_main_process else None]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def _send_object(self, payload, dst):
        dist.send_object_list([payload], dst=dst)

    def _recv_object(self, src):
        payload = [None]
        dist.recv_object_list(payload, src=src)
        return payload[0]

    def _send_lead_request(self, payload):
        if not self.is_denovo_rank:
            raise RuntimeError('_send_lead_request is only valid on denovo ranks')
        self._send_object(payload, dst=self.partner_rank)

    def _recv_lead_request(self):
        if not self.is_lead_rank:
            raise RuntimeError('_recv_lead_request is only valid on lead ranks')
        return self._recv_object(src=self.partner_rank)

    def _send_downstream_response(self, payload):
        if not self.is_lead_rank:
            raise RuntimeError('_send_downstream_response is only valid on lead ranks')
        self._send_object(payload, dst=self.partner_rank)

    def _recv_downstream_response(self):
        if not self.is_denovo_rank:
            raise RuntimeError('_recv_downstream_response is only valid on denovo ranks')
        return self._recv_object(src=self.partner_rank)

    def _send_stage_log_payload(self, payload):
        if not self.is_lead_rank or not self.is_stage_main:
            raise RuntimeError('_send_stage_log_payload is only valid on lead main rank')
        self._send_object(payload, dst=0)

    def _recv_stage_log_payload(self):
        if not self.is_denovo_rank or not self.is_stage_main:
            raise RuntimeError('_recv_stage_log_payload is only valid on denovo main rank')
        return self._recv_object(src=self.lead_main_rank)

    def _broadcast_denovo_specs(self, group_specs):
        payload = [None]
        if self.is_stage_main:
            payload[0] = serialize_denovo_specs(
                expand_denovo_group_specs(group_specs, self.config.denovo_num_generations)
            )
        dist.broadcast_object_list(payload, src=self.denovo_ranks[0], group=self.denovo_group)
        return deserialize_denovo_specs(payload[0])

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

    def _build_lead_specs(self, valid_seed_smiles, cycle_seed):
        rng = random.Random(cycle_seed + 7919 + self.stage_local_rank * 1000)
        lead_specs = []
        for seed_smiles in valid_seed_smiles:
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

    def _select_lead_seed_requests(
        self,
        denovo_reward_records,
        local_seed_base_rewards,
        cycle_seed,
        local_start,
    ):
        if not self.is_denovo_rank:
            raise RuntimeError('_select_lead_seed_requests is only valid on denovo ranks')

        local_seed_entries = []
        for record, base_reward in zip(denovo_reward_records, local_seed_base_rewards.tolist()):
            local_seed_entries.append(
                {
                    'is_valid': bool(record.is_valid and record.smiles is not None),
                    'smiles': record.smiles,
                    'base_reward': float(base_reward),
                }
            )
        gathered_seed_entries = self._stage_all_gather_objects(local_seed_entries)
        payload = [None]
        if self.is_stage_main:
            try:
                global_seed_entries = [item for shard in gathered_seed_entries for item in shard]
                if len(global_seed_entries) != self.denovo_global_sample_count:
                    raise ValueError(
                        'Global de novo seed entry count mismatch for process_group_ddp: '
                        f'{len(global_seed_entries)} vs {self.denovo_global_sample_count}'
                    )
                selected_group_indices, selected_seed_entries, selected_seed_mask = select_full_denovo_groups_for_lead(
                    seed_entries=global_seed_entries,
                    denovo_group_size=self.config.denovo_num_generations,
                    lead_num_seed_groups=self.lead_num_groups_global,
                    seed=cycle_seed + 15485863,
                )
                payload[0] = {
                    'error': None,
                    'selected_group_indices': selected_group_indices,
                    'selected_seed_mask': selected_seed_mask,
                    'request_shards': partition_selected_seed_entries(
                        selected_seed_entries,
                        self.stage_group_size,
                    ),
                    'default_downstream_base_rewards': [item['base_reward'] for item in global_seed_entries],
                }
            except Exception as exc:
                payload[0] = {'error': str(exc)}
        dist.broadcast_object_list(payload, src=self.denovo_ranks[0], group=self.denovo_group)
        selection_payload = payload[0]
        if selection_payload is None:
            raise RuntimeError('Lead seed selection payload was not broadcast for process_group_ddp')

        local_end = local_start + len(denovo_reward_records)
        if selection_payload.get('error') is not None:
            local_selected_seed_mask = torch.zeros(
                len(denovo_reward_records),
                device=self.device,
                dtype=torch.bool,
            )
            request_shard = []
        else:
            local_selected_seed_mask = torch.tensor(
                selection_payload['selected_seed_mask'][local_start:local_end],
                device=self.device,
                dtype=torch.bool,
            )
            request_shard = selection_payload['request_shards'][self.stage_local_rank]
            if len(request_shard) != self.lead_num_groups_local:
                raise ValueError(
                    'Per-rank selected lead seed count mismatch for process_group_ddp: '
                    f'{len(request_shard)} vs {self.lead_num_groups_local}'
                )
        return selection_payload, request_shard, local_selected_seed_mask

    def _record_text_logs(self, rows):
        if not self.config.log_completions or not rows:
            return
        gathered = self._stage_all_gather_objects(rows)
        if self.is_stage_main:
            for shard in gathered:
                self._textual_logs.extend(shard)

    def _append_metric(self, key, value):
        self._metrics[key].append(float(value))

    def _record_rollout_metrics(self, metrics):
        self._last_rollout_metrics = dict(metrics)
        for key, value in metrics.items():
            self._append_metric(key, value)

    def _record_stage_loss_metrics(self, metrics):
        prefix = self.stage
        gathered_ratio = self._stage_gather_tensor(metrics['ratio_mean'].detach().reshape(1))
        self._append_metric(f'{prefix}/ratio_mean', torch.nanmean(gathered_ratio).item())

        gathered_low = self._stage_gather_tensor(metrics['clip_ratio_low_mean'].detach().reshape(1))
        self._append_metric(f'{prefix}/clip_ratio/low_mean', torch.nanmean(gathered_low).item())
        self._append_metric(f'{prefix}/clip_ratio/low_min', _nanreduce(gathered_low, mode='min'))

        gathered_high = self._stage_gather_tensor(metrics['clip_ratio_high_mean'].detach().reshape(1))
        self._append_metric(f'{prefix}/clip_ratio/high_mean', torch.nanmean(gathered_high).item())
        self._append_metric(f'{prefix}/clip_ratio/high_max', _nanreduce(gathered_high, mode='max'))

        gathered_region = self._stage_gather_tensor(metrics['clip_ratio_region_mean'].detach().reshape(1))
        self._append_metric(f'{prefix}/clip_ratio/region_mean', torch.nanmean(gathered_region).item())

        if 'kl_mean' in metrics:
            gathered_kl = self._stage_gather_tensor(metrics['kl_mean'].detach().reshape(1))
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

    def _prepare_inputs(self, mode='train'):
        if self._buffered_inputs is None or self._buffer_iteration >= self.config.num_iterations:
            generated = self._generate_and_score_stage(mode=mode)
            self._record_rollout_metrics(generated['metrics'])
            if self.is_denovo_rank:
                self._buffered_inputs = split_tensor_dict(
                    generated['inputs'],
                    self.config.gradient_accumulation_steps,
                )
            else:
                self._buffered_inputs = self._split_lead_inputs(generated['inputs'])
            self._buffer_metadata = {'buffer_cycle': int(generated['buffer_cycle'])}
            self._buffer_iteration = 0
        return self._buffered_inputs, self._buffer_iteration

    def _clear_rollout_buffer(self):
        self._buffered_inputs = None
        self._buffer_iteration = 0

    def _merge_stage_metrics(self, denovo_metrics, lead_metrics):
        if denovo_metrics['step'] != lead_metrics['step']:
            raise ValueError(
                f"stage metric step mismatch: {denovo_metrics['step']} vs {lead_metrics['step']}"
            )
        if denovo_metrics['buffer_cycle'] != lead_metrics['buffer_cycle']:
            raise ValueError(
                'stage metric buffer_cycle mismatch: '
                f"{denovo_metrics['buffer_cycle']} vs {lead_metrics['buffer_cycle']}"
            )
        merged = dict(denovo_metrics)
        for key, value in lead_metrics.items():
            if key in {'step', 'buffer_cycle'}:
                continue
            merged[key] = value
        return merged

    def _log_metrics(self, split, metrics, extra_rows=None):
        if not self.is_main_process:
            return

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
        if self._wandb is not None:
            self._wandb.log(metrics, step=self.global_step)
        for row in extra_rows or ():
            write_jsonl(self.text_logs_path, row)

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

    def _build_denovo_rollout_metrics(
        self,
        denovo_reward_records,
        denovo_rollout,
        local_seed_base_rewards,
        local_downstream_base_rewards,
        local_denovo_advantages,
        global_denovo_rewards,
        global_denovo_reward_std,
        denovo_zero_std_ratio,
        local_selected_seed_mask,
    ):
        gathered_denovo_valid = self._stage_gather_tensor(
            torch.tensor([float(record.is_valid) for record in denovo_reward_records], device=self.device)
        )
        gathered_denovo_alert = self._stage_gather_tensor(
            torch.tensor([float(record.alert_hit) for record in denovo_reward_records], device=self.device)
        )
        gathered_denovo_qed = self._stage_gather_tensor(
            torch.tensor(
                [float('nan') if record.qed is None else float(record.qed) for record in denovo_reward_records],
                device=self.device,
            )
        )
        gathered_denovo_sa = self._stage_gather_tensor(
            torch.tensor(
                [float('nan') if record.sa is None else float(record.sa) for record in denovo_reward_records],
                device=self.device,
            )
        )
        gathered_denovo_sa_score = self._stage_gather_tensor(
            torch.tensor(
                [float('nan') if record.sa_score is None else float(record.sa_score) for record in denovo_reward_records],
                device=self.device,
            )
        )
        gathered_denovo_soft = self._stage_gather_tensor(
            torch.tensor(
                [float('nan') if record.soft_reward is None else float(record.soft_reward) for record in denovo_reward_records],
                device=self.device,
            )
        )
        gathered_completion_lengths = self._stage_gather_tensor(denovo_rollout.completion_mask.sum(dim=1).float())
        gathered_seed_base_rewards = self._stage_gather_tensor(local_seed_base_rewards.detach())
        gathered_downstream_base_rewards = self._stage_gather_tensor(local_downstream_base_rewards.detach())
        gathered_advantages = self._stage_gather_tensor(local_denovo_advantages.detach())
        gathered_selected_seed_mask = self._stage_gather_tensor(local_selected_seed_mask.float())

        return {
            'denovo/reward_mean': global_denovo_rewards.mean().item(),
            'denovo/reward_std': global_denovo_reward_std.mean().item(),
            'denovo/advantage_mean': gathered_advantages.mean().item(),
            'denovo/zero_std_ratio': denovo_zero_std_ratio,
            'denovo/completion_length': gathered_completion_lengths.mean().item(),
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

    def _build_lead_rollout_metrics(
        self,
        lead_records,
        lead_rollout,
        local_lead_rewards,
        global_lead_reward_std,
        local_lead_advantages,
        lead_zero_std_ratio,
    ):
        global_lead_rewards, _ = self._stage_gather_variable_scalars(local_lead_rewards)
        if global_lead_rewards.numel() == 0:
            return {
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

        all_valid_lists = self._stage_all_gather_objects([float(item['record'].is_valid) for item in lead_records])
        all_alert_lists = self._stage_all_gather_objects([float(item['record'].alert_hit) for item in lead_records])
        all_qed_lists = self._stage_all_gather_objects(
            [float('nan') if item['record'].qed is None else float(item['record'].qed) for item in lead_records]
        )
        all_sa_lists = self._stage_all_gather_objects(
            [float('nan') if item['record'].sa is None else float(item['record'].sa) for item in lead_records]
        )
        all_sa_score_lists = self._stage_all_gather_objects(
            [float('nan') if item['record'].sa_score is None else float(item['record'].sa_score) for item in lead_records]
        )
        all_soft_lists = self._stage_all_gather_objects(
            [float('nan') if item['record'].soft_reward is None else float(item['record'].soft_reward) for item in lead_records]
        )
        all_sim_lists = self._stage_all_gather_objects(
            [float('nan') if item['sim'] is None else float(item['sim']) for item in lead_records]
        )
        all_base_lists = self._stage_all_gather_objects([float(item['base_reward']) for item in lead_records])
        all_length_lists = self._stage_all_gather_objects(
            [] if lead_rollout is None else lead_rollout.completion_mask.sum(dim=1).detach().cpu().tolist()
        )
        all_adv_lists = self._stage_all_gather_objects(local_lead_advantages.detach().cpu().tolist())

        global_valid = torch.tensor([value for shard in all_valid_lists for value in shard], device=self.device)
        global_alert = torch.tensor([value for shard in all_alert_lists for value in shard], device=self.device)
        global_qed = torch.tensor([value for shard in all_qed_lists for value in shard], device=self.device)
        global_sa = torch.tensor([value for shard in all_sa_lists for value in shard], device=self.device)
        global_sa_score = torch.tensor(
            [value for shard in all_sa_score_lists for value in shard],
            device=self.device,
        )
        global_soft = torch.tensor([value for shard in all_soft_lists for value in shard], device=self.device)
        global_sim = torch.tensor([value for shard in all_sim_lists for value in shard], device=self.device)
        global_base = torch.tensor([value for shard in all_base_lists for value in shard], device=self.device)
        global_lengths = torch.tensor(
            [value for shard in all_length_lists for value in shard],
            device=self.device,
            dtype=torch.float32,
        )
        global_advantages = torch.tensor(
            [value for shard in all_adv_lists for value in shard],
            device=self.device,
        )

        return {
            'lead/reward_mean': global_lead_rewards.mean().item(),
            'lead/reward_std': global_lead_reward_std.mean().item(),
            'lead/advantage_mean': global_advantages.mean().item(),
            'lead/zero_std_ratio': lead_zero_std_ratio,
            'lead/completion_length': global_lengths.mean().item(),
            'lead/valid_fraction': global_valid.mean().item(),
            'lead/alert_hit_fraction': global_alert.mean().item(),
            'lead/invalid_fraction': 1.0 - global_valid.mean().item(),
            'lead/base_reward_mean': global_base.mean().item(),
            'lead/rewards/qed_mean': _nanmean(global_qed),
            'lead/rewards/sa_mean': _nanmean(global_sa),
            'lead/rewards/sa_score_mean': _nanmean(global_sa_score),
            'lead/rewards/soft_mean': _nanmean(global_soft),
            'lead/rewards/sim_mean': _nanmean(global_sim),
        }

    def _generate_and_score_denovo(self, mode):
        buffer_cycle = self.generation_cycle_idx
        cycle_seed = self.config.seed + buffer_cycle * 10000

        if self.is_stage_main:
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

        local_start = self.stage_local_rank * self.denovo_local_sample_count
        local_end = (self.stage_local_rank + 1) * self.denovo_local_sample_count
        local_denovo_specs = expanded_specs[local_start:local_end]
        denovo_rollout_seed = cycle_seed + self.stage_local_rank * 1000
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

        selection_payload, local_lead_seed_entries, local_selected_seed_mask = self._select_lead_seed_requests(
            denovo_reward_records=denovo_reward_records,
            local_seed_base_rewards=local_seed_base_rewards,
            cycle_seed=cycle_seed,
            local_start=local_start,
        )
        self._send_lead_request(
            {
                'buffer_cycle': buffer_cycle,
                'cycle_seed': cycle_seed,
                'seed_count': len(denovo_reward_records),
                'selection_error': selection_payload.get('error'),
                'selected_seed_entries': local_lead_seed_entries,
            }
        )

        downstream_response = self._recv_downstream_response()
        if downstream_response.get('error') is not None:
            raise ValueError(downstream_response['error'])
        gathered_selected_seed_rewards = self._stage_all_gather_objects(
            downstream_response['selected_seed_rewards']
        )
        merged_downstream_base_rewards = merge_selected_seed_downstream_base_rewards(
            default_downstream_base_rewards=selection_payload['default_downstream_base_rewards'],
            selected_seed_rewards=[
                item for shard in gathered_selected_seed_rewards for item in shard
            ],
            expected_selected_seed_count=self.lead_num_groups_global,
        )
        local_downstream_base_rewards = torch.tensor(
            merged_downstream_base_rewards[local_start:local_end],
            device=self.device,
            dtype=torch.float32,
        )
        local_denovo_rewards = combine_seed_rewards(
            seed_base_rewards=local_seed_base_rewards,
            downstream_base_rewards=local_downstream_base_rewards,
            alpha=self.config.denovo_reward_alpha,
        )

        global_denovo_rewards = self._stage_gather_tensor(local_denovo_rewards).detach()
        global_denovo_advantages, global_denovo_reward_std, denovo_zero_std_ratio = compute_grouped_advantages(
            rewards=global_denovo_rewards,
            num_generations=self.config.denovo_num_generations,
            scale_rewards=self.config.scale_rewards,
        )
        local_denovo_advantages = global_denovo_advantages[local_start:local_end].to(self.device)

        denovo_mask_seeds = self._sample_mask_seeds()
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

        rollout_metrics = self._build_denovo_rollout_metrics(
            denovo_reward_records=denovo_reward_records,
            denovo_rollout=denovo_rollout,
            local_seed_base_rewards=local_seed_base_rewards,
            local_downstream_base_rewards=local_downstream_base_rewards,
            local_denovo_advantages=local_denovo_advantages,
            global_denovo_rewards=global_denovo_rewards,
            global_denovo_reward_std=global_denovo_reward_std,
            denovo_zero_std_ratio=denovo_zero_std_ratio,
            local_selected_seed_mask=local_selected_seed_mask,
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
        self._record_text_logs(log_rows)

        self.generation_cycle_idx = buffer_cycle + 1
        return {
            'buffer_cycle': buffer_cycle,
            'metrics': rollout_metrics,
            'inputs': {
                'prompt_ids': denovo_rollout.prompt_ids,
                'completion_ids': denovo_rollout.completion_ids,
                'completion_mask': denovo_rollout.completion_mask,
                'advantages': local_denovo_advantages,
                'old_per_token_logps': denovo_old_per_token_logps,
                'ref_per_token_logps': denovo_ref_per_token_logps,
                'mask_seeds': denovo_mask_seeds,
            },
        }

    def _generate_and_score_lead(self, mode):
        lead_request = self._recv_lead_request()
        buffer_cycle = int(lead_request['buffer_cycle'])
        cycle_seed = int(lead_request['cycle_seed'])
        selection_error = lead_request.get('selection_error')
        if selection_error is not None:
            self._send_downstream_response({'error': selection_error})
            raise ValueError(selection_error)

        selected_seed_entries = list(lead_request['selected_seed_entries'])
        if len(selected_seed_entries) != self.lead_num_groups_local:
            raise ValueError(
                'Per-rank selected lead seed count mismatch in lead stage: '
                f'{len(selected_seed_entries)} vs {self.lead_num_groups_local}'
            )
        selected_seed_smiles = [item['smiles'] for item in selected_seed_entries]

        lead_specs = self._build_lead_specs(selected_seed_smiles, cycle_seed)
        if len(lead_specs) != self.lead_local_sample_count:
            raise ValueError(
                'Local lead rollout spec count mismatch in lead stage: '
                f'{len(lead_specs)} vs {self.lead_local_sample_count}'
            )
        lead_rollout_seed = cycle_seed + 500000 + self.stage_local_rank * 1000
        lead_rollout = self.lead_policy.rollout_specs(
            specs=lead_specs,
            generation_batch_size=self.config.lead_generation_batch_size,
            seed=lead_rollout_seed,
        )
        lead_records = self._score_lead_records(lead_rollout.seed_smiles, lead_rollout.smiles)
        local_lead_rewards = [item['reward'] for item in lead_records]
        if len(local_lead_rewards) != self.lead_local_sample_count:
            raise ValueError(
                'Local lead reward count mismatch in lead stage: '
                f'{len(local_lead_rewards)} vs {self.lead_local_sample_count}'
            )
        local_lead_base_rewards_tensor = torch.tensor(
            [item['base_reward'] for item in lead_records],
            device=self.device,
            dtype=torch.float32,
        )
        local_selected_seed_rewards = aggregate_selected_seed_downstream_base_rewards(
            selected_seed_entries=selected_seed_entries,
            lead_base_rewards=local_lead_base_rewards_tensor,
            lead_num_generations=self.config.lead_num_generations,
            downstream_topk=self.config.downstream_topk,
        )
        self._send_downstream_response(
            {
                'error': None,
                'selected_seed_rewards': local_selected_seed_rewards,
            }
        )

        global_lead_rewards, local_lead_start = self._stage_gather_variable_scalars(local_lead_rewards)
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

        lead_mask_seeds = self._sample_mask_seeds()
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

        rollout_metrics = self._build_lead_rollout_metrics(
            lead_records=lead_records,
            lead_rollout=lead_rollout,
            local_lead_rewards=local_lead_rewards,
            global_lead_reward_std=global_lead_reward_std,
            local_lead_advantages=local_lead_advantages,
            lead_zero_std_ratio=lead_zero_std_ratio,
        )

        if lead_rollout is not None:
            log_rows = []
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

        self.generation_cycle_idx = buffer_cycle + 1
        return {
            'buffer_cycle': buffer_cycle,
            'metrics': rollout_metrics,
            'inputs': {
                'input_ids': lead_rollout.input_ids.detach().clone(),
                'completion_mask': lead_rollout.completion_mask.detach().clone(),
                'advantages': local_lead_advantages,
                'old_per_token_logps': lead_old_per_token_logps,
                'ref_per_token_logps': lead_ref_per_token_logps,
                'mask_seeds': lead_mask_seeds,
                'has_samples': True,
            },
        }

    def _generate_and_score_stage(self, mode):
        if self.is_denovo_rank:
            return self._generate_and_score_denovo(mode=mode)
        return self._generate_and_score_lead(mode=mode)

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
        return compute_clipped_grpo_loss(
            new_log_probs=per_token_logps,
            old_log_probs=old_per_token_logps,
            advantages=inputs['advantages'],
            completion_mask=inputs['completion_mask'],
            epsilon=self.config.denovo_epsilon,
            ref_log_probs=ref_per_token_logps,
            beta=self.config.denovo_beta,
        )

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
        return compute_clipped_grpo_loss(
            new_log_probs=per_token_logps,
            old_log_probs=old_per_token_logps,
            advantages=inputs['advantages'],
            completion_mask=lead_completion_mask,
            epsilon=self.config.lead_epsilon,
            ref_log_probs=ref_per_token_logps,
            beta=self.config.lead_beta,
        )

    def _compute_stage_loss(self, inputs, iteration_idx):
        if self.is_denovo_rank:
            return self._compute_denovo_loss(inputs, iteration_idx)
        return self._compute_lead_loss(inputs, iteration_idx)

    def _maybe_no_sync(self, chunk_idx):
        if self.stage_group_size <= 1 or chunk_idx + 1 == self.config.gradient_accumulation_steps:
            return nullcontext()
        return self.policy.model.backbone.no_sync()

    def _sync_stage_reference(self):
        if self.is_denovo_rank:
            if (
                self.config.denovo_sync_ref_model
                and self.global_step % self.config.denovo_ref_model_sync_steps == 0
            ):
                self.denovo_reference.sync_from(
                    self.denovo_policy,
                    alpha=self.config.denovo_ref_model_mixup_alpha,
                )
            return
        if (
            self.config.lead_sync_ref_model
            and self.global_step % self.config.lead_ref_model_sync_steps == 0
        ):
            self.lead_reference.sync_from(
                self.lead_policy,
                alpha=self.config.lead_ref_model_mixup_alpha,
            )

    def _stage_grad_norm(self):
        return torch.nn.utils.clip_grad_norm_(
            self.policy.model.backbone.parameters(),
            self.config.max_grad_norm,
        )

    def _append_optimizer_metrics(self, grad_norm):
        prefix = self.stage
        self._append_metric(f'{prefix}/grad_norm', float(grad_norm))
        self._append_metric(f'{prefix}/lr', float(self.scheduler.get_last_lr()[0]))

    def _checkpoint_dir(self):
        return os.path.join(self.output_dir, f'checkpoint-{self.global_step:06d}')

    def _save_stage_checkpoint(self, checkpoint_dir):
        if self.is_denovo_rank and self.is_stage_main:
            self.denovo_policy.save_checkpoint(
                os.path.join(checkpoint_dir, 'denovo_model.ckpt'),
                step=self.global_step,
                accelerator=None,
            )
            torch.save(self.denovo_optimizer.state_dict(), os.path.join(checkpoint_dir, 'denovo_optimizer.pt'))
            torch.save(self.denovo_scheduler.state_dict(), os.path.join(checkpoint_dir, 'denovo_scheduler.pt'))
            torch.save(
                self.denovo_reference.get_backbone_state_dict(),
                os.path.join(checkpoint_dir, 'denovo_reference_backbone.pt'),
            )
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
            with open(os.path.join(checkpoint_dir, 'backend_state.json'), 'w') as handle:
                json.dump(
                    {
                        'distributed_backend': 'process_group_ddp',
                        'world_size': self.world_size,
                        'stage_group_size': self.stage_group_size,
                    },
                    handle,
                    sort_keys=True,
                    indent=2,
                )
            return

        if self.is_lead_rank and self.is_stage_main:
            self.lead_policy.save_checkpoint(
                os.path.join(checkpoint_dir, 'lead_model.ckpt'),
                step=self.global_step,
                accelerator=None,
            )
            torch.save(self.lead_optimizer.state_dict(), os.path.join(checkpoint_dir, 'lead_optimizer.pt'))
            torch.save(self.lead_scheduler.state_dict(), os.path.join(checkpoint_dir, 'lead_scheduler.pt'))
            torch.save(
                self.lead_reference.get_backbone_state_dict(),
                os.path.join(checkpoint_dir, 'lead_reference_backbone.pt'),
            )

    def _save_checkpoint(self):
        checkpoint_dir = self._checkpoint_dir()
        self._world_barrier()
        if self.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
        self._world_barrier()
        self._save_stage_checkpoint(checkpoint_dir)
        self._world_barrier()
        if self.is_main_process:
            maybe_trim_checkpoints(self.output_dir, self.config.save_total_limit)
        self._world_barrier()

    def _load_policy_checkpoint(self, policy, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        backbone_state = {
            key[len('backbone.'):]: value
            for key, value in checkpoint['state_dict'].items()
            if key.startswith('backbone.')
        }
        policy.load_backbone_state_dict(backbone_state)
        policy.load_ema_state(checkpoint.get('ema'))

    def _load_checkpoint(self, checkpoint_dir):
        trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
        backend_state_path = os.path.join(checkpoint_dir, 'backend_state.json')
        ensure_exists(trainer_state_path, 'trainer state')
        ensure_exists(backend_state_path, 'process-group backend state')
        with open(trainer_state_path) as handle:
            trainer_state = json.load(handle)
        with open(backend_state_path) as handle:
            backend_state = json.load(handle)

        expected_backend = 'process_group_ddp'
        actual_backend = backend_state.get('distributed_backend')
        if actual_backend != expected_backend:
            raise ValueError(
                f'Checkpoint backend mismatch: expected {expected_backend}, found {actual_backend!r}'
            )
        if int(backend_state.get('world_size', -1)) != self.world_size:
            raise ValueError(
                'Checkpoint world_size mismatch for process_group_ddp backend: '
                f"{backend_state.get('world_size')} vs current {self.world_size}"
            )
        if int(backend_state.get('stage_group_size', -1)) != self.stage_group_size:
            raise ValueError(
                'Checkpoint stage_group_size mismatch for process_group_ddp backend: '
                f"{backend_state.get('stage_group_size')} vs current {self.stage_group_size}"
            )

        if self.is_denovo_rank:
            ensure_exists(os.path.join(checkpoint_dir, 'denovo_model.ckpt'), 'de novo model checkpoint')
            ensure_exists(os.path.join(checkpoint_dir, 'denovo_optimizer.pt'), 'de novo optimizer state')
            ensure_exists(os.path.join(checkpoint_dir, 'denovo_scheduler.pt'), 'de novo scheduler state')
            ensure_exists(os.path.join(checkpoint_dir, 'denovo_reference_backbone.pt'), 'de novo reference state')
            self._load_policy_checkpoint(self.denovo_policy, os.path.join(checkpoint_dir, 'denovo_model.ckpt'))
            self.denovo_optimizer.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, 'denovo_optimizer.pt'), map_location='cpu', weights_only=False)
            )
            self.denovo_scheduler.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, 'denovo_scheduler.pt'), map_location='cpu', weights_only=False)
            )
            self.denovo_reference.load_backbone_state_dict(
                torch.load(
                    os.path.join(checkpoint_dir, 'denovo_reference_backbone.pt'),
                    map_location='cpu',
                    weights_only=False,
                )
            )
        else:
            ensure_exists(os.path.join(checkpoint_dir, 'lead_model.ckpt'), 'lead model checkpoint')
            ensure_exists(os.path.join(checkpoint_dir, 'lead_optimizer.pt'), 'lead optimizer state')
            ensure_exists(os.path.join(checkpoint_dir, 'lead_scheduler.pt'), 'lead scheduler state')
            ensure_exists(os.path.join(checkpoint_dir, 'lead_reference_backbone.pt'), 'lead reference state')
            self._load_policy_checkpoint(self.lead_policy, os.path.join(checkpoint_dir, 'lead_model.ckpt'))
            self.lead_optimizer.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, 'lead_optimizer.pt'), map_location='cpu', weights_only=False)
            )
            self.lead_scheduler.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, 'lead_scheduler.pt'), map_location='cpu', weights_only=False)
            )
            self.lead_reference.load_backbone_state_dict(
                torch.load(
                    os.path.join(checkpoint_dir, 'lead_reference_backbone.pt'),
                    map_location='cpu',
                    weights_only=False,
                )
            )

        self.global_step = int(trainer_state['global_step'])
        self.generation_cycle_idx = int(trainer_state['generation_cycle_idx'])
        self._last_train_metrics = trainer_state.get('last_metrics')
        self._buffer_iteration = 0
        self._buffered_inputs = None
        self._buffer_metadata = None
        self._last_rollout_metrics = None
        self._world_barrier()

    def train(self, resume_from_checkpoint=None):
        if resume_from_checkpoint is not None:
            logger.info('Resuming from checkpoint: %s', resume_from_checkpoint)
            self._load_checkpoint(resume_from_checkpoint)

        while self.global_step < self.config.max_steps:
            buffered_inputs, iteration_idx = self._prepare_inputs(mode='train')
            self.optimizer.zero_grad(set_to_none=True)

            for chunk_idx in range(self.config.gradient_accumulation_steps):
                inputs = buffered_inputs[chunk_idx]
                with self._maybe_no_sync(chunk_idx):
                    stage_loss, stage_metrics = self._compute_stage_loss(inputs, iteration_idx)
                    (stage_loss / self.config.gradient_accumulation_steps).backward()
                self._record_stage_loss_metrics(stage_metrics)

            grad_norm = self._stage_grad_norm()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.policy.update_ema()
            self._append_optimizer_metrics(grad_norm)

            self.global_step += 1
            self._buffer_iteration += 1
            if self._buffer_iteration >= self.config.num_iterations:
                self._clear_rollout_buffer()

            self._sync_stage_reference()

            should_log = self.global_step == 1 and self.config.logging_first_step
            should_log = should_log or (self.global_step % self.config.logging_steps == 0)
            if should_log:
                stage_metrics = self._consume_logged_metrics()
                payload = {
                    'metrics': stage_metrics,
                    'rows': list(self._textual_logs),
                }
                self._textual_logs = []
                if self.is_denovo_rank and self.is_stage_main:
                    lead_payload = self._recv_stage_log_payload()
                    merged_metrics = self._merge_stage_metrics(payload['metrics'], lead_payload['metrics'])
                    self._last_train_metrics = merged_metrics
                    self._log_metrics(
                        'train',
                        merged_metrics,
                        extra_rows=payload['rows'] + lead_payload['rows'],
                    )
                elif self.is_lead_rank and self.is_stage_main:
                    self._send_stage_log_payload(payload)

            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint()

            self._world_barrier()

        if self._metrics:
            stage_metrics = self._consume_logged_metrics()
            payload = {
                'metrics': stage_metrics,
                'rows': list(self._textual_logs),
            }
            self._textual_logs = []
            if self.is_denovo_rank and self.is_stage_main:
                lead_payload = self._recv_stage_log_payload()
                self._last_train_metrics = self._merge_stage_metrics(payload['metrics'], lead_payload['metrics'])
            elif self.is_lead_rank and self.is_stage_main:
                self._send_stage_log_payload(payload)

        final_metrics = self._broadcast_main_object(self._last_train_metrics if self.is_main_process else None)
        self._last_train_metrics = final_metrics
        return TrainResult(metrics=final_metrics or {})

    def evaluate(self):
        raise NotImplementedError('Joint pipeline evaluation is not implemented')

    def log_metrics(self, split, metrics):
        self._log_metrics(split, metrics)

    def save_metrics(self, split, metrics):
        if not self.is_main_process:
            return
        path = os.path.join(self.output_dir, f'{split}_results.json')
        with open(path, 'w') as handle:
            json.dump(metrics, handle, sort_keys=True, indent=2)

    def save_state(self):
        if not self.is_main_process:
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
        self._world_barrier()
        if self.is_denovo_rank and self.is_stage_main:
            self.denovo_policy.save_checkpoint(
                os.path.join(output_dir, 'final_denovo_model.ckpt'),
                step=self.global_step,
                accelerator=None,
            )
        if self.is_lead_rank and self.is_stage_main:
            self.lead_policy.save_checkpoint(
                os.path.join(output_dir, 'final_lead_model.ckpt'),
                step=self.global_step,
                accelerator=None,
            )
        self._world_barrier()

    def close(self):
        self.base_reward_model.close()
        if self._wandb is not None:
            self._wandb.finish()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
