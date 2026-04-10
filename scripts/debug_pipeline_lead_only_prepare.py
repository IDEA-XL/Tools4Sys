import argparse
import logging
import os
import random
import sys

import torch
import torch.distributed as dist

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from genmol.rl.cpgrpo import compute_clipped_grpo_loss, compute_grouped_advantages
from genmol.rl.lead_policy import LeadOptCpGRPOPolicy
from genmol.rl.lead_reward import compute_similarity
from genmol.rl.lead_specs import LeadOptSpec
from genmol.rl.pipeline_reward import topk_mean
from genmol.rl.pipeline_trainer import JointTrainConfig, load_config
from genmol.rl.policy import GenMolCpGRPOPolicy
from genmol.rl.reward import MolecularReward
from genmol.rl.specs import (
    deserialize_specs as deserialize_denovo_specs,
    expand_group_specs as expand_denovo_group_specs,
    sample_group_specs as sample_denovo_group_specs,
    serialize_specs as serialize_denovo_specs,
)


logger = logging.getLogger(__name__)


def configure_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )


def broadcast_denovo_specs(group_specs, num_generations, accelerator):
    payload = [None]
    if accelerator.is_main_process:
        payload[0] = serialize_denovo_specs(expand_denovo_group_specs(group_specs, num_generations))
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    return deserialize_denovo_specs(payload[0])


def all_gather_objects(payload, world_size):
    if not dist.is_available() or not dist.is_initialized():
        return [payload]
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, payload)
    return gathered


def gather_variable_scalars(values, accelerator):
    local_values = [float(item) for item in values]
    gathered = all_gather_objects(local_values, accelerator.num_processes)
    local_start = sum(len(item) for item in gathered[:accelerator.process_index])
    flat = [value for shard in gathered for value in shard]
    if not flat:
        return torch.empty((0,), device=accelerator.device, dtype=torch.float32), local_start
    return torch.tensor(flat, device=accelerator.device, dtype=torch.float32), local_start


def score_lead_records(base_reward_model, sim_weight, seed_smiles_list, candidate_smiles_list):
    base_records = base_reward_model.score(candidate_smiles_list)
    combined = []
    for seed_smiles, base_record in zip(seed_smiles_list, base_records):
        if not base_record.is_valid or base_record.smiles is None:
            combined.append({'reward': -1.0, 'base_reward': -1.0, 'sim': None, 'record': base_record})
            continue
        sim = compute_similarity(seed_smiles, base_record.smiles)
        if sim is None:
            combined.append({'reward': -1.0, 'base_reward': -1.0, 'sim': None, 'record': base_record})
            continue
        combined.append(
            {
                'reward': float(base_record.reward) + sim_weight * float(sim),
                'base_reward': float(base_record.reward),
                'sim': float(sim),
                'record': base_record,
            }
        )
    return combined


def build_lead_specs(config, seed_records, cycle_seed, process_index):
    rng = random.Random(cycle_seed + 7919 + process_index * 1000)
    lead_specs = []
    valid_seed_indices = []
    for seed_idx, seed_record in enumerate(seed_records):
        if not seed_record.is_valid or seed_record.smiles is None:
            continue
        valid_seed_indices.append(seed_idx)
        for _ in range(config.lead_num_generations):
            lead_specs.append(
                LeadOptSpec(
                    seed_smiles=seed_record.smiles,
                    mutation_seed=rng.randrange(2**31),
                    generation_temperature=config.lead_generation_temperature,
                    randomness=config.lead_randomness,
                    min_seed_len=config.lead_min_seed_len,
                )
            )
    return lead_specs, valid_seed_indices


def sample_mask_seed(device, random_masking):
    if random_masking:
        return int(torch.randint(0, 2**12, (1,), device=device).item())
    return 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    if os.environ.get('GENMOL_DEBUG_ANOMALY', '1') == '1':
        torch.autograd.set_detect_anomaly(True)

    config: JointTrainConfig = load_config(args.config)
    configure_logging()

    ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=config.ddp_broadcast_buffers)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='bf16' if config.bf16 else 'no',
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device
    world_size = accelerator.num_processes
    set_seed(config.seed, device_specific=True)

    denovo_local_sample_count = config.denovo_per_device_train_batch_size
    denovo_global_sample_count = denovo_local_sample_count * world_size
    if denovo_global_sample_count % config.denovo_num_generations != 0:
        raise ValueError(
            'global de novo seed batch must be divisible by denovo_num_generations: '
            f'{denovo_global_sample_count} vs {config.denovo_num_generations}'
        )
    denovo_num_groups_global = denovo_global_sample_count // config.denovo_num_generations

    denovo_policy = GenMolCpGRPOPolicy(
        checkpoint_path=config.denovo_init_ckpt_path,
        device=device,
        bf16=config.bf16,
        trainable=False,
    )
    lead_policy = LeadOptCpGRPOPolicy(
        checkpoint_path=config.lead_init_ckpt_path,
        device=device,
        bf16=config.bf16,
        trainable=True,
        score_chunk_size=config.lead_rescore_chunk_size,
    )
    lead_reference = LeadOptCpGRPOPolicy(
        checkpoint_path=config.lead_ref_ckpt_path,
        device=device,
        bf16=config.bf16,
        trainable=False,
        score_chunk_size=config.lead_rescore_chunk_size,
    )
    if config.lead_gradient_checkpointing:
        lead_policy.enable_gradient_checkpointing(config.lead_gradient_checkpointing_kwargs)
    lead_policy.train()

    lead_optimizer = torch.optim.AdamW(
        lead_policy.trainable_parameters(),
        lr=config.lead_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
    )
    lead_policy.model.backbone, lead_optimizer = accelerator.prepare(
        lead_policy.model.backbone,
        lead_optimizer,
    )

    reward_model = MolecularReward()
    print(
        f'trainer_ready rank={accelerator.process_index} '
        f'world_size={world_size} denovo_prepared=no lead_prepared=yes'
    )
    sys.stdout.flush()

    try:
        cycle_seed = config.seed
        if accelerator.is_main_process:
            group_specs = sample_denovo_group_specs(
                num_groups=denovo_num_groups_global,
                generation_temperature=config.denovo_generation_temperature,
                randomness=config.denovo_randomness,
                min_add_len=config.denovo_min_add_len,
                seed=cycle_seed,
                max_completion_length=config.denovo_max_completion_length,
            )
        else:
            group_specs = []

        expanded_specs = broadcast_denovo_specs(group_specs, config.denovo_num_generations, accelerator)
        local_start = accelerator.process_index * denovo_local_sample_count
        local_end = (accelerator.process_index + 1) * denovo_local_sample_count
        local_denovo_specs = expanded_specs[local_start:local_end]

        denovo_rollout = denovo_policy.rollout_specs(
            specs=local_denovo_specs,
            generation_batch_size=config.denovo_generation_batch_size,
            seed=cycle_seed + accelerator.process_index * 1000,
        )
        denovo_reward_records = reward_model.score(denovo_rollout.smiles)

        lead_specs, valid_seed_indices = build_lead_specs(
            config=config,
            seed_records=denovo_reward_records,
            cycle_seed=cycle_seed,
            process_index=accelerator.process_index,
        )
        if not lead_specs:
            raise RuntimeError('No valid seed molecules generated for lead debug batch')

        lead_rollout = lead_policy.rollout_specs(
            specs=lead_specs,
            generation_batch_size=config.lead_generation_batch_size,
            seed=cycle_seed + 500000 + accelerator.process_index * 1000,
        )
        lead_records = score_lead_records(
            base_reward_model=reward_model,
            sim_weight=config.sim_weight,
            seed_smiles_list=lead_rollout.seed_smiles,
            candidate_smiles_list=lead_rollout.smiles,
        )
        local_lead_rewards = [item['reward'] for item in lead_records]
        local_lead_base_rewards = torch.tensor(
            [item['base_reward'] for item in lead_records],
            device=device,
            dtype=torch.float32,
        )

        expected = len(valid_seed_indices) * config.lead_num_generations
        if local_lead_base_rewards.numel() != expected:
            raise ValueError(
                'Lead base reward tensor size does not match grouped valid-seed layout: '
                f'{local_lead_base_rewards.numel()} vs {expected}'
            )
        _ = [
            topk_mean(
                local_lead_base_rewards[group_idx * config.lead_num_generations:(group_idx + 1) * config.lead_num_generations],
                k=config.downstream_topk,
            )
            for group_idx, _seed_idx in enumerate(valid_seed_indices)
        ]

        global_lead_rewards, local_lead_start = gather_variable_scalars(local_lead_rewards, accelerator)
        global_lead_advantages, _global_lead_reward_std, _lead_zero_std_ratio = compute_grouped_advantages(
            rewards=global_lead_rewards,
            num_generations=config.lead_num_generations,
            scale_rewards=config.scale_rewards,
        )
        local_lead_advantages = global_lead_advantages[
            local_lead_start:local_lead_start + len(local_lead_rewards)
        ].to(device)

        lead_mask_seed = sample_mask_seed(device, config.random_masking)
        lead_ref_per_token_logps = lead_reference.per_token_logps(
            input_ids=lead_rollout.input_ids.detach().clone().unsqueeze(0),
            completion_mask=lead_rollout.completion_mask.detach().clone(),
            mask_seeds=[lead_mask_seed],
            gradient_accumulation_steps=1,
            requires_grad=False,
        )

        print(
            f"batch_ready rank={accelerator.process_index} "
            f"has_samples=True shape={tuple(lead_rollout.input_ids.shape)}"
        )
        sys.stdout.flush()

        lead_input_ids = lead_rollout.input_ids.detach().clone()
        lead_completion_mask = lead_rollout.completion_mask.detach().clone()
        per_token_logps = lead_policy.per_token_logps(
            input_ids=lead_input_ids.unsqueeze(0),
            completion_mask=lead_completion_mask,
            mask_seeds=[lead_mask_seed],
            gradient_accumulation_steps=1,
            requires_grad=True,
        )
        loss, step_metrics = compute_clipped_grpo_loss(
            new_log_probs=per_token_logps,
            old_log_probs=per_token_logps.detach(),
            advantages=local_lead_advantages,
            completion_mask=lead_completion_mask,
            epsilon=config.lead_epsilon,
            ref_log_probs=lead_ref_per_token_logps,
            beta=config.lead_beta,
        )
        print(
            f"lead_loss_ready rank={accelerator.process_index} "
            f"loss={float(loss.detach().cpu()):.6f} "
            f"ratio_mean={float(step_metrics['ratio_mean'].detach().cpu()):.6f}"
        )
        sys.stdout.flush()

        lead_optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        print(f'lead_backward_ok rank={accelerator.process_index}')
        sys.stdout.flush()
    finally:
        reward_model.close()


if __name__ == '__main__':
    main()
