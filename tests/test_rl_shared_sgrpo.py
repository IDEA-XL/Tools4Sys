import torch

from rl_shared.sgrpo import compute_grouped_advantages, compute_sgrpo_advantages


def test_compute_grouped_advantages_matches_leave_one_out():
    rewards = torch.tensor([1.0, 2.0, 10.0, 14.0])
    advantages, _, zero_std_ratio = compute_grouped_advantages(rewards, num_generations=2, scale_rewards=False)
    assert torch.allclose(advantages, torch.tensor([-1.0, 1.0, -4.0, 4.0]))
    assert zero_std_ratio == 0.0


def test_compute_sgrpo_advantages_combines_rollout_and_group_terms():
    rollout_rewards = torch.tensor([1.0, 3.0, 5.0, 9.0])
    group_rewards = torch.tensor([0.2, 0.8])
    final_advantages, expanded_group_advantages, rollout_advantages, metrics = compute_sgrpo_advantages(
        rollout_rewards=rollout_rewards,
        group_rewards=group_rewards,
        num_generations=2,
        supergroup_num_groups=2,
        group_advantage_weight=0.5,
        scale_rewards=False,
    )
    assert final_advantages.shape == rollout_rewards.shape
    assert expanded_group_advantages.shape == rollout_rewards.shape
    assert rollout_advantages.shape == rollout_rewards.shape
    assert metrics['group_advantage_weight'] == 0.5
