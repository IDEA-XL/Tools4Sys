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


def test_compute_sgrpo_advantages_gates_group_reward_by_thresholds():
    rollout_rewards = torch.tensor([1.0, 3.0, 5.0, 9.0])
    group_rewards = torch.tensor([0.2, 0.8])
    group_mean_individual_rewards = {
        'qed': torch.tensor([0.90, 0.84]),
        'sa_score': torch.tensor([0.80, 0.90]),
    }

    _, _, _, metrics = compute_sgrpo_advantages(
        rollout_rewards=rollout_rewards,
        group_rewards=group_rewards,
        num_generations=2,
        supergroup_num_groups=2,
        group_advantage_weight=0.5,
        scale_rewards=False,
        hierarchy='advantage_sum',
        group_mean_individual_rewards=group_mean_individual_rewards,
        individual_reward_thresholds={'qed': 0.85, 'sa_score': 0.75},
    )

    assert metrics['active_threshold_count'] == 2.0
    assert metrics['group_reward_indicator_mean'] == 0.5
    assert metrics['group_reward_raw_mean'] == 0.5
    assert abs(metrics['group_reward_mean'] - 0.1) < 1e-6


def test_compute_sgrpo_advantages_reward_sum_uses_combined_rewards():
    rollout_rewards = torch.tensor([1.0, 3.0, 5.0, 9.0])
    group_rewards = torch.tensor([2.0, 4.0])
    weight = 0.25
    combined_rewards = torch.tensor([1.25, 2.75, 4.75, 7.75])
    expected_advantages, _, _ = compute_grouped_advantages(
        combined_rewards,
        num_generations=4,
        scale_rewards=False,
    )

    final_advantages, expanded_group_advantages, rollout_advantages, metrics = compute_sgrpo_advantages(
        rollout_rewards=rollout_rewards,
        group_rewards=group_rewards,
        num_generations=2,
        supergroup_num_groups=2,
        group_advantage_weight=weight,
        scale_rewards=False,
        hierarchy='reward_sum',
    )

    assert torch.allclose(final_advantages, expected_advantages)
    assert metrics['hierarchy_reward_sum_enabled'] == 1.0
    assert abs(metrics['combined_reward_mean'] - combined_rewards.mean().item()) < 1e-6
    assert expanded_group_advantages.shape == rollout_rewards.shape
    assert rollout_advantages.shape == rollout_rewards.shape
