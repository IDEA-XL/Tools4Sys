import argparse
import json

from progen2.rewards import CompositeProteinReward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequences-json', required=True)
    parser.add_argument('--rewards-config-json', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--default-reward-batch-size', type=int, default=None)
    args = parser.parse_args()

    with open(args.sequences_json) as handle:
        sequences = json.load(handle)
    with open(args.rewards_config_json) as handle:
        reward_config = json.load(handle)

    reward = CompositeProteinReward(
        reward_config,
        device=args.device,
        default_reward_batch_size=args.default_reward_batch_size,
    )
    reward.calibrate(sequences[: min(4096, len(sequences))])
    values, metrics = reward.score(sequences)
    print(json.dumps({'scores': values, 'metrics': metrics}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
