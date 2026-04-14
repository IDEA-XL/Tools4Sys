import argparse
import json
import os
import random
import sys

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))

from genmol.mm.crossdocked import load_crossdocked_manifest
from genmol.mm.policy import PocketPrefixCpGRPOPolicy
from genmol.rl.specs import sample_group_specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--manifest_path', required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_pockets', type=int, default=32)
    parser.add_argument('--num_samples_per_pocket', type=int, default=1)
    parser.add_argument('--generation_batch_size', type=int, default=2048)
    parser.add_argument('--generation_temperature', type=float, default=1.0)
    parser.add_argument('--randomness', type=float, default=0.3)
    parser.add_argument('--min_add_len', type=int, default=60)
    parser.add_argument('--max_completion_length', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()

    entries, _ = load_crossdocked_manifest(args.manifest_path, args.split)
    if args.num_pockets <= 0:
        raise ValueError('num_pockets must be positive')
    if args.num_samples_per_pocket <= 0:
        raise ValueError('num_samples_per_pocket must be positive')

    rng = random.Random(args.seed)
    if args.num_pockets > len(entries):
        raise ValueError(f'num_pockets exceeds available manifest entries: {args.num_pockets} vs {len(entries)}')
    selected_entries = rng.sample(entries, args.num_pockets)
    expanded_entries = []
    for entry in selected_entries:
        expanded_entries.extend([entry] * args.num_samples_per_pocket)

    specs = sample_group_specs(
        num_groups=len(expanded_entries),
        generation_temperature=args.generation_temperature,
        randomness=args.randomness,
        min_add_len=args.min_add_len,
        max_completion_length=args.max_completion_length,
        seed=args.seed,
    )
    policy = PocketPrefixCpGRPOPolicy(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        bf16=args.bf16,
        trainable=False,
    )
    pocket_raw_embeddings, pocket_mask = policy.get_pocket_raw_embeddings(
        [entry['pocket_coords'] for entry in expanded_entries]
    )
    rollout = policy.rollout_specs(
        specs=specs,
        pocket_raw_embeddings=pocket_raw_embeddings,
        pocket_mask=pocket_mask,
        generation_batch_size=min(args.generation_batch_size, len(expanded_entries)),
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as handle:
        for entry, spec, safe_string, smiles in zip(expanded_entries, specs, rollout.safe_strings, rollout.smiles):
            handle.write(
                json.dumps(
                    {
                        'source_index': int(entry['source_index']),
                        'split': entry['split'],
                        'residue_count': int(entry['residue_count']),
                        'spec': spec.__dict__,
                        'safe': safe_string,
                        'smiles': smiles,
                    },
                    sort_keys=True,
                )
                + '\n'
            )


if __name__ == '__main__':
    main()
