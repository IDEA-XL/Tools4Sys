import argparse

from progen2.modeling.wrapper import OfficialProGen2CausalLM
from progen2.rl.policy import ProGen2Policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--official-code-dir', required=True)
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--tokenizer-path', required=True)
    parser.add_argument('--prompt', default='1')
    parser.add_argument('--num-return-sequences', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint-subdir', default='float16')
    args = parser.parse_args()

    model = OfficialProGen2CausalLM(
        official_code_dir=args.official_code_dir,
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_path=args.tokenizer_path,
        checkpoint_subdir=args.checkpoint_subdir,
        device=args.device,
        use_fp16=args.device.startswith('cuda'),
    )
    policy = ProGen2Policy(model, trainable=False)
    rollout = policy.generate_rollouts(
        [args.prompt],
        num_return_sequences=args.num_return_sequences,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        seed=42,
    )
    for sequence in rollout.protein_sequences:
        print(sequence)


if __name__ == '__main__':
    main()
