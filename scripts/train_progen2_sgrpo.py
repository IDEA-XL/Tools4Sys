import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from progen2.rl.trainer import ProGen2SGRPOTrainer, load_config, resolve_output_dir


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    args = parser.parse_args()

    config = load_config(args.config_path)
    output_dir = resolve_output_dir(config, args.config_path)
    trainer = ProGen2SGRPOTrainer(config=config, output_dir=output_dir)
    trainer.train()


if __name__ == '__main__':
    main()
