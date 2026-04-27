import argparse
import logging
import os
import sys

import yaml

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))

from genmol.mm.trainer import (
    PocketPrefixCpGRPOTrainer,
    find_last_checkpoint,
    load_config,
    resolve_output_dir,
)


logger = logging.getLogger(__name__)


def configure_logging(level_name):
    log_level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume_from_checkpoint', default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logging(config.log_level)
    output_dir = resolve_output_dir(config, args.config)
    is_main_process = int(os.environ.get('RANK', '0')) == 0

    os.makedirs(output_dir, exist_ok=True)
    last_checkpoint = find_last_checkpoint(output_dir)
    if is_main_process and args.resume_from_checkpoint is None and last_checkpoint is not None:
        logger.info('Checkpoint detected, resuming training at %s.', last_checkpoint)

    if (
        is_main_process
        and args.resume_from_checkpoint is None
        and last_checkpoint is None
        and os.listdir(output_dir)
        and not config.overwrite_output_dir
    ):
        raise FileExistsError(f'output_dir already exists and is non-empty: {output_dir}')

    if args.resume_from_checkpoint is None and last_checkpoint is not None:
        resume_from_checkpoint = last_checkpoint
    else:
        resume_from_checkpoint = args.resume_from_checkpoint

    if is_main_process:
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as handle:
            yaml.safe_dump(config.__dict__, handle, sort_keys=False)

    trainer = PocketPrefixCpGRPOTrainer(config=config, output_dir=output_dir)
    try:
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        metrics = train_result.metrics
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()
        trainer.save_model(output_dir)

        if config.do_eval:
            eval_metrics = trainer.evaluate()
            trainer.log_metrics('eval', eval_metrics)
            trainer.save_metrics('eval', eval_metrics)
    finally:
        trainer.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        logger.exception('Pocket-prefix mm RL training failed')
        raise
