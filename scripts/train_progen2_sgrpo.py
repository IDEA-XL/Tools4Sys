import logging
import os
from pathlib import Path

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def _configure_triton_cache_dir() -> None:
    base_dir = Path(
        os.environ.get(
            'TRITON_CACHE_DIR',
            f"/tmp/{os.environ.get('USER', 'unknown')}/triton_cache",
        )
    )
    job_id = os.environ.get('SLURM_JOB_ID', 'nojid')
    local_rank = os.environ.get('LOCAL_RANK', '0')
    cache_dir = base_dir / f"slurm{job_id}" / f"rank{local_rank}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(cache_dir)


_configure_triton_cache_dir()

from progen2.rl.trainer import (
    ProGen2SGRPOTrainer,
    find_last_checkpoint,
    load_config,
    resolve_output_dir,
)


def main():
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('--resume_from_checkpoint', default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    )

    config = load_config(args.config_path)
    resume_from_checkpoint = (
        os.path.abspath(args.resume_from_checkpoint)
        if args.resume_from_checkpoint is not None
        else None
    )
    output_dir = resolve_output_dir(
        config,
        args.config_path,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    is_main_process = int(os.environ.get('RANK', '0')) == 0
    os.makedirs(output_dir, exist_ok=True)
    last_checkpoint = find_last_checkpoint(output_dir)
    if is_main_process and resume_from_checkpoint is None and last_checkpoint is not None:
        logging.getLogger(__name__).info('Checkpoint detected, resuming training at %s.', last_checkpoint)
    if (
        is_main_process
        and resume_from_checkpoint is None
        and last_checkpoint is None
        and os.listdir(output_dir)
        and not config.overwrite_output_dir
    ):
        raise FileExistsError(f'output_dir already exists and is non-empty: {output_dir}')
    if resume_from_checkpoint is None and last_checkpoint is not None:
        resume_from_checkpoint = last_checkpoint
    if is_main_process:
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as handle:
            yaml.safe_dump(config.__dict__, handle, sort_keys=False)
    trainer = ProGen2SGRPOTrainer(config=config, output_dir=output_dir)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == '__main__':
    main()
