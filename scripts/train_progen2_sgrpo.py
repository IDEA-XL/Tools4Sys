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

from progen2.rl.trainer import ProGen2SGRPOTrainer, load_config, resolve_output_dir


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    )

    config = load_config(args.config_path)
    output_dir = resolve_output_dir(config, args.config_path)
    trainer = ProGen2SGRPOTrainer(config=config, output_dir=output_dir)
    trainer.train()


if __name__ == '__main__':
    main()
