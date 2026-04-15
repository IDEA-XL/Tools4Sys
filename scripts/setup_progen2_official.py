import argparse
import tarfile
from pathlib import Path

import requests


OFFICIAL_BASE = 'https://raw.githubusercontent.com/salesforce/progen/main/progen2'
CHECKPOINT_BASE = 'https://storage.googleapis.com/sfr-progen-research/checkpoints'


def download(url, destination):
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('wb') as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    download(f'{OFFICIAL_BASE}/tokenizer.json', output_dir / 'tokenizer.json')
    download(f'{OFFICIAL_BASE}/models/progen/configuration_progen.py', output_dir / 'models' / 'progen' / 'configuration_progen.py')
    download(f'{OFFICIAL_BASE}/models/progen/modeling_progen.py', output_dir / 'models' / 'progen' / 'modeling_progen.py')
    (output_dir / 'models' / '__init__.py').write_text('')
    (output_dir / 'models' / 'progen' / '__init__.py').write_text('')
    (output_dir / 'prompts_unconditional.txt').write_text('1\n')

    if args.model:
        checkpoint_url = f'{CHECKPOINT_BASE}/{args.model}.tar.gz'
        destination = output_dir / 'checkpoints' / f'{args.model}.tar.gz'
        download(checkpoint_url, destination)
        extract_dir = output_dir / 'checkpoints' / args.model
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(destination, 'r:gz') as handle:
            handle.extractall(extract_dir)
        print(extract_dir)
    else:
        print(output_dir)


if __name__ == '__main__':
    main()
