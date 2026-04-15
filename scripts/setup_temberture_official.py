import argparse
from pathlib import Path

import requests


API_BASE = 'https://api.github.com/repos/ibmm-unibe-ch/TemBERTure/contents'
HEADERS = {'Accept': 'application/vnd.github+json'}
DEFAULT_PATHS = [
    'README.md',
    'temBERTure/requirements.txt',
    'temBERTure/temBERTure.py',
    'temBERTure/temBERTure_TM',
]


def _fetch_json(path):
    url = f'{API_BASE}/{path}?ref=main' if path else f'{API_BASE}?ref=main'
    response = requests.get(url, headers=HEADERS, timeout=120)
    response.raise_for_status()
    return response.json()


def _download_file(download_url, destination):
    response = requests.get(download_url, timeout=120, stream=True)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('wb') as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def _download_path(path, output_root):
    payload = _fetch_json(path)
    if isinstance(payload, dict):
        if payload['type'] != 'file':
            raise ValueError(f'Expected file payload for {path}, got {payload['type']!r}')
        destination = output_root / payload['path']
        _download_file(payload['download_url'], destination)
        return
    for item in payload:
        if item['type'] == 'file':
            destination = output_root / item['path']
            _download_file(item['download_url'], destination)
        elif item['type'] == 'dir':
            _download_path(item['path'], output_root)
        else:
            raise ValueError(f'Unsupported TemBERTure repo item type {item['type']!r} for {item['path']}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in DEFAULT_PATHS:
        _download_path(path, output_dir)
    print(output_dir)


if __name__ == '__main__':
    main()
