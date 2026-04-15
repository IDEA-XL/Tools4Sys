import argparse
import subprocess
import sys
from pathlib import Path


OVERLAY_PACKAGES = [
    'transformers==4.51.3',
    'adapters==1.2.0',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overlay-dir', required=True)
    args = parser.parse_args()

    overlay_dir = Path(args.overlay_dir).expanduser().resolve()
    overlay_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        '-m',
        'pip',
        'install',
        '--upgrade',
        '--target',
        str(overlay_dir),
        *OVERLAY_PACKAGES,
    ]
    subprocess.check_call(command)
    print(overlay_dir)


if __name__ == '__main__':
    main()
