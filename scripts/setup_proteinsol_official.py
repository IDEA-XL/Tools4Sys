import argparse
import zipfile
from pathlib import Path

import requests


PROTEINSOL_ZIP_URL = 'https://protein-sol.manchester.ac.uk/cgi-bin/utilities/download_sequence_code.php'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / 'protein-sol.zip'

    response = requests.get(PROTEINSOL_ZIP_URL, timeout=120, stream=True)
    response.raise_for_status()
    with archive_path.open('wb') as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)

    with zipfile.ZipFile(archive_path) as handle:
        handle.extractall(output_dir)

    script_path = output_dir / 'protein-sol-sequence-prediction-software' / 'multiple_prediction_wrapper_export.sh'
    if not script_path.is_file():
        raise RuntimeError(f'Protein-Sol wrapper script not found after extraction: {script_path}')
    script_path.chmod(0o755)
    print(output_dir)


if __name__ == '__main__':
    main()
