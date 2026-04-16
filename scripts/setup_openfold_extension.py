import argparse
import importlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REQUIRED_SOURCE_PATHS = [
    'setup.py',
    'openfold/__init__.py',
    'openfold/utils/kernel/csrc/softmax_cuda.cpp',
    'openfold/utils/kernel/csrc/softmax_cuda_kernel.cu',
]


def validate_source_dir(source_dir: Path):
    missing = [str(source_dir / path) for path in REQUIRED_SOURCE_PATHS if not (source_dir / path).exists()]
    if missing:
        raise FileNotFoundError(
            'OpenFold source directory is incomplete. Missing required files: ' + ', '.join(missing)
        )


def validate_overlay_dir(overlay_dir: Path):
    if not overlay_dir.exists():
        raise FileNotFoundError(f'Python overlay directory not found: {overlay_dir}')
    if not overlay_dir.is_dir():
        raise NotADirectoryError(f'Python overlay path is not a directory: {overlay_dir}')


def patch_source_tree(build_root: Path):
    setup_path = build_root / 'setup.py'
    original = setup_path.read_text()
    if "'-std=c++14'" not in original:
        raise RuntimeError(f'Expected OpenFold setup.py to request -std=c++14, but did not find that token in {setup_path}')
    patched = original.replace("'-std=c++14'", "'-std=c++17'")
    setup_path.write_text(patched)


def build_extension(source_dir: Path, work_dir: Path):
    build_root = work_dir / 'openfold_build'
    shutil.copytree(source_dir, build_root)
    patch_source_tree(build_root)
    subprocess.check_call([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=build_root)
    matches = sorted(build_root.glob('attn_core_inplace_cuda*.so'))
    if not matches:
        raise FileNotFoundError(
            f'OpenFold build did not produce attn_core_inplace_cuda*.so under {build_root}'
        )
    return build_root, matches[0]


def install_overlay(build_root: Path, extension_path: Path, overlay_dir: Path):
    target_openfold = overlay_dir / 'openfold'
    if target_openfold.exists():
        shutil.rmtree(target_openfold)
    shutil.copytree(build_root / 'openfold', target_openfold)
    for existing in overlay_dir.glob('attn_core_inplace_cuda*.so'):
        existing.unlink()
    shutil.copy2(extension_path, overlay_dir / extension_path.name)


def verify_overlay(overlay_dir: Path):
    sys.path.insert(0, str(overlay_dir))
    try:
        module = importlib.import_module('attn_core_inplace_cuda')
        importlib.import_module('openfold')
    finally:
        try:
            sys.path.remove(str(overlay_dir))
        except ValueError:
            pass
    print(module.__file__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--overlay-dir', required=True)
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    overlay_dir = Path(args.overlay_dir).expanduser().resolve()
    validate_source_dir(source_dir)
    validate_overlay_dir(overlay_dir)

    with tempfile.TemporaryDirectory(prefix='openfold_ext_build_') as tmpdir:
        build_root, extension_path = build_extension(source_dir, Path(tmpdir))
        install_overlay(build_root, extension_path, overlay_dir)
    verify_overlay(overlay_dir)


if __name__ == '__main__':
    main()
