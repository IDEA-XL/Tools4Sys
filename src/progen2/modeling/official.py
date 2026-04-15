import importlib
import os
import sys
from contextlib import contextmanager


_MODELING_IMPORT = 'models.progen.modeling_progen'


def _required_paths(official_code_dir):
    return [
        os.path.join(official_code_dir, 'models'),
        os.path.join(official_code_dir, 'models', '__init__.py'),
        os.path.join(official_code_dir, 'models', 'progen'),
        os.path.join(official_code_dir, 'models', 'progen', '__init__.py'),
        os.path.join(official_code_dir, 'models', 'progen', 'configuration_progen.py'),
        os.path.join(official_code_dir, 'models', 'progen', 'modeling_progen.py'),
        os.path.join(official_code_dir, 'tokenizer.json'),
    ]


def ensure_official_code_dir(official_code_dir):
    if not official_code_dir:
        raise ValueError('official_code_dir is required')
    missing = [path for path in _required_paths(official_code_dir) if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            'official ProGen2 code directory is incomplete. Missing required files: '
            + ', '.join(missing)
        )


@contextmanager
def official_code_import_path(official_code_dir):
    ensure_official_code_dir(official_code_dir)
    inserted = False
    if official_code_dir not in sys.path:
        sys.path.insert(0, official_code_dir)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(official_code_dir)
            except ValueError:
                pass


def load_official_modeling_module(official_code_dir):
    with official_code_import_path(official_code_dir):
        return importlib.import_module(_MODELING_IMPORT)


def get_progen_model_class(official_code_dir):
    module = load_official_modeling_module(official_code_dir)
    if not hasattr(module, 'ProGenForCausalLM'):
        raise ImportError(f'Official module {_MODELING_IMPORT} does not define ProGenForCausalLM')
    return module.ProGenForCausalLM
