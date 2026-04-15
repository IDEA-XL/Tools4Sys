import importlib
import math
import os
import sys
import types
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


def _install_transformers_model_parallel_compat():
    module_name = 'transformers.utils.model_parallel_utils'
    if module_name in sys.modules:
        return
    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError:
        pass

    compat_module = types.ModuleType(module_name)

    def assert_device_map(device_map, num_blocks):
        if not isinstance(device_map, dict) or not device_map:
            raise ValueError('device_map must be a non-empty dict')
        if int(num_blocks) < 0:
            raise ValueError(f'num_blocks must be non-negative, got {num_blocks!r}')
        expected = set(range(int(num_blocks)))
        seen = []
        for device, block_ids in device_map.items():
            if not isinstance(block_ids, (list, tuple)):
                raise ValueError(
                    f'device_map[{device!r}] must be a list or tuple of block ids, '
                    f'got {type(block_ids).__name__}'
                )
            for block_id in block_ids:
                seen.append(int(block_id))
        seen_set = set(seen)
        duplicates = sorted(block_id for block_id in seen_set if seen.count(block_id) > 1)
        missing = sorted(expected - seen_set)
        extra = sorted(seen_set - expected)
        if duplicates or missing or extra:
            raise ValueError(
                'Invalid device_map for ProGen2 model parallelism: '
                f'duplicates={duplicates}, missing={missing}, extra={extra}'
            )

    def get_device_map(num_blocks, devices):
        num_blocks = int(num_blocks)
        if num_blocks < 0:
            raise ValueError(f'num_blocks must be non-negative, got {num_blocks!r}')
        devices = list(devices)
        if not devices:
            raise ValueError('devices must be non-empty')
        blocks_per_device = int(math.ceil(num_blocks / len(devices))) if num_blocks else 0
        device_map = {}
        start = 0
        for device in devices:
            end = min(start + blocks_per_device, num_blocks)
            device_map[device] = list(range(start, end))
            start = end
        return device_map

    compat_module.assert_device_map = assert_device_map
    compat_module.get_device_map = get_device_map
    sys.modules[module_name] = compat_module


@contextmanager
def official_code_import_path(official_code_dir):
    ensure_official_code_dir(official_code_dir)
    _install_transformers_model_parallel_compat()
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
