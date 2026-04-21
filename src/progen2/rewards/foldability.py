import importlib
import sys
import types

import torch

from progen2.rewards.common import iter_chunks, move_model_to_device, release_model, validate_batch_size


def _ensure_openfold_lightning_compat():
    try:
        from pytorch_lightning.utilities.seed import seed_everything  # noqa: F401
        return
    except ImportError:
        pass
    try:
        from lightning_fabric.utilities.seed import seed_everything as lf_seed_everything
    except ImportError as exc:
        raise RuntimeError(
            'ESMFold foldability scoring requires a seed_everything implementation from either '
            'pytorch_lightning.utilities.seed or lightning_fabric.utilities.seed'
        ) from exc
    shim = types.ModuleType('pytorch_lightning.utilities.seed')
    shim.seed_everything = lf_seed_everything
    sys.modules['pytorch_lightning.utilities.seed'] = shim


def _ensure_openfold_deepspeed_compat():
    try:
        import deepspeed
    except ImportError:
        return
    if hasattr(deepspeed.utils, 'is_initialized'):
        return
    if not hasattr(deepspeed, 'comm') or not hasattr(deepspeed.comm, 'is_initialized'):
        raise RuntimeError(
            'OpenFold compatibility requires deepspeed.comm.is_initialized when '
            'deepspeed.utils.is_initialized is unavailable'
        )
    deepspeed.utils.is_initialized = deepspeed.comm.is_initialized


def _require_openfold_attention_core_extension():
    try:
        importlib.import_module('attn_core_inplace_cuda')
    except ImportError as exc:
        raise RuntimeError(
            'ESMFold foldability scoring requires the compiled OpenFold attention extension '
            'attn_core_inplace_cuda to be preinstalled and importable from the runtime PYTHONPATH'
        ) from exc


class ESMFoldFoldabilityScorer:
    def __init__(self, device='cpu', batch_size=1, num_recycles=1):
        try:
            import esm
        except ImportError as exc:
            raise ImportError('esm is required for ESMFold foldability scoring') from exc
        _ensure_openfold_lightning_compat()
        _ensure_openfold_deepspeed_compat()
        _require_openfold_attention_core_extension()
        try:
            import openfold  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                'ESMFold foldability scoring requires the openfold package to be preinstalled '
                'and importable from the runtime PYTHONPATH before starting ProGen2 SGRPO'
            ) from exc

        self.esm = esm
        self.device = torch.device(device)
        self.batch_size = validate_batch_size(batch_size, field_name='foldability.batch_size')
        if num_recycles is not None:
            try:
                num_recycles = int(num_recycles)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'foldability.num_recycles must be a non-negative integer or null, got {num_recycles!r}'
                ) from exc
            if num_recycles < 0:
                raise ValueError(
                    f'foldability.num_recycles must be a non-negative integer or null, got {num_recycles!r}'
                )
        self.num_recycles = num_recycles
        self.model = None
        self.last_move_to_device_sec = 0.0
        self.last_release_to_cpu_sec = 0.0

    def _ensure_loaded(self):
        if self.model is None:
            self.model = self.esm.pretrained.esmfold_v1()
            self.model.eval()
        self.last_move_to_device_sec = move_model_to_device(self.model, self.device)

    def release(self):
        self.last_release_to_cpu_sec = release_model(self.model, self.device)

    @torch.no_grad()
    def score_raw(self, sequences):
        if not sequences:
            raise ValueError('sequences must be non-empty')
        outputs = []
        self._ensure_loaded()
        for chunk in iter_chunks(sequences, self.batch_size):
            inference = self.model.infer(chunk, num_recycles=self.num_recycles)
            if 'mean_plddt' not in inference:
                raise ValueError('ESMFold inference did not return mean_plddt')
            mean_plddt = torch.as_tensor(inference['mean_plddt'], dtype=torch.float32)
            if mean_plddt.numel() != len(chunk):
                raise RuntimeError(
                    'ESMFold inference returned a different number of mean_plddt values than inputs '
                    f'for the current chunk: {mean_plddt.numel()} != {len(chunk)}'
                )
            outputs.extend((mean_plddt / 100.0).detach().cpu().tolist())
        return outputs
