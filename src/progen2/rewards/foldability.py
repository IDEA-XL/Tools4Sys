import sys
import types

import torch

from progen2.rewards.common import iter_chunks, release_model, validate_batch_size


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


class ESMFoldFoldabilityScorer:
    def __init__(self, device='cpu', batch_size=1):
        try:
            import esm
        except ImportError as exc:
            raise ImportError('esm is required for ESMFold foldability scoring') from exc
        _ensure_openfold_lightning_compat()
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
        self.model = None

    def _ensure_loaded(self):
        if self.model is None:
            self.model = self.esm.pretrained.esmfold_v1()
            self.model.eval()
        self.model.to(self.device)

    def release(self):
        release_model(self.model, self.device)

    @torch.no_grad()
    def score_raw(self, sequences):
        if not sequences:
            raise ValueError('sequences must be non-empty')
        outputs = []
        self._ensure_loaded()
        for chunk in iter_chunks(sequences, self.batch_size):
            for sequence in chunk:
                inference = self.model.infer(sequence)
                if 'mean_plddt' not in inference:
                    raise ValueError('ESMFold inference did not return mean_plddt')
                outputs.append(float(inference['mean_plddt']) / 100.0)
        return outputs
