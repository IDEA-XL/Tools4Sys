import torch


class ESMFoldFoldabilityScorer:
    def __init__(self, device='cpu'):
        try:
            import esm
        except ImportError as exc:
            raise ImportError('esm is required for ESMFold foldability scoring') from exc

        self.device = torch.device(device)
        self.model = esm.pretrained.esmfold_v1()
        self.model.eval().to(self.device)

    @torch.no_grad()
    def score_raw(self, sequences):
        if not sequences:
            raise ValueError('sequences must be non-empty')
        outputs = []
        for sequence in sequences:
            inference = self.model.infer(sequence)
            if 'mean_plddt' not in inference:
                raise ValueError('ESMFold inference did not return mean_plddt')
            outputs.append(float(inference['mean_plddt']) / 100.0)
        return outputs
