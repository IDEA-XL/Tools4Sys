import torch


class ESMPocketEncoder:
    def __init__(self, model_name='esm_if1_gvp4_t16_142M_UR50', device='cpu', freeze=True):
        try:
            import esm
            from esm.inverse_folding.util import get_encoder_output
        except ImportError as exc:
            raise ImportError(
                'Official ESM-IF dependencies are required for multimodal GenMol. '
                'Install the multimodal extras before using pocket_prefix_mm.'
            ) from exc

        model_loader = getattr(esm.pretrained, model_name, None)
        if model_loader is None:
            raise ValueError(f'Unsupported ESM-IF model_name: {model_name}')

        self.model_name = model_name
        self.model, self.alphabet = model_loader()
        self._get_encoder_output = get_encoder_output
        self.device = torch.device(device)
        self.model.eval()
        self.model.to(self.device)
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
        self.freeze = bool(freeze)

        embedding_dim = None
        encoder = getattr(self.model, 'encoder', None)
        if encoder is not None:
            embed_tokens = getattr(encoder, 'embed_tokens', None)
            if embed_tokens is not None and hasattr(embed_tokens, 'embedding_dim'):
                embedding_dim = int(embed_tokens.embedding_dim)
        if embedding_dim is None:
            raise ValueError('Failed to infer ESM-IF encoder embedding dimension')
        self.embedding_dim = embedding_dim

    def to(self, device):
        device = torch.device(device)
        if device != self.device:
            self.device = device
            self.model.to(self.device)
        return self

    def encode(self, pocket_coords_batch):
        if not isinstance(pocket_coords_batch, list) or not pocket_coords_batch:
            raise ValueError('pocket_coords_batch must be a non-empty list')

        outputs = []
        with torch.no_grad():
            for batch_idx, coords in enumerate(pocket_coords_batch):
                if not torch.is_tensor(coords):
                    coords = torch.as_tensor(coords, dtype=torch.float32)
                coords = coords.to(dtype=torch.float32)
                if coords.dim() != 3 or coords.size(1) != 3 or coords.size(2) != 3:
                    raise ValueError(
                        'Each pocket coordinate tensor must have shape [num_residues, 3, 3], '
                        f'got {list(coords.shape)} at batch index {batch_idx}'
                    )
                encoded = self._get_encoder_output(
                    self.model,
                    self.alphabet,
                    coords.detach().cpu().numpy(),
                )
                if encoded.dim() != 2:
                    raise ValueError(
                        f'Expected ESM-IF encoder output to have shape [num_residues, embedding_dim], got {list(encoded.shape)}'
                    )
                outputs.append(encoded.detach())
        return outputs
