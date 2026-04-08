import itertools
import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch
from torch.nn.parallel import DistributedDataParallel

from genmol.model import GenMol
from genmol.rl.cpgrpo import compute_coupled_log_probs
from genmol.utils.bracket_safe_converter import bracketsafe2safe
from genmol.utils.utils_chem import safe_to_smiles


@dataclass
class RolloutBatch:
    token_ids: torch.Tensor
    completion_mask: torch.Tensor
    safe_strings: list[str]
    smiles: list[str | None]


def _move_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _move_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_cpu(item) for item in value)
    return value


class GenMolCpGRPOPolicy:
    def __init__(self, checkpoint_path, device, precision='bf16', trainable=True):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.precision = precision
        self.model = GenMol.load_from_checkpoint(checkpoint_path, map_location='cpu')
        self.model.to(self.device)

        if self.model.ema:
            self.model.ema.move_shadow_params_to_device(self.device)
            self.model.ema.copy_to(itertools.chain(self.model.backbone.parameters()))
        self.mask_index = self.model.mask_index
        self.bos_index = self.model.bos_index
        self.eos_index = self.model.eos_index
        self.pad_index = self.model.tokenizer.pad_token_id
        self.use_bracket_safe = bool(self.model.config.training.get('use_bracket_safe'))

        if not trainable:
            self.freeze()

    def freeze(self):
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @property
    def autocast_context(self):
        if self.device.type != 'cuda':
            return nullcontext()
        if self.precision != 'bf16':
            return nullcontext()
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)

    def _unwrap_backbone(self):
        backbone = self.model.backbone
        if isinstance(backbone, DistributedDataParallel):
            return backbone.module
        return backbone

    def trainable_parameters(self):
        return self._unwrap_backbone().parameters()

    def update_ema(self):
        if self.model.ema:
            self.model.ema.update(itertools.chain(self._unwrap_backbone().parameters()))

    @contextmanager
    def backbone_eval_mode(self):
        backbone = self.model.backbone
        was_training = backbone.training
        backbone.eval()
        try:
            yield
        finally:
            backbone.train(was_training)

    def forward_logits(self, input_ids):
        attention_mask = input_ids != self.pad_index
        with self.autocast_context:
            logits = self.model.backbone(input_ids, attention_mask=attention_mask)['logits']
        return logits.float()

    def coupled_log_probs(self, token_ids, completion_mask, num_iterations, base_seed, requires_grad):
        seeds = [int(base_seed) + idx for idx in range(num_iterations)]

        def score_fn(batch):
            return self.forward_logits(batch)

        with self.backbone_eval_mode():
            return compute_coupled_log_probs(
                score_fn=score_fn,
                token_ids=token_ids,
                completion_mask=completion_mask,
                mask_token_id=self.mask_index,
                seeds=seeds,
                requires_grad=requires_grad,
            )

    def _decode_safe_strings(self, token_ids):
        return self.model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    def _decode_smiles(self, safe_strings):
        smiles_list = []
        for safe_string in safe_strings:
            try:
                if self.use_bracket_safe:
                    smiles = safe_to_smiles(bracketsafe2safe(safe_string), fix=True)
                else:
                    smiles = safe_to_smiles(safe_string, fix=True)
            except Exception:
                smiles = None

            if smiles:
                smiles = sorted(smiles.split('.'), key=len)[-1]
            smiles_list.append(smiles)
        return smiles_list

    def rollout_group(self, spec, num_samples, seed):
        if num_samples <= 0:
            raise ValueError('num_samples must be positive')

        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)

        token_ids = torch.full(
            (num_samples, spec.add_seq_len + 2),
            fill_value=self.mask_index,
            device=self.device,
            dtype=torch.long,
        )
        token_ids[:, 0] = self.bos_index
        token_ids[:, -1] = self.eos_index
        completion_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        completion_mask[:, 1:-1] = True

        with self.backbone_eval_mode():
            with torch.no_grad():
                x = token_ids
                num_steps = max(self.model.mdlm.get_num_steps_confidence(x), 2)
                attention_mask = x != self.pad_index
                for step_idx in range(num_steps):
                    logits = self.forward_logits(x)
                    x = self.model.mdlm.step_confidence(
                        logits,
                        x,
                        step_idx,
                        num_steps,
                        spec.softmax_temp,
                        spec.randomness,
                    )

        safe_strings = self._decode_safe_strings(x)
        smiles = self._decode_smiles(safe_strings)
        return RolloutBatch(
            token_ids=x,
            completion_mask=completion_mask,
            safe_strings=safe_strings,
            smiles=smiles,
        )

    def save_checkpoint(self, path, step):
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = _move_to_cpu(self.model.state_dict())
        checkpoint['state_dict'] = {
            key.replace('backbone.module.', 'backbone.'): value
            for key, value in state_dict.items()
        }
        checkpoint['global_step'] = int(step)
        checkpoint['epoch'] = 0
        checkpoint['optimizer_states'] = []
        checkpoint['lr_schedulers'] = []

        if self.model.ema:
            checkpoint['ema'] = _move_to_cpu(self.model.ema.state_dict())

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
