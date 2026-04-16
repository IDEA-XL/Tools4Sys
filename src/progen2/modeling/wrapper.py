import os
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel

from progen2.checkpoint import PROGEN2_SGRPO_VARIANT, stamp_checkpoint_variant
from progen2.modeling.official import get_progen_model_class
from progen2.modeling.tokenizer import OfficialProGen2Tokenizer


def resolve_checkpoint_load_dir(checkpoint_dir, checkpoint_subdir):
    if not checkpoint_dir:
        raise ValueError('checkpoint_dir is required')
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    if checkpoint_subdir:
        load_dir = os.path.join(checkpoint_dir, checkpoint_subdir)
    else:
        load_dir = checkpoint_dir
    if not os.path.isdir(load_dir):
        raise FileNotFoundError(f'ProGen2 checkpoint directory not found: {load_dir}')
    return load_dir


class OfficialProGen2CausalLM:
    def __init__(
        self,
        *,
        official_code_dir,
        checkpoint_dir,
        tokenizer_path,
        checkpoint_subdir=None,
        device='cpu',
        use_fp16=False,
        autocast_dtype=None,
    ):
        self.device = torch.device(device)
        self.official_code_dir = str(official_code_dir)
        self.checkpoint_dir = resolve_checkpoint_load_dir(checkpoint_dir, checkpoint_subdir)
        self.tokenizer = OfficialProGen2Tokenizer(tokenizer_path)
        self.autocast_dtype = autocast_dtype

        model_cls = get_progen_model_class(self.official_code_dir)
        load_kwargs = {'low_cpu_mem_usage': True}
        if use_fp16 and self.device.type == 'cuda':
            load_kwargs['torch_dtype'] = torch.float16
        self.model = model_cls.from_pretrained(self.checkpoint_dir, **load_kwargs)
        self.model.to(self.device)

    def _root_model(self):
        model = self.model
        if isinstance(model, DistributedDataParallel):
            model = model.module
        while hasattr(model, 'module'):
            model = model.module
        return model

    @property
    def autocast_context(self):
        if self.device.type != 'cuda' or self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type='cuda', dtype=self.autocast_dtype)

    def train(self):
        self._root_model().train()

    def eval(self):
        self._root_model().eval()

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)

    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self._root_model().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

    def trainable_parameters(self):
        return self._root_model().parameters()

    def save_checkpoint(self, checkpoint_dir, extra_state=None):
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._root_model().save_pretrained(checkpoint_dir)
        extra_state = dict(extra_state or {})
        stamp_checkpoint_variant(extra_state, PROGEN2_SGRPO_VARIANT)
        torch.save(extra_state, os.path.join(checkpoint_dir, 'trainer_state.pt'))

    @staticmethod
    def load_trainer_state(checkpoint_dir):
        path = os.path.join(checkpoint_dir, 'trainer_state.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(f'ProGen2 trainer_state.pt not found: {path}')
        return torch.load(path, map_location='cpu', weights_only=False)
