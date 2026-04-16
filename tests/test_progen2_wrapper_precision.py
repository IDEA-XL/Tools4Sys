from contextlib import nullcontext

import torch

from progen2.modeling.wrapper import OfficialProGen2CausalLM


def test_autocast_context_is_disabled_without_autocast_dtype():
    model = OfficialProGen2CausalLM.__new__(OfficialProGen2CausalLM)
    model.device = torch.device('cuda')
    model.autocast_dtype = None
    context = model.autocast_context
    assert isinstance(context, nullcontext)


def test_autocast_context_uses_requested_dtype():
    model = OfficialProGen2CausalLM.__new__(OfficialProGen2CausalLM)
    model.device = torch.device('cuda')
    model.autocast_dtype = torch.bfloat16
    context = model.autocast_context
    assert getattr(context, 'fast_dtype', None) == torch.bfloat16
