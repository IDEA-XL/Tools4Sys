import random
from dataclasses import dataclass

import safe as sf
import torch

from genmol.rl.lead_cpgrpo import get_per_token_logps_full
from genmol.rl.policy import GenMolCpGRPOPolicy
from genmol.utils.utils_chem import Slicer


@dataclass
class LeadRolloutBatch:
    input_ids: torch.Tensor
    completion_mask: torch.Tensor
    specs: list
    seed_smiles: list[str]
    safe_strings: list[str]
    smiles: list[str | None]


class LeadOptCpGRPOPolicy(GenMolCpGRPOPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slicer = Slicer()

    def _encode_seed(self, seed_smiles):
        encoded_smiles = sf.SAFEConverter(slicer=self.slicer, ignore_stereo=True).encoder(seed_smiles, allow_empty=True)
        token_ids = self.model.tokenizer(
            [encoded_smiles],
            return_tensors='pt',
            truncation=True,
            max_length=self.model.config.model.max_position_embeddings,
        )['input_ids'][0]
        return token_ids

    def _build_prompt_from_seed(self, seed_smiles, mutation_seed, min_seed_len):
        seed_ids = self._encode_seed(seed_smiles)
        rng = random.Random(int(mutation_seed))
        max_position_embeddings = self.model.config.model.max_position_embeddings

        if seed_ids.numel() < min_seed_len:
            num_insert_mask = max(1, min_seed_len - seed_ids.numel() + 1)
            mask_start_idx = seed_ids.numel() - 1
            mask_end_idx = seed_ids.numel() - 1
        else:
            dot_positions = (seed_ids == self.model.tokenizer('.')['input_ids'][1]).nonzero(as_tuple=True)[0].tolist()
            special_token_idx = [0] + dot_positions + [seed_ids.numel() - 1]
            fragment_idx = rng.randint(0, len(special_token_idx) - 2)
            mask_start_idx = special_token_idx[fragment_idx] + 1
            mask_end_idx = special_token_idx[fragment_idx + 1]
            num_insert_mask = rng.randint(5, 15)
        max_insert = max_position_embeddings - seed_ids.numel() + mask_end_idx - mask_start_idx
        num_insert_mask = max(1, min(num_insert_mask, max_insert))

        new_ids = torch.hstack(
            [
                seed_ids[:mask_start_idx],
                torch.full((num_insert_mask,), self.mask_index, dtype=torch.long),
                seed_ids[mask_end_idx:],
            ]
        )
        completion_mask = torch.zeros_like(new_ids, dtype=torch.bool)
        completion_mask[mask_start_idx:mask_start_idx + num_insert_mask] = True
        return new_ids, completion_mask

    def per_token_logps(
        self,
        input_ids,
        completion_mask,
        mask_seeds,
        gradient_accumulation_steps,
        requires_grad,
    ):
        def score_fn(batch):
            return self.forward_logits(batch)

        return get_per_token_logps_full(
            score_fn=score_fn,
            input_ids=input_ids,
            completion_mask=completion_mask,
            mask_token_id=self.mask_index,
            mask_seeds=mask_seeds,
            gradient_accumulation_steps=gradient_accumulation_steps,
            requires_grad=requires_grad,
            score_chunk_size=128,
        )

    def rollout_specs(self, specs, generation_batch_size, seed):
        if not specs:
            raise ValueError('specs must be non-empty')
        if generation_batch_size <= 0:
            raise ValueError('generation_batch_size must be positive')

        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)

        chunk_outputs = []
        chunk_masks = []
        chunk_seed_smiles = []

        with self.backbone_eval_mode():
            with torch.no_grad():
                for start in range(0, len(specs), generation_batch_size):
                    chunk_specs = specs[start:start + generation_batch_size]
                    prompt_rows = []
                    prompt_masks = []
                    for spec in chunk_specs:
                        prompt_ids, completion_mask = self._build_prompt_from_seed(
                            seed_smiles=spec.seed_smiles,
                            mutation_seed=spec.mutation_seed,
                            min_seed_len=spec.min_seed_len,
                        )
                        prompt_rows.append(prompt_ids)
                        prompt_masks.append(completion_mask)
                        chunk_seed_smiles.append(spec.seed_smiles)

                    pad_len = max(row.numel() for row in prompt_rows)
                    token_ids = torch.full(
                        (len(chunk_specs), pad_len),
                        fill_value=self.pad_index,
                        dtype=torch.long,
                        device=self.device,
                    )
                    completion_mask = torch.zeros(
                        (len(chunk_specs), pad_len),
                        dtype=torch.bool,
                        device=self.device,
                    )
                    for row_idx, (row_ids, row_mask) in enumerate(zip(prompt_rows, prompt_masks)):
                        row_len = row_ids.numel()
                        token_ids[row_idx, :row_len] = row_ids.to(self.device)
                        completion_mask[row_idx, :row_len] = row_mask.to(self.device)

                    x = token_ids
                    num_steps = max(self.model.mdlm.get_num_steps_confidence(x), 2)
                    for step_idx in range(num_steps):
                        logits = self.forward_logits(x)
                        x = self.model.mdlm.step_confidence(
                            logits,
                            x,
                            step_idx,
                            num_steps,
                            chunk_specs[0].generation_temperature,
                            chunk_specs[0].randomness,
                        )

                    chunk_outputs.append(x.detach().clone())
                    chunk_masks.append(completion_mask.detach().clone())

        token_ids = torch.cat(chunk_outputs, dim=0)
        completion_mask = torch.cat(chunk_masks, dim=0)
        safe_strings = self._decode_safe_strings(token_ids)
        smiles = self._decode_smiles(safe_strings)
        return LeadRolloutBatch(
            input_ids=token_ids,
            completion_mask=completion_mask,
            specs=list(specs),
            seed_smiles=chunk_seed_smiles,
            safe_strings=safe_strings,
            smiles=smiles,
        )
