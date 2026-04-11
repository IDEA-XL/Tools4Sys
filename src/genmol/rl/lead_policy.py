import random
from dataclasses import dataclass

import safe as sf
import torch
import torch.nn.functional as F

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


def select_valid_mutation_interval(special_token_idx, seed_length, max_position_embeddings, rng):
    if seed_length <= 0:
        raise ValueError(f'seed_length must be positive, got {seed_length}')
    if max_position_embeddings <= 0:
        raise ValueError(
            f'max_position_embeddings must be positive, got {max_position_embeddings}'
        )
    if seed_length > max_position_embeddings:
        raise ValueError(
            'seed_length exceeds model maximum context: '
            f'{seed_length} vs {max_position_embeddings}'
        )
    if len(special_token_idx) < 2:
        raise ValueError(
            'special_token_idx must contain at least two boundary markers, got '
            f'{special_token_idx}'
        )

    candidate_intervals = []
    for left_idx, right_idx in zip(special_token_idx[:-1], special_token_idx[1:]):
        mask_start_idx = int(left_idx) + 1
        mask_end_idx = int(right_idx)
        removable_width = mask_end_idx - mask_start_idx
        max_insert = max_position_embeddings - seed_length + removable_width
        if max_insert >= 1:
            candidate_intervals.append((mask_start_idx, mask_end_idx, int(max_insert)))

    if not candidate_intervals:
        raise ValueError(
            'No valid mutation interval can keep the prompt within model context: '
            f'seed_length={seed_length} max_position_embeddings={max_position_embeddings} '
            f'special_token_idx={special_token_idx}'
        )

    return candidate_intervals[rng.randrange(len(candidate_intervals))]


class LeadOptCpGRPOPolicy(GenMolCpGRPOPolicy):
    def __init__(self, *args, score_chunk_size=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.slicer = Slicer()
        if score_chunk_size <= 0:
            raise ValueError('score_chunk_size must be positive')
        self.score_chunk_size = int(score_chunk_size)

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
            max_insert = max_position_embeddings - seed_ids.numel() + mask_end_idx - mask_start_idx
        else:
            dot_positions = (seed_ids == self.model.tokenizer('.')['input_ids'][1]).nonzero(as_tuple=True)[0].tolist()
            special_token_idx = [0] + dot_positions + [seed_ids.numel() - 1]
            mask_start_idx, mask_end_idx, max_insert = select_valid_mutation_interval(
                special_token_idx=special_token_idx,
                seed_length=int(seed_ids.numel()),
                max_position_embeddings=int(max_position_embeddings),
                rng=rng,
            )
            num_insert_mask = rng.randint(5, 15)
        if max_insert < 1:
            raise ValueError(
                'No room to insert mask tokens without exceeding model context: '
                f'seed_smiles={seed_smiles!r} seed_length={seed_ids.numel()} '
                f'mask_start_idx={mask_start_idx} mask_end_idx={mask_end_idx} '
                f'max_position_embeddings={max_position_embeddings}'
            )
        num_insert_mask = min(num_insert_mask, max_insert)

        new_ids = torch.hstack(
            [
                seed_ids[:mask_start_idx],
                torch.full((num_insert_mask,), self.mask_index, dtype=torch.long),
                seed_ids[mask_end_idx:],
            ]
        )
        if new_ids.numel() > max_position_embeddings:
            raise ValueError(
                'Constructed lead prompt exceeds model context: '
                f'{new_ids.numel()} vs {max_position_embeddings} for seed {seed_smiles!r}'
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
            score_chunk_size=self.score_chunk_size,
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

        max_seq_len = max(chunk.size(1) for chunk in chunk_outputs)
        padded_outputs = []
        padded_masks = []
        for output_chunk, mask_chunk in zip(chunk_outputs, chunk_masks):
            pad_width = max_seq_len - output_chunk.size(1)
            if pad_width < 0:
                raise ValueError('pad_width must be non-negative')
            if pad_width > 0:
                output_chunk = F.pad(output_chunk, (0, pad_width), value=self.pad_index)
                mask_chunk = F.pad(mask_chunk, (0, pad_width), value=False)
            padded_outputs.append(output_chunk)
            padded_masks.append(mask_chunk)

        token_ids = torch.cat(padded_outputs, dim=0)
        completion_mask = torch.cat(padded_masks, dim=0)
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
