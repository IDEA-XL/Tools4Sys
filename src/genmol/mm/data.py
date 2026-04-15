from dataclasses import dataclass

import torch

from genmol.mm.crossdocked import load_crossdocked_manifest
from genmol.utils.bracket_safe_converter import safe2bracketsafe
from genmol.utils.utils_data import get_tokenizer


@dataclass(frozen=True)
class ManifestConfig:
    manifest_path: str
    split: str
    max_total_positions: int


class PocketPrefixManifestDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, split):
        self.entries, self.stats = load_crossdocked_manifest(manifest_path=manifest_path, split=split)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self.entries[index]


class PocketPrefixCollator:
    def __init__(self, config):
        self.tokenizer = get_tokenizer()
        self.max_length = int(config.model.max_position_embeddings)
        self.use_bracket_safe = bool(config.training.get('use_bracket_safe'))

    def __call__(self, examples):
        if not examples:
            raise ValueError('examples must be non-empty')

        safe_strings = []
        pocket_sequences = []
        pocket_coords = []
        source_indices = []
        for example in examples:
            safe_string = example['safe']
            if self.use_bracket_safe:
                safe_string = safe2bracketsafe(safe_string)
            safe_strings.append(safe_string)
            pocket_sequences.append(example['pocket_sequence'])
            pocket_coords.append(torch.as_tensor(example['pocket_coords'], dtype=torch.float32))
            source_indices.append(int(example['source_index']))

        batch = self.tokenizer(
            safe_strings,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        del batch['token_type_ids']
        batch['pocket_sequences'] = pocket_sequences
        batch['pocket_coords'] = pocket_coords
        batch['source_indices'] = torch.tensor(source_indices, dtype=torch.long)
        return batch


def get_multimodal_dataloader(config, split=None, shuffle=None):
    dataset_split = split or str(config.multimodal_data.split)
    dataset = PocketPrefixManifestDataset(
        manifest_path=str(config.multimodal_data.manifest_path),
        split=dataset_split,
    )
    collator = PocketPrefixCollator(config)
    if shuffle is None:
        shuffle = dataset_split == 'train'

    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': int(config.loader.batch_size),
        'collate_fn': collator,
        'num_workers': int(config.loader.num_workers),
        'pin_memory': bool(config.loader.pin_memory),
        'shuffle': bool(shuffle),
        'persistent_workers': bool(config.loader.get('persistent_workers', config.loader.num_workers > 0)),
    }
    if int(config.loader.num_workers) > 0 and config.loader.get('prefetch_factor') is not None:
        dataloader_kwargs['prefetch_factor'] = int(config.loader.prefetch_factor)
    return torch.utils.data.DataLoader(**dataloader_kwargs)
