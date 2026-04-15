import os

import torch


def load_prompt_texts(prompt_path):
    if not prompt_path:
        raise ValueError('prompt_path is required')
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f'prompt file not found: {prompt_path}')
    with open(prompt_path) as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    if not prompts:
        raise ValueError(f'prompt file is empty: {prompt_path}')
    return prompts


class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, prompt_path):
        self.prompts = load_prompt_texts(prompt_path)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index]
