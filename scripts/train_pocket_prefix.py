import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import hydra
import lightning as L
import omegaconf
import torch

from genmol.mm.data import get_multimodal_dataloader
from genmol.mm.model import PocketPrefixGenMol
from genmol.utils.utils_data import get_last_checkpoint


omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)


@hydra.main(version_base=None, config_path='../configs', config_name='base_pocket_prefix')
def train(config):
    wandb_logger = None
    if config.wandb.name is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            **config.wandb,
        )

    if config.training.get('use_bracket_safe'):
        config.model.vocab_size += 2

    model = PocketPrefixGenMol(config)
    ckpt_path = get_last_checkpoint(config.callback.dirpath)
    if ckpt_path is None:
        model.load_backbone_from_unimodal_checkpoint(str(config.multimodal.init_unimodal_ckpt))

    train_dataloader = get_multimodal_dataloader(config, split=str(config.multimodal_data.split), shuffle=True)
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=[hydra.utils.instantiate(config.callback)],
        strategy=hydra.utils.instantiate(
            {
                '_target_': 'lightning.pytorch.strategies.DDPStrategy',
                'find_unused_parameters': False,
            }
        ),
        logger=wandb_logger,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dataloader, ckpt_path=ckpt_path)


if __name__ == '__main__':
    train()
