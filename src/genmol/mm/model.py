import itertools

import hydra.utils
import lightning as L
import torch
from bionemo.moco.distributions.prior import DiscreteMaskedPrior
from bionemo.moco.distributions.time import UniformTimeDistribution
from bionemo.moco.interpolants import MDLM
from bionemo.moco.schedules.noise.continuous_noise_transforms import LogLinearExpNoiseTransform
from torch.nn.parallel import DistributedDataParallel
from transformers import BertForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig

from genmol.mm.checkpoint import (
    POCKET_PREFIX_MM_VARIANT,
    load_checkpoint_payload,
    require_multimodal_checkpoint,
    require_unimodal_checkpoint,
    stamp_checkpoint_variant,
)
from genmol.mm.pocket_encoder import ESMPocketEncoder
from genmol.mm.prefix import extract_molecule_logits, pack_prefix_conditioning
from genmol.mm.projector import PocketPrefixProjector
from genmol.utils.ema import ExponentialMovingAverage
from genmol.utils.utils_data import get_tokenizer
from genmol.utils.utils_moco import AntitheticUniformTimeDistribution
from genmol.utils.utils_save import clean_checkpoint, fast_forward_info


class PocketPrefixGenMol(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model_variant = str(config.model_variant)
        if self.model_variant != POCKET_PREFIX_MM_VARIANT:
            raise ValueError(
                f'Expected model_variant={POCKET_PREFIX_MM_VARIANT!r}, got {self.model_variant!r}'
            )

        self.tokenizer = get_tokenizer()
        self.mask_index = self.tokenizer.mask_token_id
        self.bos_index = self.tokenizer.bos_token_id
        self.eos_index = self.tokenizer.eos_token_id
        tokenizer_vocab_size = int(len(self.tokenizer))
        configured_vocab_size = int(self.config.model.vocab_size)
        if configured_vocab_size != tokenizer_vocab_size:
            raise ValueError(
                'pocket_prefix_mm config.model.vocab_size must match total tokenizer size len(tokenizer): '
                f'{configured_vocab_size} vs {tokenizer_vocab_size}'
            )

        self.backbone = BertForMaskedLM(BertConfig.from_dict(dict(self.config.model)))
        self.pocket_encoder = ESMPocketEncoder(
            model_name=str(self.config.pocket_encoder.model_name),
            device='cpu',
            freeze=bool(self.config.pocket_encoder.freeze),
        )
        self.projector = PocketPrefixProjector(
            input_dim=int(self.pocket_encoder.embedding_dim),
            hidden_size=int(self.config.model.hidden_size),
        )
        self.max_total_positions = int(self.config.conditioning.max_total_positions)
        if self.max_total_positions != int(self.config.model.max_position_embeddings):
            raise ValueError(
                'conditioning.max_total_positions must equal model.max_position_embeddings for pocket_prefix_mm: '
                f'{self.max_total_positions} vs {self.config.model.max_position_embeddings}'
            )
        if str(self.config.conditioning.overlength_policy) != 'drop_sample':
            raise ValueError(
                f"Only conditioning.overlength_policy='drop_sample' is supported, got {self.config.conditioning.overlength_policy!r}"
            )
        if str(self.config.conditioning.projector_type) != 'mlp2':
            raise ValueError(
                f"Only conditioning.projector_type='mlp2' is supported, got {self.config.conditioning.projector_type!r}"
            )

        if self.config.training.antithetic_sampling:
            time_distribution = AntitheticUniformTimeDistribution(sampling_eps=self.config.training.sampling_eps)
        else:
            time_distribution = UniformTimeDistribution()
        prior = DiscreteMaskedPrior(num_classes=tokenizer_vocab_size, mask_dim=self.mask_index)
        noise_schedule = LogLinearExpNoiseTransform()
        self.mdlm = MDLM(
            time_distribution=time_distribution,
            prior_distribution=prior,
            noise_schedule=noise_schedule,
        )

        if self.config.training.ema > 0:
            self.ema = ExponentialMovingAverage(self._ema_parameters(), decay=self.config.training.ema)
        else:
            self.ema = None

    def _ema_parameters(self):
        return itertools.chain(self.backbone.parameters(), self.projector.parameters())

    def _trainable_parameters(self):
        return itertools.chain(self.backbone.parameters(), self.projector.parameters())

    def on_load_checkpoint(self, checkpoint):
        require_multimodal_checkpoint(checkpoint, checkpoint_path='<in-memory-lightning-checkpoint>')
        if self.ema:
            self.ema.load_state_dict(checkpoint['ema'])
        self.fast_forward_epochs, self.fast_forward_batches = fast_forward_info(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        stamp_checkpoint_variant(checkpoint, POCKET_PREFIX_MM_VARIANT)
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
        clean_checkpoint(checkpoint, self.trainer.accumulate_grad_batches)
        if 'sampler' not in checkpoint:
            checkpoint['sampler'] = {}
        if hasattr(self.trainer.train_dataloader.sampler, 'state_dict'):
            sampler_state_dict = self.trainer.train_dataloader.sampler.state_dict()
            checkpoint['sampler']['random_state'] = sampler_state_dict.get('random_state', None)
        else:
            checkpoint['sampler']['random_state'] = None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._trainable_parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
        )
        scheduler = hydra.utils.instantiate(
            {
                '_target_': 'transformers.get_constant_schedule_with_warmup',
                'num_warmup_steps': 2500,
            },
            optimizer=optimizer,
        )
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'name': 'lr',
        }
        return [optimizer], [scheduler_dict]

    def on_train_start(self):
        self.backbone.train()
        self.pocket_encoder.to(self.device)
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(self._ema_parameters())

    @staticmethod
    def _unwrap_module(module):
        if isinstance(module, DistributedDataParallel):
            module = module.module
        while hasattr(module, 'module'):
            module = module.module
        return module

    def _token_embeddings(self, input_ids):
        embedding_layer = self._unwrap_module(self.backbone).get_input_embeddings()
        return embedding_layer(input_ids)

    def encode_pocket_batch(self, pocket_coords):
        self.pocket_encoder.to(self.device)
        encoded = self.pocket_encoder.encode(pocket_coords)
        return encoded

    def project_pocket_batch(self, raw_pocket_embeddings):
        projected = []
        for embedding in raw_pocket_embeddings:
            projected.append(self.projector(embedding.to(device=self.device, dtype=torch.float32)))
        return projected

    def _raw_pocket_from_padded(self, pocket_raw_embeddings, pocket_mask):
        if pocket_raw_embeddings.dim() != 3:
            raise ValueError('pocket_raw_embeddings must have shape [batch, max_prefix_len, esm_dim]')
        if pocket_mask.dim() != 2:
            raise ValueError('pocket_mask must have shape [batch, max_prefix_len]')
        if pocket_raw_embeddings.size(0) != pocket_mask.size(0):
            raise ValueError('pocket_raw_embeddings and pocket_mask batch sizes must match')
        output = []
        for batch_idx in range(pocket_raw_embeddings.size(0)):
            valid = pocket_mask[batch_idx].to(dtype=torch.bool)
            output.append(pocket_raw_embeddings[batch_idx, valid])
        return output

    def forward_conditioned_logits(self, input_ids, attention_mask, pocket_coords):
        raw_pocket_embeddings = self.encode_pocket_batch(pocket_coords)
        return self.forward_conditioned_logits_from_raw_pocket_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pocket_raw_embeddings=raw_pocket_embeddings,
        )

    def forward_conditioned_logits_from_padded_raw_pocket(self, input_ids, attention_mask, pocket_raw_embeddings, pocket_mask):
        raw_pocket_embeddings = self._raw_pocket_from_padded(pocket_raw_embeddings, pocket_mask)
        return self.forward_conditioned_logits_from_raw_pocket_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pocket_raw_embeddings=raw_pocket_embeddings,
        )

    def forward_conditioned_logits_from_raw_pocket_embeddings(self, input_ids, attention_mask, pocket_raw_embeddings):
        if attention_mask is None:
            attention_mask = input_ids != self.tokenizer.pad_token_id
        token_embeddings = self._token_embeddings(input_ids)
        projected_prefixes = self.project_pocket_batch(pocket_raw_embeddings)
        packed = pack_prefix_conditioning(
            token_embeddings=token_embeddings,
            token_attention_mask=attention_mask,
            prefix_embeddings=projected_prefixes,
            max_total_positions=self.max_total_positions,
        )
        with torch.amp.autocast('cuda', dtype=torch.float32):
            logits = self.backbone(
                inputs_embeds=packed.inputs_embeds,
                attention_mask=packed.attention_mask,
                token_type_ids=torch.zeros_like(packed.attention_mask, dtype=torch.long),
                position_ids=packed.position_ids,
            )['logits']
        return extract_molecule_logits(logits.float(), packed.token_positions)

    def forward(self, input_ids, attention_mask=None, pocket_coords=None):
        if pocket_coords is None:
            raise ValueError('pocket_coords are required for pocket_prefix_mm forward')
        return self.forward_conditioned_logits(input_ids, attention_mask=attention_mask, pocket_coords=pocket_coords)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pocket_coords = batch['pocket_coords']
        t = self.mdlm.sample_time(input_ids.shape[0])
        xt = self.mdlm.forward_process(input_ids, t)
        logits = self.forward_conditioned_logits(
            input_ids=xt,
            attention_mask=attention_mask,
            pocket_coords=pocket_coords,
        )
        if self.config.training.global_mean_loss:
            loss = self.mdlm.loss(logits, input_ids, xt, t, mask=attention_mask, global_mean=True)
        else:
            loss = self.mdlm.loss(logits, input_ids, xt, t, mask=attention_mask).mean()
        self.log(
            name='train_loss',
            value=loss.item(),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def load_backbone_from_unimodal_checkpoint(self, checkpoint_path):
        checkpoint = load_checkpoint_payload(checkpoint_path)
        require_unimodal_checkpoint(checkpoint, checkpoint_path)
        if 'state_dict' not in checkpoint:
            raise ValueError(f'Checkpoint missing state_dict: {checkpoint_path}')
        backbone_state = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('backbone.'):
                backbone_state[key[len('backbone.'):]] = value
        if not backbone_state:
            raise ValueError(f'No backbone parameters found in unimodal checkpoint: {checkpoint_path}')
        self.backbone.load_state_dict(backbone_state, strict=True)
