import lightning as L
import torch


class CUDAMemoryMonitor(L.pytorch.callbacks.Callback):
    def __init__(self, enforce_max_reserved_ratio=None):
        self.enforce_max_reserved_ratio = enforce_max_reserved_ratio

    def on_train_start(self, trainer, pl_module):
        if trainer.strategy.root_device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        device = trainer.strategy.root_device
        if device.type != 'cuda':
            return

        peak_reserved = torch.cuda.max_memory_reserved(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        peak_reserved_gib = peak_reserved / float(1024 ** 3)
        peak_reserved_ratio = peak_reserved / float(total_memory)

        pl_module.log(
            name='train_peak_reserved_gib',
            value=peak_reserved_gib,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )
        pl_module.log(
            name='train_peak_reserved_ratio',
            value=peak_reserved_ratio,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )

        if (
            self.enforce_max_reserved_ratio is not None
            and peak_reserved_ratio > float(self.enforce_max_reserved_ratio)
        ):
            raise RuntimeError(
                'Peak CUDA reserved memory ratio exceeded configured limit: '
                f'{peak_reserved_ratio:.4f} > {float(self.enforce_max_reserved_ratio):.4f}'
            )
