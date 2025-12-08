from abc import ABC, abstractmethod
from typing import Any, final

import torch
from lightning import LightningModule

from src.models.components.loss_fns.base_loss_fn import BaseLossFn


class BaseModel(LightningModule, ABC):
    def __init__(
            self,
            freezing_strategy: list[str],
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            loss_fn: BaseLossFn,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn', 'eo_encoder', 'prediction_head', 'text_encoder'])

        # Loss
        self.loss_fn = loss_fn

    @final
    def freezer(self) -> None:
        for part in self.hparams.freezing_strategy:
            for param in self.__getattr__(part).parameters():
                param.requires_grad = False

    @abstractmethod
    def forward(
            self,
            batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def _step(
            self,
            batch: dict[str, torch.Tensor],
            mode: str='train',
    ) -> torch.Tensor:
        pass

    @final
    def training_step(
            self,
            batch: dict[str, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, 'train')

    @final
    def validation_step(
            self,
            batch: dict[str, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, 'val')

    @final
    def test_step(
            self,
            batch: dict[str, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, 'test')

    @final
    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}