from typing import override, Dict, Tuple

import torch

from src.models.base_model import BaseModel
from src.models.components.eo_encoders.base_eo_encoder import BaseEOEncoder
from src.models.components.loss_fns.base_loss_fn import BaseLossFn
from src.models.components.pred_heads.linear_pred_head import BasePredictionHead
from src.models.components.text_encoders.base_text_encoder import BaseTextEncoder


class TextAlignmentModel(BaseModel):
    def __init__(
            self,
            eo_encoder: BaseEOEncoder,
            text_encoder: BaseTextEncoder,
            num_classes: int,
            freezing_strategy: list[str],
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            loss_fn: BaseLossFn,
            prediction_head: BasePredictionHead | None = None,
    ) -> None:
        super().__init__(freezing_strategy, optimizer, scheduler, loss_fn)
        for part in freezing_strategy:
            assert part in ['eo_encoder', 'prediction_head'], ValueError(f"Unknown freezing strategy for {part} part")

        # Encoders configuration
        self.eo_encoder = eo_encoder
        self.text_encoder = text_encoder

        # Prediction head
        self.prediction_head = prediction_head
        if self.prediction_head is not None:
            self.prediction_head.set_dim(input_dim=self.eo_encoder.output_dim, output_dim=num_classes)
            self.prediction_head.configure_nn()

        # Freezing requested parts
        self.freezer()

    @override
    def forward(
            self,
            batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eo_feats = self.eo_encoder(batch)
        text_feats = self.text_encoder(batch)
        return eo_feats, text_feats

    @override
    def _step(
            self,
            batch: Dict[str, torch.Tensor],
            mode: str='train'
    ) -> torch.Tensor:
        eo_feats, text_feats = self.forward(batch)
        loss = self.loss_fn(eo_feats, text_feats)

        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

if __name__ == "__main__":
    _ = TextAlignmentModel(None, None, None, None, None, None, None)