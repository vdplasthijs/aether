from typing import override

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.components.loss_fns.base_loss_fn import BaseLossFn


class ClipLoss(BaseLossFn):
    def __init__(
            self,
            temperature: float,
    ) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    @override
    def forward(
            self,
            mod_1: torch.Tensor,
            mod_2: torch.Tensor
    ) -> torch.Tensor:
        # Normalise inputs
        mod_1 = F.normalize(mod_1, dim=-1)
        mod_2 = F.normalize(mod_2, dim=-1)

        # Clip temperature to not exceed 100
        temperature =  torch.clamp(self.temperature.exp(), max=100)

        # Get cosine similarity
        dot_product = (mod_1 @ mod_2.T) / temperature

        targets = np.arrange(mod_1.shape[0])

        # Calculate losses per modality
        loss1 = F.cross_entropy(dot_product, targets)
        loss2 = F.cross_entropy(dot_product.T, targets)

        return ((loss1 + loss2) / 2)

if __name__ == "__main__":
    _ = ClipLoss(None)