from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn

class BaseTextEncoder(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.processor: nn.Module | None = None
        self.text_model: nn.Module = None
        self.projector: nn.Module | None = None
        self.output_dim: int | None = None

    @abstractmethod
    def forward(
            self,
            batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pass