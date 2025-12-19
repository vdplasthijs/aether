from typing import Dict, override
import os

import torch
from transformers import CLIPProcessor, CLIPModel

from src.models.components.text_encoders.base_text_encoder import BaseTextEncoder


class ClipTextEncoder(BaseTextEncoder):
    def __init__(
            self,
            hf_cache_dir: str = '../.cache'
    ) -> None:
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(
            'openai/clip-vit-large-patch14',
            use_fast=True,
            cache_dir=hf_cache_dir,
        )
        self.model = CLIPModel.from_pretrained(
            'openai/clip-vit-large-patch14',
            cache_dir=hf_cache_dir,
        )

        self.output_dim = self.model.config.text_config.projection_dim

    @override
    def forward(
            self,
            batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Get text inputs
        text_input = batch.get('text')

        # Tokenize and embed
        text_tokens = self.processor(text=text_input, return_tensors='pt', padding=True)
        device = next(self.model.parameters()).device
        text_tokens = {k: v.to(device) for k, v in text_tokens.items()}

        text_embeds = self.model.get_text_features(**text_tokens)

        # Project
        if self.extra_projector is not None:
            text_embeds = self.extra_projector(text_embeds)

        return text_embeds




