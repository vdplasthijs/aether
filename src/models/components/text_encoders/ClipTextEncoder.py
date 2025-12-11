from typing import Dict, override

import torch
from transformers import CLIPProcessor, CLIPModel

from src.models.components.text_encoders.base_text_encoder import BaseTextEncoder


class ClipTextEncoder(BaseTextEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')

        model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
        self.model = model.text_model
        self.projector = model.text_projection

        self.output_dim = model.config.text_config.projection_dim

    @override
    def forward(
            self,
            batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Get text inputs
        text_input = batch.get('text')
        #TODO: turn numericals into text

        # Tokenize and embed
        text_tokens = self.processor(text=text_input, return_tensors='pt', padding=True)
        text_embeds = self.model(**text_tokens)[0] # from pooled cls

        # Project
        if self.projector is not None:
            text_embeds = self.projector(text_embeds)

        return text_embeds



