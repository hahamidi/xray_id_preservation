import torch
import torch.nn as nn

import os
from typing import Dict, List, Union


from transformers import AutoTokenizer, CLIPTextModel


class LabelToToken(nn.Module):
    """
    Reads a label vector from the batch and projects it into tokens.
    Expects batch["labels"] or another key specified in config.
    """
    def __init__(self, in_features: int, out_features: int, ip_num_tokens_label: int, from_key: str = "labels"):
        super().__init__()
        self.num_tokens = ip_num_tokens_label
        self.out_features = out_features
        self.from_key = from_key

        self.proj = nn.Linear(in_features, out_features * self.num_tokens)

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch[self.from_key]          # e.g. batch["labels"], shape [B, in_features]
        b = x.size(0)
        out = self.proj(x)                # [B, num_tokens*out_features]
        return out.view(b, self.num_tokens, self.out_features)  # [B, T, D]


class ImageToToken(nn.Module):
    """
    Reads an image embedding from the batch and projects it into tokens.
    Expects batch["image_embeds"] or another key specified in config.
    """
    def __init__(self, in_features: int, out_features: int, ip_num_tokens_image: int, from_key: str = "image_embeds"):
        super().__init__()
        self.num_tokens = ip_num_tokens_image
        self.out_features = out_features
        self.from_key = from_key

        self.proj = nn.Linear(in_features, out_features * self.num_tokens)

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch[self.from_key]          # e.g. batch["image_embeds"], shape [B, in_features]
        b = x.size(0)
        out = self.proj(x)                # [B, num_tokens*out_features]
        return out.view(b, self.num_tokens, self.out_features)  # [B, T, D]
    
  



class SimpleSDTextEncoder(nn.Module):
    """
    Simple Stable Diffusion-style text encoder (single CLIP).
    Reads text strings from batch[from_key], tokenizes, and returns embeddings [B, T, D].

    Example config:
      target: encoders.text_encoders.SimpleSDTextEncoder
      params:
        model_name_or_path: openai/clip-vit-base-patch32
        max_length: 77
        from_key: captions
    """

    def __init__(
        self,
        model_name_or_path: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
        from_key: str = "captions",
        pad_to_max_length: bool = True,
        tokenizers_parallelism: bool = False,
    ):
        super().__init__()

        os.environ["TOKENIZERS_PARALLELISM"] = "true" if tokenizers_parallelism else "false"

        self.from_key = from_key
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length

        # smaller CLIP backbone (base instead of large/huge)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name_or_path)

    @property
    def device(self) -> torch.device:
        return next(self.text_encoder.parameters()).device

    def forward(self, batch: Dict[str, Union[List[str], torch.Tensor]]) -> torch.Tensor:
        texts = batch[self.from_key]  # assume list[str]
        tokens = self.tokenizer(
            texts,
            padding=("max_length" if self.pad_to_max_length else True),
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        outputs = self.text_encoder(**tokens, return_dict=True)
        hidden = outputs.last_hidden_state  # [B, T, D]

        return hidden
    

if __name__ == "__main__":
    # Example usage
    label_encoder = LabelToToken(in_features=14, out_features=2048, ip_num_tokens_label=4, from_key="labels")
    image_encoder = ImageToToken(in_features=1024, out_features=2048, ip_num_tokens_image=4, from_key="image_embeds")
    text_encoder = SimpleSDTextEncoder(model_name_or_path="openai/clip-vit-base-patch32", max_length=77, from_key="captions")
    
    # Dummy batch
    batch = {
        "labels": torch.randn(2, 14),  # 2 samples, 14 features
        "image_embeds": torch.randn(2, 1024),  # 2 samples, 1024 features
        "captions": ["A sample caption", "Another caption"]
    }

    label_tokens = label_encoder(batch)
    image_tokens = image_encoder(batch)
    text_tokens = text_encoder(batch)

    print("Label Tokens Shape:", label_tokens.shape)
    print("Image Tokens Shape:", image_tokens.shape)
    print("Text Tokens Shape:", text_tokens.shape)
    # Output shapes should be:
    # Label Tokens Shape: [2, 4, 2048]
    # Image Tokens Shape: [2, 4, 2048]
    # Text Tokens Shape: [2, 77, D] where D is the hidden size of the text encoder