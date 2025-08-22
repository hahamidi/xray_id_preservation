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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        out = self.proj(x)                # [B, num_tokens*out_features]
        return out.view(b, self.num_tokens, self.out_features)  # [B, T, D]
    
  
from typing import Dict, List, Union
import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class SimpleSDTextEncoder(nn.Module):
    """
    SD 1.5-compatible CLIP text encoder.
    Returns last_hidden_state [B, T, D] with D=768 and T<=77.
    """

    def __init__(
        self,
        use_sd15_repo: bool = True,
        sd15_repo_id: str = "runwayml/stable-diffusion-v1-5",
        clip_l14_id: str = "openai/clip-vit-large-patch14",
        max_length: int = 77,
        from_key: str = "captions",
        pad_to_max_length: bool = True,
        tokenizers_parallelism: bool = False,
        torch_dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "true" if tokenizers_parallelism else "false"

        self.from_key = from_key
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length

        if use_sd15_repo:
            # EXACTLY the tokenizer/weights SD 1.5 ships with
            self.tokenizer = CLIPTokenizer.from_pretrained(sd15_repo_id, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(sd15_repo_id, subfolder="text_encoder", torch_dtype=torch_dtype)
        else:
            # Same architecture/weights for text tower (ViT-L/14)
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_l14_id)
            self.text_encoder = CLIPTextModel.from_pretrained(clip_l14_id, torch_dtype=torch_dtype)

        self.text_encoder.to(device)
        # Ensure tokenizer length behavior matches SD 1.x expectations
        try:
            self.tokenizer.model_max_length = max_length
        except Exception:
            pass

        # Optional: hard sanity checks against the SD 1.5 config you wanted
        self._assert_sd15_text_config()

    @property
    def device(self) -> torch.device:
        return next(self.text_encoder.parameters()).device

    @torch.no_grad()
    def tokenize(self, x: Union[List[str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            tokens = {"input_ids": x}
        elif isinstance(x, list) and (len(x) == 0 or isinstance(x[0], str)):
            tokens = self.tokenizer(
                x,
                padding=("max_length" if self.pad_to_max_length else True),
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            raise TypeError(f"Unsupported input for tokenize(): {type(x)}")

        return {k: v.to(self.device) for k, v in tokens.items()}

    def forward(self, x: Union[Dict, List[str], torch.Tensor]) -> torch.Tensor:
        if (isinstance(x, dict) and "input_ids" in x) or isinstance(x, torch.Tensor):
            tokens = x if isinstance(x, dict) else {"input_ids": x}
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
        else:
            tokens = self.tokenize(x)

        outputs = self.text_encoder(**tokens, return_dict=True)
        return outputs.last_hidden_state  # [B, T(<=77), 768]

    # --- internals ---
    def _assert_sd15_text_config(self):
        cfg = self.text_encoder.config
        # Exactly what you listed earlier for SD 1.5
        assert getattr(cfg, "hidden_size", None) == 768, f"hidden_size {cfg.hidden_size}"
        assert getattr(cfg, "num_hidden_layers", None) == 12, f"layers {cfg.num_hidden_layers}"
        assert getattr(cfg, "num_attention_heads", None) == 12, f"heads {cfg.num_attention_heads}"
        assert getattr(cfg, "intermediate_size", None) == 3072, f"mlp {cfg.intermediate_size}"
        assert getattr(cfg, "max_position_embeddings", None) == 77, f"max_pos {cfg.max_position_embeddings}"
        assert getattr(cfg, "vocab_size", None) == 49408, f"vocab {cfg.vocab_size}"
        assert getattr(cfg, "projection_dim", None) == 768, f"proj {cfg.projection_dim}"
        # HF sometimes aliases quick_gelu as gelu_new
        assert getattr(cfg, "hidden_act", None) in ("quick_gelu", "gelu_new"), f"act {cfg.hidden_act}"
    

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