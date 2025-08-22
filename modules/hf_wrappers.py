# hf_wrappers.py
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
import torch

def _maybe_dtype(x):
    return getattr(torch, x) if isinstance(x, str) else x


# -------- existing loaders --------

class UNetFromHF:
    def __new__(cls, repo_id="runwayml/stable-diffusion-v1-5", subfolder="unet",
                torch_dtype=None, **_ignored):
        return UNet2DConditionModel.from_pretrained(
            repo_id, subfolder=subfolder, torch_dtype=_maybe_dtype(torch_dtype)
        )

class FirstStageFromHF:
    def __new__(cls, repo_id="runwayml/stable-diffusion-v1-5", subfolder="vae",
                torch_dtype=None, **_ignored):
        return AutoencoderKL.from_pretrained(
            repo_id, subfolder=subfolder, torch_dtype=_maybe_dtype(torch_dtype)
        )

class SchedulerFromHF:
    def __new__(cls, repo_id="runwayml/stable-diffusion-v1-5", subfolder="scheduler",
                **_ignored):
        return DDPMScheduler.from_pretrained(repo_id, subfolder=subfolder)


# -------- NEW: split tokenizer / encoder loaders --------

class TextTokenizerFromHF:
    """Load SD-1.5 tokenizer (BPE) from the Hub."""
    def __new__(cls, repo_id="runwayml/stable-diffusion-v1-5",
                subfolder="tokenizer", **_ignored):
        return CLIPTokenizer.from_pretrained(repo_id, subfolder=subfolder)

class TextEncoderFromHF:
    """Load SD-1.5 CLIP text encoder (ViT-L/14) from the Hub."""
    def __new__(cls, repo_id="runwayml/stable-diffusion-v1-5",
                subfolder="text_encoder", torch_dtype=None, device="cpu",
                **_ignored):
        model = CLIPTextModel.from_pretrained(
            repo_id, subfolder=subfolder, torch_dtype=_maybe_dtype(torch_dtype)
        )
        return model.to(device)
