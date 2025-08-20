# --- std lib ---
import os
import time
import argparse
from pathlib import Path
import yaml
import warnings
from typing import Any, Dict, Optional, Tuple

# --- torch / accelerate ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

# --- diffusers / transformers (core building blocks) ---
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer
from utils_idp import report_module

# --- your project-specific imports (placeholders / TODOs) ---
# from your_project.data import DAVIS_Dataset, collate_fn
# from your_project.encoders import DynamicEncoder
# from modify_instantID import FrozenDinoV2Encoder
# from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
# from ip_adapter.utils import is_torch2_available


# ------------------------------------------------------------
# Utility: resolve "module.ClassName" strings to actual classes
# ------------------------------------------------------------
def locate(target: str):
    """
    Import a dotted path and return the attribute/class.
    Example: "torch.nn.Linear" -> torch.nn.Linear
    """
    parts = target.split(".")
    module_path, cls_name = ".".join(parts[:-1]), parts[-1]
    mod = __import__(module_path, fromlist=[cls_name])
    return getattr(mod, cls_name)


def instantiate_from_target(target: str, params: Optional[Dict] = None):
    """
    Generic factory: instantiate any class by target string + kwargs.
    """
    cls = locate(target)
    return cls(**(params or {}))


# ------------------------------------------------------------
# Optional helpers for freezing and weight loading
# ------------------------------------------------------------
def maybe_freeze(module: Optional[nn.Module], train_flag: bool):
    """
    If module exists: set requires_grad to train_flag.
    """
    if module is None:
        return
    module.requires_grad_(bool(train_flag))


def maybe_load_state_dict(module: Optional[nn.Module], path: Optional[str], strict: bool = False):
    """
    If module & path provided: load a state_dict from disk.
    """
    if module is None or not path:
        return
    sd = torch.load(path, map_location="cpu")
    module.load_state_dict(sd, strict=strict)


from pydoc import locate as pydoc_locate


def set_ip_adapters(unet: UNet2DConditionModel, adapters_cfg: Optional[Dict], device, dtype) -> nn.Module:
    if not adapters_cfg:
        return nn.ModuleList()

    # --- validate & locate target
    target_path = adapters_cfg.get("target")
    if not target_path:
        raise ValueError("adapters_cfg must include 'target' (full import path to adapter class).")
    target_cls = pydoc_locate(target_path)
    if target_cls is None:
        raise ImportError(f"Could not locate adapter class: '{target_path}'")

    base_params = dict(adapters_cfg.get("params", {}))
    unet_sd = unet.state_dict()
    attn_procs = {}
    adapters = []

    for name, proc in list(unet.attn_processors.items()):
        is_self_attn = name.endswith("attn1.processor")

        hidden_size = None
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks."):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[-(block_id + 1)]
        elif name.startswith("down_blocks."):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            warnings.warn(f"Skipping unknown attention processor location: {name}")
            attn_procs[name] = proc
            continue

        if is_self_attn:
            attn_procs[name] = proc
            continue

        cross_attention_dim = getattr(unet.config, "cross_attention_dim", None)

        params = dict(base_params)
        params.update(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

        adapter = target_cls(**params)
        if hasattr(adapter, "to"):
            adapter = adapter.to(device=device, dtype=dtype)

        # --- inline robust warm start
        prefix = name.rsplit(".processor", 1)[0]
        src_k, src_v = f"{prefix}.to_k.weight", f"{prefix}.to_v.weight"

        state = {}
        for ad_key, ad_tensor in adapter.state_dict().items():
            if ad_key.endswith("weight"):
                if "k" in ad_key and src_k in unet_sd and unet_sd[src_k].shape == ad_tensor.shape:
                    state[ad_key] = unet_sd[src_k]
                elif "v" in ad_key and src_v in unet_sd and unet_sd[src_v].shape == ad_tensor.shape:
                    state[ad_key] = unet_sd[src_v]

        if state:
            missing, unexpected = adapter.load_state_dict(state, strict=False)
            if missing or unexpected:
                warnings.warn(f"Partial warm start at {name}: missing={missing}, unexpected={unexpected}")

        attn_procs[name] = adapter
        adapters.append(adapter)

    unet.set_attn_processor(attn_procs)

    out = nn.ModuleList(adapters)
    if hasattr(out, "to"):
        out = out.to(device=device, dtype=dtype)
    return out


# ------------------------------------------------------------
# Model wrapper that concatenates text/image/label tokens
# ------------------------------------------------------------
class MultiAdaptorSDXL(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        noisy_latents,
        timesteps,
        text_tokens,
        added_cond_kwargs=None,
        image_tokens=None,
        label_tokens=None,
    ):
        seqs = []
        if text_tokens is not None:
            seqs.append(text_tokens)
        if image_tokens is not None:
            seqs.append(image_tokens)
        if label_tokens is not None:
            seqs.append(label_tokens)

        encoder_hidden_states = torch.cat(seqs, dim=1) if len(seqs) else None

        unet_out = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=(added_cond_kwargs or {}),
        )
        return unet_out.sample if hasattr(unet_out, "sample") else unet_out


# ------------------------------------------------------------
# YAML config parser & builder
# ------------------------------------------------------------
def build_from_yaml(config_path: str):
    with open(config_path, "r") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f)

    try:
        mcfg = raw_cfg["model"]
        sch_params = mcfg["scheduler_config"]["params"]
        vae_params = mcfg["first_stage_config"]["params"]
        unet_params = mcfg["unet_config"]["params"]
    except KeyError as e:
        raise KeyError(f"Missing required config key in YAML: {e}.") from e

    scheduler = DDPMScheduler.from_config(sch_params)
    vae = AutoencoderKL.from_config(vae_params)
    unet = UNet2DConditionModel.from_config(unet_params)

    cond_encoder = None
    cond_cfg = mcfg.get("cond_stage_config")
    if cond_cfg:
        cond_encoder = instantiate_from_target(
            cond_cfg["target"], cond_cfg.get("params", {})
        )

    token_conditioners_cfg = mcfg.get("token_conditioners", {}) or {}

    label_to_token_model = None
    label_cfg = token_conditioners_cfg.get("label_to_token")
    if label_cfg:
        label_to_token_model = instantiate_from_target(
            label_cfg["target"], label_cfg.get("params", {})
        )

    image_to_token_model = None
    embed_cfg = token_conditioners_cfg.get("image_to_token")
    if embed_cfg:
        image_to_token_model = instantiate_from_target(
            embed_cfg["target"], embed_cfg.get("params", {})
        )

    toggles: Dict[str, Any] = mcfg.get("trainable")
    if toggles is None:
        toggles = {}
        warnings.warn("No 'trainable' section found; defaulting to train all.")

    weights: Dict[str, Any] = mcfg.get("weights")
    if weights is None:
        weights = {}
        warnings.warn("No 'weights' section found; assuming fresh init.")

    adapter_cfg: Optional[Dict[str, Any]] = mcfg.get("adapters_config")

    return (
        raw_cfg, adapter_cfg, scheduler, vae, unet, cond_encoder,
        image_to_token_model, label_to_token_model, toggles, weights
    )


# ------------------------------------------------------------
# Arg parsing
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--logging_dir", type=str, default="logs")
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    p.add_argument("--noise_offset", type=float, default=None)
    return p.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    (
        raw_cfg, adapter_cfg, scheduler, vae, unet, cond_encoder,
        image_to_token_model, label_to_token_model, toggles, weights
    ) = build_from_yaml(args.config)

    weight_dtype = (
        torch.float16 if accelerator.mixed_precision == "fp16"
        else torch.bfloat16 if accelerator.mixed_precision == "bf16"
        else torch.float32
    )
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device)
    if cond_encoder is not None:
        cond_encoder.to(accelerator.device, dtype=weight_dtype)
    if image_to_token_model is not None:
        image_to_token_model.to(accelerator.device, dtype=weight_dtype)
    if label_to_token_model is not None:
        label_to_token_model.to(accelerator.device, dtype=weight_dtype)

    adapter_modules = set_ip_adapters(
        unet=unet, device=accelerator.device, dtype=weight_dtype, adapters_cfg=adapter_cfg
    )

    maybe_freeze(unet, toggles.get("unet", True))
    maybe_freeze(cond_encoder, toggles.get("cond_encoder", False))
    maybe_freeze(image_to_token_model, toggles.get("image_to_token", True))
    maybe_freeze(label_to_token_model, toggles.get("label_to_token", True))
    maybe_freeze(adapter_modules, toggles.get("adapters", True))
    vae.requires_grad_(False)

    maybe_load_state_dict(unet, weights.get("main_model") or weights.get("unet"))
    maybe_load_state_dict(vae, weights.get("vae"))
    maybe_load_state_dict(cond_encoder, weights.get("cond_encoder"))
    maybe_load_state_dict(image_to_token_model, weights.get("image_to_token"))
    maybe_load_state_dict(label_to_token_model, weights.get("label_to_token"))
    maybe_load_state_dict(adapter_modules, weights.get("adapters"))

    model = MultiAdaptorSDXL(unet=unet)
    # print report here what is state of created objects
        # print report here what is state of created objects
    def _fmt(n: int) -> str:
        # format numbers in human-readable units
        if n >= 1e9:
            return f"{n/1e9:.2f}B"
        elif n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.2f}K"
        return str(n)



    print("\n========== MODEL REPORT ==========")
    report_module("UNet", unet)
    report_module("VAE", vae)
    report_module("CondEncoder", cond_encoder)
    report_module("ImageToToken", image_to_token_model)
    report_module("LabelToToken", label_to_token_model)
    report_module("Adapters", adapter_modules)

    print(f"[REPORT] MultiAdaptorSDXL wrapper: {model.__class__.__name__}")
    print("=================================\n")






if __name__ == "__main__":
    main()