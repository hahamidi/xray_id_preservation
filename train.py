"""
High-level training template (pseudocode) for a Diffusion+Adapters setup.

INPUTS
- YAML config (matches your schema) describing model/data/weights.
- Dataset on disk (paths in the YAML).
- Optional pretrained weights (UNet, VAE, text encoder, adapters).

OUTPUTS
- Checkpoints saved under output_dir/checkpoint-<global_step>.
- Logs in output_dir/logs for the chosen tracker (e.g., TensorBoard).
"""

# --- std lib ---
import os
import time
import argparse
from pathlib import Path
import yaml

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

# --- your project-specific imports (placeholders / TODOs) ---
# from your_project.data import DAVIS_Dataset, collate_fn
# from your_project.encoders import DynamicEncoder                      # (optional)
# from modify_instantID import FrozenDinoV2Encoder                       # image encoder
# from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
# from ip_adapter.utils import is_torch2_available
# (We will wire these up in the concrete implementation.)

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


def instantiate_from_target(target: str, params: dict | None):
    """
    Generic factory: instantiate any class by target string + kwargs.
    """
    cls = locate(target)
    return cls(**(params or {}))


# ------------------------------------------------------------
# Optional helpers for freezing and weight loading
# ------------------------------------------------------------
def maybe_freeze(module: nn.Module | None, train_flag: bool):
    """
    If module exists: set requires_grad to train_flag.
    """
    if module is None:
        return
    module.requires_grad_(bool(train_flag))


def maybe_load_state_dict(module: nn.Module | None, path: str | None, strict: bool = False):
    """
    If module & path provided: load a state_dict from disk.
    """
    if module is None or not path:
        return
    sd = torch.load(path, map_location="cpu")
    module.load_state_dict(sd, strict=strict)


from pydoc import locate as pydoc_locate
import warnings
import torch.nn as nn

def set_ip_adapters(unet: UNet2DConditionModel, adapters_cfg: dict, device, dtype) -> nn.Module:
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
    # optional: allow remapping kv names via config
    kv_names = adapters_cfg.get("kv_param_names", {"k": "to_k_ip.weight", "v": "to_v_ip.weight"})

    # read once before swap
    unet_sd = unet.state_dict()
    attn_procs = {}
    adapters = []

    for name, proc in list(unet.attn_processors.items()):
        is_self_attn = name.endswith("attn1.processor")

        # try to derive hidden_size from name; skip unknowns instead of hard-fail
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
            attn_procs[name] = proc  # keep existing
            continue

        if is_self_attn:
            # leave self-attn unchanged; if you *must* reset:
            # attn_procs[name] = AttnProcessor().to(device=device, dtype=dtype)
            attn_procs[name] = proc
            continue

        # try to infer cross_attention_dim from the module if possible
        cross_attention_dim = getattr(unet, "config", None)
        if cross_attention_dim is not None:
            cross_attention_dim = unet.config.cross_attention_dim
        # if you want to be safer, introspect the underlying attention module here.

        params = dict(base_params)
        params.update(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

        adapter = target_cls(**params)
        if hasattr(adapter, "to"):
            adapter = adapter.to(device=device, dtype=dtype)

        # warm start: map UNet's to_k/to_v to adapter's expected names
        prefix = name.rsplit(".processor", 1)[0]
        k_key, v_key = f"{prefix}.to_k.weight", f"{prefix}.to_v.weight"
        state = {}
        if k_key in unet_sd and kv_names.get("k"):
            state[kv_names["k"]] = unet_sd[k_key]
        if v_key in unet_sd and kv_names.get("v"):
            state[kv_names["v"]] = unet_sd[v_key]
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
    """
    Wrap UNet and the token projectors:
      - text_tokens: [B, T_txt, D_text]
      - image_embeds: [B, T_img, D_img]  -> project -> [B, T_img, D_text]
      - label_embeds: [B, T_lab, D_lab]  -> project -> [B, T_lab, D_text]
    """
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        noisy_latents,                # [B,4,H/8,W/8]
        timesteps,                    # [B]
        text_tokens,                  # [B,T_txt,D_text] (can be None if UNet expects raw ids)
        added_cond_kwargs=None,       # SDXL "time_ids" etc.
        image_tokens=None,            # [B,T_img,D_img]
        label_tokens=None,            # [B,T_lab,D_lab]
    ):
        # Concatenate condition sequences that unet.cross-attn will attend to
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
import warnings
from typing import Any, Dict, Optional, Tuple

def build_from_yaml(config_path: str):
    """
    Returns (in order):
      raw_cfg               : the raw YAML dict
      adapter_cfg           : dict or None
      scheduler             : DDPMScheduler
      vae                   : AutoencoderKL
      unet                  : UNNet2DConditionModel
      text_encoder          : optional encoder returning [B,T,D]
      image_to_token_model  : optional projector for image/embed -> text dim
      label_to_token_model  : optional projector for label -> text dim
      toggles               : dict of trainable flags (empty => train all)
      weights               : dict of weight paths (empty => none)
    """
    with open(config_path, "r") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f)

    try:
        mcfg = raw_cfg["model"]
        sch_params = mcfg["scheduler_config"]["params"]
        vae_params = mcfg["first_stage_config"]["params"]
        unet_params = mcfg["unet_config"]["params"]
    except KeyError as e:
        raise KeyError(f"Missing required config key in YAML: {e}. "
                       f"Expected model.scheduler_config/first_stage_config/unet_config with 'params'.") from e

    scheduler = DDPMScheduler.from_config(sch_params)
    vae       = AutoencoderKL.from_config(vae_params)
    unet      = UNet2DConditionModel.from_config(unet_params)

    # Optional text/cond encoder
    cond_encoder = None
    cond_cfg = mcfg.get("cond_stage_config")
    if cond_cfg:
        cond_encoder = instantiate_from_target(
            cond_cfg["target"], cond_cfg.get("params", {})
        )

    # -------------------------------
    # Global token conditioners
    # -------------------------------
    token_conditioners_cfg = mcfg.get("token_conditioners", {}) or {}

    label_to_token_model = None
    label_cfg = token_conditioners_cfg.get("label_to_token")
    if label_cfg:
        label_to_token_model = instantiate_from_target(
            label_cfg["target"], label_cfg.get("params", {})
        )

    image_to_token_model = None
    embed_cfg = token_conditioners_cfg.get("embed_to_token")
    if embed_cfg:
        image_to_token_model = instantiate_from_target(
            embed_cfg["target"], embed_cfg.get("params", {})
        )

    # Trainable toggles & weight paths
    toggles: Dict[str, Any] = mcfg.get("trainable")
    if toggles is None:
        toggles = {}
        warnings.warn("No 'trainable' section found in model config; "
                      "defaulting to 'train all' semantics.")

    weights: Dict[str, Any] = mcfg.get("weights")
    if weights is None:
        weights = {}
        warnings.warn("No 'weights' section found in model config; "
                      "assuming fresh init (no pretrained weights).")

    adapter_cfg: Optional[Dict[str, Any]] = mcfg.get("adapters_config")

    return (
        raw_cfg, adapter_cfg, scheduler, vae, unet, cond_encoder,
        image_to_token_model, label_to_token_model, toggles, weights
    )

# ------------------------------------------------------------
# Arg parsing (only the interface, defaults can be tuned later)
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
# Main (high-level flow only; details marked as TODO)
# ------------------------------------------------------------
def main():
    # 1) Parse CLI args and set up accelerate/logging
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # 2) Build modules from YAML (configs only—no weights yet)
    (
        raw_cfg, adapter_cfg, scheduler, vae, unet, cond_encoder,
        image_to_token_model, label_to_token_model, toggles, weights

    ) = build_from_yaml(args.config)



    # TODO: your image encoder (frozen DINOv2) — returns per-image embeddings
    # image_encoder = FrozenDinoV2Encoder(DINOv2_weight_path=cfg["model"]["params"].get("dino_weight_path"))
    # image_encoder.requires_grad_(False)

    # 4) Device & dtype policy
    weight_dtype = (
        torch.float16 if accelerator.mixed_precision == "fp16"
        else torch.bfloat16 if accelerator.mixed_precision == "bf16"
        else torch.float32
    )
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device)  # keep VAE in fp32 typically
    if cond_encoder is not None:
        cond_encoder.to(accelerator.device, dtype=weight_dtype)
    if image_to_token_model is not None:
        image_to_token_model.to(accelerator.device, dtype=weight_dtype)
    if label_to_token_model is not None:
        label_to_token_model.to(accelerator.device, dtype=weight_dtype)

    # 5) Install IP-Adapters into UNet (sets custom attention processors)
    adapter_modules = set_ip_adapters(
        unet=unet, device=accelerator.device, dtype=weight_dtype, adapters_cfg=adapter_cfg
    )

    # 6) Freeze/train toggles
    maybe_freeze(unet,          toggles.get("unet", True))
    maybe_freeze(cond_encoder,  toggles.get("cond_encoder", False))
    maybe_freeze(image_to_token_model, toggles.get("image_to_token", True))
    maybe_freeze(label_to_token_model, toggles.get("label_to_token", True))
    maybe_freeze(adapter_modules,  toggles.get("adapters", True))
    vae.requires_grad_(False)

    # 7) Optional: load weights after modules exist
    maybe_load_state_dict(unet,            weights.get("main_model") or weights.get("unet"))
    maybe_load_state_dict(vae,             weights.get("vae"))
    # maybe_load_state_dict(text_encoder,  weights.get("text_encoder"))
    # maybe_load_state_dict(image_proj_model, weights.get("image_proj"))
    # maybe_load_state_dict(label_proj_model, weights.get("label_proj"))
    maybe_load_state_dict(adapter_modules,  weights.get("ip_adapters"))

    # 8) Wrap model to handle multi-conditions
    model = MultiAdaptorSDXL(
        unet=unet,
        image_proj_model=image_proj_model,
        label_proj_model=label_proj_model,
        adapter_modules=adapter_modules,
    )

    # 9) Optimizer over trainable params
    param_groups = []
    if any(p.requires_grad for p in model.unet.parameters()):
        param_groups.append(model.unet.parameters())
    if text_encoder is not None and any(p.requires_grad for p in text_encoder.parameters()):
        param_groups.append(text_encoder.parameters())
    if image_proj_model is not None and any(p.requires_grad for p in image_proj_model.parameters()):
        param_groups.append(image_proj_model.parameters())
    if label_proj_model is not None and any(p.requires_grad for p in label_proj_model.parameters()):
        param_groups.append(label_proj_model.parameters())
    if adapter_modules is not None and any(p.requires_grad for p in adapter_modules.parameters()):
        param_groups.append(adapter_modules.parameters())

    optimizer = torch.optim.AdamW(
        params=(p for group in param_groups for p in group),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # 10) Dataset / DataLoader (your dataset must return the fields used below)
    # train_dataset = DAVIS_Dataset(
    #     tokenizer=tokenizer,
    #     tokenizer_2=tokenizer,  # kept for compatibility
    #     size=cfg["data"]["params"].get("resolution", 1024),
    #     image_root_path=cfg["data"]["params"]["data_root_path"],
    #     pairs_file_path=cfg["data"]["params"]["pairs_file_path"],
    # )
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     batch_size=args.train_batch_size,
    #     num_workers=args.dataloader_num_workers,
    # )

    # 11) Prepare with accelerator
    # model, optimizer, train_dataloader, image_encoder = accelerator.prepare(
    #     model, optimizer, train_dataloader, image_encoder
    # )

    # 12) Training loop (outline)
    global_step = 0
    for epoch in range(args.num_train_epochs):
        # for step, batch in enumerate(train_dataloader):
        #     with accelerator.accumulate(model):
        #         # (a) Encode images to latents (no grad)
        #         # latents: encode with VAE, scale by vae.config.scaling_factor
        #         # (b) Sample noise & timesteps, add noise via scheduler
        #         # (c) Build condition tokens:
        #         #     - text_tokens = text_encoder(input_ids)
        #         #     - image_embeds = image_encoder(...) -> (maybe drop some)
        #         #     - label_embeds = batch.get("label_embeds", None)
        #         # (d) SDXL added cond (time_ids) from batch (original_size, crops, etc.)
        #         # (e) Forward UNet -> noise_pred
        #         # (f) Loss = MSE(noise_pred, noise)
        #         # (g) backward + step + zero_grad
        #         pass

        #     # (h) Save periodic checkpoints
        #     if accelerator.is_main_process and (global_step % args.save_steps == 0):
        #         save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        #         accelerator.save_state(save_path)

        #     global_step += 1
        pass  # end epoch

    # 13) (Optional) Save final state
    # if accelerator.is_main_process:
    #     accelerator.save_state(os.path.join(args.output_dir, "checkpoint-final"))


if __name__ == "__main__":
    main()