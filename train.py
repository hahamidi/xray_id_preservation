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


# ------------------------------------------------------------
# Adapters install helper (config-driven)
# ------------------------------------------------------------
def set_ip_adapters(
    unet: UNet2DConditionModel,
    adapters_cfg: dict,
    device,
    dtype,
) -> nn.Module:
    """
    Create and inject adapters into each UNet cross-attention layer,
    using class + params from adapters_cfg["target"] and ["params"].

    Example YAML section:

    adapters_config:
      target: ip_adapter.attention_processor.IPAttnProcessor   # class path
      params:
        scale: 1.0
        num_tokens: 4

    Returns
    -------
    adapters : nn.Module
        ModuleList of the instantiated adapter modules (trainable ones).
    """
    if adapters_cfg is None:
        return nn.ModuleList()

    target_cls = locate(adapters_cfg["target"])
    base_params = adapters_cfg.get("params", {})

    unet_sd = unet.state_dict()
    attn_procs = {}
    adapters = []

    for name in list(unet.attn_processors.keys()):
        is_self_attn = name.endswith("attn1.processor")
        cross_attention_dim = None if is_self_attn else unet.config.cross_attention_dim

        # hidden size per block
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks."):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[-(block_id + 1)]
        elif name.startswith("down_blocks."):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            raise ValueError(f"Unknown attention processor location: {name}")

        if cross_attention_dim is None:
            # leave self-attention as vanilla AttnProcessor (parameter-free)
            attn_procs[name] = AttnProcessor().to(device=device, dtype=dtype)
        else:
            # merge base params with per-layer info
            params = dict(base_params)
            params.update(dict(hidden_size=hidden_size,
                               cross_attention_dim=cross_attention_dim))

            # build adapter from config
            adapter = target_cls(**params).to(device=device, dtype=dtype)

            # warm-start with original k/v weights if present
            prefix = name.rsplit(".processor", 1)[0]
            k_key, v_key = f"{prefix}.to_k.weight", f"{prefix}.to_v.weight"
            state = {}
            if k_key in unet_sd:
                state["to_k_ip.weight"] = unet_sd[k_key]
            if v_key in unet_sd:
                state["to_v_ip.weight"] = unet_sd[v_key]
            if state:
                adapter.load_state_dict(state, strict=False)

            attn_procs[name] = adapter
            adapters.append(adapter)

    # install processors into UNet
    unet.set_attn_processor(attn_procs)

    return nn.ModuleList(adapters).to(device=device, dtype=dtype)


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
    def __init__(self, unet, image_proj_model=None, label_proj_model=None, adapter_modules=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.label_proj_model = label_proj_model
        self.adapter_modules = adapter_modules  # holds IP-attn parameters

    def forward(
        self,
        noisy_latents,                # [B,4,H/8,W/8]
        timesteps,                    # [B]
        text_tokens,                  # [B,T_txt,D_text] (can be None if UNet expects raw ids)
        added_cond_kwargs=None,       # SDXL "time_ids" etc.
        image_embeds=None,            # [B,T_img,D_img]
        label_embeds=None,            # [B,T_lab,D_lab]
    ):
        # Concatenate condition sequences that unet.cross-attn will attend to
        seqs = []
        if text_tokens is not None:
            seqs.append(text_tokens)
        if image_embeds is not None and self.image_proj_model is not None:
            seqs.append(self.image_proj_model(image_embeds))
        if label_embeds is not None and self.label_proj_model is not None:
            seqs.append(self.label_proj_model(label_embeds))

        encoder_hidden_states = torch.cat(seqs, dim=1) if len(seqs) else None

        unet_out = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=(added_cond_kwargs or {}),
        )
        return unet_out.sample if hasattr(unet_out, "sample") else unet_out


# ------------------------------------------------------------
# Config loader/builder: constructs core modules from YAML
# ------------------------------------------------------------
def build_from_yaml(config_path: str):
    """
    Returns:
      cfg                    : raw YAML dict
      scheduler              : DDPMScheduler
      vae                    : AutoencoderKL
      unet                   : UNet2DConditionModel
      text_encoder           : optional encoder returning [B,T,D]
      image_proj_model       : projector for image embeds -> text dim
      label_proj_model       : projector for label embeds -> text dim
      toggles                : dict of trainable flags
      weights                : dict of weight paths
      tokenizer_path         : string (e.g., SDXL tokenizer subfolder)
      ip_scale, ip_num_tokens: adapter settings
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    mcfg = cfg["model"]["params"]

    # Core modules from *config dicts* (no pretrained checkpoints yet)
    scheduler = DDPMScheduler.from_config(mcfg["scheduler_config"]["params"])
    vae       = AutoencoderKL.from_config(mcfg["first_stage_config"]["params"])
    unet      = UNet2DConditionModel.from_config(mcfg["unet_config"]["params"])

    # Optional: dynamic text encoder (your class should return token embeddings)
    text_enc_cfg = mcfg.get("cond_stage_config")
    text_encoder = None
    if text_enc_cfg and text_enc_cfg.get("target"):
        text_encoder = instantiate_from_target(text_enc_cfg["target"], text_enc_cfg.get("params", {}))

    # Adapter/projector config
    adapters_cfg  = mcfg.get("adapters_config", {})
    text_dim      = unet.config.cross_attention_dim
    ip_scale      = adapters_cfg.get("ip_scale_image", 1)  # or unified ip_scale
    ip_num_tokens = adapters_cfg.get("ip_num_tokens_image", 4)

    # Image projector
    img_in_dim = adapters_cfg.get("image_in_dim")
    image_proj_model = None
    if img_in_dim is not None:
        proj_cfg = adapters_cfg.get("image_projector", {
            "target": "torch.nn.Linear",
            "params": {"in_features": img_in_dim, "out_features": text_dim, "bias": True}
        })
        image_proj_model = instantiate_from_target(proj_cfg["target"], proj_cfg.get("params", {}))

    # Label projector
    lab_in_dim = adapters_cfg.get("label_in_dim")
    label_proj_model = None
    if lab_in_dim is not None:
        lab_proj_cfg = adapters_cfg.get("label_projector", {
            "target": "torch.nn.Linear",
            "params": {"in_features": lab_in_dim, "out_features": text_dim, "bias": True}
        })
        label_proj_model = instantiate_from_target(lab_proj_cfg["target"], lab_proj_cfg.get("params", {}))

    # Trainable toggles & weight paths
    toggles = mcfg.get("trainable", {
        "unet": True, "text_encoder": False, "image_proj": True, "label_proj": True, "ip_adapters": True
    })
    weights = mcfg.get("weights", {})
    tokenizer_path = mcfg.get("tokenizer_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0")

    return (
        cfg, scheduler, vae, unet, text_encoder, image_proj_model, label_proj_model,
        toggles, weights, tokenizer_path, ip_scale, ip_num_tokens
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
        cfg, noise_scheduler, vae, unet, text_encoder,
        image_proj_model, label_proj_model, toggles, weights,
        tokenizer_path, ip_scale, ip_num_tokens
    ) = build_from_yaml(args.config)

    # 3) Tokenizer and encoders (text + image)
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, subfolder="tokenizer")

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
    if text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    if image_proj_model is not None:
        image_proj_model.to(accelerator.device, dtype=weight_dtype)
    if label_proj_model is not None:
        label_proj_model.to(accelerator.device, dtype=weight_dtype)

    # 5) Install IP-Adapters into UNet (sets custom attention processors)
    adapter_modules = set_ip_adapters(
        unet=unet, device=accelerator.device, dtype=weight_dtype,
        scale=ip_scale, num_tokens=ip_num_tokens
    )

    # 6) Freeze/train toggles
    maybe_freeze(unet,          toggles.get("unet", True))
    maybe_freeze(text_encoder,  toggles.get("text_encoder", False))
    maybe_freeze(image_proj_model, toggles.get("image_proj", True))
    maybe_freeze(label_proj_model, toggles.get("label_proj", True))
    maybe_freeze(adapter_modules,  toggles.get("ip_adapters", True))
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