# accelerate launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision fp16 --dynamo_backend no train.py --config ./configs/base_LDM.yaml --train_batch_size 64
import argparse
import os
import time
import warnings
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer

from tqdm.auto import tqdm

from utils_idp import report_module, fmt_params
from utils_vis import ensure_dir, run_viz_hook

# ------------------------------------------------------------
# Utility: resolve "module.ClassName" strings to actual classes
# ------------------------------------------------------------
def locate(target: str):
    parts = target.split(".")
    module_path, cls_name = ".".join(parts[:-1]), parts[-1]
    mod = __import__(module_path, fromlist=[cls_name])
    return getattr(mod, cls_name)

def instantiate_from_target(target: str, params: Optional[Dict] = None):
    cls = locate(target)
    return cls(**(params or {}))

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def maybe_freeze(module: Optional[nn.Module], train_flag: bool):
    if module is None:
        return
    module.requires_grad_(bool(train_flag))

def maybe_load_state_dict(module: Optional[nn.Module], path: Optional[str], strict: bool = False):
    if module is None or not path:
        return
    sd = torch.load(path, map_location="cpu")
    module.load_state_dict(sd, strict=strict)

from pydoc import locate as pydoc_locate

# WHY:
# This function attaches IP adapters to a UNet model’s attention processors.
# Adapters are inserted only for cross-attention layers (not self-attention).
# This allows fine-tuning or modifying the model’s attention mechanism
# without retraining the whole UNet. It also supports warm-starting adapter
# weights from existing UNet key/value projections, which stabilizes training.
# here we want to set a set of new key/value projections for extra tokens ( image emnbeddings, label embeddings ...)
# the key and value projections are initialized from the UNet's existing key/value projections fro this new new projections
# see the adapter class in adapters.py

def set_ip_adapters(unet: UNet2DConditionModel, adapters_cfg: Optional[Dict], device, dtype) -> nn.Module:
    if not adapters_cfg:
        return nn.ModuleList()
    target_path = adapters_cfg.get("target")
    if not target_path:
        raise ValueError("adapters_cfg must include 'target'.")
    target_cls = pydoc_locate(target_path)
    if target_cls is None:
        raise ImportError(f"Could not locate adapter class: '{target_path}'")

    # base parameters for all adapters
    base_params = dict(adapters_cfg.get("params", {}))
    unet_sd = unet.state_dict()
    attn_procs = {}
    adapters = []

    for name, proc in list(unet.attn_processors.items()):
        is_self_attn = name.endswith("attn1.processor")

        # infer hidden size from processor location in UNet
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

        # skip self-attention processors (only target cross-attention)
        if is_self_attn:
            attn_procs[name] = proc
            continue

        cross_attention_dim = getattr(unet.config, "cross_attention_dim", None)
        params = dict(base_params)
        params.update(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

        # create adapter module
        adapter = target_cls(**params).to(device=device, dtype=dtype)

        # warm start adapter with UNet's key/value weights if shapes match
        prefix = name.rsplit(".processor", 1)[0]
        src_k, src_v = f"{prefix}.to_k.weight", f"{prefix}.to_v.weight"
        state = {}
        for ad_key, ad_tensor in adapter.state_dict().items():
            if ad_key.endswith("weight"):
                if "k" in ad_key and src_k in unet_sd and unet_sd[src_k].shape == ad_tensor.shape:
                    state[ad_key] = unet_sd[src_k].clone()
                elif "v" in ad_key and src_v in unet_sd and unet_sd[src_v].shape == ad_tensor.shape:
                    state[ad_key] = unet_sd[src_v].clone()
        if state:
            adapter.load_state_dict(state, strict=False)

        attn_procs[name] = adapter
        adapters.append(adapter)

    # replace UNet attention processors with adapters
    unet.set_attn_processor(attn_procs)
    return nn.ModuleList(adapters).to(device=device, dtype=dtype)


# WHY:
# This wrapper extends a UNet to accept multiple conditioning inputs (text, image, labels).
# Instead of handling them separately, it concatenates them into a single encoder_hidden_states
# sequence for conditioning the UNet. This makes the model flexible in mixing modalities.

# ------------------------------------------------------------
# Wrapper
# ------------------------------------------------------------
class MultiAdaptorSDXL(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, noisy_latents, timesteps, text_tokens,
                added_cond_kwargs=None, image_tokens=None, label_tokens=None):
        seqs = []
        # collect all available conditioning sequences
        if text_tokens is not None:
            seqs.append(text_tokens)
        if image_tokens is not None:
            seqs.append(image_tokens)
        if label_tokens is not None:
            seqs.append(label_tokens)

        # concatenate along sequence dimension, or None if no conditioning
        encoder_hidden_states = torch.cat(seqs, dim=1) if len(seqs) else None

        # pass through wrapped UNet with extended conditioning
        unet_out = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=(added_cond_kwargs or {})
        )

        # some UNet outputs wrap the tensor in .sample
        return unet_out.sample if hasattr(unet_out, "sample") else unet_out
    

# why:
# This function builds the model components from a YAML configuration file.
# It loads the scheduler, VAE, UNet, and optional conditioning encoders.
# it is like creating all objects we need from config file
# ------------------------------------------------------------
# YAML builder
# ------------------------------------------------------------
def build_from_yaml(config_path: str):
    with open(config_path, "r") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f)
    mcfg = raw_cfg["model"]
    scheduler = DDPMScheduler.from_config(mcfg["scheduler_config"]["params"])
    vae = AutoencoderKL.from_config(mcfg["first_stage_config"]["params"])
    unet = UNet2DConditionModel.from_config(mcfg["unet_config"]["params"])
    cond_encoder = None
    cond_cfg = mcfg.get("cond_stage_config")
    if cond_cfg:
        cond_encoder = instantiate_from_target(cond_cfg["target"], cond_cfg.get("params", {}))
    token_conditioners_cfg = mcfg.get("token_conditioners", {}) or {}
    label_to_token_model = None
    if token_conditioners_cfg.get("label_to_token"):
        label_to_token_model = instantiate_from_target(
            token_conditioners_cfg["label_to_token"]["target"],
            token_conditioners_cfg["label_to_token"].get("params", {})
        )
    image_to_token_model = None
    if token_conditioners_cfg.get("image_to_token"):
        image_to_token_model = instantiate_from_target(
            token_conditioners_cfg["image_to_token"]["target"],
            token_conditioners_cfg["image_to_token"].get("params", {})
        )
    toggles: Dict[str, Any] = mcfg.get("trainable", {})
    weights: Dict[str, Any] = mcfg.get("weights", {})
    adapter_cfg: Optional[Dict[str, Any]] = mcfg.get("adapters_config")
    return (raw_cfg, adapter_cfg, scheduler, vae, unet, cond_encoder,
            image_to_token_model, label_to_token_model, toggles, weights)

# ------------------------------------------------------------
# Argparse
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--logging_dir", type=str, default="logs")
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--num_train_epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--dataloader_num_workers", type=int, default=2)
    p.add_argument("--noise_offset", type=float, default=None)

    # ------ Visualization flags (only used by viz_utils.maybe_run_viz) ------
    p.add_argument("--sample_every", type=int, default=10, help="Run viz hook every N global steps (0 disables).")
    p.add_argument("--sample_inference_steps", type=int, default=25, help="Reverse diffusion steps for sampling.")
    p.add_argument("--sample_grid_rows", type=int, default=4)
    p.add_argument("--sample_grid_cols", type=int, default=4)
    p.add_argument("--samples_dir", type=str, default="samples", help="Subdir under output_dir to save PNGs.")
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

    # ---- DTYPE RULES ----
    device_type = accelerator.device.type
    param_dtype = torch.float32
    if accelerator.mixed_precision == "fp16" and device_type != "cpu":
        compute_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16" and device_type != "cpu":
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float32

    # ---- ADAPTERS ----
    adapter_modules = set_ip_adapters(unet=unet, device=accelerator.device,
                                      dtype=param_dtype, adapters_cfg=adapter_cfg)

    # ---- FREEZE ----
    maybe_freeze(unet, toggles.get("unet", True))
    maybe_freeze(cond_encoder, toggles.get("cond_encoder", False))
    maybe_freeze(image_to_token_model, toggles.get("image_to_token", True))
    maybe_freeze(label_to_token_model, toggles.get("label_to_token", True))
    maybe_freeze(adapter_modules, toggles.get("adapters", True))
    vae.requires_grad_(False)

    # ---- LOAD WEIGHTS ----
    maybe_load_state_dict(unet, weights.get("main_model") or weights.get("unet"))
    maybe_load_state_dict(vae, weights.get("vae"))
    maybe_load_state_dict(cond_encoder, weights.get("cond_encoder"))
    maybe_load_state_dict(image_to_token_model, weights.get("image_to_token"))
    maybe_load_state_dict(label_to_token_model, weights.get("label_to_token"))
    maybe_load_state_dict(adapter_modules, weights.get("adapters"))

    model = MultiAdaptorSDXL(unet=unet)

    if accelerator.is_main_process:
        print("\n========== MODEL REPORT ==========")
        report_module("UNet", unet)
        report_module("VAE", vae)
        report_module("CondEncoder", cond_encoder)
        report_module("ImageToToken", image_to_token_model)
        report_module("LabelToToken", label_to_token_model)
        report_module("Adapters", adapter_modules)
        print("=================================\n")

    # ---- OPTIMIZER ----
    def collect_trainable_params(*modules):
        return [p for m in modules if m is not None for p in m.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        collect_trainable_params(unet, cond_encoder, image_to_token_model, label_to_token_model),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ---- DATA ----
    Train_dataset = instantiate_from_target(
        raw_cfg["datasets"]['train_dataset']["target"],
        params=raw_cfg["datasets"]['train_dataset'].get("params", {})
    )
    train_dataloader = DataLoader(
        Train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # ---- PREP (multi-GPU aware) ----
    model.train()
    if cond_encoder is not None: cond_encoder.train()
    if image_to_token_model is not None: image_to_token_model.train()
    if label_to_token_model is not None: label_to_token_model.train()

    train_dataloader, model, vae, optimizer, cond_encoder, image_to_token_model, label_to_token_model = accelerator.prepare(
        train_dataloader, model, vae, optimizer, cond_encoder, image_to_token_model, label_to_token_model
    )

    logger = get_logger(__name__)
    global_step = 0
    scaling_factor = getattr(getattr(vae, "config", vae), "scaling_factor", 0.18215)

    # ---- TRAIN LOOP ----
    for epoch in range(args.num_train_epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_main_process)
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                images = batch["images"].to(dtype=torch.float32, non_blocking=True)
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample() * scaling_factor
                latents = latents.to(dtype=compute_dtype)

                noise = torch.randn_like(latents)
                if args.noise_offset is not None:
                    noise = noise + args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device, dtype=latents.dtype
                    )
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device, dtype=torch.long
                )
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                ids = cond_encoder.tokenize(batch['captions'])
                ids = {k: v.to(non_blocking=True) for k, v in ids.items()}
                ce_grads = cond_encoder is not None and any(p.requires_grad for p in cond_encoder.parameters())
                with torch.set_grad_enabled(ce_grads):
                    text_tokens = cond_encoder(ids)
                text_tokens = text_tokens.to(dtype=compute_dtype)

                image_embeddings = batch["image_embeds"].to(dtype=compute_dtype, non_blocking=True)
                image_tokens = image_to_token_model(image_embeddings)
                label_embeddings = batch["labels"].to(dtype=compute_dtype, non_blocking=True)
                label_tokens = label_to_token_model(label_embeddings)

                with torch.autocast(device_type=device_type, dtype=compute_dtype):
                    noise_pred = model(
                        noisy_latents=noisy_latents,
                        timesteps=timesteps,
                        text_tokens=text_tokens,
                        image_tokens=image_tokens,
                        label_tokens=label_tokens,
                    )
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            progress_bar.set_postfix({"loss": loss.item()})
            if accelerator.is_main_process and (global_step % 50 == 0):
                logger.info(f"[epoch {epoch} | step {step} | gs {global_step}] loss={loss.item():.4f}")
            if accelerator.is_main_process and (global_step % args.save_steps == 0):
                accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{global_step}"))


            # Check cadence & process
            if accelerator.is_main_process \
            and args.sample_every > 0 \
            and (global_step % args.sample_every == 0):
                # <-- call your visualization function here
                run_viz_hook(
                    accelerator=accelerator,
                    global_step=global_step,
                    args=args,
                    batch=batch,
                    model=model,
                    vae=vae,
                    scheduler_train=scheduler,
                    cond_encoder=cond_encoder,
                    image_to_token_model=image_to_token_model,
                    label_to_token_model=label_to_token_model,
                    compute_dtype=compute_dtype,
                    scaling_factor=scaling_factor,
                )

                

if __name__ == "__main__":
    main()