# utils_vis.py

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn


# ---------- filesystem ----------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------- tensor <-> image ----------

def tensor_to_uint8_img(t: torch.Tensor) -> Image.Image:
    """
    Accepts a tensor in CHW with range [0,1] (float32/float16) or [-1,1].
    Returns a PIL Image (RGB).
    """
    if t.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(t.shape)}")
    t = t.detach().to(dtype=torch.float32).cpu()
    if t.min() < 0.0:  # assume [-1,1] -> map to [0,1]
        t = (t + 1.0) * 0.5
    t = t.clamp(0.0, 1.0)
    arr = (t.numpy() * 255.0).round().astype(np.uint8)  # C,H,W
    arr = np.transpose(arr, (1, 2, 0))                  # H,W,C
    return Image.fromarray(arr)


# ---------- grids (dependency-free; no torchvision) ----------

def save_grid(
    images: Sequence[torch.Tensor] | Sequence[Image.Image],
    rows: int,
    cols: int,
    out_path: str,
    padding: int = 2,
    pad_color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    """
    Save a rows x cols grid. Each element can be a CHW torch tensor in [0,1] or a PIL Image.
    All images must share the same H,W. If fewer than rows*cols images are provided,
    the remaining tiles stay blank (pad_color).
    """
    if rows * cols < 1:
        raise ValueError("rows*cols must be >= 1")

    # Convert all to PIL
    pil_imgs: List[Image.Image] = []
    for im in images[: rows * cols]:
        if isinstance(im, Image.Image):
            pil_imgs.append(im)
        elif torch.is_tensor(im):
            pil_imgs.append(tensor_to_uint8_img(im))
        else:
            raise TypeError(f"Unsupported image type: {type(im)}")

    if not pil_imgs:
        raise ValueError("No images to save.")

    w, h = pil_imgs[0].size
    for p in pil_imgs:
        if p.size != (w, h):
            raise ValueError("All images must have identical spatial size.")

    grid_w = cols * w + (cols - 1) * padding
    grid_h = rows * h + (rows - 1) * padding
    canvas = Image.new("RGB", (grid_w, grid_h), pad_color)

    for idx, img in enumerate(pil_imgs):
        r = idx // cols
        c = idx % cols
        x = c * (w + padding)
        y = r * (h + padding)
        canvas.paste(img, (x, y))

    ensure_dir(os.path.dirname(out_path) or ".")
    canvas.save(out_path)


# ---------- VAE decode helper ----------

@torch.no_grad()
def decode_latents(
    vae: nn.Module,
    latents: torch.Tensor,
    scaling_factor: float,
) -> torch.Tensor:
    """
    VAE decode that:
    - divides by scaling_factor
    - decodes
    - clamps to [0,1]
    - returns float32 CHW images on CPU
    Input:  B,C,H,W latents (any dtype/device)
    Output: B,3,H*,W* in [0,1], float32, CPU
    """
    x = (latents / float(scaling_factor)).to(dtype=torch.float32)
    out = vae.decode(x)
    decoded = out.sample if hasattr(out, "sample") else out
    decoded = decoded.clamp(-1, 1)
    decoded = ((decoded + 1.0) * 0.5).clamp(0.0, 1.0)  # [-1,1] -> [0,1]
    return decoded.to(dtype=torch.float32, device="cpu")


# ---------- indexing helpers ----------

def evenly_spaced_indices(n: int, k: int) -> List[int]:
    """
    Return k indices in [0, n-1], approximately evenly spaced, always including first and last.
    """
    if k <= 1:
        return [0]
    arr = np.linspace(0, max(0, n - 1), num=k)
    return [int(round(v)) for v in arr]


def choose_random_indices(pool_size: int, count: int) -> List[int]:
    """
    Choose 'count' unique indices in [0, pool_size). No torch.Generator; works with old PyTorch.
    """
    count = min(pool_size, max(1, count))
    return torch.randperm(pool_size)[:count].tolist()


# ---------- schedulers ----------

def _make_sd1x_scheduler(
    *,
    name: str,
    num_inference_steps: int,
    device: torch.device,
):
    """
    Create an SD1.x-compatible inference scheduler.
    Supported names: 'ddim', 'euler', 'euler_a', 'pndm', 'ddpm'
    """
    name = (name or "euler_a").lower()  # CHANGED: default to Euler-ancestral
    if name == "euler_a":  # NEW: Euler ancestral + Karras sigmas
        from diffusers.schedulers import EulerAncestralDiscreteScheduler
        sch = EulerAncestralDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
    elif name == "euler":
        from diffusers.schedulers import EulerDiscreteScheduler
        sch = EulerDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
    elif name == "pndm":
        from diffusers.schedulers import PNDMScheduler
        sch = PNDMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
    elif name == "ddpm":
        from diffusers import DDPMScheduler
        sch = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            prediction_type="epsilon", steps_offset=1, clip_sample=False, set_alpha_to_one=False
        )
    else:  # 'ddim'
        from diffusers.schedulers import DDIMScheduler
        sch = DDIMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
            prediction_type="epsilon", steps_offset=1, clip_sample=False, set_alpha_to_one=False
        )
    sch.set_timesteps(num_inference_steps, device=device)
    return sch


# ---------- CFG rescale (helps with crunchy highlights) ----------

def _cfg_rescale(eps: torch.Tensor, eps_u: torch.Tensor, rescale: float = 0.7) -> torch.Tensor:
    # NEW
    dims = list(range(1, eps.ndim))
    std_ratio = eps.std(dim=dims, keepdim=True) / (eps_u.std(dim=dims, keepdim=True) + 1e-8)
    return rescale * (eps / (std_ratio + 1e-8)) + (1.0 - rescale) * eps


# ---------- main entry: visualization hook ----------

@torch.no_grad()
def run_viz_hook(
    *,
    accelerator,  # accelerate.Accelerator
    global_step: int,
    args,  # must carry the viz args below
    batch: Dict[str, Any],
    model: nn.Module,  # your MultiAdaptorSDXL (UNet-like)
    vae: nn.Module,
    scheduler_train,  # diffusers.DDPMScheduler (training scheduler)
    cond_encoder: Optional[nn.Module],
    tokenizer: Optional[nn.Module],
    image_to_token_model: Optional[nn.Module],
    label_to_token_model: Optional[nn.Module],
    compute_dtype: torch.dtype,
    scaling_factor: float,
) -> None:
    """
    Produces 3 PNGs under {args.output_dir}/{args.samples_dir}/:
      - gs_{step}_grid.png
      - gs_{step}_noising_row.png
      - gs_{step}_denoising_row.png

    All visualization logic is isolated here; train.py just calls this function.
    """

    device = accelerator.device
    device_type = device.type
    out_dir = os.path.join(args.output_dir, args.samples_dir)
    ensure_dir(out_dir)

    # Save & switch modes
    model_was_training = model.training
    model.eval()
    cond_was_training = cond_encoder.training if cond_encoder is not None else None
    if cond_encoder is not None:
        cond_encoder.eval()
    vae_was_training = vae.training
    vae.eval()

    # ---------- helpers (inner) ----------
    def _encode_text(captions: List[str]) -> torch.Tensor:
            max_len = getattr(args, "max_length", 77)
            toks = tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            toks = {k: v.to(device=device, non_blocking=True) for k, v in toks.items()}
            # fp32 encode, return last_hidden_state
            outputs = cond_encoder(**toks, return_dict=True)
            return outputs.last_hidden_state  # [B, T, 768] (fp32)


    def _image_tokens(image_embeds: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if image_embeds is None or image_to_token_model is None:
            return None
        with torch.autocast(device_type=device_type, dtype=compute_dtype):
            return image_to_token_model(image_embeds.to(device=device, dtype=compute_dtype, non_blocking=True))

    def _label_tokens(label_embeds: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if label_embeds is None or label_to_token_model is None:
            return None
        with torch.autocast(device_type=device_type, dtype=compute_dtype):
            return label_to_token_model(label_embeds.to(device=device, dtype=compute_dtype, non_blocking=True))

    def _fresh_infer_scheduler():
        steps = int(getattr(args, "sample_inference_steps", 36))   # CHANGED: 36 by default
        name = getattr(args, "sample_scheduler", "euler_a")        # CHANGED: Euler-a default
        return _make_sd1x_scheduler(name=name, num_inference_steps=steps, device=device)

    def _noise_like(shape, seed: Optional[int]) -> torch.Tensor:
        g = None
        if seed is not None:
            g = torch.Generator(device=device).manual_seed(int(seed))
        return torch.randn(shape, device=device, dtype=compute_dtype, generator=g)

    # ===== A) Grid of fresh samples conditioned on THIS random batch =====
    images_bchw = batch["images"]  # B,3,H,W (on CPU or GPU)
    B, _, H, W = images_bchw.shape
    grid_rows = getattr(args, "sample_grid_rows", 4)
    grid_cols = getattr(args, "sample_grid_cols", 4)
    grid_B = min(B, grid_rows * grid_cols)
    sel_idxs = choose_random_indices(B, grid_B)

    # Latent shape from image size (H/8 x W/8, C=4)
    latent_shape = (grid_B, 4, H // 8, W // 8)

    # Conditioning (CFG)
    captions = [batch["captions"][i] for i in sel_idxs]
    neg_prompt = getattr(
        args, "sample_negative_prompt",
        "lowres, worst quality, low quality, blurry, jpeg artifacts"   # CHANGED: helpful default
    )
    guidance_scale = float(getattr(args, "sample_guidance_scale", 7.5))  # CHANGED: 7.5 default
    cfg_rescale = float(getattr(args, "sample_cfg_rescale", 0.7))        # NEW
    seed = getattr(args, "sample_seed", None)

    text_cond = _encode_text(captions).to(dtype=compute_dtype)
    text_uncond = _encode_text([neg_prompt] * grid_B).to(dtype=compute_dtype)

    # Optional extra tokens (duplicate for CFG if present)
    img_tokens = _image_tokens(batch.get("image_embeds", None)[sel_idxs] if "image_embeds" in batch else None)
    lbl_tokens = _label_tokens(batch.get("labels", None)[sel_idxs] if "labels" in batch else None)
    img_tokens_uncond = None if img_tokens is None else img_tokens.clone()
    lbl_tokens_uncond = None if lbl_tokens is None else lbl_tokens.clone()

    # Reverse diffusion sampling (CFG, init sigma, scale_model_input)
    sch = _fresh_infer_scheduler()
    latents = _noise_like(latent_shape, seed=seed)
    latents = latents * sch.init_noise_sigma

    with torch.autocast(device_type=device_type, dtype=compute_dtype):
        for t in sch.timesteps:
            lat_in = torch.cat([latents, latents], dim=0)
            lat_in = sch.scale_model_input(lat_in, t)

            txt = torch.cat([text_uncond, text_cond], dim=0)
            img_tok = None if img_tokens is None else torch.cat([img_tokens_uncond, img_tokens], dim=0)
            lbl_tok = None if lbl_tokens is None else torch.cat([lbl_tokens_uncond, lbl_tokens], dim=0)

            out = model(
                noisy_latents=lat_in,
                timesteps=t,
                text_tokens=txt,
                image_tokens=img_tok,
                label_tokens=lbl_tok,
            )
            eps = out.sample if hasattr(out, "sample") else out
            eps_u, eps_c = eps.chunk(2, dim=0)
            eps = eps_u + guidance_scale * (eps_c - eps_u)
            if cfg_rescale > 0.0:  # NEW
                eps = _cfg_rescale(eps, eps_u, rescale=cfg_rescale)

            latents = sch.step(eps, t, latents).prev_sample

    imgs = decode_latents(vae, latents, scaling_factor)  # B,3,H,W (CPU)
    grid_path = os.path.join(out_dir, f"gs_{global_step:07d}_grid.png")
    save_grid([imgs[i] for i in range(imgs.shape[0])], rows=grid_rows, cols=grid_cols, out_path=grid_path)

    # ===== B) Forward (noising) row: 5 panels =====
    idx0 = choose_random_indices(B, 1)[0]
    img0 = images_bchw[idx0:idx0+1].to(device=device, dtype=torch.float32)  # (1,3,H,W)
    # CHANGED: normalize to [-1,1] for VAE.encode
    img0_norm = (img0 * 2.0 - 1.0).clamp(-1.0, 1.0)
    with torch.autocast(device_type=device_type, dtype=compute_dtype):
        z0 = vae.encode(img0_norm).latent_dist.sample() * scaling_factor  # z ~ q(z|x) scaled

    fixed_noise = torch.randn_like(z0)

    t_forward = torch.linspace(
        0, scheduler_train.config.num_train_timesteps - 1,
        steps=5, dtype=torch.long, device=device
    )
    z_list_forward = []
    for t in t_forward:
        z_t = scheduler_train.add_noise(z0, fixed_noise, t)
        z_list_forward.append(z_t)

    imgs_forward = [decode_latents(vae, z, scaling_factor)[0] for z in z_list_forward]
    forward_path = os.path.join(out_dir, f"gs_{global_step:07d}_noising_row.png")
    save_grid(imgs_forward, rows=1, cols=5, out_path=forward_path)

    # ===== C) Reverse (denoising) row: 5 panels (CFG & proper scaling) =====
    captions1 = [batch["captions"][idx0]]
    tt_cond = _encode_text(captions1).to(dtype=compute_dtype)
    tt_uncond = _encode_text([neg_prompt]).to(dtype=compute_dtype)
    it1 = _image_tokens(batch.get("image_embeds", None)[idx0:idx0+1] if "image_embeds" in batch else None)
    lt1 = _label_tokens(batch.get("labels", None)[idx0:idx0+1] if "labels" in batch else None)

    sch2 = _fresh_infer_scheduler()
    z = _noise_like(z0.shape, seed=seed)
    z = z * sch2.init_noise_sigma
    snap_ids = set(evenly_spaced_indices(len(sch2.timesteps), 5))
    snaps: List[torch.Tensor] = []

    with torch.autocast(device_type=device_type, dtype=compute_dtype):
        for i, t in enumerate(sch2.timesteps):
            z_in = torch.cat([z, z], dim=0)
            z_in = sch2.scale_model_input(z_in, t)
            txt = torch.cat([tt_uncond, tt_cond], dim=0)
            img_tok = None if it1 is None else torch.cat([it1, it1], dim=0)
            lbl_tok = None if lt1 is None else torch.cat([lt1, lt1], dim=0)

            out = model(
                noisy_latents=z_in,
                timesteps=t,
                text_tokens=txt,
                image_tokens=img_tok,
                label_tokens=lbl_tok,
            )
            eps = out.sample if hasattr(out, "sample") else out
            eps_u, eps_c = eps.chunk(2, dim=0)
            eps = eps_u + guidance_scale * (eps_c - eps_u)
            if cfg_rescale > 0.0:
                eps = _cfg_rescale(eps, eps_u, rescale=cfg_rescale)

            z = sch2.step(eps, t, z).prev_sample
            if i in snap_ids:
                snaps.append(z.clone())

    imgs_denoise = [decode_latents(vae, zz, scaling_factor)[0] for zz in snaps[:5]]
    denoise_path = os.path.join(out_dir, f"gs_{global_step:07d}_denoising_row.png")
    save_grid(imgs_denoise, rows=1, cols=5, out_path=denoise_path)

    # Restore training modes
    if model_was_training: model.train()
    if cond_encoder is not None and cond_was_training: cond_encoder.train()
    if vae_was_training: vae.train()
