import os, time, json, itertools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel


# -------------------------
# minimal, plug-and-play wrapper
# -------------------------
class XRay_SDXL(nn.Module):
    def __init__(self, unet, image_proj_model=None, label_proj_model=None, label_token_embed=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.label_proj_model = label_proj_model
        self.label_token_embed = label_token_embed

    def forward(
        self,
        noisy_latents,
        timesteps,
        text_tokens,                 # [B, T_txt, D_txt]
        added_cond,                  # dict for UNet added_cond_kwargs (e.g., time_ids, pooled)
        image_embeds=None,           # [B, T_img, D_img]
        label_embeds=None,           # [B, T_lab, D_lab]
        labels=None,                 # [B]
    ):
        tokens = [text_tokens]
        if image_embeds is not None and self.image_proj_model is not None:
            ip = self.image_proj_model(image_embeds)     # -> [B, T_img, D_txt]
            tokens.append(ip)
        if label_embeds is not None and self.label_proj_model is not None:
            lp = self.label_proj_model(label_embeds)     # -> [B, T_lab, D_txt]
            tokens.append(lp)
        if labels is not None and self.label_token_embed is not None:
            lt = self.label_token_embed(labels).unsqueeze(1)  # -> [B, 1, D_txt]
            tokens.append(lt)

        encoder_hidden_states = torch.cat(tokens, dim=1)

        out = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond,
        )
        return out.sample if hasattr(out, "sample") else out


# -------------------------
# simple, dynamic text encoder
# returns (token_embeddings [B,T,D], pooled [B,D_pool])
# -------------------------
class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size, text_dim, pooled_dim=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, text_dim)
        self.pooled_proj = nn.Linear(text_dim, pooled_dim or text_dim)

    def forward(self, input_ids):
        tok = self.embed(input_ids)                  # [B,T,D]
        pooled = tok.mean(dim=1)                     # [B,D]
        pooled = self.pooled_proj(pooled)            # [B,Dp]
        return tok, pooled


def build_text_encoder_from_config(cfg):
    return SimpleTextEncoder(
        vocab_size=cfg.get("vocab_size", 30522),
        text_dim=cfg["text_dim"],
        pooled_dim=cfg.get("pooled_dim", cfg["text_dim"]),
    )


# -------------------------
# config-based constructors (no weights required)
# expects a single JSON file with keys: "scheduler", "vae", "unet", "text_encoder"
# -------------------------
def load_models_from_config(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    scheduler = DDPMScheduler.from_config(cfg["scheduler"])
    vae = AutoencoderKL.from_config(cfg["vae"])
    unet = UNet2DConditionModel.from_config(cfg["unet"])
    text_encoder = build_text_encoder_from_config(cfg["text_encoder"])

    return scheduler, vae, unet, text_encoder, cfg


# -------------------------
# main training
# modes:
#   - pretrain_text: train UNet with text only
#   - finetune_adapter: freeze UNet, train adapters (image/label)
# -------------------------
def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )
    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        print("output dir ready")

    noise_scheduler, vae, unet, text_encoder, cfg = load_models_from_config(args.model_config)

    text_dim = cfg["text_encoder"]["text_dim"]
    assert text_dim == unet.config.cross_attention_dim, "text_dim must match UNet cross_attention_dim"

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.train_mode == "pretrain_text":
        unet.requires_grad_(True)
        image_proj_model = None
        label_proj_model = None
        label_token_embed = None
    else:
        unet.requires_grad_(False)
        image_in_dim = args.image_in_dim or cfg.get("image_in_dim", text_dim)
        label_in_dim = args.label_in_dim or cfg.get("label_in_dim", text_dim)
        image_proj_model = nn.Linear(image_in_dim, text_dim) if args.use_image_adapter else None
        label_proj_model = nn.Linear(label_in_dim, text_dim) if args.use_label_proj else None
        label_token_embed = nn.Embedding(args.num_labels, text_dim) if args.use_label_tokens else None

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    if image_proj_model is not None:
        image_proj_model.to(accelerator.device, dtype=weight_dtype)
    if label_proj_model is not None:
        label_proj_model.to(accelerator.device, dtype=weight_dtype)
    if label_token_embed is not None:
        label_token_embed.to(accelerator.device)

    model = XRay_SDXL(unet, image_proj_model, label_proj_model, label_token_embed)

    if args.train_mode == "pretrain_text":
        params_to_opt = model.unet.parameters()
    else:
        parts = []
        if model.image_proj_model is not None:
            parts.append(model.image_proj_model.parameters())
        if model.label_proj_model is not None:
            parts.append(model.label_proj_model.parameters())
        if model.label_token_embed is not None:
            parts.append(model.label_token_embed.parameters())
        params_to_opt = itertools.chain(*parts) if parts else []

    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # your dataset must provide:
    # images [B,3,H,W], text_input [B,T], original_size [B,2], yyxx [B,4]
    # for adapter finetune (optional): image_embeds [B,T_img,D_img], label_embeds [B,T_lab,D_lab], labels [B]
    train_dataset = YourXRayDataset(
        size=args.resolution,
        data_root_path=args.data_root_path,
        pairs_file_path=args.pairs_file_path,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    global_step = 0
    for epoch in range(args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin

            with accelerator.accumulate(model):
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                noise = torch.randn_like(latents)
                if args.noise_offset:
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=accelerator.device, dtype=weight_dtype
                    )

                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    text_tokens, pooled = text_encoder(batch["text_input"].to(accelerator.device))

                add_time_ids = torch.cat(
                    [batch["original_size"].to(accelerator.device), batch["yyxx"].to(accelerator.device)],
                    dim=1,
                ).to(accelerator.device, dtype=weight_dtype)
                added_cond = {"text_embeds": pooled, "time_ids": add_time_ids}

                if args.train_mode == "pretrain_text":
                    image_embeds = None
                    label_embeds = None
                    labels = None
                else:
                    image_embeds = batch.get("image_embeds", None)
                    if image_embeds is not None:
                        image_embeds = image_embeds.to(accelerator.device, dtype=weight_dtype)

                    label_embeds = batch.get("label_embeds", None)
                    if label_embeds is not None:
                        label_embeds = label_embeds.to(accelerator.device, dtype=weight_dtype)

                    labels = batch.get("labels", None)
                    if labels is not None:
                        labels = labels.to(accelerator.device)

                noise_pred = model(
                    noisy_latents,
                    timesteps,
                    text_tokens.to(accelerator.device, dtype=weight_dtype),
                    added_cond,
                    image_embeds=image_embeds,
                    label_embeds=label_embeds,
                    labels=labels,
                )

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print(
                        "epoch {}, step {}, load_data_time {:.4f}, step_time {:.4f}, loss {:.6f}".format(
                            epoch, step, load_data_time, time.perf_counter() - begin, avg_loss
                        )
                    )

            global_step += 1
            if accelerator.is_main_process and global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)

            begin = time.perf_counter()


if __name__ == "__main__":
    main()