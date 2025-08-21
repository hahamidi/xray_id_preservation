# Stable Diffusion Training Framework with Adapter

This repo provides a **config-driven training setup** for Stable Diffusion XL with:

- **IP-Adapters** (per-layer attention processors).
- Extra conditioning modules:
  - `label_to_token`
  - `embed_to_token`

---

## 📦 Installation

```bash
git clone <your-repo>
cd <your-repo>
pip install -r requirements.txt
```

Requirements include:
- torch
- diffusers
- transformers
- accelerate
- pyyaml

---

## 🚀 Training

```bash
accelerate launch train.py \
  --config configs/example.yaml \
  --output_dir ./output
```

### Key CLI Flags
- `--train_batch_size`
- `--num_train_epochs`
- `--mixed_precision` (`no`, `fp16`, `bf16`)
- `--save_steps`
- `--report_to` (`tensorboard`, `wandb`, etc.)

---

## 🗂 Config Structure

Training is controlled by YAML configs, e.g.:

```yaml
model:
  params:
    unet_config: {...}
    first_stage_config: {...}
    adapters_config: {...}
    token_conditioners:
      label_to_token: {...}
      embed_to_token: {...}
data:
  params:
    resolution: 1024
    data_root_path: /path/to/dataset
    pairs_file_path: /path/to/train.txt
```

- `unet_config`: defines the diffusion UNet.  
- `first_stage_config`: defines the VAE.  
- `adapters_config`: per-layer IP-Adapters, image/label projectors.  
- `token_conditioners`: extra modules for conditioning tokens.  

See `configs/example.yaml` for a template.

---

## 💾 Checkpoints

Checkpoints are saved in:

```
output/checkpoint-<step>/
```

Each folder contains:
- Model weights  
- Optimizer state  
- Scheduler state  
- Accelerate state  

---

## 🧩 Extending

- Add new per-layer adapters in `adapters_config`.  
- Add new conditioning modules under `token_conditioners`.  
- Swap out VAE, UNet, or schedulers by editing the YAML config.  

---

## 🔗 Architecture

High-level flow:

```
[text tokens] + [label_to_token(label vecs)] + [embed_to_token(embeds)]
        ↓
   concatenated conditioning sequence
        ↓
   UNet with IP-Adapters (per-layer cross-attn injections)
        ↓
       latents
        ↓
       VAE → images
```

- IP-Adapters = per-layer modules injected into UNet cross-attention.  
- Label-to-token & embed-to-token = separate modules, independent of adapters.  
- Modules can be frozen or trained depending on config flags.  

---

## 📝 Notes

- Modular design for research flexibility.  
- Not optimized for inference speed.  
- Works with Accelerate for distributed training.  