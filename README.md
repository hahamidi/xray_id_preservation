Here’s a complete README.md template you can drop into your repo and adjust as needed:

# Stable Diffusion XL with IP-Adapters and Extra Conditioning

This project is a **config-driven training framework** for [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), extended with:

- **IP-Adapters** (per-layer attention processors).
- **Extra conditioning modules**:
  - `label_to_token`: maps label vectors → token embeddings.
  - `embed_to_token`: maps external embeddings → token embeddings.

The framework is designed to be modular: every component (UNet, VAE, schedulers, adapters, projectors) is defined in YAML and can be swapped or frozen without touching the training code.

---

## 📌 Features

- Configurable UNet, VAE, and schedulers (via YAML).
- Support for **per-layer adapters** injected into UNet cross-attention.
- External conditioning through **label/projector/embedding modules**.
- Integration with [🤗 Accelerate](https://github.com/huggingface/accelerate).
- Configurable checkpoint loading/saving.

---

## ⚙️ Installation

Clone and install dependencies:

```bash
git clone <your-repo>
cd <your-repo>
pip install -r requirements.txt

Requirements include:
	•	torch
	•	diffusers
	•	transformers
	•	accelerate
	•	pyyaml

⸻

🗂 Config Structure

Training is controlled by a YAML config file.
Example:

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

	•	unet_config: defines the diffusion UNet.
	•	first_stage_config: defines the VAE (encoder/decoder).
	•	adapters_config: per-layer IP-Adapters, image/label projectors.
	•	token_conditioners: extra modules for conditioning tokens.

See configs/example.yaml for a full template.

⸻

🚀 Training

Launch training with Accelerate:

accelerate launch train.py \
  --config configs/example.yaml \
  --output_dir ./output \
  --train_batch_size 2 \
  --num_train_epochs 100

Key CLI flags
	•	--mixed_precision: no, fp16, bf16
	•	--report_to: tensorboard, wandb, etc.
	•	--save_steps: checkpoint saving interval

⸻

📦 Checkpoints

Checkpoints are saved to:

output_dir/checkpoint-<global_step>/

Each contains:
	•	model weights
	•	optimizer state
	•	scheduler state
	•	Accelerate state

⸻

🧩 Extending
	•	Add new per-layer adapters by editing adapters_config.
	•	Add new conditioning modules under token_conditioners.
	•	Replace VAE, UNet, or schedulers by swapping their target class in YAML.

⸻

🔗 Architecture

High-level flow:

[text tokens] + [label_to_token(label vecs)] + [embed_to_token(embeds)] 
        ↓
   concatenated conditioning sequence
        ↓
   UNet (with IP-Adapters are injected into every cross-attention layer of the UNet. Each modality (e.g., image, label, embedding) has its own dedicated adapter branch, providing modality-specific cross-attention into the UNet, while the base UNet weights remain unchanged.)
        ↓
       latents
        ↓
       VAE → images


⸻

📝 Notes
	•	IP-Adapters are per-layer modules, created and injected into UNet cross-attention.
	•	Label-to-token and embed-to-token are separate modules, not tied to per-layer adapters.
	•	Each module can be frozen or trained depending on config flags.
	•	This repo is designed for research flexibility, not production inference speed.

