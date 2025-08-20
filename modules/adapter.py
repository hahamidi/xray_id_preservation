import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    import xformers
    import xformers.ops
    xformers_available = True
except Exception as e:
    xformers_available = False

class IPAttnProcessor(nn.Module):
    r"""
    Attention processor supporting text + image + label tokens.

    Inputs are expected to be concatenated as:
        encoder_hidden_states = [ text_tokens | image_tokens | label_tokens ]

    Args:
        hidden_size (int): q/k/v head dim per head (Diffusers UNet cross-attn hidden size)
        cross_attention_dim (int): D of encoder_hidden_states tokens (text/image/label) (e.g., 2048 in SDXL)
        # Back-compat (image)
        scale (float): alias for `scale_image` (default 1.0)
        num_tokens (int): alias for `num_tokens_image` (default 4)
        # New
        scale_image (float): weight for image tokens
        scale_label (float): weight for label tokens
        num_tokens_image (int): number of image tokens appended to the end
        num_tokens_label (int): number of label tokens appended to the end
        share_ip_kv (bool): if True, share KV projections between image/label (uses image layers)
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        *, # above parameters come first from the base Unet. below are based on the config file
        scale_image: Optional[float] = 1.0,
        scale_label: Optional[float] = 1.0,
        num_tokens_image: int = 4,
        num_tokens_label: int = 4,
        share_ip_kv: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        # Back-compat defaults
        self.num_tokens_image = num_tokens_image
        self.num_tokens_label = num_tokens_label
        self.scale_image = scale_image
        self.scale_label = scale_label
        self.share_ip_kv = share_ip_kv

        in_dim = cross_attention_dim or hidden_size

        # Image KV (keep original names for warm-start compatibility)
        self.to_k_ip = nn.Linear(in_dim, hidden_size, bias=False)  # image
        self.to_v_ip = nn.Linear(in_dim, hidden_size, bias=False)  # image

        # Label KV (separate layers unless sharing is requested)
        if share_ip_kv:
            self.to_k_lbl = self.to_k_ip
            self.to_v_lbl = self.to_v_ip
        else:
            self.to_k_lbl = nn.Linear(in_dim, hidden_size, bias=False)
            self.to_v_lbl = nn.Linear(in_dim, hidden_size, bias=False)

    def forward(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        region_control=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        # --- Determine batch/seq lengths (use encoder seq if provided) ---
        if encoder_hidden_states is None:
            batch_size, sequence_length, _ = hidden_states.shape
        else:
            batch_size, sequence_length, _ = encoder_hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # --- Queries ---
        query = attn.to_q(hidden_states)

        # --- Split encoder tokens into [text | image | label] ---
        if encoder_hidden_states is None:
            enc_text = hidden_states
            enc_img = None
            enc_lbl = None
        else:
            n_img = int(self.num_tokens_image or 0)
            n_lbl = int(self.num_tokens_label or 0)
            n_extra = n_img + n_lbl
            if n_extra > 0:
                end_text = encoder_hidden_states.shape[1] - n_extra
                if end_text < 0:
                    raise ValueError(
                        f"num_tokens_image({n_img}) + num_tokens_label({n_lbl}) exceeds encoder seq len "
                        f"{encoder_hidden_states.shape[1]}"
                    )
                enc_text = encoder_hidden_states[:, :end_text, :]
                enc_img = encoder_hidden_states[:, end_text:end_text + n_img, :] if n_img > 0 else None
                enc_lbl = encoder_hidden_states[:, end_text + n_img:, :] if n_lbl > 0 else None
            else:
                enc_text = encoder_hidden_states
                enc_img = None
                enc_lbl = None

            if attn.norm_cross:
                enc_text = attn.norm_encoder_hidden_states(enc_text)

        # --- Standard cross-attn on text tokens ---
        key = attn.to_k(enc_text)
        value = attn.to_v(enc_text)

        q = attn.head_to_batch_dim(query)
        k = attn.head_to_batch_dim(key)
        v = attn.head_to_batch_dim(value)

        if xformers_available:
            out_text = self._memory_efficient_attention_xformers(q, k, v, attention_mask)
        else:
            attn_probs = attn.get_attention_scores(q, k, attention_mask)
            out_text = torch.bmm(attn_probs, v)
        out_text = attn.batch_to_head_dim(out_text)

        # --- Extra cross-attn: IMAGE tokens ---
        out_img = 0
        if enc_img is not None and self.num_tokens_image > 0 and self.scale_image != 0:
            k_img = self.to_k_ip(enc_img)
            v_img = self.to_v_ip(enc_img)
            k_img = attn.head_to_batch_dim(k_img)
            v_img = attn.head_to_batch_dim(v_img)

            if xformers_available:
                img_ctx = self._memory_efficient_attention_xformers(q, k_img, v_img, None)
            else:
                img_probs = attn.get_attention_scores(q, k_img, None)
                img_ctx = torch.bmm(img_probs, v_img)
            out_img = attn.batch_to_head_dim(img_ctx)

            # Optional region control (apply only to image stream, as in your original)
            # if len(region_control.prompt_image_conditioning) == 1:
            #     region_mask = region_control.prompt_image_conditioning[0].get('region_mask', None)
            #     if region_mask is not None:
            #         H, W = region_mask.shape[:2]
            #         ratio = (H * W / q.shape[1]) ** 0.5
            #         mask = F.interpolate(region_mask[None, None], scale_factor=1 / ratio, mode='nearest').reshape([1, -1, 1])
            #     else:
            #         mask = torch.ones_like(out_img)
            #     out_img = out_img * mask

        # --- Extra cross-attn: LABEL tokens ---
        out_lbl = 0
        if enc_lbl is not None and self.num_tokens_label > 0 and self.scale_label != 0:
            k_lbl = self.to_k_lbl(enc_lbl)
            v_lbl = self.to_v_lbl(enc_lbl)
            k_lbl = attn.head_to_batch_dim(k_lbl)
            v_lbl = attn.head_to_batch_dim(v_lbl)

            if xformers_available:
                lbl_ctx = self._memory_efficient_attention_xformers(q, k_lbl, v_lbl, None)
            else:
                lbl_probs = attn.get_attention_scores(q, k_lbl, None)
                lbl_ctx = torch.bmm(lbl_probs, v_lbl)
            out_lbl = attn.batch_to_head_dim(lbl_ctx)

        # --- Combine streams ---
        hidden_states = out_text
        if isinstance(out_img, torch.Tensor):
            hidden_states = hidden_states + self.scale_image * out_img
        if isinstance(out_lbl, torch.Tensor):
            hidden_states = hidden_states + self.scale_label * out_lbl

        # --- Output projection + residual ---
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        return xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
    


# ---- Minimal fake "attn" object to simulate Diffusers CrossAttention ----
class DummyAttn:
    def __init__(self, hidden_size, cross_attention_dim, num_heads=8):
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(hidden_size, hidden_size), nn.Dropout(0.0)])

        self.group_norm = None
        self.spatial_norm = None
        self.norm_cross = False
        self.residual_connection = True
        self.rescale_output_factor = 1.0

    def prepare_attention_mask(self, mask, seq_len, batch_size):
        return None

    def norm_encoder_hidden_states(self, enc):
        return enc

    def head_to_batch_dim(self, tensor):
        b, s, d = tensor.shape
        return tensor.view(b, s, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(b * self.num_heads, s, self.head_dim)

    def batch_to_head_dim(self, tensor):
        bnh, s, d = tensor.shape
        b = bnh // self.num_heads
        return tensor.view(b, self.num_heads, s, d).permute(0, 2, 1, 3).reshape(b, s, self.num_heads * d)

    def get_attention_scores(self, q, k, mask):
        attn_weights = torch.bmm(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
        return torch.softmax(attn_weights, dim=-1)

# ---- Quick test ----
if __name__ == "__main__":
    torch.manual_seed(0)

    bsz = 2
    seq_text, seq_img, seq_lbl = 16, 4, 3
    hidden = 64
    cross_dim = 64

    # tokens
    text_tokens = torch.randn(bsz, seq_text, cross_dim)
    image_tokens = torch.randn(bsz, seq_img, cross_dim)
    label_tokens = torch.randn(bsz, seq_lbl, cross_dim)
    encoder_hidden_states = torch.cat([text_tokens, image_tokens, label_tokens], dim=1)

    hidden_states = torch.randn(bsz, 20, hidden)

    # dummy cross-attn
    attn = DummyAttn(hidden_size=hidden, cross_attention_dim=cross_dim, num_heads=4)

    # processor
    proc = IPAttnProcessor(
        hidden_size=hidden,
        cross_attention_dim=cross_dim,
        num_tokens_image=seq_img,
        num_tokens_label=seq_lbl,
        scale_image=1.0,
        scale_label=0.5,
    )

    out = proc(attn, hidden_states, encoder_hidden_states)
    print("Output shape:", out.shape)