"""
STDiT2: Spatial-Temporal Diffusion Transformer v2.

Unified model supporting both single-channel (stage 2) and multi-channel
(stage 1, e.g. RGB + depth + segmentation) inputs via the `multi_channels`
parameter.
"""

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from .stdit2_blocks import (
    Attention,
    CaptionEmbedder,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    PositionEmbedding2D,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_layernorm,
    t2i_modulate,
)


def auto_grad_checkpoint(fn, *args, **kwargs):
    """Call ``fn`` with gradient checkpointing when the model is training."""
    if torch.is_grad_enabled() and torch.is_autocast_enabled():
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False, **kwargs)
    return fn(*args, **kwargs)


# =====================================================================
# STDiT2 Transformer Block
# =====================================================================


class STDiT2Block(nn.Module):
    """Single spatial-temporal DiT block with adaptive layer norm (adaLN)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        enable_flashattn: bool = False,
        enable_layernorm_kernel: bool = False,
        rope=None,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.enable_flashattn = enable_flashattn

        # Spatial self-attention
        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=enable_flashattn,
            qk_norm=qk_norm,
        )
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        # Cross-attention
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)

        # MLP
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=approx_gelu,
            drop=0,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Temporal self-attention
        self.norm_temp = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn_temp = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=enable_flashattn,
            rope=rope,
            qk_norm=qk_norm,
        )
        self.scale_shift_table_temporal = nn.Parameter(torch.randn(3, hidden_size) / hidden_size**0.5)

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        """Select between ``x`` and ``masked_x`` based on a per-frame mask."""
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(
        self,
        x,
        y,
        t,
        t_tmp,
        mask=None,
        x_mask=None,
        t0=None,
        t0_tmp=None,
        T=None,
        S=None,
        box_func=None,
        box=None,
        inflated=True,
    ):
        B, N, C = x.shape
        is_fifo = t.ndim == 3 and t_tmp.ndim == 3

        # --- Compute adaLN modulation parameters ---
        if is_fifo:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None, None] + t.reshape(B, T, 6, -1)
            ).chunk(6, dim=2)
            shift_tmp, scale_tmp, gate_tmp = (
                self.scale_shift_table_temporal[None, None] + t_tmp.reshape(B, T, 3, -1)
            ).chunk(3, dim=2)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
            shift_tmp, scale_tmp, gate_tmp = (
                self.scale_shift_table_temporal[None] + t_tmp.reshape(B, 3, -1)
            ).chunk(3, dim=1)

        if x_mask is not None:
            if is_fifo:
                shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                    self.scale_shift_table[None, None] + t0.reshape(B, T, 6, -1)
                ).chunk(6, dim=2)
                shift_tmp_zero, scale_tmp_zero, gate_tmp_zero = (
                    self.scale_shift_table_temporal[None, None] + t0_tmp.reshape(B, T, 3, -1)
                ).chunk(3, dim=2)
            else:
                shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                    self.scale_shift_table[None] + t0.reshape(B, 6, -1)
                ).chunk(6, dim=1)
                shift_tmp_zero, scale_tmp_zero, gate_tmp_zero = (
                    self.scale_shift_table_temporal[None] + t0_tmp.reshape(B, 3, -1)
                ).chunk(3, dim=1)

        # --- Spatial self-attention ---
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa, is_fifo=is_fifo)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm1(x), shift_msa_zero, scale_msa_zero, is_fifo=is_fifo)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        if inflated:
            x_s = rearrange(x_m, "(B V) (T S) C -> (B T) (V S) C", T=T, S=S, V=6)
        else:
            x_s = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
        x_s = self.attn(x_s)
        if inflated:
            x_s = rearrange(x_s, "(B T) (V S) C -> (B V) (T S) C", T=T, S=S)
        else:
            x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=T, S=S)

        if x_mask is not None:
            if is_fifo:
                gate_msa_zero = rearrange(gate_msa_zero, "B T N C -> (B T) N C")
                gate_msa = rearrange(gate_msa, "B T N C -> (B T) N C")
                x_s = rearrange(x_s, "B (T S) C -> (B T) S C", T=T)
                x_s_zero = gate_msa_zero * x_s
                x_s = gate_msa * x_s
                x_s_zero = rearrange(x_s_zero, "(B T) S C -> B (T S) C", T=T)
                x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=T)
            else:
                x_s_zero = gate_msa_zero * x_s
                x_s = gate_msa * x_s
            x_s = self.t_mask_select(x_mask, x_s, x_s_zero, T, S)
        else:
            if is_fifo:
                gate_msa = rearrange(gate_msa, "B T N C -> (B T) N C")
                x_s = rearrange(x_s, "B (T S) C -> (B T) S C", T=T)
                x_s = gate_msa * x_s
                x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=T)
            else:
                x_s = gate_msa * x_s
        x = x + self.drop_path(x_s)

        # --- Temporal self-attention ---
        x_m = t2i_modulate(self.norm_temp(x), shift_tmp, scale_tmp, is_fifo=is_fifo)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm_temp(x), shift_tmp_zero, scale_tmp_zero, is_fifo=is_fifo)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        x_t = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
        x_t = self.attn_temp(x_t)
        x_t = rearrange(x_t, "(B S) T C -> B (T S) C", T=T, S=S)
        if x_mask is not None:
            x_t_zero = gate_tmp_zero * x_t
            x_t = gate_tmp * x_t
            x_t = self.t_mask_select(x_mask, x_t, x_t_zero, T, S)
        else:
            if is_fifo:
                gate_tmp = rearrange(gate_tmp, "B T N C -> (B T) N C")
                x_t = rearrange(x_t, "B (T S) C -> (B T) S C", T=T)
                x_t = gate_tmp * x_t
                x_t = rearrange(x_t, "(B T) S C -> B (T S) C", T=T)
            else:
                x_t = gate_tmp * x_t
        x = x + self.drop_path(x_t)

        # --- Optional grounding / box conditioning ---
        if box_func is not None and box is not None:
            x = box_func((x, box))

        # --- Cross-attention ---
        x = rearrange(x, "B (T S) C -> (B T) S C", T=T, S=S)
        x = x + self.cross_attn(x, y, mask)
        x = rearrange(x, "(B T) S C -> B (T S) C", T=T, S=S)

        # --- MLP ---
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp, is_fifo=is_fifo)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero, is_fifo=is_fifo)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

        x_mlp = self.mlp(x_m)
        if x_mask is not None:
            x_mlp_zero = gate_mlp_zero * x_mlp
            x_mlp = gate_mlp * x_mlp
            x_mlp = self.t_mask_select(x_mask, x_mlp, x_mlp_zero, T, S)
        else:
            if is_fifo:
                gate_mlp = rearrange(gate_mlp, "B T N C -> (B T) N C")
                x_mlp = rearrange(x_mlp, "B (T S) C -> (B T) S C", T=T)
                x_mlp = gate_mlp * x_mlp
                x_mlp = rearrange(x_mlp, "(B T) S C -> B (T S) C", T=T)
            else:
                x_mlp = gate_mlp * x_mlp
        x = x + self.drop_path(x_mlp)

        return x


# =====================================================================
# STDiT2 Main Model
# =====================================================================


class STDiT2(nn.Module):
    """Spatial-Temporal Diffusion Transformer v2.

    Args:
        multi_channels: Number of channel groups. Set to 1 for standard
            diffusion (stage 2). Set to >1 (e.g. 3 for RGB+depth+seg) to
            create an additional ``x_embedder_fuse`` and expand the output
            channels accordingly (stage 1).
    """

    def __init__(
        self,
        input_size=(None, None, None),
        input_sq_size=32,
        in_channels=4,
        multi_channels=1,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        no_temporal_pos_emb=False,
        caption_channels=4096,
        model_max_length=120,
        dtype=torch.bfloat16,
        freeze=None,
        qk_norm=False,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.multi_channels = multi_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dtype = dtype
        self.no_temporal_pos_emb = no_temporal_pos_emb
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.enable_flashattn = enable_flashattn
        self.enable_layernorm_kernel = enable_layernorm_kernel

        # Dynamic spatial input support
        self.patch_size = patch_size
        self.input_size = input_size
        self.input_sq_size = input_sq_size
        self.pos_embed = PositionEmbedding2D(hidden_size)

        # Patch embedders
        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        if multi_channels > 1:
            self.x_embedder_fuse = PatchEmbed3D(
                patch_size=patch_size,
                in_chans=in_channels * multi_channels,
                embed_dim=hidden_size,
            )

        # Timestep / caption embedders
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.t_block_temp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )

        # Transformer blocks
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.rope = RotaryEmbedding(dim=self.hidden_size // self.num_heads)
        self.blocks = nn.ModuleList(
            [
                STDiT2Block(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path_rates[i],
                    enable_flashattn=self.enable_flashattn,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    rope=self.rope.rotate_queries_or_keys,
                    qk_norm=qk_norm,
                )
                for i in range(self.depth)
            ]
        )

        # Final output layer
        self.final_layer = T2IFinalLayer(
            hidden_size, np.prod(self.patch_size), self.out_channels * self.multi_channels
        )

        # Resolution / aspect-ratio / frame-count embedders
        assert self.hidden_size % 3 == 0, "hidden_size must be divisible by 3"
        self.csize_embedder = SizeEmbedder(self.hidden_size // 3)
        self.ar_embedder = SizeEmbedder(self.hidden_size // 3)
        self.fl_embedder = SizeEmbedder(self.hidden_size)
        self.fps_embedder = SizeEmbedder(self.hidden_size)

        # Initialization
        self.initialize_weights()
        self.initialize_temporal()
        if freeze is not None:
            assert freeze in ["not_temporal", "text"]
            if freeze == "not_temporal":
                self.freeze_not_temporal()
            elif freeze == "text":
                self.freeze_text()

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """Reverse patch embedding: [B, N, C] -> [B, C_out, T, H, W]."""
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels * self.multi_channels,
        )
        # Remove padding
        x = x[:, :, :R_t, :R_h, :R_w]
        return x

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        x,
        timestep,
        y,
        mask=None,
        x_mask=None,
        num_frames=None,
        height=None,
        width=None,
        ar=None,
        fps=None,
    ):
        """
        Args:
            x: Latent video tensor of shape ``[B, C, T, H, W]``.
            timestep: Diffusion timesteps of shape ``[B]``.
            y: Caption embeddings of shape ``[B, 1, N_token, C]``.
            mask: Prompt token mask of shape ``[B, N_token]``.
            x_mask: Per-frame mask of shape ``[B, T]`` (optional).
            num_frames, height, width, ar, fps: Conditioning scalars.

        Returns:
            Output latent of shape ``[B, C_out, T, H, W]``.
        """
        B = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        # === Process conditioning info ===
        hw = torch.cat([height[:, None], width[:, None]], dim=1)
        rs = (height[0].item() * width[0].item()) ** 0.5
        csize = self.csize_embedder(hw, B)

        ar = ar.unsqueeze(1)
        ar = self.ar_embedder(ar, B)
        data_info = torch.cat([csize, ar], dim=1)

        fl = num_frames.unsqueeze(1)
        fps = fps.unsqueeze(1)
        fl = self.fl_embedder(fl, B)
        fl = fl + self.fps_embedder(fps, B)

        # === Spatial layout ===
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)
        S = H * W
        scale = rs / self.input_sq_size
        base_size = round(S**0.5)
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # === Patch embedding ===
        x = self.x_embedder(x)
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb
        x = rearrange(x, "B T S C -> B (T S) C")

        # === AdaIN timestep conditioning ===
        t = self.t_embedder(timestep, dtype=x.dtype)
        t_spc = t + data_info
        t_tmp = t + fl
        t_spc_mlp = self.t_block(t_spc)
        t_tmp_mlp = self.t_block_temp(t_tmp)
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0_spc = t0 + data_info
            t0_tmp = t0 + fl
            t0_spc_mlp = self.t_block(t0_spc)
            t0_tmp_mlp = self.t_block_temp(t0_tmp)
        else:
            t0_spc = None
            t0_tmp = None
            t0_spc_mlp = None
            t0_tmp_mlp = None

        # === Caption embedding ===
        y = self.y_embedder(y, self.training)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        # === Transformer blocks ===
        for block in self.blocks:
            x = auto_grad_checkpoint(
                block, x, y, t_spc_mlp, t_tmp_mlp, y_lens, x_mask,
                t0_spc_mlp, t0_tmp_mlp, T, S,
            )

        # === Output ===
        x = self.final_layer(x, t, x_mask, t0_spc, T, S)
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)
        x = x.to(torch.float32)
        return x

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-init patch embedders
        nn.init.constant_(self.x_embedder.proj.weight, 0)
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        if self.multi_channels > 1:
            nn.init.constant_(self.x_embedder_fuse.proj.weight, 0)
            nn.init.constant_(self.x_embedder_fuse.proj.bias, 0)

        # Timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        nn.init.normal_(self.t_block_temp[1].weight, std=0.02)

        # Caption embedding MLP
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out cross-attention output projections
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def initialize_temporal(self):
        for block in self.blocks:
            nn.init.constant_(block.attn_temp.proj.weight, 0)
            nn.init.constant_(block.attn_temp.proj.bias, 0)

    def freeze_not_temporal(self):
        for n, p in self.named_parameters():
            if "attn_temp" not in n:
                p.requires_grad = False

    def freeze_text(self):
        for n, p in self.named_parameters():
            if "cross_attn" in n:
                p.requires_grad = False


# =====================================================================
# Factory
# =====================================================================


def STDiT2_XL_2(from_pretrained=None, **kwargs):
    """Create an STDiT2-XL/2 model (depth=28, hidden=1152, patch=1x2x2)."""
    model = STDiT2(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
    if from_pretrained is not None:
        ckpt = torch.load(from_pretrained, map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt:
            ckpt = ckpt["model"]
        model.load_state_dict(ckpt, strict=False)
    return model
