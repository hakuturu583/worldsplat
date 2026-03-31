# --------------------------------------------------------
# STDiT2 Building Blocks
# --------------------------------------------------------
# References:
# PixArt: https://github.com/PixArt-alpha/PixArt-alpha
# Latte:  https://github.com/Vchitect/Latte
# DiT:    https://github.com/facebookresearch/DiT/tree/main
# GLIDE:  https://github.com/openai/glide-text2im
# MAE:    https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import functools
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import Mlp

try:
    import xformers.ops
except ImportError:
    xformers = None


approx_gelu = lambda: nn.GELU(approximate="tanh")


# ===============================================
# Feed-Forward Network
# ===============================================


class GEGLU(nn.Module):
    """Gated GLU activation with GELU gating."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """Feed-forward network with optional GLU gating."""

    def __init__(self, dim: int, dim_out: int = None, mult: int = 4, glu: bool = False, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        if glu:
            project_in = GEGLU(dim, inner_dim)
        else:
            project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===============================================
# Normalization
# ===============================================


class LlamaRMSNorm(nn.Module):
    """RMS normalization (equivalent to T5LayerNorm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def get_layernorm(hidden_size: int, eps: float, affine: bool, use_kernel: bool):
    """Return FusedLayerNorm (apex) if use_kernel=True, else standard LayerNorm."""
    if use_kernel:
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(hidden_size, elementwise_affine=affine, eps=eps)
        except ImportError:
            raise RuntimeError("FusedLayerNorm not available. Please install apex.")
    else:
        return nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)


# ===============================================
# Modulation Helpers
# ===============================================


def modulate(norm_func, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation: norm(x) * (1 + scale) + shift.

    Args:
        norm_func: Normalization function (e.g., LayerNorm).
        x: Input tensor of shape (B, N, D).
        shift: Shift tensor of shape (B, D).
        scale: Scale tensor of shape (B, D).
    """
    dtype = x.dtype
    x = norm_func(x.to(torch.float32)).to(dtype)
    x = x * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
    return x.to(dtype)


def t2i_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, is_fifo: bool = False) -> torch.Tensor:
    """Text-to-image adaptive modulation with optional FIFO (per-frame) support.

    When is_fifo=False, shift/scale have shape (B, 1, C) and are broadcast.
    When is_fifo=True, shift/scale have shape (B, T, N, C) for per-frame modulation.
    """
    if not is_fifo:
        return x * (1 + scale) + shift
    B, T, N, C = shift.shape
    x = rearrange(x, "B (T S) C -> (B T) S C", T=T)
    shift = rearrange(shift, "B T N C -> (B T) N C")
    scale = rearrange(scale, "B T N C -> (B T) N C")
    x = x * (1 + scale) + shift
    x = rearrange(x, "(B T) S C -> B (T S) C", T=T)
    return x


# ===============================================
# Patch Embedding
# ===============================================


class PatchEmbed3D(nn.Module):
    """3D video patch embedding with automatic padding.

    Args:
        patch_size: Temporal and spatial patch sizes (T, H, W).
        in_chans: Number of input channels.
        embed_dim: Embedding dimension.
        norm_layer: Optional normalization layer.
        flatten: If True, flatten spatial dims to sequence.
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, D, H, W = x.size()
        # Pad to make dimensions divisible by patch_size
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # (B, C, T, H, W)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, C, T, H, W) -> (B, N, C)
        return x


# ===============================================
# Attention Layers
# ===============================================


class Attention(nn.Module):
    """Multi-head self-attention with optional RoPE and flash attention.

    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        qkv_bias: If True, add bias to qkv projection.
        qk_norm: If True, apply RMSNorm to q and k.
        attn_drop: Attention dropout rate.
        proj_drop: Output projection dropout rate.
        norm_layer: Normalization layer for qk_norm.
        enable_flashattn: If True, use flash attention when beneficial.
        rope: Optional rotary position embedding module.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flashattn: bool = False,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn = enable_flashattn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Flash attn is not memory efficient for small sequences (empirical)
        enable_flashattn = self.enable_flashattn and (N > B)

        qkv = self.qkv(x)
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.rope:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)
        q, k = self.q_norm(q), self.k_norm(k)

        if enable_flashattn:
            from flash_attn import flash_attn_func

            # (B, H, N, D) -> (B, N, H, D)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
            )
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.to(torch.float32).softmax(dim=-1).to(dtype)
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not enable_flashattn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention using xformers memory-efficient attention.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        attn_drop: Attention dropout rate.
        proj_drop: Output projection dropout rate.
    """

    def __init__(self, d_model: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Args:
            x: Image tokens of shape (B, N, C).
            cond: Conditioning tokens of shape (B, S, C).
            mask: Optional list of sequence lengths for block-diagonal masking.
        """
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossViewAttention(nn.Module):
    """Left-neighbor cross-view attention for surround-view cameras.

    Each view attends to its left neighbor (circularly: view 0 attends to view 5).
    Designed for 6 surrounding cameras in autonomous driving setups.

    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        qkv_bias: If True, add bias to qkv projection.
        qk_norm: If True, apply normalization to q and k.
        attn_drop: Attention dropout rate.
        proj_drop: Output projection dropout rate.
        norm_layer: Normalization layer for qk_norm.
        enable_flashattn: If True, use flash attention.
        view_num: Number of camera views (default: 6).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        enable_flashattn: bool = False,
        view_num: int = 6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn = enable_flashattn
        self.view_num = view_num

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        if self.enable_flashattn:
            qkv_permute_shape = (2, 0, 1, 3, 4)
        else:
            qkv_permute_shape = (2, 0, 3, 1, 4)
        qkv = qkv.view(qkv_shape).permute(qkv_permute_shape)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Build circular left-neighbor index: [5, 0, 1, 2, 3, 4]
        left_former_idx = torch.arange(self.view_num) - 1
        left_former_idx[0] = self.view_num - 1

        # Remap k, v to attend to left neighbor
        k = rearrange(k, "(B V) ... -> B V ...", V=self.view_num)
        v = rearrange(v, "(B V) ... -> B V ...", V=self.view_num)
        k = k[:, left_former_idx]
        v = v[:, left_former_idx]
        k = rearrange(k, "B V ... -> (B V) ...")
        v = rearrange(v, "B V ... -> (B V) ...")

        if self.enable_flashattn:
            from flash_attn import flash_attn_func

            x = flash_attn_func(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
            )
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.to(torch.float32).softmax(dim=-1).to(dtype)
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not self.enable_flashattn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GatedSelfAttention(nn.Module):
    """Gated self-attention for object grounding.

    Fuses object tokens into the main sequence via gated attention and MLP.
    Gates are initialized to zero for stable training.

    Args:
        d_model: Model dimension.
        d_cond: Conditioning (object) token dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, d_model: int, d_cond: int, num_heads: int):
        super().__init__()
        self.linear = nn.Linear(d_cond, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, num_heads=num_heads, qkv_bias=False)
        self.mlp = FeedForward(d_model, glu=True)
        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))

    def forward(self, x):
        """
        Args:
            x: Tuple of (image_tokens, object_tokens).
               image_tokens: (B, N, C), object_tokens: (B, M, d_cond).
        Returns:
            Gated-fused image tokens of shape (B, N, C).
        """
        x, objs = x
        B, N, C = x.shape
        objs = self.linear(objs)
        x = x + torch.tanh(self.alpha_attn) * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, :N, :]
        x = x + torch.tanh(self.alpha_dense) * self.mlp(self.norm2(x))
        return x


# ===============================================
# Embedding Layers
# ===============================================


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations via sinusoidal encoding + MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.

        Args:
            t: 1-D tensor of N indices (may be fractional).
            dim: Output embedding dimension.
            max_period: Controls the minimum frequency.

        Returns:
            Tensor of shape (N, dim).
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        return self.mlp(t_freq)


class SizeEmbedder(TimestepEmbedder):
    """Embeds scalar size/resolution/FPS/frame-count values into vector representations.

    Extends TimestepEmbedder to handle multi-dimensional size inputs.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    def forward(self, s: torch.Tensor, bs: int) -> torch.Tensor:
        """
        Args:
            s: Size tensor of shape (B,) or (B, D) where D is the number of dimensions.
            bs: Batch size (s is repeated if s.shape[0] != bs).
        """
        if s.ndim == 1:
            s = s[:, None]
        assert s.ndim == 2
        if s.shape[0] != bs:
            s = s.repeat(bs // s.shape[0], 1)
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
        s_emb = self.mlp(s_freq)
        s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return s_emb

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class CaptionEmbedder(nn.Module):
    """Caption embedding with classifier-free guidance dropout.

    Args:
        in_channels: Input caption feature dimension.
        hidden_size: Output embedding dimension.
        uncond_prob: Probability of dropping captions for CFG during training.
        act_layer: Activation layer for the MLP.
        token_num: Number of caption tokens.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        uncond_prob: float,
        act_layer=nn.GELU(approximate="tanh"),
        token_num: int = 120,
    ):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            drop=0,
        )
        self.register_buffer(
            "y_embedding",
            torch.randn(token_num, in_channels) / in_channels**0.5,
        )
        self.uncond_prob = uncond_prob

    def token_drop(self, caption: torch.Tensor, force_drop_ids=None) -> torch.Tensor:
        """Drop captions to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption: torch.Tensor, train: bool, force_drop_ids=None) -> torch.Tensor:
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


# ===============================================
# Position Embeddings
# ===============================================


class PositionEmbedding2D(nn.Module):
    """2D sinusoidal position embedding with LRU caching."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "dim must be divisible by 4"
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_sin_cos_emb(self, t: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("i,d->id", t, self.inv_freq)
        emb_cos = torch.cos(out)
        emb_sin = torch.sin(out)
        return torch.cat((emb_sin, emb_cos), dim=-1)

    @functools.lru_cache(maxsize=512)
    def _get_cached_emb(
        self,
        device: torch.device,
        dtype: torch.dtype,
        h: int,
        w: int,
        scale: float = 1.0,
        base_size: Optional[int] = None,
    ) -> torch.Tensor:
        grid_h = torch.arange(h, device=device) / scale
        grid_w = torch.arange(w, device=device) / scale
        if base_size is not None:
            grid_h *= base_size / h
            grid_w *= base_size / w
        grid_h, grid_w = torch.meshgrid(grid_w, grid_h, indexing="ij")
        grid_h = grid_h.t().reshape(-1)
        grid_w = grid_w.t().reshape(-1)
        emb_h = self._get_sin_cos_emb(grid_h)
        emb_w = self._get_sin_cos_emb(grid_w)
        return torch.concat([emb_h, emb_w], dim=-1).unsqueeze(0).to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
        scale: Optional[float] = 1.0,
        base_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self._get_cached_emb(x.device, x.dtype, h, w, scale, base_size)


# ===============================================
# Final Layer
# ===============================================


class T2IFinalLayer(nn.Module):
    """Final prediction layer with AdaLN modulation (PixArt-style).

    Args:
        hidden_size: Model dimension.
        num_patch: Number of values per patch (product of patch spatial dims).
        out_channels: Number of output channels.
        d_t: Default temporal dimension (can be overridden in forward).
        d_s: Default spatial dimension (can be overridden in forward).
    """

    def __init__(self, hidden_size: int, num_patch: int, out_channels: int, d_t: int = None, d_s: int = None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels
        self.d_t = d_t
        self.d_s = d_s

    def t_mask_select(self, x_mask: torch.Tensor, x: torch.Tensor, masked_x: torch.Tensor, T: int, S: int):
        """Select between x and masked_x based on temporal mask."""
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor, x_mask=None, t0=None, T=None, S=None):
        if T is None:
            T = self.d_t
        if S is None:
            S = self.d_s
        is_fifo = t.ndim == 3

        if is_fifo:
            shift, scale = (self.scale_shift_table[None, None] + t[:, :, None]).chunk(2, dim=2)
        else:
            shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)

        x = t2i_modulate(self.norm_final(x), shift, scale, is_fifo=is_fifo)

        if x_mask is not None:
            if is_fifo:
                shift_zero, scale_zero = (self.scale_shift_table[None, None] + t0[:, :, None]).chunk(2, dim=2)
            else:
                shift_zero, scale_zero = (self.scale_shift_table[None] + t0[:, None]).chunk(2, dim=1)
            x_zero = t2i_modulate(self.norm_final(x), shift_zero, scale_zero)
            x = self.t_mask_select(x_mask, x, x_zero, T, S)

        x = self.linear(x)
        return x


# ===============================================
# Sinusoidal Position Embedding Utilities
# ===============================================


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size, cls_token: bool = False, extra_tokens: int = 0, scale: float = 1.0, base_size=None
) -> np.ndarray:
    """Generate 2D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension.
        grid_size: Int or tuple (H, W) for the spatial grid.
        cls_token: If True, prepend zeros for cls/extra tokens.
        extra_tokens: Number of extra tokens to prepend.
        scale: Position scaling factor.
        base_size: Optional base size for normalization.

    Returns:
        Array of shape [grid_size*grid_size, embed_dim] or
        [extra_tokens + grid_size*grid_size, embed_dim].
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / scale
    if base_size is not None:
        grid_h *= base_size / grid_size[0]
        grid_w *= base_size / grid_size[1]
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Generate 2D sincos embeddings from a meshgrid."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed(embed_dim: int, length: int, scale: float = 1.0) -> np.ndarray:
    """Generate 1D sinusoidal positional embeddings."""
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sincos embeddings from position array.

    Args:
        embed_dim: Output dimension (must be even).
        pos: Position array of shape (M,) or (M, 1).

    Returns:
        Array of shape (M, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)
