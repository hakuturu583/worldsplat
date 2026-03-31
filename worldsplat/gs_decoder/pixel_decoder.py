"""Pixel-aligned Gaussian decoder: predicts per-pixel Gaussian splat parameters."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LatentDecoder_SD(nn.Module):
    """Pixel-level Gaussian decoder.

    Takes image features (latent + depth + plucker embeddings) and predicts
    per-pixel 3D Gaussian parameters: position, RGB, opacity, rotation, scale,
    and a segmentation head for dynamic/static separation.

    Args:
        in_embed_dim: Dimension of the Plucker-to-embedding linear layer.
        num_cams: Number of camera views.
        near: Near depth bound.
        far: Far depth bound.
        use_checkpoint: Whether to use gradient checkpointing.
        decoder: GSDecoder module instance.
        use_real_depth: If True, predict depth as input_depth + residual.
        use_pluk: If True, concatenate Plucker embeddings to input features.
    """

    def __init__(
        self,
        in_embed_dim: int = 3,
        num_cams: int = 6,
        near: float = 0.1,
        far: float = 200.0,
        use_checkpoint: bool = False,
        decoder: nn.Module = None,
        use_real_depth: bool = False,
        use_pluk: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.plucker_to_embed = nn.Linear(6, in_embed_dim)
        self.cams_embeds = nn.Parameter(torch.Tensor(num_cams, in_embed_dim))

        self.num_cams = num_cams
        self.near = near
        self.far = far

        self.decoder = decoder
        # Gaussian channels: rgb(3) + disp(1) + opacity(1) + scale(3) + rotation(4) + xyz_offset(3) + seg(2)
        self.gs_channels = 17
        self.opt_act = torch.sigmoid
        self.scale_act = lambda x: torch.exp(x) * 0.01
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = torch.sigmoid
        self.use_real_depth = use_real_depth
        self.use_pluk = use_pluk

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(
        self,
        img_feats: torch.Tensor,
        depths_in: torch.Tensor,
        confs_in: torch.Tensor,
        pluckers: torch.Tensor,
        origins: torch.Tensor,
        directions: torch.Tensor,
        status: str = "train",
    ):
        """Predict per-pixel Gaussian parameters.

        Args:
            img_feats: (B, V, C, H, W) image features.
            depths_in: (B, V, H, W) input depth maps.
            confs_in: (B, V, H, W) depth confidence maps.
            pluckers: (B, V, C_pluk, H, W) Plucker embeddings.
            origins: (B, V, H, W, 3) ray origins.
            directions: (B, V, H, W, 3) ray directions.
            status: "train" or "test".

        Returns:
            gaussians: (B, N, 14) Gaussian parameters [means, rgb, opacity, rotation, scale].
            depths: (B, V, H, W) predicted depth maps.
            segs: (B, V, H, W, 2) segmentation logits (static/dynamic).
        """
        if self.use_pluk:
            img_feats = torch.cat([img_feats, pluckers], dim=2)

        bs = origins.shape[0]
        v = img_feats.shape[1]

        depths_in = rearrange(depths_in, "b v h w -> (b v) () h w")
        confs_in = rearrange(confs_in, "b v h w -> (b v) () h w")

        gaussians = self.decoder(img_feats.detach())

        _, _, h, w = gaussians.shape
        gaussians = rearrange(
            gaussians, "(b v) (n c) h w -> b (v h w n) c",
            b=bs, v=v, n=1, c=self.gs_channels,
        )

        rgbs, disp, opacity, scales, rotations, xyz_offset, segs = gaussians.split(
            (3, 1, 1, 3, 4, 3, 2), dim=-1
        )
        opacities = self.opt_act(opacity)
        scales = self.scale_act(scales)
        rotations = self.rot_act(rotations)
        rgbs = self.rgb_act(rgbs)

        depths_in = rearrange(depths_in, "(b v) c h w -> b (v h w) c", b=bs, v=v)
        if self.use_real_depth:
            depths = torch.clamp(depths_in + disp, min=self.near)
        else:
            depths = 1.0 / (
                disp.sigmoid() * (1.0 / self.near - 1.0 / self.far) + 1.0 / self.far
            )

        origins = rearrange(origins, "b v h w c -> b (v h w) c")
        directions = rearrange(directions, "b v h w c -> b (v h w) c")

        means = origins + directions * depths + xyz_offset

        depths = depths.squeeze(2)
        depths = rearrange(depths, "b (v h w) -> b v h w", b=bs, v=v, h=h, w=w)
        segs = rearrange(segs, "b (v h w) c -> b v h w c", b=bs, v=v, h=h, w=w)

        gaussians = torch.cat([means, rgbs, opacities, rotations, scales], dim=-1)

        return gaussians, depths, segs
