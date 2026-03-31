"""LatentGaussianDecoder: unified Gaussian splatting decoder model.

This module flattens the original 3-level inheritance hierarchy
(OmniGaussian -> LatentGaussian_V2 -> LatentGaussian_Source) into a single class.

Key functionality:
- VAE encoding of input images, depth, and segmentation maps
- Per-pixel Gaussian prediction via a UNet-based decoder
- 4D Gaussian aggregation with static/dynamic decomposition
- Multi-view rendering and loss computation
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .gaussian_renderer import GaussianRenderer
from .losses import LPIPS
from .utils import maybe_resize


# ---------------------------------------------------------------------------
# Simple VAE Encoder (wraps diffusers AutoencoderKL)
# ---------------------------------------------------------------------------

class SimpleAutoencoderKL(nn.Module):
    """Thin wrapper around diffusers AutoencoderKL for encoding images to latent space.

    Args:
        from_pretrained: HuggingFace model name or local path.
        micro_batch_size: If set, encode in micro-batches to save memory.
        cache_dir: Cache directory for the pretrained model.
        local_files_only: Whether to use only local files.
    """

    def __init__(
        self,
        from_pretrained: str = None,
        micro_batch_size: int = None,
        cache_dir: str = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        from diffusers.models import AutoencoderKL

        self.module = AutoencoderKL.from_pretrained(
            from_pretrained, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.out_channels = self.module.config.latent_channels
        self.patch_size = (1, 8, 8)
        self.micro_batch_size = micro_batch_size

    def encode(self, x: Tensor) -> Tensor:
        """Encode multi-view images to latent representations.

        Args:
            x: (B, V, C, H, W) input images.

        Returns:
            (B, V, C_latent, H/8, W/8) latent features.
        """
        B = x.shape[0]
        x = rearrange(x, "B V C H W -> (B V) C H W")

        if self.micro_batch_size is None:
            x = self.module.encode(x).latent_dist.sample().mul_(0.18215)
        else:
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = self.module.encode(x[i : i + bs]).latent_dist.sample().mul_(0.18215)
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)

        x = rearrange(x, "(B V) C H W -> B V C H W", B=B)
        return x

    def decode(self, x: Tensor) -> Tensor:
        """Decode latent representations back to images.

        Args:
            x: (B, V, C_latent, H/8, W/8) latent features.

        Returns:
            (B, V, C, H, W) decoded images.
        """
        B = x.shape[0]
        x = rearrange(x, "B V C H W -> (B V) C H W")

        if self.micro_batch_size is None:
            x = self.module.decode(x / 0.18215).sample
        else:
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = self.module.decode(x[i : i + bs] / 0.18215).sample
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)

        x = rearrange(x, "(B V) C H W -> B V C H W", B=B)
        return x

    def get_latent_size(self, input_size):
        return [
            input_size[i] // self.patch_size[i] if input_size[i] is not None else None
            for i in range(3)
        ]


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class LatentGaussianDecoder(nn.Module):
    """Latent Gaussian Decoder for 4D scene reconstruction.

    Encodes multi-view images into VAE latent space, predicts per-pixel 3D Gaussians,
    separates static/dynamic Gaussians, and renders novel views.

    Args:
        encoder: SimpleAutoencoderKL instance for VAE encoding.
        pixel_gs: LatentDecoder_SD instance for per-pixel Gaussian prediction.
        camera_args: Dict with ``resolution``, ``znear``, ``zfar``, etc.
        loss_args: Dict with loss weights and configuration.
        dataset_params: Dict with ``pc_range`` etc.
        warm_depth_iter: Number of warm-up iterations (depth-only supervision).
        depth_sup_type: Depth supervision type: ``"m3d"``, ``"lidar"``, or ``"fuse"``.
        encode_seg: Whether to encode segmentation maps via VAE.
        encode_depth: Whether to encode depth maps via VAE.
        num_views: Number of camera views per frame.
        single: If True, render each frame independently (no static/dynamic decomposition).
    """

    def __init__(
        self,
        encoder: nn.Module,
        pixel_gs: nn.Module,
        camera_args: dict,
        loss_args: dict,
        dataset_params: dict = None,
        warm_depth_iter: int = 0,
        depth_sup_type: str = "m3d",
        encode_seg: bool = True,
        encode_depth: bool = True,
        num_views: int = 6,
        single: bool = False,
    ):
        super().__init__()

        self.encoder = encoder
        self.pixel_gs = pixel_gs
        self.camera_args = camera_args
        self.loss_args = loss_args
        self.dataset_params = dataset_params
        self.warm_depth_iter = warm_depth_iter
        self.depth_sup_type = depth_sup_type
        self.encode_seg = encode_seg
        self.encode_depth = encode_depth
        self.num_views = num_views
        self.single = single

        self.renderer = GaussianRenderer(self.device, **camera_args)

        # Perceptual loss
        if self.loss_args.get("weight_perceptual", 0) > 0:
            self.perceptual_loss = LPIPS().eval()
        else:
            self.perceptual_loss = None

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def plucker_embedder_ds(
        self,
        rays_o: Tensor,
        rays_d: Tensor,
        downsample_factor: int = 8,
    ) -> Tensor:
        """Compute downsampled Plucker embeddings from ray origins and directions.

        Args:
            rays_o: (B, V, H, W, 3) ray origins.
            rays_d: (B, V, H, W, 3) ray directions.
            downsample_factor: Spatial downsampling factor.

        Returns:
            (B, V, 9, H_ds, W_ds) Plucker embeddings [origins, directions, moments].
        """
        rays_o = rays_o.permute(0, 1, 4, 2, 3)
        rays_d = rays_d.permute(0, 1, 4, 2, 3)

        if downsample_factor > 1:
            rays_o = rays_o[..., ::downsample_factor, ::downsample_factor]
            rays_d = rays_d[..., ::downsample_factor, ::downsample_factor]

        plucker = torch.cat(
            [rays_o, rays_d, torch.cross(rays_o, rays_d, dim=2)], dim=2
        )
        return plucker

    def normalize_seg(self, seg: Tensor) -> Tensor:
        """Normalize segmentation maps for VAE encoding."""
        assert seg.dim() == 4
        norm_seg = rearrange(seg, "b v h w -> b v () h w")
        norm_seg = norm_seg.repeat(1, 1, 3, 1, 1)
        return norm_seg

    def normalize_depth(self, depth: Tensor, near: float = 0.0, far: float = 200.0) -> Tensor:
        """Normalize depth maps to [-1, 1] for VAE encoding."""
        assert depth.dim() == 4
        norm_depth = torch.clamp(depth, min=near, max=far)
        norm_depth = 2 * norm_depth / far - 1.0
        norm_depth = rearrange(norm_depth, "b v h w -> b v () h w")
        norm_depth = norm_depth.repeat(1, 1, 3, 1, 1)
        return norm_depth

    def extract_img_feat(
        self,
        img: Tensor,
        seg: Tensor,
        depth: Tensor,
    ) -> List[Tensor]:
        """Extract VAE-encoded features from images, segmentation, and depth.

        All inputs are encoded with a frozen VAE and concatenated along the channel dim.

        Args:
            img: (B, V, 3, H, W) input images.
            seg: (B, V, H, W) segmentation maps.
            depth: (B, V, H, W) depth maps.

        Returns:
            List containing a single tensor of shape (B, V, C_cat, H/8, W/8).
        """
        with torch.no_grad():
            img_feats = self.encoder.encode(img)

            if self.encode_seg:
                norm_seg = self.normalize_seg(seg)
                seg_feats = self.encoder.encode(norm_seg)
            else:
                seg_feats = torch.zeros_like(img_feats)

            if self.encode_depth:
                norm_depth = self.normalize_depth(depth)
                depth_feats = self.encoder.encode(norm_depth)
            else:
                depth_feats = torch.zeros_like(img_feats)

            feats = torch.cat([img_feats, depth_feats, seg_feats], dim=2)

        return [feats]

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def get_data(self, batch: dict) -> dict:
        """Extract and move batch data to the model device."""
        device_id = self.device
        data_dict = {}

        # Input images
        data_dict["imgs"] = batch["inputs"]["rgb"].to(device_id, dtype=self.dtype)

        # Input rays and camera parameters
        rays_o = batch["inputs_pix"]["rays_o"].to(device_id, dtype=self.dtype)
        rays_d = batch["inputs_pix"]["rays_d"].to(device_id, dtype=self.dtype)
        data_dict["rays_o"] = rays_o
        data_dict["rays_d"] = rays_d
        data_dict["pluckers"] = self.plucker_embedder_ds(rays_o, rays_d)
        data_dict["fxs"] = batch["inputs_pix"]["fx"].to(device_id, dtype=self.dtype)
        data_dict["fys"] = batch["inputs_pix"]["fy"].to(device_id, dtype=self.dtype)
        data_dict["cxs"] = batch["inputs_pix"]["cx"].to(device_id, dtype=self.dtype)
        data_dict["cys"] = batch["inputs_pix"]["cy"].to(device_id, dtype=self.dtype)
        data_dict["c2ws"] = batch["inputs_pix"]["c2w"].to(device_id, dtype=self.dtype)
        data_dict["cks"] = batch["inputs_pix"]["ck"].to(device_id, dtype=self.dtype)
        data_dict["depths"] = batch["inputs_pix"]["depth_m"].to(device_id, dtype=self.dtype)
        data_dict["confs"] = batch["inputs_pix"]["conf_m"].to(device_id, dtype=self.dtype)

        # Output targets (normalize RGB from [-1, 1] to [0, 1])
        data_dict["output_imgs"] = (batch["outputs"]["rgb"].to(device_id, dtype=self.dtype) + 1.0) / 2.0
        data_dict["output_depths"] = batch["outputs"]["depth"].to(device_id, dtype=self.dtype)
        data_dict["output_depths_m"] = batch["outputs"]["depth_m"].to(device_id, dtype=self.dtype)
        data_dict["output_confs_m"] = batch["outputs"]["conf_m"].to(device_id, dtype=self.dtype)
        data_dict["output_positions"] = (
            batch["outputs"]["rays_o"] + batch["outputs"]["rays_d"] * batch["outputs"]["depth_m"].unsqueeze(-1)
        ).to(device_id, dtype=self.dtype)
        data_dict["output_rays_o"] = batch["outputs"]["rays_o"].to(device_id, dtype=self.dtype)
        data_dict["output_rays_d"] = batch["outputs"]["rays_d"].to(device_id, dtype=self.dtype)
        data_dict["output_c2ws"] = batch["outputs"]["c2w"].to(device_id, dtype=self.dtype)

        # Depth references
        data_dict["input_depths_r"] = batch["inputs_pix"]["depth_r"].to(device_id, dtype=self.dtype)
        data_dict["output_depths_r"] = batch["outputs"]["depth_r"].to(device_id, dtype=self.dtype)

        # Segmentation
        data_dict["input_segs_m"] = batch["inputs_pix"]["input_segs_m"].to(device_id, dtype=self.dtype)
        data_dict["output_segs_m"] = batch["outputs"]["output_segs_m"].to(device_id, dtype=self.dtype)

        try:
            data_dict["bin_token"] = batch["bin_token"]
        except (KeyError, TypeError):
            data_dict["bin_token"] = ""

        return data_dict

    # ------------------------------------------------------------------
    # Rendering with static/dynamic decomposition
    # ------------------------------------------------------------------

    def render_each_frame(
        self,
        dynamic_mask: Tensor,
        gaussians_all: Tensor,
        render_c2w: Tensor,
        render_cks: Tensor,
    ) -> dict:
        """Render each frame independently (without static/dynamic splitting).

        Args:
            dynamic_mask: (B, num_views, H, W) per-pixel dynamic mask.
            gaussians_all: (B, F*N, C) all Gaussian parameters.
            render_c2w: (B, F*V, 4, 4) camera-to-world matrices.
            render_cks: (B, F*V, 3, 3) camera intrinsics.

        Returns:
            Dict with rendered images, alpha, and depth.
        """
        bs = dynamic_mask.shape[0]
        frame_num = dynamic_mask.shape[1] // self.num_views

        render_c2w = rearrange(render_c2w, "b (f v) h w -> b f v h w", f=frame_num)
        render_cks = rearrange(render_cks, "b (f v) h w -> b f v h w", f=frame_num)

        render_pkg_list = []
        for b in range(bs):
            gaussians_b = rearrange(gaussians_all[b], "(f n) c -> f n c", f=frame_num)
            frame_results = []
            for f in range(frame_num):
                pkg = self.renderer.render(
                    gaussians=gaussians_b[f].unsqueeze(0),
                    c2w=render_c2w[b][f].unsqueeze(0),
                    cks=render_cks[b][f].unsqueeze(0),
                    rays_o=None, rays_d=None,
                )
                frame_results.append(pkg)

            merged = {k: torch.cat([d[k] for d in frame_results], dim=1) for k in frame_results[0]}
            render_pkg_list.append(merged)

        return {k: torch.cat([d[k] for d in render_pkg_list], dim=0) for k in render_pkg_list[0]}

    def render_with_static_dynamic(
        self,
        dynamic_mask: Tensor,
        gaussians_all: Tensor,
        render_c2w: Tensor,
        render_cks: Tensor,
    ) -> dict:
        """Render with static/dynamic Gaussian decomposition.

        Static Gaussians are shared across all frames. Dynamic Gaussians are
        per-frame and only include those marked as dynamic by the segmentation mask.

        Args:
            dynamic_mask: (B, num_views, H, W) per-pixel dynamic mask.
            gaussians_all: (B, F*N, C) all Gaussian parameters.
            render_c2w: (B, F*V, 4, 4) camera-to-world matrices.
            render_cks: (B, F*V, 3, 3) camera intrinsics.

        Returns:
            Dict with rendered images, alpha, and depth.
        """
        bs = dynamic_mask.shape[0]
        frame_num = dynamic_mask.shape[1] // self.num_views

        render_c2w = rearrange(render_c2w, "b (f v) h w -> b f v h w", f=frame_num)
        render_cks = rearrange(render_cks, "b (f v) h w -> b f v h w", f=frame_num)
        dynamic_mask_flat = rearrange(dynamic_mask, "b v h w -> b (v h w)")

        render_pkg_list = []
        for b in range(bs):
            static_mask = ~dynamic_mask_flat[b].bool()
            gaussians_static = gaussians_all[b][static_mask]

            gaussians_b = rearrange(gaussians_all[b], "(f n) c -> f n c", f=frame_num)
            dyn_mask_b = rearrange(dynamic_mask_flat[b], "(f n) -> f n", f=frame_num).bool()

            frame_results = []
            for f in range(frame_num):
                gaussians_dynamic = gaussians_b[f][dyn_mask_b[f]]
                if gaussians_dynamic.numel() == 0:
                    gs_current = gaussians_static.unsqueeze(0)
                else:
                    gs_current = torch.cat([gaussians_static, gaussians_dynamic], dim=0).unsqueeze(0)

                pkg = self.renderer.render(
                    gaussians=gs_current,
                    c2w=render_c2w[b][f].unsqueeze(0),
                    cks=render_cks[b][f].unsqueeze(0),
                    rays_o=None, rays_d=None,
                )
                frame_results.append(pkg)

            merged = {k: torch.cat([d[k] for d in frame_results], dim=1) for k in frame_results[0]}
            render_pkg_list.append(merged)

        return {k: torch.cat([d[k] for d in render_pkg_list], dim=0) for k in render_pkg_list[0]}

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    @staticmethod
    def bce_loss(pred: Tensor, gt: Tensor, reduction: str = "mean") -> Tensor:
        """Cross-entropy loss for binary segmentation."""
        assert pred.shape[:-1] == gt.shape
        assert pred.shape[-1] == 2
        pred_flat = pred.view(-1, 2)
        gt_flat = gt.view(-1).long()
        return nn.CrossEntropyLoss(reduction=reduction)(pred_flat, gt_flat)

    @staticmethod
    def silog_loss(
        depth_est: Tensor,
        depth_gt: Tensor,
        variance_focus: float = 0.85,
        eps: float = 1e-6,
        conf_m: Tensor = None,
    ) -> Tensor:
        """Scale-invariant logarithmic depth loss."""
        d = torch.log(depth_est + eps) - torch.log(depth_gt + eps)
        if conf_m is not None:
            d = d * conf_m
        return torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: dict,
        split: str = "train",
        iter: int = 0,
        iter_end: int = 100000,
    ) -> Tuple:
        """Forward training/validation pass.

        Returns:
            Tuple of (loss, loss_terms, render_pkg_fuse, render_pkg_pixel,
                       render_pkg_volume, gaussians_all, gaussians_pixel,
                       gaussians_volume, data_dict).
        """
        iter_thresh = iter >= self.warm_depth_iter
        data_dict = self.get_data(batch)
        img = data_dict["imgs"]
        bs = img.shape[0]

        # Encode features
        img_feats = self.extract_img_feat(
            img=img, seg=data_dict["input_segs_m"], depth=data_dict["depths"]
        )

        # Predict per-pixel Gaussians
        gaussians_pixel, pred_depths, pred_segs = self.pixel_gs(
            img_feats[0],
            data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
            data_dict["rays_o"], data_dict["rays_d"],
        )

        dynamic_mask = pred_segs.argmax(-1)
        data_dict["segmask"] = dynamic_mask

        gaussians_volume = None
        gaussians_all = gaussians_pixel

        bs = gaussians_all.shape[0]
        render_c2w = data_dict["output_c2ws"]
        render_cks = data_dict["cks"]

        # -- Loss bookkeeping --
        loss = 0.0
        loss_terms = {}

        def set_loss(key, split_name, loss_value, loss_weight=1.0):
            loss_terms[f"{split_name}/loss_{key}"] = loss_value.item()

        # Segmentation loss
        if self.loss_args.get("weight_seg", 0) > 0:
            seg_loss = self.bce_loss(pred_segs, data_dict["input_segs_m"])
            loss = loss + self.loss_args["weight_seg"] * seg_loss
            set_loss("seg", split, seg_loss, self.loss_args["weight_seg"])

        # Warm-up phase: depth-only losses
        if not iter_thresh:
            render_pkg_fuse, render_pkg_pixel, render_pkg_volume = None, None, None

            if self.loss_args.get("weight_depth_abs", 0) > 0:
                loss = loss + self._compute_depth_loss(
                    pred_depths, data_dict, set_loss, split, is_warmup=True
                )

            return (
                loss, loss_terms, render_pkg_fuse, render_pkg_pixel,
                render_pkg_volume, gaussians_all, gaussians_pixel, gaussians_volume, data_dict,
            )

        # Full rendering phase
        if self.single:
            render_pkg_fuse = self.render_each_frame(
                dynamic_mask=dynamic_mask.detach(),
                gaussians_all=gaussians_all,
                render_c2w=render_c2w,
                render_cks=render_cks,
            )
        else:
            render_pkg_fuse = self.render_with_static_dynamic(
                dynamic_mask=dynamic_mask.detach(),
                gaussians_all=gaussians_all,
                render_c2w=render_c2w,
                render_cks=render_cks,
            )

        render_pkg_pixel, render_pkg_volume = None, None

        # Ground-truth data
        rgb_gt = data_dict["output_imgs"]
        data_dict["rgb_gt"] = rgb_gt
        depth_m_gt = data_dict["output_depths_m"]
        conf_m_gt = data_dict["output_confs_m"]
        data_dict["depth_m_gt"] = depth_m_gt
        data_dict["conf_m_gt"] = conf_m_gt

        # RGB reconstruction loss
        if self.loss_args.get("weight_recon", 0) > 0:
            loss_type = self.loss_args.get("recon_loss_type", "l1")
            if loss_type == "l1":
                rec_loss = torch.abs(rgb_gt - render_pkg_fuse["image"])
            else:
                rec_loss = (rgb_gt - render_pkg_fuse["image"]) ** 2
            loss = loss + rec_loss.mean() * self.loss_args["weight_recon"]
            set_loss("recon", split, rec_loss.mean(), self.loss_args["weight_recon"])

        # Perceptual loss
        if self.loss_args.get("weight_perceptual", 0) > 0:
            resolution = self.camera_args["resolution"]
            p_inp_pred = maybe_resize(
                render_pkg_fuse["image"].reshape(-1, 3, resolution[0], resolution[1]),
                tgt_reso=self.loss_args["perceptual_resolution"],
            )
            p_inp_gt = maybe_resize(
                rgb_gt.reshape(-1, 3, resolution[0], resolution[1]),
                tgt_reso=self.loss_args["perceptual_resolution"],
            )
            p_loss = self.perceptual_loss(p_inp_pred, p_inp_gt)
            p_loss = rearrange(p_loss, "(b v) c h w -> b v c h w", b=bs).mean()
            loss = loss + p_loss * self.loss_args["weight_perceptual"]
            set_loss("perceptual", split, p_loss, self.loss_args["weight_perceptual"])

        # Depth loss on rendered depth
        if self.loss_args.get("weight_depth_abs", 0) > 0:
            rendered_depths = render_pkg_fuse["depth"].squeeze(2)
            loss = loss + self._compute_depth_loss(
                rendered_depths, data_dict, set_loss, split, is_warmup=False
            )

        return (
            loss, loss_terms, render_pkg_fuse, render_pkg_pixel,
            render_pkg_volume, gaussians_all, gaussians_pixel, gaussians_volume, data_dict,
        )

    def _compute_depth_loss(
        self, pred_depths, data_dict, set_loss_fn, split, is_warmup=False
    ) -> Tensor:
        """Compute depth loss based on the configured supervision type."""
        depth_loss = torch.tensor(0.0, device=self.device)
        w = self.loss_args["weight_depth_abs"]

        if self.depth_sup_type == "m3d":
            d_loss = torch.abs(pred_depths - data_dict["depths"])
            d_loss = (d_loss * data_dict["confs"]).mean()
            depth_loss = w * d_loss
            set_loss_fn("depth_abs", split, d_loss, w)

        elif self.depth_sup_type == "lidar":
            mask = data_dict["input_depths_r"] > 0.1
            d_loss = self.silog_loss(pred_depths[mask], data_dict["input_depths_r"][mask])
            depth_loss = w * d_loss
            set_loss_fn("depth_abs", split, d_loss, w)

        elif self.depth_sup_type == "fuse":
            depth_mask = data_dict["input_depths_r"] > 0.1
            d_lidar = self.silog_loss(pred_depths[depth_mask], data_dict["input_depths_r"][depth_mask])
            d_ps = torch.abs(pred_depths - data_dict["depths"])
            d_ps = d_ps * data_dict["confs"]
            d_ps = d_ps.mean()
            d_loss = d_lidar + d_ps
            depth_loss = w * d_loss
            set_loss_fn("depth_abs", split, d_loss, w)

        return depth_loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward_test(self, batch: dict):
        """Forward pass for inference.

        Returns:
            rgb: (B, V, 3, H, W) rendered images.
            depth: (B, V, 1, H, W) rendered depth maps.
        """
        data_dict = self.get_data(batch)
        img = data_dict["imgs"]
        bs = img.shape[0]

        img_feats = self.extract_img_feat(
            img=img, seg=data_dict["input_segs_m"], depth=data_dict["depths"]
        )

        gaussians_pixel, pred_depths, pred_segs = self.pixel_gs(
            img_feats[0],
            data_dict["depths"], data_dict["confs"], data_dict["pluckers"],
            data_dict["rays_o"], data_dict["rays_d"],
        )

        dynamic_mask = pred_segs.argmax(-1)
        gaussians_all = gaussians_pixel

        render_c2w = data_dict["output_c2ws"]
        render_cks = data_dict["cks"]

        if self.single:
            render_pkg = self.render_each_frame(
                dynamic_mask=dynamic_mask.detach(),
                gaussians_all=gaussians_all,
                render_c2w=render_c2w,
                render_cks=render_cks,
            )
        else:
            render_pkg = self.render_with_static_dynamic(
                dynamic_mask=dynamic_mask.detach(),
                gaussians_all=gaussians_all,
                render_c2w=render_c2w,
                render_cks=render_cks,
            )

        return render_pkg["image"].cpu(), render_pkg["depth"].cpu()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict, val_result_savedir: str, iter: int = 0):
        """Run validation and save visual results."""
        (loss_val, loss_terms, render_pkg_fuse, _, _, gaussians_all,
         gaussians_pixel, _, batch_data) = self.forward(batch, "val", iter=iter)
        if render_pkg_fuse is not None:
            self.save_val_results(
                batch_data, render_pkg_fuse, gaussians_all, val_result_savedir
            )
        return loss_terms

    def save_val_results(
        self,
        batch_gt: dict,
        render_pkg_fuse: dict,
        gaussians_all: Tensor,
        save_dir: str,
    ):
        """Save validation visualizations to disk."""
        os.makedirs(save_dir, exist_ok=True)
        batch_size = render_pkg_fuse["image"].shape[0]
        n_views = render_pkg_fuse["image"].shape[1]

        rgbs_gt = batch_gt["output_imgs"].cpu()
        depths_m_gt = batch_gt["output_depths_m"]
        depths_m_gt = (depths_m_gt / 255.0).unsqueeze(2).repeat(1, 1, 3, 1, 1).cpu()
        seg_gt = batch_gt["output_segs_m"].unsqueeze(2).repeat(1, 1, 3, 1, 1).cpu()
        seg_pred = batch_gt["segmask"].unsqueeze(2).repeat(1, 1, 3, 1, 1).cpu()

        for i in range(batch_size):
            sample_dir = os.path.join(save_dir, f"sample-{i}")
            os.makedirs(sample_dir, exist_ok=True)

            for v in range(n_views):
                rgb = render_pkg_fuse["image"][i, v].cpu()
                depth = render_pkg_fuse["depth"][i, v]
                depth_vis = depth.repeat(3, 1, 1).cpu() / 255.0

                cat_gt = torch.cat([rgbs_gt[i, v], depths_m_gt[i, v], seg_gt[i, v]], dim=-1)
                cat_pred = torch.cat([rgb, depth_vis, seg_pred[i, v]], dim=-1)
                grid = torch.cat([cat_gt, cat_pred], dim=1)
                grid = (grid.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
                imageio.imwrite(os.path.join(sample_dir, f"{v}.png"), grid)

            # Save Gaussians as PLY
            gs_reformat = torch.cat([
                gaussians_all[i : i + 1, :, 0:3],
                gaussians_all[i : i + 1, :, 6:7],
                gaussians_all[i : i + 1, :, 11:14],
                gaussians_all[i : i + 1, :, 7:11],
                gaussians_all[i : i + 1, :, 3:6],
            ], dim=-1)
            self.renderer.save_ply(gs_reformat, os.path.join(sample_dir, f"sample-{i}.ply"))

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self, lr: float):
        """Configure optimizer with frozen encoder parameters.

        Args:
            lr: Base learning rate.

        Returns:
            List with a single AdamW optimizer.
        """
        encoder_params = set(id(p) for p in self.encoder.parameters())
        base_params = [p for p in self.parameters() if id(p) not in encoder_params]
        opt = torch.optim.AdamW(
            [{"params": base_params}],
            lr=lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8,
        )
        return [opt]
