#    Copyright (C) 2026 Xiaomi Corporation.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# -*- coding: utf-8 -*-

"""
WorldSplat Inference Pipeline.

Chains the three stages of the WorldSplat framework:
  Stage 1: 4D-Aware Diffusion -- generates multi-modal latents (image, depth, seg)
  Stage 2: GS Decoder -- converts latents to 4D Gaussians, renders novel views
  Stage 3: Enhanced Diffusion -- refines the rendered views

Usage:
    python tools/inference.py \
        --config configs/inference.yaml \
        --output_dir OUTPUT/inference \
        --stage1_ckpt path/to/stage1 \
        --gs_decoder_ckpt path/to/gs_decoder \
        --stage2_ckpt path/to/stage2 \
        --vae_pretrained path/to/vae
"""

import argparse
import os

import torch
from omegaconf import OmegaConf

from worldsplat.utils import instantiate_from_config, to_torch_dtype


def parse_args():
    parser = argparse.ArgumentParser(description="WorldSplat Inference Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Inference config YAML")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--stage1_ckpt", type=str, required=True, help="Stage 1 diffusion checkpoint")
    parser.add_argument("--gs_decoder_ckpt", type=str, required=True, help="GS decoder checkpoint")
    parser.add_argument("--stage2_ckpt", type=str, required=True, help="Stage 2 diffusion checkpoint")
    parser.add_argument("--vae_pretrained", type=str, required=True, help="Path to pretrained VAE")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_sampling_steps", type=int, default=100)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    return parser.parse_args()


@torch.no_grad()
def run_stage1_diffusion(config, model, vae, scheduler, prompt_data, device, dtype):
    """
    Stage 1: 4D-Aware Diffusion.

    Given text prompts and optional conditioning (bounding boxes, layout),
    generate multi-modal latents z = [z_img, z_depth, z_seg].

    Returns:
        z_multimodal: Tensor of shape [B*V, C_img+C_depth+C_seg, T, H, W]
    """
    model.eval()

    model_args = {
        "y": prompt_data["caption_feature"].to(device),
        "mask": prompt_data["attention_mask"].to(device),
        "height": prompt_data["height"].to(device, dtype),
        "width": prompt_data["width"].to(device, dtype),
        "ar": prompt_data["ar"].to(device, dtype),
        "num_frames": prompt_data["num_frames"].to(device, dtype),
        "fps": prompt_data["fps"].to(device, dtype),
    }

    if "data_info" in prompt_data:
        model_args["data_info"] = prompt_data["data_info"]

    z_multimodal = scheduler.sample(
        model,
        z_size=(model.in_channels, 64, 64),
        device=device,
        additional_args=model_args,
        dtype=dtype,
    )

    return z_multimodal


@torch.no_grad()
def run_gs_decoder(config, gs_decoder, z_multimodal, vae, camera_params, device, dtype):
    """
    Stage 2: Gaussian Splatting Decoder.

    Decode multi-modal latents into per-pixel Gaussian parameters,
    then render novel views via differentiable splatting.

    Args:
        z_multimodal: Latent tensor [B*V, C, T, H, W] from Stage 1.
        camera_params: Dict with camera intrinsics/extrinsics for novel views.

    Returns:
        rendered_views: Rendered images at novel viewpoints [B, V_novel, C, H, W].
        render_map: Conditioning signal for Stage 3 [B*V_novel, C, T, H, W].
    """
    gs_decoder.eval()

    # Split multi-modal latents
    C_per_modal = z_multimodal.shape[1] // 3
    z_img = z_multimodal[:, :C_per_modal]
    z_depth = z_multimodal[:, C_per_modal : 2 * C_per_modal]
    z_seg = z_multimodal[:, 2 * C_per_modal :]

    # Decode latents to pixel space
    decoded_img = vae.decode(z_img)
    decoded_depth = vae.decode(z_depth)

    # GS decoder: latents -> Gaussian parameters -> rendered novel views
    gs_output = gs_decoder(
        image=decoded_img,
        depth=decoded_depth,
        camera_params=camera_params,
    )

    rendered_views = gs_output["rendered_views"]
    render_map = gs_output["render_map"]

    return rendered_views, render_map


@torch.no_grad()
def run_stage2_diffusion(config, model, vae, scheduler, render_map, prompt_data, device, dtype):
    """
    Stage 3: Enhanced Diffusion.

    Refine the rendered novel views using the render_map as control condition.

    Args:
        render_map: Conditioning from GS decoder [B*V, C, T, H, W].

    Returns:
        refined_images: Enhanced output images [B*V, 3, H_out, W_out].
    """
    model.eval()

    # Encode render_map as control signal
    c = vae.encode(render_map)

    model_args = {
        "y": prompt_data["caption_feature"].to(device),
        "mask": prompt_data["attention_mask"].to(device),
        "height": prompt_data["height"].to(device, dtype),
        "width": prompt_data["width"].to(device, dtype),
        "ar": prompt_data["ar"].to(device, dtype),
        "num_frames": prompt_data["num_frames"].to(device, dtype),
        "fps": prompt_data["fps"].to(device, dtype),
        "c": c,
    }

    if "data_info" in prompt_data:
        model_args["data_info"] = prompt_data["data_info"]

    z_refined = scheduler.sample(
        model,
        z_size=(model.in_channels, 64, 64),
        device=device,
        additional_args=model_args,
        dtype=dtype,
    )

    refined_images = vae.decode(z_refined)
    return refined_images


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = to_torch_dtype(args.dtype)
    os.makedirs(args.output_dir, exist_ok=True)

    config = OmegaConf.load(args.config)

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    print("Loading VAE...")
    from worldsplat.diffusion.models.videovae import VideoAutoencoderKL
    vae = VideoAutoencoderKL(
        from_pretrained=args.vae_pretrained, micro_batch_size=4, local_files_only=True,
    ).to(device, dtype).eval()

    print("Loading Stage 1 diffusion model...")
    stage1_model = instantiate_from_config(config.stage1_model).to(device, dtype)
    stage1_state = torch.load(args.stage1_ckpt, map_location="cpu")
    stage1_model.load_state_dict(stage1_state, strict=False)
    stage1_model.eval()

    from worldsplat.diffusion.schedulers.iddpm_cfg import IDDPM
    stage1_scheduler = IDDPM(
        num_sampling_steps=args.num_sampling_steps, cfg_scale=args.cfg_scale,
    )

    print("Loading GS Decoder...")
    from worldsplat.gs_decoder import build_gs_decoder
    gs_decoder = build_gs_decoder(config.gs_decoder_model).to(device, dtype)
    gs_state = torch.load(args.gs_decoder_ckpt, map_location="cpu")
    gs_decoder.load_state_dict(gs_state, strict=False)
    gs_decoder.eval()

    print("Loading Stage 2 diffusion model...")
    stage2_model = instantiate_from_config(config.stage2_model).to(device, dtype)
    stage2_state = torch.load(args.stage2_ckpt, map_location="cpu")
    stage2_model.load_state_dict(stage2_state, strict=False)
    stage2_model.eval()

    stage2_scheduler = IDDPM(
        num_sampling_steps=args.num_sampling_steps, cfg_scale=args.cfg_scale,
    )

    # ------------------------------------------------------------------
    # Prepare input data
    # ------------------------------------------------------------------
    print("Preparing input data...")
    input_dataset = instantiate_from_config(config.input_data)
    # prompt_data and camera_params should be prepared from the dataset/config
    # This is application-specific and depends on the driving scenario

    for idx, sample in enumerate(input_dataset):
        prompt_data = sample["prompt_data"]
        camera_params = sample["camera_params"]

        # Stage 1: Generate multi-modal latents
        print(f"[Sample {idx}] Running Stage 1: 4D-Aware Diffusion...")
        z_multimodal = run_stage1_diffusion(
            config, stage1_model, vae, stage1_scheduler, prompt_data, device, dtype,
        )

        # Stage 2: GS Decoder -> novel view rendering
        print(f"[Sample {idx}] Running Stage 2: GS Decoder...")
        rendered_views, render_map = run_gs_decoder(
            config, gs_decoder, z_multimodal, vae, camera_params, device, dtype,
        )

        # Stage 3: Enhanced Diffusion refinement
        print(f"[Sample {idx}] Running Stage 3: Enhanced Diffusion...")
        refined_images = run_stage2_diffusion(
            config, stage2_model, vae, stage2_scheduler, render_map, prompt_data, device, dtype,
        )

        # Save outputs
        save_dir = os.path.join(args.output_dir, f"sample_{idx:04d}")
        os.makedirs(save_dir, exist_ok=True)

        from torchvision.utils import save_image
        save_image(
            refined_images, os.path.join(save_dir, "refined.png"),
            nrow=6, normalize=True, value_range=(-1, 1),
        )
        save_image(
            rendered_views, os.path.join(save_dir, "rendered.png"),
            nrow=6, normalize=True, value_range=(-1, 1),
        )
        print(f"[Sample {idx}] Saved to {save_dir}")

    print("Inference complete.")


if __name__ == "__main__":
    main()
