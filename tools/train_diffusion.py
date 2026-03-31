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
Unified diffusion training script for WorldSplat.

Stage 1 (4D-Aware Diffusion): Encodes image + depth + segmentation via VAE,
    concatenates z_img + z_depth + z_seg as the diffusion target.
Stage 2 (Enhanced Diffusion): Encodes image via VAE, uses render_map from
    the GS decoder as the control condition.

Usage:
    # Stage 1: 4D-aware multi-modal diffusion
    torchrun --nproc_per_node=8 tools/train_diffusion.py \
        --stage 1 --config configs/stage1.yaml --save_dir OUTPUT/stage1

    # Stage 2: render-conditioned enhancement diffusion
    torchrun --nproc_per_node=8 tools/train_diffusion.py \
        --stage 2 --config configs/stage2.yaml --save_dir OUTPUT/stage2
"""

import argparse
import json
import os
import shutil
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

from worldsplat.diffusion.models.videovae import VideoAutoencoderKL
from worldsplat.diffusion.schedulers.rflow import RFLOW
from worldsplat.diffusion.models.mask_generator import MaskGenerator
from worldsplat.utils import (
    all_reduce_mean,
    create_logger,
    format_numel_str,
    get_data_parallel_group,
    get_model_numel,
    instantiate_from_config,
    save_training_state,
    set_data_parallel_group,
    set_grad_checkpoint,
    set_sequence_parallel_group,
    to_torch_dtype,
)
from worldsplat.utils.lr_scheduler import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def batch_to_device(batch, device, dtype):
    """Move all tensor values in a dict to device/dtype."""
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device, dtype)
    return batch


@torch.no_grad()
def stack_batch(batch, stage):
    """Reshape batch from [B, V, ...] to [B*V, ...] for multi-view processing."""
    B, view_num = batch["image"].shape[:2]

    batch["image"] = rearrange(batch["image"], "b v t c ... -> (b v) c t ...")
    batch["height"] = rearrange(batch["height"], "b v ... -> (b v) ...")
    batch["width"] = rearrange(batch["width"], "b v ... -> (b v) ...")
    batch["ar"] = rearrange(batch["ar"], "b v ... -> (b v) ...")
    batch["num_frames"] = rearrange(batch["num_frames"], "b v ... -> (b v) ...")
    batch["fps"] = rearrange(batch["fps"], "b v ... -> (b v) ...")
    batch["hed_edge"] = rearrange(batch["hed_edge"], "b v t c ... -> (b v) c t ...")
    batch["mask"] = rearrange(batch["mask"], "b v ... -> (b v) ...")
    batch["boxes_2d"] = rearrange(batch["boxes_2d"], "b v t ... -> (b v t) ...")
    batch["masks"] = rearrange(batch["masks"], "b v t ... -> (b v t) ...")
    batch["gt_category_2d"] = rearrange(batch["gt_category_2d"], "b v t ... -> (b v t) ...")
    batch["heading"] = rearrange(batch["heading"], "b v t ... -> (b v t) ...")
    batch["instance_id"] = rearrange(batch["instance_id"], "b v t ... -> (b v t) ...")
    batch["positive_embedding"] = rearrange(batch["positive_embedding"], "b v t ... -> (b v t) ...")
    batch["caption_feature"] = rearrange(batch["caption_feature"], "b v t ... -> (b v t) ...")
    batch["attention_mask"] = rearrange(batch["attention_mask"], "b v t ... -> (b v t) ...")

    if stage == 1:
        batch["depth_map"] = rearrange(batch["depth_map"], "b v t c ... -> (b v) c t ...")
        batch["seg_map"] = rearrange(batch["seg_map"], "b v t c ... -> (b v) c t ...")
    elif stage == 2:
        batch["render_map"] = rearrange(batch["render_map"], "b v t c ... -> (b v) c t ...")

    num_frames = batch["image"].shape[2]

    # Reorder string lists to match [B*V*T] layout
    for key in ("caption", "save_name", "filename"):
        if key not in batch:
            continue
        result = []
        for i in range(B):
            for j in range(view_num):
                for k in range(num_frames):
                    result.append(batch[key][k * view_num + j][i])
        batch[key] = result

    return batch


@torch.no_grad()
def get_input_stage1(batch, vae, device, dtype):
    """Stage 1: encode image + depth + seg, concatenate latents."""
    z_img = vae.encode(batch["image"].to(device, dtype))
    z_depth = vae.encode(batch["depth_map"].to(device, dtype))
    z_seg = vae.encode(batch["seg_map"].to(device, dtype))
    z = torch.cat([z_img, z_depth, z_seg], dim=1)

    bg = vae.encode(batch["hed_edge"].to(device, dtype))

    model_args = {
        "y": batch["caption_feature"].to(device),
        "mask": batch["attention_mask"].to(device),
        "height": batch["height"].to(device, dtype),
        "width": batch["width"].to(device, dtype),
        "ar": batch["ar"].to(device, dtype),
        "num_frames": batch["num_frames"].to(device, dtype),
        "fps": batch["fps"].to(device, dtype),
        "c": bg,
        "box_mask": batch["masks"].to(device, torch.int64),
    }
    return z, model_args


@torch.no_grad()
def get_input_stage2(batch, vae, device, dtype):
    """Stage 2: encode image, use render_map as control condition."""
    z = vae.encode(batch["image"].to(device, dtype))
    bg = vae.encode(batch["render_map"].to(device, dtype))

    model_args = {
        "y": batch["caption_feature"].to(device),
        "mask": batch["attention_mask"].to(device),
        "height": batch["height"].to(device, dtype),
        "width": batch["width"].to(device, dtype),
        "ar": batch["ar"].to(device, dtype),
        "num_frames": batch["num_frames"].to(device, dtype),
        "fps": batch["fps"].to(device, dtype),
        "c": bg,
        "box_mask": torch.zeros_like(batch["masks"], dtype=torch.int64, device=device),
    }
    return z, model_args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="WorldSplat Diffusion Training")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                        help="Training stage: 1 = 4D-aware diffusion, 2 = enhanced diffusion")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--plugin", type=str, default="zero2", choices=["zero2", "zero2-seq"])
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--sp_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_bucket_build_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--vae_pretrained", type=str, required=True, help="Path to pretrained VAE")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1500)
    parser.add_argument("--scheduler_type", type=str, default="constant", choices=["constant", "cosine"])
    parser.add_argument("--grad_checkpoint", action="store_true", default=True)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=300)
    parser.add_argument("--max_keep_ckpts", type=int, default=5)
    parser.add_argument("--copy_blocks_num", type=int, default=13)
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to pretrained model checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    # Distributed setup
    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(args.seed)
    coordinator = DistCoordinator()
    device = get_current_device()
    dtype = to_torch_dtype(args.dtype)

    # Output directory
    os.makedirs(args.save_dir, exist_ok=True)
    shutil.copy2(args.config, os.path.join(args.save_dir, "config.yaml"))
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Logger (only rank 0 writes)
    if coordinator.is_master():
        logger = create_logger(args.save_dir)
        logger.info(f"Experiment directory: {args.save_dir}")
        logger.info(f"Training stage: {args.stage}")
    else:
        logger = create_logger(None)

    # ColossalAI plugin
    if args.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2, precision=args.dtype, initial_scale=2**16, max_norm=args.grad_clip,
        )
        set_data_parallel_group(dist.group.WORLD)
    elif args.plugin == "zero2-seq":
        from worldsplat.diffusion.acceleration.plugin import ZeroSeqParallelPlugin
        plugin = ZeroSeqParallelPlugin(
            sp_size=args.sp_size, stage=2, precision=args.dtype,
            initial_scale=2**16, max_norm=args.grad_clip,
        )
        set_sequence_parallel_group(plugin.sp_group)
        set_data_parallel_group(plugin.dp_group)

    booster = Booster(plugin=plugin)

    # Load config
    config = OmegaConf.load(args.config)

    # Dataset
    dataset = instantiate_from_config(config.train_dataset_names)
    logger.info(f"Dataset contains {len(dataset)} samples.")

    from worldsplat.data.opensora_dataloader_var import prepare_dataloader
    dataloader_args = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=args.prefetch_factor,
    )
    bucket_config = config.get("bucket_config", None)
    dataloader, sampler = prepare_dataloader(
        bucket_config=bucket_config,
        num_bucket_build_workers=args.num_bucket_build_workers,
        **dataloader_args,
    )

    # VAE (frozen)
    vae = VideoAutoencoderKL(
        from_pretrained=args.vae_pretrained, micro_batch_size=4, local_files_only=True,
    )
    vae = vae.to(device, dtype)

    # Diffusion model
    model = instantiate_from_config(config.model)
    model = model.to(device, dtype)

    # Wrap with ControlNet
    from worldsplat.diffusion.models.controlnet import ControlSTDiT2Half
    model = ControlSTDiT2Half(
        model, copy_blocks_num=args.copy_blocks_num,
        grounding_tokenizer=config.grounding_tokenizer,
    )

    # Load pretrained weights if provided
    if args.ckpt_path is not None:
        state_dict = torch.load(args.ckpt_path, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint from {args.ckpt_path}")
        if missing_keys:
            logger.info(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            logger.info(f"Unexpected keys: {len(unexpected_keys)}")
        dist.barrier()

    # Freeze / unfreeze parameters based on stage
    if args.stage == 1:
        for name, param in model.named_parameters():
            if "base_model" in name:
                param.requires_grad = True
            elif "controlnet" in name or "position_net" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif args.stage == 2:
        for name, param in model.named_parameters():
            if "base_model" in name and "attn" in name:
                param.requires_grad = True
            elif "controlnet" in name or "position_net" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        f"Trainable params: {format_numel_str(model_numel_trainable)}, "
        f"Total params: {format_numel_str(model_numel)}"
    )

    grounding_tokenizer_input = instantiate_from_config(config.grounding_tokenizer_input)
    model = model.to(device, dtype)

    # Scheduler
    scheduler = RFLOW(use_timestep_transform=True, sample_method="logit-normal")

    # Optimizer
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        adamw_mode=True,
    )

    # LR scheduler
    num_steps_per_epoch = len(dataloader)
    total_iters = args.epochs * num_steps_per_epoch
    if args.scheduler_type == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_iters,
        )
    elif args.scheduler_type == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps,
        )
    else:
        lr_scheduler = None

    # Mask generator (optional)
    mask_generator = None
    if "mask_ratios" in config:
        mask_generator = MaskGenerator(config.mask_ratios, config.condition_frames_max)
        logger.info(f"Mask ratios: {config.mask_ratios}")

    # Gradient checkpointing
    if args.grad_checkpoint:
        set_grad_checkpoint(model)

    # Boost for distributed training
    model.train()
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Model boosted for distributed training")

    # Select input function based on stage
    get_input_fn = get_input_stage1 if args.stage == 1 else get_input_stage2

    # Training loop
    start_epoch = 0
    log_step = 0
    running_loss = 0.0
    logger.info(f"Training for {args.epochs} epochs with {num_steps_per_epoch} steps per epoch")

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info(f"Beginning epoch {epoch}...")

        with tqdm(
            enumerate(dataloader_iter),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            total=num_steps_per_epoch,
        ) as pbar:
            for step, batch in pbar:
                batch = stack_batch(batch, stage=args.stage)
                x_start, model_args = get_input_fn(batch, vae, device, dtype)

                # Grounding input
                grounding_input = grounding_tokenizer_input.prepare(batch)
                batch_to_device(grounding_input, device, dtype)
                model_args["data_info"] = {"box": grounding_input}

                # Mask
                if mask_generator is not None:
                    mask = mask_generator.get_masks(x_start)
                    model_args["x_mask"] = mask
                else:
                    mask = None

                # Forward
                loss_term = scheduler.training_losses(
                    model, x_start, model_kwargs=model_args, mask=mask,
                )
                loss = loss_term["loss"].mean()

                # Backward
                booster.backward(loss=loss, optimizer=optimizer)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()

                # Logging
                all_reduce_mean(loss)
                running_loss += loss.item()
                global_step = epoch * num_steps_per_epoch + step
                log_step += 1

                if coordinator.is_master() and (global_step + 1) % args.log_every == 0:
                    avg_loss = running_loss / log_step
                    pbar.set_postfix({"loss": avg_loss, "step": step, "global_step": global_step})
                    running_loss = 0
                    log_step = 0

                # Checkpointing
                if args.ckpt_every > 0 and (global_step + 1) % args.ckpt_every == 0:
                    save_training_state(
                        booster, model, optimizer, lr_scheduler,
                        epoch, step + 1, global_step + 1, args.batch_size,
                        coordinator, args.save_dir, sampler=sampler,
                    )
                    # Cleanup old checkpoints
                    if coordinator.is_master():
                        all_ckpts = [
                            f for f in os.listdir(args.save_dir) if f.startswith("epoch")
                        ]
                        if len(all_ckpts) > args.max_keep_ckpts:
                            all_ckpts.sort(
                                key=lambda x: os.path.getmtime(os.path.join(args.save_dir, x))
                            )
                            for old in all_ckpts[: len(all_ckpts) - args.max_keep_ckpts]:
                                shutil.rmtree(os.path.join(args.save_dir, old))
                                logger.info(f"Deleted old checkpoint: {old}")
                    dist.barrier()
                    logger.info(
                        f"Saved checkpoint at epoch {epoch} step {step + 1} "
                        f"global_step {global_step + 1}"
                    )

        sampler.reset()


if __name__ == "__main__":
    main()
