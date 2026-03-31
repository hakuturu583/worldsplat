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
Training script for the Gaussian Splatting Decoder.

Converts multi-modal latents from the diffusion model into 4D Gaussians
and optimizes via differentiable rendering losses.

Usage:
    accelerate launch tools/train_gs_decoder.py \
        --config configs/gs_decoder.py --work_dir OUTPUT/gs_decoder
"""

import argparse
import inspect
import logging
import math
import os
import os.path as osp
import time

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, ProjectConfiguration, set_seed
from datetime import timedelta
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_logger(log_file=None, is_main_process=False, log_level=logging.INFO):
    """Create a logger for the main process only."""
    if not is_main_process:
        return None
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s  %(levelname)5s  %(message)s")
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def create_dataset(dataset_class, config_dict, split):
    """Instantiate a dataset class, filtering config keys to match __init__ signature."""
    valid_args = inspect.signature(dataset_class.__init__).parameters
    valid_args = {k for k in valid_args if k != "self"}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_args}
    return dataset_class(**filtered_config, split=split)


def load_config(path):
    """Load a Python config file (mmengine-style or plain dict)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Try to extract a Config-like object or fall back to module attributes
    if hasattr(mod, "Config"):
        return mod.Config
    # Build a simple namespace from module-level variables
    from types import SimpleNamespace
    cfg_dict = {k: v for k, v in vars(mod).items() if not k.startswith("_")}
    return SimpleNamespace(**cfg_dict)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="WorldSplat GS Decoder Training")
    parser.add_argument("--config", type=str, required=True, help="Path to Python config file")
    parser.add_argument("--work_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--resume_from", type=str, default="", help="Checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    try:
        from mmengine.config import Config
        cfg = Config.fromfile(args.config)
    except ImportError:
        cfg = load_config(args.config)

    cfg.work_dir = args.work_dir

    # Accelerator setup
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.work_dir,
        logging_dir=os.path.join(cfg.work_dir, "logs"),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="worldsplat-gs-decoder",
            init_kwargs={"wandb": {"name": cfg.exp_name}},
        )

    if cfg.seed is not None:
        set_seed(cfg.seed + accelerator.local_process_index)

    # Logger
    if accelerator.is_main_process:
        os.makedirs(args.work_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(args.work_dir, f"{timestamp}.log")
    os.makedirs(osp.dirname(log_file), exist_ok=True)
    logger = create_logger(
        log_file=log_file, is_main_process=accelerator.is_main_process,
    )
    if logger is not None:
        logger.info(f"Config: {cfg}")

    # Build model
    from worldsplat.gs_decoder import build_gs_decoder
    model = build_gs_decoder(cfg.model).to(accelerator.device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if logger is not None:
        logger.info(f"Number of trainable params: {n_parameters}")

    # Optimizer
    optimizers = model.configure_optimizers(cfg.lr)
    optimizer = optimizers[0]

    # LR scheduler: warmup + cosine annealing
    warm_up = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        1 / (cfg.warmup_steps * accelerator.num_processes),
        1,
        total_iters=cfg.warmup_steps * accelerator.num_processes,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.max_train_steps * accelerator.num_processes,
        eta_min=cfg.lr * 0.1,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warm_up, cosine_scheduler],
        milestones=[cfg.warmup_steps * accelerator.num_processes],
    )

    # Datasets
    dataset_config = cfg.dataset_params
    import worldsplat.data.gs_decoder_dataset as datasets
    dataset_cls = getattr(datasets, dataset_config.dataset_name)
    train_dataset = create_dataset(dataset_cls, dataset_config, split="train")
    val_dataset = create_dataset(dataset_cls, dataset_config, split="val")

    train_dataloader = DataLoader(
        train_dataset, dataset_config.batch_size_train, shuffle=True,
        num_workers=dataset_config.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset, dataset_config.batch_size_val, shuffle=False,
        num_workers=dataset_config.num_workers_val,
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.gradient_accumulation_steps
    )

    # Resume from checkpoint
    epoch = 0
    global_iter = 0
    first_epoch = 0
    resume_step = -1

    resume_path = args.resume_from or getattr(cfg, "resume_from", "")
    if resume_path:
        if resume_path == "latest":
            dirs = [d for d in os.listdir(cfg.work_dir) if d.startswith("checkpoint")]
            if dirs:
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                resume_path = dirs[-1]
            else:
                resume_path = None
        else:
            resume_path = os.path.basename(resume_path)

        if resume_path:
            accelerator.load_state(
                osp.join(cfg.work_dir, resume_path), map_location="cpu", strict=False,
            )
            global_iter = int(resume_path.split("-")[1])
            first_epoch = global_iter // num_update_steps_per_epoch
            resume_step = global_iter % num_update_steps_per_epoch
            if logger is not None:
                logger.info(f"Resumed from epoch {first_epoch}, iter {global_iter}")

    # Training
    max_num_epochs = cfg.max_epochs
    print_freq = cfg.print_freq

    while epoch < max_num_epochs:
        model.train()
        data_time_s = time.time()
        time_s = time.time()

        for i_iter, batch in enumerate(train_dataloader):
            data_time_e = time.time()

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss, log, _, _, _, _, _, _, _ = model.module.forward(
                    batch, "train", iter=global_iter, iter_end=cfg.max_train_steps,
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), cfg.grad_max_norm,
                    )
                optimizer.step()
                scheduler.step()

            accelerator.wait_for_everyone()

            if accelerator.sync_gradients and accelerator.is_main_process:
                # Save checkpoint
                if global_iter % cfg.save_freq == 0:
                    save_file_name = os.path.join(
                        os.path.abspath(args.work_dir), f"checkpoint-{global_iter}",
                    )
                    accelerator.save_state(save_file_name)
                    if logger is not None:
                        logger.info(f"Saved checkpoint to {save_file_name}")

                # Validation
                if global_iter % cfg.val_freq == 0:
                    model.eval()
                    for i_iter_val, batch_val in enumerate(val_dataloader):
                        val_save_dir = osp.join(
                            cfg.work_dir, "validation",
                            f"step-{global_iter}/batch-{i_iter_val}",
                        )
                        log_val = model.module.validation_step(
                            batch_val, val_save_dir, iter=global_iter,
                        )
                        log.update(log_val)
                    model.train()

            time_e = time.time()

            # Print loss
            if global_iter % print_freq == 0 and accelerator.is_main_process:
                lr = optimizer.param_groups[0]["lr"]
                losses_str = ", ".join(f"{k}: {v:.3f}" for k, v in log.items())
                if logger is not None:
                    logger.info(
                        f"[TRAIN] Epoch {epoch} Iter {i_iter}/{len(train_dataloader)}: "
                        f"Loss: {loss.item():.3f}, {losses_str}, "
                        f"grad_norm: {grad_norm:.1f}, lr: {lr:.7f}, "
                        f"time: {time_e - time_s:.3f} ({data_time_e - data_time_s:.3f})"
                    )

            global_iter += 1
            accelerator.log(log, step=global_iter)
            data_time_s = time.time()
            time_s = time.time()

        epoch += 1

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
