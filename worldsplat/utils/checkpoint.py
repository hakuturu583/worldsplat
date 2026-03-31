import json
import logging
import os
from collections.abc import Iterable
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.booster import Booster
from colossalai.checkpoint_io import GeneralCheckpointIO
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    Only rank-0 produces real output; other ranks get a silent logger.
    """
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------

def set_grad_checkpoint(model, use_fp32_attention=False, gc_step=1):
    """Enable gradient checkpointing for all modules in the model."""
    assert isinstance(model, nn.Module)

    def set_attr(module):
        module.grad_checkpointing = True
        module.fp32_attention = use_fp32_attention
        module.grad_checkpointing_step = gc_step

    model.apply(set_attr)


def auto_grad_checkpoint(module, *args, **kwargs):
    """Automatically apply gradient checkpointing if enabled on the module."""
    if getattr(module, "grad_checkpointing", False):
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module, gc_step, *args, **kwargs)
    return module(*args, **kwargs)


# ---------------------------------------------------------------------------
# Checkpoint loading / saving
# ---------------------------------------------------------------------------

def load_checkpoint(model, ckpt_path, save_as_pt=False):
    """Load model weights from a sharded ColossalAI checkpoint directory."""
    model_dir = os.path.join(ckpt_path, "model")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {model_dir}")

    ckpt_io = GeneralCheckpointIO()
    ckpt_io.load_model(model, model_dir)
    logging.info(f"Successfully loaded model from {model_dir}")

    if save_as_pt:
        save_path = os.path.join(ckpt_path, "model_ckpt.pt")
        torch.save(model.state_dict(), save_path)
        logging.info(f"Model checkpoint saved to {save_path}")


def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def load_training_state(
    booster: Booster,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    load_dir: str,
    sampler=None,
) -> Tuple[int, int, int]:
    """Load full training state (model, optimizer, lr_scheduler, running states)."""
    booster.load_model(model, os.path.join(load_dir, "model"))
    booster.load_optimizer(optimizer, os.path.join(load_dir, "optimizer"))
    if lr_scheduler is not None:
        booster.load_lr_scheduler(lr_scheduler, os.path.join(load_dir, "lr_scheduler"))
    running_states = load_json(os.path.join(load_dir, "running_states.json"))
    if sampler is not None:
        sampler.load_state_dict(torch.load(os.path.join(load_dir, "sampler")))
    dist.barrier()
    return (
        running_states["epoch"],
        running_states["step"],
        running_states["sample_start_index"],
    )


def save_training_state(
    booster: Booster,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    epoch: int,
    step: int,
    global_step: int,
    batch_size: int,
    coordinator,
    save_dir: str,
    sampler=None,
):
    """Save full training state to a checkpoint directory."""
    save_dir = os.path.join(save_dir, f"epoch{epoch}-global_step{global_step}")
    os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)

    booster.save_model(model, os.path.join(save_dir, "model"), shard=True)
    booster.save_optimizer(
        optimizer, os.path.join(save_dir, "optimizer"), shard=True, size_per_shard=4096
    )
    if lr_scheduler is not None:
        booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))

    sampler_start_idx = step * batch_size if batch_size is not None else None
    running_states = {
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "sample_start_index": sampler_start_idx,
    }
    if coordinator.is_master():
        save_json(running_states, os.path.join(save_dir, "running_states.json"))
        if sampler is not None:
            torch.save(sampler.state_dict(step), os.path.join(save_dir, "sampler"))
    dist.barrier()
