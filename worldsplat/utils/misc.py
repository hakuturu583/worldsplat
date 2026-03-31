import importlib
from typing import Tuple

import torch
import torch.distributed as dist


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor and divide by world size to get the mean."""
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


def format_numel_str(numel: int) -> str:
    """Format a number of elements into a human-readable string (K/M/B)."""
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


def get_model_numel(model: torch.nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params) for a model."""
    num_params = 0
    num_params_trainable = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_trainable += p.numel()
    return num_params, num_params_trainable


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def to_torch_dtype(dtype):
    """Convert a string dtype name to a torch.dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    elif isinstance(dtype, str):
        dtype_mapping = {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "half": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype not in dtype_mapping:
            raise ValueError(f"Unsupported dtype string: {dtype}")
        return dtype_mapping[dtype]
    else:
        raise ValueError(f"Unsupported dtype type: {type(dtype)}")


def instantiate_from_config(config):
    """Instantiate an object from a config dict with 'target' and optional 'params' keys."""
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    """Import and return an object from a fully qualified string path."""
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
