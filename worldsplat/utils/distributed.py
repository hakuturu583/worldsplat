import torch.distributed as dist

# ---------------------------------------------------------------------------
# Global parallel group registry
# ---------------------------------------------------------------------------

_GLOBAL_PARALLEL_GROUPS = dict()


def set_data_parallel_group(group: dist.ProcessGroup):
    _GLOBAL_PARALLEL_GROUPS["data"] = group


def get_data_parallel_group():
    return _GLOBAL_PARALLEL_GROUPS.get("data", None)


def set_sequence_parallel_group(group: dist.ProcessGroup):
    _GLOBAL_PARALLEL_GROUPS["sequence"] = group


def get_sequence_parallel_group():
    return _GLOBAL_PARALLEL_GROUPS.get("sequence", None)


# ---------------------------------------------------------------------------
# Synchronization helpers
# ---------------------------------------------------------------------------

def synchronize():
    """Barrier that is safe to call even when distributed is not initialized."""
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
