"""Frame mask generator for conditional video diffusion training."""

import math
import random

import torch


class MaskGenerator:
    """Generates per-frame binary masks for training with partial frame conditioning.

    Supported mask types:
        - mask_no: all frames are noised (no conditioning frames).
        - mask_quarter_head: a random number of leading frames are kept clean.

    Args:
        mask_ratios: dict mapping mask type names to their sampling probabilities (must sum to 1).
        condition_frames_max: maximum number of leading frames to keep as conditioning.
    """

    VALID_MASK_NAMES = ["mask_no", "mask_quarter_head"]

    def __init__(self, mask_ratios, condition_frames_max):
        assert all(
            name in self.VALID_MASK_NAMES for name in mask_ratios.keys()
        ), f"mask_name should be one of {self.VALID_MASK_NAMES}, got {list(mask_ratios.keys())}"
        assert all(
            0 <= ratio <= 1 for ratio in mask_ratios.values()
        ), f"mask_ratio values should be in [0, 1], got {list(mask_ratios.values())}"
        assert math.isclose(
            sum(mask_ratios.values()), 1.0, abs_tol=1e-6
        ), f"sum of mask_ratios should be 1, got {sum(mask_ratios.values())}"

        self.mask_ratios = mask_ratios
        self.condition_frames_max = condition_frames_max

    def get_mask(self, x):
        """Sample a single frame mask for input tensor x (shape: ..., T, ...)."""
        mask_type = random.random()
        mask_name = None
        prob_acc = 0.0
        for mask, mask_ratio in self.mask_ratios.items():
            prob_acc += mask_ratio
            if mask_type < prob_acc:
                mask_name = mask
                break

        num_frames = x.shape[2]
        mask = torch.ones(num_frames, dtype=torch.bool, device=x.device)
        if num_frames <= 1:
            return mask

        if mask_name == "mask_quarter_head":
            random_size = random.randint(1, self.condition_frames_max)
            mask[:random_size] = 0

        return mask

    def get_masks(self, x, num_views=6):
        """Generate masks for a batch where every ``num_views`` consecutive samples share the same mask."""
        masks = []
        batch_size = x.shape[0] // num_views
        assert batch_size * num_views == x.shape[0]
        for _ in range(batch_size):
            mask = self.get_mask(x)
            mask = mask.unsqueeze(0).repeat(num_views, 1)
            masks.append(mask)
        return torch.cat(masks, dim=0)
