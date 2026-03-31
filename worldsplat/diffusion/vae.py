"""VideoAutoencoderKL: a wrapper around diffusers AutoencoderKL for video (5-D) tensors."""

import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL
from einops import rearrange


class VideoAutoencoderKL(nn.Module):
    """Wraps a 2-D AutoencoderKL to handle 5-D video tensors (B, C, T, H, W).

    Each frame is encoded/decoded independently.  An optional micro_batch_size
    controls memory usage by processing frames in chunks.
    """

    def __init__(self, from_pretrained=None, micro_batch_size=None, cache_dir=None, local_files_only=False):
        super().__init__()
        self.module = AutoencoderKL.from_pretrained(
            from_pretrained, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.out_channels = self.module.config.latent_channels
        self.patch_size = (1, 8, 8)
        self.micro_batch_size = micro_batch_size

    def encode(self, x):
        """Encode video tensor (B, C, T, H, W) to latent space."""
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")

        if self.micro_batch_size is None:
            x = self.module.encode(x).latent_dist.sample().mul_(0.18215)
        else:
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = self.module.encode(x[i : i + bs]).latent_dist.sample().mul_(0.18215)
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)

        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def decode(self, x):
        """Decode latent tensor (B, C, T, H, W) back to pixel space."""
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")

        if self.micro_batch_size is None:
            x = self.module.decode(x / 0.18215).sample
        else:
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = self.module.decode(x[i : i + bs] / 0.18215).sample
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)

        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def get_latent_size(self, input_size):
        """Compute latent spatial dimensions from input dimensions."""
        latent_size = []
        for i in range(3):
            latent_size.append(input_size[i] // self.patch_size[i] if input_size[i] is not None else None)
        return latent_size
