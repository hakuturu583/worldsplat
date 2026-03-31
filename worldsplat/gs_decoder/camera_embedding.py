"""Camera embedding utilities: Plucker ray computation and camera pose recovery."""

import torch
from einops import rearrange, repeat
from torch import nn


def meshgrid(spatial_shape, normalized=True, indexing="ij", device=None):
    """Create evenly spaced position coordinates for the given spatial shape.

    Returns:
        Tensor of shape (*spatial_shape, len(spatial_shape)).
    """
    if normalized:
        axis_coords = [torch.linspace(-1.0, 1.0, steps=s, device=device) for s in spatial_shape]
    else:
        axis_coords = [torch.linspace(0, s - 1, steps=s, device=device) for s in spatial_shape]
    grid_coords = torch.meshgrid(*axis_coords, indexing=indexing)
    return torch.stack(grid_coords, dim=-1)


def get_plucker_rays(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    h: int = 32,
    w: int = 32,
    stride: int = 8,
    is_diffusion: bool = False,
) -> torch.Tensor:
    """Compute Plucker ray embeddings from camera extrinsics and intrinsics.

    Args:
        extrinsics: (B, V, 4, 4) camera-to-world matrices.
        intrinsics: (B, 3, 3) or (B, V, 3, 3) camera intrinsics.
        h, w: Output spatial dimensions.
        stride: Downsampling stride applied to intrinsics.
        is_diffusion: If True, omit camera origins from output.

    Returns:
        Tensor of shape (B, V, C, h, w) with C=9 (origins+directions+moments)
        or C=6 (directions+moments) when is_diffusion=True.
    """
    b, v = extrinsics.shape[:2]

    updated_intrinsics = intrinsics.clone().unsqueeze(1) if len(intrinsics.shape) == 3 else intrinsics.clone()
    updated_intrinsics[..., 0, 0] *= 1 / stride
    updated_intrinsics[..., 0, 2] *= 1 / stride
    updated_intrinsics[..., 1, 1] *= 1 / stride
    updated_intrinsics[..., 1, 2] *= 1 / stride

    pixel_coords = meshgrid((w, h), normalized=False, indexing="xy", device=extrinsics.device)
    ones = torch.ones((h, w, 1), device=extrinsics.device)
    pixel_coords = torch.cat([pixel_coords, ones], dim=-1)
    pixel_coords = rearrange(pixel_coords, "h w c -> c (h w)")
    pixel_coords = repeat(pixel_coords, "... -> b v ...", b=b, v=v)

    rots, trans = extrinsics.split([3, 1], dim=-1)

    directions = rots @ updated_intrinsics.inverse() @ pixel_coords
    directions = directions / directions.norm(dim=2, keepdim=True)
    directions = rearrange(directions, "b v c (h w) -> b v c h w", h=h, w=w)
    cam_origins = repeat(trans.squeeze(-1), "b v c -> b v c h w", h=h, w=w)
    moments = torch.cross(cam_origins, directions, dim=2)

    output = [] if is_diffusion else [cam_origins]
    output.append(directions)
    output.append(moments)

    return torch.cat(output, dim=2)


def intersect_skew_lines_high_dim(p: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Find the intersection of skew lines in high dimensions.

    Reference: https://en.wikipedia.org/wiki/Skew_lines
    """
    dim = p.shape[-1]
    eye = torch.eye(dim, device=p.device, dtype=p.dtype)[None, None, None]
    I_min_cov = eye - (r[..., None] * r[..., None, :])
    sum_proj = I_min_cov.matmul(p[..., None]).sum(dim=-3)
    p_intersect = torch.linalg.lstsq(I_min_cov.sum(dim=-3), sum_proj).solution[..., 0]
    return p_intersect


def compute_optimal_rotation_alignment(
    A: torch.Tensor, B: torch.Tensor
) -> torch.Tensor:
    """Compute optimal rotation R that minimizes ||A - B @ R||_F.

    Args:
        A: (3, H, W)
        B: (3, H, W)

    Returns:
        R: (3, 3)
    """
    A = rearrange(A, "c h w -> (h w) c")
    B = rearrange(B, "c h w -> (h w) c")

    H = B.T @ A
    U, _, Vh = torch.linalg.svd(H, full_matrices=True)
    s = torch.linalg.det(U @ Vh)
    S_prime = torch.diag(torch.tensor([1, 1, torch.sign(s)], device=A.device))
    return U @ S_prime @ Vh
