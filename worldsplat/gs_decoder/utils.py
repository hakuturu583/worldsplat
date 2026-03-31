"""Utility functions for camera operations, ray computation, and batch processing."""

import math
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Camera helpers (merged from camera.py and ops.py)
# ---------------------------------------------------------------------------

def create_camera_to_world_matrix(
    elevation: float, azimuth: float, cam_dist: float = 1.0
) -> np.ndarray:
    """Create a 4x4 camera-to-world matrix from elevation/azimuth angles."""
    elevation = np.radians(elevation)
    azimuth = np.radians(azimuth)
    x = np.cos(elevation) * np.cos(azimuth) * cam_dist
    y = np.cos(elevation) * np.sin(azimuth) * cam_dist
    z = np.sin(elevation) * cam_dist

    camera_pos = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 0, 1])

    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up)

    cam2world = np.eye(4)
    cam2world[:3, :3] = np.array([right, new_up, -forward]).T
    cam2world[:3, 3] = camera_pos
    return cam2world


def convert_opengl_to_blender(camera_matrix):
    if isinstance(camera_matrix, np.ndarray):
        flip_yz = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        return np.dot(flip_yz, camera_matrix)
    else:
        flip_yz = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        )
        if camera_matrix.ndim == 3:
            flip_yz = flip_yz.unsqueeze(0)
        return torch.matmul(flip_yz.to(camera_matrix), camera_matrix)


def normalize_camera(camera_matrix):
    """Normalize camera location onto a unit sphere."""
    if isinstance(camera_matrix, np.ndarray):
        camera_matrix = camera_matrix.reshape(-1, 4, 4)
        translation = camera_matrix[:, :3, 3]
        translation = translation / (np.linalg.norm(translation, axis=1, keepdims=True) + 1e-8)
        camera_matrix[:, :3, 3] = translation
    else:
        camera_matrix = camera_matrix.reshape(-1, 4, 4)
        translation = camera_matrix[:, :3, 3]
        translation = translation / (torch.norm(translation, dim=1, keepdim=True) + 1e-8)
        camera_matrix[:, :3, 3] = translation
    return camera_matrix


def rescale_intrinsic(camera_intrinsic, src_res, tgt_res=(1.0, 1.0)):
    """Rescale camera intrinsic parameters to fit a target image resolution."""
    src_h, src_w = src_res
    tgt_h, tgt_w = tgt_res
    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    cx = camera_intrinsic[0, 2]
    cy = camera_intrinsic[1, 2]
    scale_h, scale_w = tgt_h / src_h, tgt_w / src_w
    fx_s, fy_s, cx_s, cy_s = fx * scale_w, fy * scale_h, cx * scale_w, cy * scale_h
    intrinsic_scaled = [[fx_s, 0, cx_s], [0, fy_s, cy_s], [0, 0, 1]]
    return intrinsic_scaled, fx_s, fy_s, cx_s, cy_s


# ---------------------------------------------------------------------------
# Gaussian-splatting projection helpers
# ---------------------------------------------------------------------------

def convert_pose(C2W: Tensor) -> Tensor:
    """Flip y/z axes for Gaussian splatting convention."""
    flip_yz = torch.eye(4, device=C2W.device)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    return torch.matmul(C2W, flip_yz)


def get_projection_matrix_gaussian(
    znear: float, zfar: float, fovX: float, fovY: float, device: str = "cuda"
) -> Tensor:
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def get_cam_info_gaussian(
    c2w: Tensor, fovx: Tensor, fovy: Tensor, znear: float, zfar: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute world-to-view, full projection, and camera center for Gaussian rasterization."""
    c2w = convert_pose(c2w)
    world_view_transform = torch.inverse(c2w.float()).transpose(0, 1).cuda().float()
    projection_matrix = (
        get_projection_matrix_gaussian(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy)
        .transpose(0, 1)
        .cuda()
        .float()
    )
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0).float()
    camera_center = world_view_transform.inverse()[3, :3]
    return world_view_transform, full_proj_transform, camera_center


def transform_c2w(c2w: Tensor) -> Tensor:
    """Convert camera-to-world to world-to-camera with y/z flip (for gsplat)."""
    flip_yz = torch.eye(4, device=c2w.device)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    c2w = torch.matmul(c2w, flip_yz)
    w2c = torch.linalg.inv(c2w)
    return w2c


# ---------------------------------------------------------------------------
# Ray computation
# ---------------------------------------------------------------------------

def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
) -> Tensor:
    """Get ray directions for all pixels in camera coordinate."""
    pixel_center = 0.5 if use_pixel_centers else 0

    if isinstance(focal, float):
        fx, fy = focal, focal
        cx, cy = W / 2, H / 2
    else:
        fx, fy = focal
        assert principal is not None
        cx, cy = principal

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )
    directions = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)
    return directions


def get_rays(
    directions: Tensor,
    c2w: Tensor,
    keepdim: bool = False,
    noise_scale: float = 0.0,
    normalize: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Rotate ray directions from camera coordinate to world coordinate."""
    assert directions.shape[-1] == 3

    if directions.ndim == 2:
        if c2w.ndim == 2:
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)
        rays_o = c2w[:, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 3:
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(-1)
            rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(-1)
            rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 4:
        assert c2w.ndim == 3
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(-1)
        rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)

    if noise_scale > 0:
        rays_o = rays_o + torch.randn(3, device=rays_o.device) * noise_scale
        rays_d = rays_d + torch.randn(3, device=rays_d.device) * noise_scale

    if normalize:
        rays_d = F.normalize(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def chunk_batch(func: Callable, chunk_size: int, *args, **kwargs) -> Any:
    """Process a batched function call in chunks to reduce peak memory usage."""
    if chunk_size <= 0:
        return func(*args, **kwargs)

    B = None
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    assert B is not None, "No tensor found in args or kwargs."

    out = defaultdict(list)
    out_type = None
    chunk_length = 0
    for i in range(0, max(1, B), chunk_size):
        out_chunk = func(
            *[arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg for arg in args],
            **{k: arg[i : i + chunk_size] if isinstance(arg, torch.Tensor) else arg for k, arg in kwargs.items()},
        )
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, (tuple, list)):
            chunk_length = len(out_chunk)
            out_chunk = {idx: chunk for idx, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            raise TypeError(f"Unsupported return type: {type(out_chunk)}")
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            out[k].append(v)

    if out_type is None:
        return None

    out_merged: Dict[Any, Optional[torch.Tensor]] = {}
    for k, v in out.items():
        if all(vv is None for vv in v):
            out_merged[k] = None
        elif all(isinstance(vv, torch.Tensor) for vv in v):
            out_merged[k] = torch.cat(v, dim=0)
        else:
            raise TypeError(f"Unsupported types in return value: {[type(vv) for vv in v]}")

    if out_type is torch.Tensor:
        return out_merged[0]
    elif out_type in (tuple, list):
        return out_type([out_merged[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out_merged


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def maybe_resize(tensor: Tensor, tgt_reso, interp_mode: str = "bilinear") -> Tensor:
    """Optionally resize a tensor to target resolution."""
    if isinstance(tgt_reso, list):
        tensor = F.interpolate(
            tensor, size=tgt_reso, mode=interp_mode, antialias=(interp_mode == "bilinear")
        )
    else:
        if tensor.shape[-1] != tgt_reso:
            tensor = F.interpolate(
                tensor, size=(tgt_reso, tgt_reso), mode=interp_mode,
                antialias=(interp_mode == "bilinear"),
            )
    return tensor
