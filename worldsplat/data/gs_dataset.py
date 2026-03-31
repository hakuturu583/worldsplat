"""NuScenes dataset for Gaussian Splatting decoder training.

Loads multi-view camera sequences with depth, confidence, segmentation,
and camera parameters suitable for feed-forward 3D Gaussian reconstruction.
"""

import copy
import json
import os
import os.path as osp
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


# ---------------------------------------------------------------------------
# Camera geometry helpers
# ---------------------------------------------------------------------------

def _get_transform(translation, quaternion, quat=False, inverse=False):
    """Build a 4x4 SE(3) transform from translation and rotation."""
    rot = quaternion if quat else R.from_quat(quaternion).as_matrix()
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = translation
    if inverse:
        inv = np.eye(4)
        inv[:3, :3] = mat[:3, :3].T
        inv[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
        return inv
    return mat


def _resize_intrinsics(cks, tar_size, org_size=(900, 1600)):
    """Rescale camera intrinsics for a new image resolution."""
    H_org, W_org = org_size
    H_tar, W_tar = tar_size
    cks_new = cks.copy()
    cks_new[0, 0] *= W_tar / W_org
    cks_new[0, 2] *= W_tar / W_org
    cks_new[1, 1] *= H_tar / H_org
    cks_new[1, 2] *= H_tar / H_org
    return cks_new


def _sensor2keysensor(s2e, e2g, dx=0.0, dy=0.0, dz=0.0):
    """Compute sensor-to-key-sensor transforms with optional ego translation."""
    assert s2e.shape == e2g.shape and s2e.shape[-2:] == (4, 4)
    N = s2e.shape[0]
    device = s2e.device

    s2g = e2g @ s2e
    translation = torch.eye(4, device=device, dtype=s2e.dtype)
    translation[0, 3], translation[1, 3], translation[2, 3] = dx, dy, dz

    e2g0_shifted = translation @ e2g[0]
    key_s2g = e2g0_shifted @ s2e[0]
    g2key = torch.inverse(key_s2g).unsqueeze(0).expand(N, 4, 4)
    return g2key @ s2g


def _sensor2keyego(s2e, e2g, dx=0.0, dy=0.0, dz=0.0, to_opengl=False):
    """Compute sensor-to-key-ego transforms with optional ego translation."""
    assert s2e.shape == e2g.shape and s2e.shape[-2:] == (4, 4)
    N = s2e.shape[0]
    device = s2e.device

    s2g = e2g @ s2e
    translation = torch.eye(4, device=device, dtype=s2e.dtype)
    translation[0, 3], translation[1, 3], translation[2, 3] = dx, dy, dz

    e2g0_shifted = translation @ e2g[0]
    keyego_inv = torch.inverse(e2g0_shifted).unsqueeze(0).expand(N, 4, 4)
    s2k = keyego_inv @ s2g

    if to_opengl:
        flip = torch.diag(torch.tensor([-1, -1, -1, 1], dtype=s2k.dtype, device=device))
        s2k = s2k @ flip
    return s2k


def _sensor2keylidar(s2e, e2g, s2l):
    """Compute sensor-to-key-lidar transforms."""
    assert s2e.shape == e2g.shape == s2l.shape and s2e.shape[-2:] == (4, 4)
    N = s2e.shape[0]
    s2g = e2g @ s2e
    keylidar_global = e2g[0] @ s2e[0] @ torch.inverse(s2l[0])
    keylidar_inv = torch.inverse(keylidar_global).unsqueeze(0).expand(N, 4, 4)
    s2k = keylidar_inv @ s2g
    flip = torch.diag(torch.tensor([1, -1, -1, 1], dtype=s2k.dtype, device=s2k.device))
    return s2k @ flip


# ---------------------------------------------------------------------------
# Ray helpers
# ---------------------------------------------------------------------------

def _get_ray_directions(H, W, focal, principal=None):
    """Generate pixel ray directions for a pinhole camera.

    Args:
        focal: (fx, fy) or scalar.
        principal: (cx, cy) or None (defaults to image center).

    Returns:
        Tensor of shape (H, W, 3).
    """
    if isinstance(focal, (list, tuple)):
        fx, fy = focal
    else:
        fx = fy = focal
    if principal is None:
        cx, cy = W / 2.0, H / 2.0
    else:
        cx, cy = principal

    j, i = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    directions = torch.stack(
        [(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], dim=-1
    )
    return directions


def _get_rays(directions, c2w, normalize=False):
    """Compute ray origins and directions in world space.

    Args:
        directions: (N, H, W, 3) or (H, W, 3).
        c2w: (..., 4, 4).

    Returns:
        rays_o, rays_d each of shape matching directions.
    """
    rays_d = torch.einsum("...ij,...hwj->...hwi", c2w[:, :3, :3], directions)
    rays_o = c2w[:, :3, 3].unsqueeze(-2).unsqueeze(-2).expand_as(rays_d)
    if normalize:
        rays_d = F.normalize(rays_d, dim=-1)
    return rays_o, rays_d


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def _hwc3(img):
    """Ensure image is 3-channel HWC."""
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[2] == 1:
        return np.concatenate([img, img, img], axis=-1)
    if img.shape[2] == 4:
        return img[:, :, :3]
    return img


def _load_info(info, reso, prefix=None):
    """Extract image path, s2e, e2g, c2w, intrinsics from a frame info dict."""
    img_path = info["filename"]
    if prefix:
        img_path = osp.join(prefix, img_path)

    s2e = _get_transform(info["sensor2ego_translation"], info["sensor2ego_rotation"])
    e2g = _get_transform(info["ego2global_translation"], info["ego2global_rotation"])
    c2w = info["sensor2lidar_transform"]
    cks = np.array(info["camera_intrinsics"])
    cks = _resize_intrinsics(cks, tar_size=reso)
    return img_path, s2e, e2g, c2w, cks


def _load_conditions(img_paths, reso, vae_norm=False, prefix=None):
    """Load images, metric depths, confidence maps, and segmentation masks.

    Returns:
        imgs: (N, 3, H, W) float tensor.
        depths: (N, H, W) float tensor.
        confs: (N, H, W) float tensor in [0, 1].
        segs: (N, H, W) binary float tensor.
    """
    def _maybe_resize(img, tgt):
        if not isinstance(img, PIL.Image.Image):
            img = Image.fromarray(img)
        return np.array(img.resize((tgt[1], tgt[0])))

    imgs, depths, confs, segs = [], [], [], []
    for path in img_paths:
        # RGB image
        img = Image.open(osp.join(prefix, path))
        img = _maybe_resize(img, reso)
        img = _hwc3(np.array(img))
        imgs.append(img)

        # Metric depth + confidence from Metric3D-v2
        dc_path = osp.join(prefix, "depth_metric3d_v2", path.replace(".jpg", ".npz"))
        dc = np.load(dc_path)
        dpt = np.array(Image.fromarray(dc["depth"]).resize((reso[1], reso[0]), Image.BILINEAR))
        conf = np.array(Image.fromarray(dc["conf"]).resize((reso[1], reso[0]), Image.BILINEAR))
        conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
        depths.append(dpt)
        confs.append(conf)

        # Segmentation mask
        seg_path = osp.join(prefix, "segmentation_masks", path.replace(".jpg", ".png"))
        seg = np.array(Image.open(seg_path).resize((reso[1], reso[0]), Image.BILINEAR))
        segs.append(seg)

    if vae_norm:
        imgs = (torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 127.5) - 1.0
    else:
        imgs = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0

    depths = torch.from_numpy(np.stack(depths)).float()
    confs = torch.from_numpy(np.stack(confs)).float()
    segs = torch.from_numpy(np.stack(segs)).float() / 255.0
    segs = (segs >= 0.5).float()

    return imgs, depths, confs, segs


# ---------------------------------------------------------------------------
# Temporal sampling
# ---------------------------------------------------------------------------

class TemporalRandomCrop:
    """Randomly select a contiguous window of frame indices."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, total: int) -> Tuple[int, int]:
        end_max = max(0, total - self.size - 1)
        begin = random.randint(0, end_max)
        return begin, min(begin + self.size, total)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GSDecoderDataset(Dataset):
    """NuScenes dataset for Gaussian Splatting decoder.

    Args:
        data_root: Root directory containing preprocessed nuScenes data
            (images, depth, segmentation, etc.).
        split: One of ``'train'``, ``'val'``, or ``'test'``.
        infos_train: Filename of the training annotation JSON.
        infos_val: Filename of the validation annotation JSON.
        resolution: Target (H, W) for loaded images.
        vae_norm: If True, normalize images to [-1, 1]; otherwise [0, 1].
        num_frames: Number of temporal frames per sample.
        frame_interval: Stride between sampled frames.
        overfit: If True, use val set for training (debugging).
        times: Dataset repetition factor.
    """

    CAMERA_TYPES = [
        "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT",
    ]

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        infos_train: str = "nuscenes_train_seq_gs_lidar.json",
        infos_val: str = "nuscenes_val_seq_gs_lidar.json",
        resolution: List[int] = None,
        vae_norm: bool = False,
        num_frames: int = 3,
        frame_interval: int = 1,
        overfit: bool = False,
        times: int = 200,
    ):
        super().__init__()
        if resolution is None:
            resolution = [224, 400]

        self.data_root = data_root
        self.reso = resolution
        self.vae_norm = vae_norm
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.split = split
        self.overfit = overfit

        if split == "train" and overfit:
            self.data_infos = json.load(open(osp.join(data_root, infos_val)))
            self.times = times
        elif split == "train":
            self.data_infos = json.load(open(osp.join(data_root, infos_train)))
            self.times = times
        elif split == "val":
            self.data_infos = json.load(open(osp.join(data_root, infos_val)))
            self.data_infos = self.data_infos[:150:5]
            self.times = 1
        elif split == "test":
            self.data_infos = json.load(open(osp.join(data_root, infos_val)))
            self.times = 1
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.data_infos) * self.times

    def __getitem__(self, index: int) -> dict:
        index = index % len(self.data_infos)
        scene_info = self.data_infos[index]

        total_frames = len(scene_info)
        sampler = TemporalRandomCrop(self.num_frames * self.frame_interval)
        start, end = sampler(total_frames)
        assert end - start >= self.num_frames, f"Index {index}: not enough frames."

        frame_indices = np.arange(
            start, start + self.num_frames * self.frame_interval, self.frame_interval
        )
        assert frame_indices[-1] < total_frames, "Frame index out of bounds."

        if self.split != "train" or self.overfit:
            frame_indices = np.arange(0, self.num_frames * self.frame_interval, self.frame_interval)

        bin_info = [scene_info[i] for i in frame_indices]

        # Collect per-frame, per-camera data
        img_paths, s2e_list, e2g_list, c2w_list, ck_list = [], [], [], [], []
        for frame in bin_info:
            for cam in frame[:-1]:  # last element is typically lidar info
                info = copy.deepcopy(cam)
                path, s2e, e2g, c2w, cks = _load_info(info, reso=self.reso)
                img_paths.append(path)
                s2e_list.append(s2e)
                e2g_list.append(e2g)
                ck_list.append(cks)
                c2w_list.append(c2w)

        input_s2e = torch.as_tensor(np.array(s2e_list), dtype=torch.float32)
        input_e2g = torch.as_tensor(np.array(e2g_list), dtype=torch.float32)
        input_cks = torch.as_tensor(np.array(ck_list), dtype=torch.float32)
        input_c2ws = torch.as_tensor(np.array(c2w_list), dtype=torch.float32)

        # Convert c2w to OpenGL convention (flip Y and Z)
        flip_yz = torch.diag(torch.tensor([1, -1, -1, 1], dtype=input_c2ws.dtype))
        input_c2ws = input_c2ws @ flip_yz

        # Load images, depth, confidence, segmentation
        imgs, depths, confs, segs = _load_conditions(
            img_paths, self.reso, vae_norm=self.vae_norm, prefix=self.data_root
        )

        # Compute ray directions
        fxs = input_cks[:, 0, 0]
        fys = input_cks[:, 1, 1]
        cxs = input_cks[:, 0, 2]
        cys = input_cks[:, 1, 2]

        directions = torch.stack([
            _get_ray_directions(self.reso[0], self.reso[1],
                                focal=[fx.item(), fy.item()],
                                principal=[cx.item(), cy.item()])
            for fx, fy, cx, cy in zip(fxs, fys, cxs, cys)
        ])

        rays_o, rays_d = _get_rays(directions, input_c2ws, normalize=False)
        lidar_depth = torch.zeros_like(depths)
        sky_mask = torch.zeros_like(segs)

        input_dict = {"rgb": imgs}
        input_pix = {
            "depth_m": depths, "conf_m": confs,
            "ck": input_cks, "c2w": input_c2ws,
            "cx": cxs, "cy": cys, "fx": fxs, "fy": fys,
            "rays_o": rays_o, "rays_d": rays_d,
            "depth_r": lidar_depth, "input_segs_m": segs,
        }
        output_dict = {
            "rgb": imgs.clone(), "depth": depths.clone(),
            "depth_m": depths.clone(), "conf_m": confs.clone(),
            "c2w": input_c2ws.clone(),
            "rays_o": rays_o.clone(), "rays_d": rays_d.clone(),
            "depth_r": lidar_depth.clone(),
            "output_segs_m": segs.clone(), "output_skys_m": sky_mask,
        }

        return {"outputs": output_dict, "inputs": input_dict, "inputs_pix": input_pix}
