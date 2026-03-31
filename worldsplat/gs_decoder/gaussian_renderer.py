"""Gaussian splatting renderer using gsplat backend."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gsplat
from einops import rearrange
from torch import Tensor

C0 = 0.28209479177387814


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def inverse_sigmoid(x: Tensor) -> Tensor:
    return torch.log(x / (1 - x))


def build_rotation(r: Tensor) -> Tensor:
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=r.device)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s: Tensor, r: Tensor) -> Tensor:
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    return R @ L


def strip_lowerdiag(L: Tensor) -> Tensor:
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym: Tensor) -> Tensor:
    return strip_lowerdiag(sym)


def transform_c2w(c2w: Tensor) -> Tensor:
    """Convert camera-to-world matrices to world-to-camera matrices with OpenGL-to-OpenCV convention.

    Flips y and z axes, then inverts to get w2c.

    Args:
        c2w: (V, 4, 4) camera-to-world matrices.

    Returns:
        w2c: (V, 4, 4) world-to-camera matrices.
    """
    flip_yz = torch.eye(4, device=c2w.device)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    c2w = torch.matmul(c2w, flip_yz)
    w2c = torch.linalg.inv(c2w)
    return w2c


class Depth2Normal(nn.Module):
    """Compute surface normals from a depth map via finite differences."""

    def __init__(self):
        super().__init__()
        self.delzdelxkernel = torch.tensor(
            [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
        )
        self.delzdelykernel = torch.tensor(
            [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        kx = self.delzdelxkernel.view(1, 1, 3, 3).to(x.device)
        ky = self.delzdelykernel.view(1, 1, 3, 3).to(x.device)
        delzdelx = F.conv2d(x.reshape(B * C, 1, H, W), kx, padding=1).reshape(B, C, H, W)
        delzdely = F.conv2d(x.reshape(B * C, 1, H, W), ky, padding=1).reshape(B, C, H, W)
        return -torch.cross(delzdelx, delzdely, dim=1)


class GaussianRenderer:
    """Gaussian splatting renderer using the gsplat library.

    Args:
        device: Torch device.
        resolution: [H, W] output resolution.
        znear: Near clipping plane.
        zfar: Far clipping plane.
    """

    def __init__(
        self,
        device,
        resolution: list = [512, 512],
        znear: float = 0.1,
        zfar: float = 100.0,
        **kwargs,
    ):
        self.resolution = resolution
        self.znear = znear
        self.zfar = zfar
        self.bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        self.normal_module = Depth2Normal().to(device)

    def compute_Ks(self, fovx: Tensor, fovy: Tensor) -> Tensor:
        """Compute intrinsic matrices from FoV angles.

        Args:
            fovx: (B, V) horizontal FoV in radians.
            fovy: (B, V) vertical FoV in radians.

        Returns:
            Ks: (B, V, 3, 3) intrinsic matrices.
        """
        B, V = fovx.shape
        width = self.resolution[1]
        height = self.resolution[0]
        fx = width / (2 * torch.tan(fovx / 2))
        fy = height / (2 * torch.tan(fovy / 2))
        cx = width / 2.0
        cy = height / 2.0
        Ks = torch.zeros(B, V, 3, 3, device=fovx.device)
        Ks[:, :, 0, 0] = fx
        Ks[:, :, 1, 1] = fy
        Ks[:, :, 0, 2] = cx
        Ks[:, :, 1, 2] = cy
        Ks[:, :, 2, 2] = 1.0
        return Ks

    def render(
        self,
        gaussians: Tensor,
        c2w: Tensor,
        cks: Tensor,
        rays_o: Tensor = None,
        rays_d: Tensor = None,
        bg_color: Tensor = None,
        scale_modifier: float = 1.0,
    ) -> dict:
        """Render Gaussians from given camera viewpoints using gsplat.

        Gaussian tensor layout: [B, N, 14] = [means(3), rgb(3), opacity(1), rotation(4), scale(3)].

        Args:
            gaussians: (B, N, 14) Gaussian parameters.
            c2w: (B, V, 4, 4) camera-to-world matrices.
            cks: (B, V, 3, 3) camera intrinsics.
            bg_color: Optional background color override.
            scale_modifier: Scale multiplier for Gaussian scales.

        Returns:
            Dict with keys "image" (B,V,3,H,W), "alpha" (B,V,1,H,W), "depth" (B,V,1,H,W).
        """
        device = gaussians.device
        B, V = c2w.shape[:2]
        H, W = self.resolution

        list_images = []
        list_alphas = []
        list_depths = []

        for b in range(B):
            means3D = gaussians[b, :, 0:3].contiguous().float()
            rgbs = gaussians[b, :, 3:6].contiguous().float()
            opacity = gaussians[b, :, 6:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            scales = gaussians[b, :, 11:14].contiguous().float()

            scales_modified = scales * scale_modifier
            viewmats = transform_c2w(c2w[b])  # (V, 4, 4)
            Ks_b = cks[b]  # (V, 3, 3)

            render_colors_b, render_alphas_b, _ = gsplat.rasterization(
                means=means3D,
                quats=rotations,
                scales=scales_modified,
                opacities=opacity.squeeze(-1),
                colors=rgbs,
                viewmats=viewmats,
                Ks=Ks_b,
                width=W,
                height=H,
                near_plane=self.znear,
                far_plane=self.zfar,
                render_mode="RGB+D",
            )

            # Extract RGB, depth, and alpha
            images_b = torch.clamp(render_colors_b[:, :, :, :3], min=0.0, max=1.0)
            depths_b = render_colors_b[:, :, :, 3:4]
            alphas_b = render_alphas_b

            # Rearrange to (V, C, H, W)
            list_images.append(rearrange(images_b, "v h w c -> v c h w"))
            list_depths.append(rearrange(depths_b, "v h w c -> v c h w"))
            list_alphas.append(rearrange(alphas_b, "v h w c -> v c h w"))

        return {
            "image": torch.stack(list_images, dim=0),   # (B, V, 3, H, W)
            "alpha": torch.stack(list_alphas, dim=0),    # (B, V, 1, H, W)
            "depth": torch.stack(list_depths, dim=0),    # (B, V, 1, H, W)
        }

    # ------------------------------------------------------------------
    # PLY I/O
    # ------------------------------------------------------------------

    def save_ply(self, gaussians: Tensor, path: str, compatible: bool = True):
        """Save Gaussians to a PLY file.

        Expected gaussian layout: [1, N, 14] = [xyz(3), opacity(1), scale(3), rotation(4), rgb(3)].
        """
        from plyfile import PlyData, PlyElement

        assert gaussians.shape[0] == 1, "Only batch size 1 is supported."

        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float()

        # Prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        if compatible:
            opacity = inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / C0

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales_np = scales.detach().cpu().numpy()
        rotations_np = rotations.detach().cpu().numpy()

        attrs = ["x", "y", "z"]
        for i in range(f_dc.shape[1]):
            attrs.append(f"f_dc_{i}")
        attrs.append("opacity")
        for i in range(scales_np.shape[1]):
            attrs.append(f"scale_{i}")
        for i in range(rotations_np.shape[1]):
            attrs.append(f"rot_{i}")

        dtype_full = [(attr, "f4") for attr in attrs]
        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales_np, rotations_np), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path: str, compatible: bool = True) -> Tensor:
        """Load Gaussians from a PLY file.

        Returns:
            Tensor of shape (N, 14) on CPU.
        """
        from plyfile import PlyData

        plydata = PlyData.read(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float()

        if compatible:
            gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
            gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
            gaussians[..., 11:] = C0 * gaussians[..., 11:] + 0.5

        return gaussians
