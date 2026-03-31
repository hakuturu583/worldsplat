"""Loss functions: LPIPS perceptual loss and depth total-variation loss."""

import math
from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# LPIPS Perceptual Loss (VGG-based)
# ---------------------------------------------------------------------------

class LPIPS(nn.Module):
    """Learned Perceptual Image Patch Similarity (VGG-16 backbone).

    Adapted from https://github.com/richzhang/PerceptualSimilarity
    """

    def __init__(self, use_dropout: bool = True, ckpt_path: str = "checkpoints/vgg.pth"):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = VGG16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self._load_pretrained(ckpt_path)
        for param in self.parameters():
            param.requires_grad = False

    def _load_pretrained(self, ckpt_path: str):
        self.load_state_dict(
            torch.load(ckpt_path, map_location=torch.device("cpu")), strict=False
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        in0, in1 = self.scaling_layer(input), self.scaling_layer(target)
        outs0, outs1 = self.net(in0), self.net(in1)
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

        feats0, feats1 = {}, {}
        diffs = {}
        for kk in range(len(self.chns)):
            feats0[kk] = _normalize_tensor(outs0[kk])
            feats1[kk] = _normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [_spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv."""

    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class VGG16(nn.Module):
    def __init__(self, requires_grad: bool = False, pretrained: bool = True):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        VggOutputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        return VggOutputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


def _normalize_tensor(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def _spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)


# ---------------------------------------------------------------------------
# Depth Total-Variation Loss
# ---------------------------------------------------------------------------

class LossDepthTV(nn.Module):
    """Total variation loss on log-depth for regularization."""

    def __init__(
        self,
        use_second_derivative: bool = False,
        near: float = 0.1,
        far: float = 1000.0,
    ):
        super().__init__()
        self.use_second_derivative = use_second_derivative
        self.near = near
        self.far = far

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        near = math.log(self.near)
        far = math.log(self.far)
        depth = prediction.clamp(near, far)
        depth = (depth - near) / (far - near)

        depth_dx = depth.diff(dim=-1)
        depth_dy = depth.diff(dim=-2)

        if self.use_second_derivative:
            depth_dx = depth_dx.diff(dim=-1)
            depth_dy = depth_dy.diff(dim=-2)

        return depth_dx.abs().mean() + depth_dy.abs().mean()
