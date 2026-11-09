"""ResNet50-IBN-a backbone with non-local blocks + GeM/BNNeck head.

Pure-PyTorch re-implementation of the FastReID "SBS(R50-ibn)" architecture so
the public VeRi-776 checkpoint (veri_sbs_R50-ibn.pth) can be loaded without
installing FastReID. The Instance-Batch Normalisation layers make the
embedding robust to colour/brightness differences between cameras (FR-RI-05).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IBN(nn.Module):
    """Half InstanceNorm, half BatchNorm (IBN-a)."""

    def __init__(self, planes: int):
        super().__init__()
        self.half = planes // 2
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.split(x, [self.half, x.size(1) - self.half], dim=1)
        return torch.cat((self.IN(a.contiguous()), self.BN(b.contiguous())), dim=1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, with_ibn: bool, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = IBN(planes) if with_ibn else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class NonLocal(nn.Module):
    """Embedded-Gaussian style non-local block as used in FastReID."""

    def __init__(self, in_channels: int, inter_channels: int):
        super().__init__()
        self.inter = inter_channels
        self.g = nn.Conv2d(in_channels, inter_channels, 1)
        self.theta = nn.Conv2d(in_channels, inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, inter_channels, 1)
        self.W = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1), nn.BatchNorm2d(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        g_x = self.g(x).view(b, self.inter, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(b, self.inter, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(b, self.inter, -1)
        f = torch.matmul(theta_x, phi_x)
        f = f / f.size(-1)
        y = torch.matmul(f, g_x).permute(0, 2, 1).contiguous().view(b, self.inter, *x.shape[2:])
        return self.W(y) + x


class ResNetIBNBackbone(nn.Module):
    def __init__(self, layers=(3, 4, 6, 3), non_layers=(0, 2, 3, 0), last_stride: int = 1, nl_inter: int = 1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(64, layers[0], 1, True)
        self.layer2 = self._make_layer(128, layers[1], 2, True)
        self.layer3 = self._make_layer(256, layers[2], 2, True)
        self.layer4 = self._make_layer(512, layers[3], last_stride, False)
        widths = (256, 512, 1024, 2048)
        self.NL_1 = nn.ModuleList([NonLocal(widths[0], nl_inter) for _ in range(non_layers[0])])
        self.NL_2 = nn.ModuleList([NonLocal(widths[1], nl_inter) for _ in range(non_layers[1])])
        self.NL_3 = nn.ModuleList([NonLocal(widths[2], nl_inter) for _ in range(non_layers[2])])
        self.NL_4 = nn.ModuleList([NonLocal(widths[3], nl_inter) for _ in range(non_layers[3])])
        # Non-local blocks follow the last n residual blocks of a stage.
        self.nl_idx = [sorted(layers[i] - (j + 1) for j in range(non_layers[i])) for i in range(4)]

    def _make_layer(self, planes: int, blocks: int, stride: int, with_ibn: bool) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        layers = [Bottleneck(self.inplanes, planes, with_ibn, stride, downsample)]
        self.inplanes = planes * 4
        layers += [Bottleneck(self.inplanes, planes, with_ibn) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        stages = (self.layer1, self.layer2, self.layer3, self.layer4)
        nls = (self.NL_1, self.NL_2, self.NL_3, self.NL_4)
        for stage, nl, idx in zip(stages, nls, self.nl_idx):
            k = 0
            for i, block in enumerate(stage):
                x = block(x)
                if k < len(idx) and i == idx[k]:
                    x = nl[k](x)
                    k += 1
        return x


class GeneralizedMeanPoolingP(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)


class EmbeddingHead(nn.Module):
    def __init__(self, dim: int = 2048):
        super().__init__()
        self.pool_layer = GeneralizedMeanPoolingP()
        self.bottleneck = nn.Sequential(nn.BatchNorm2d(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(self.pool_layer(x))[..., 0, 0]


class ResNetIBNReID(nn.Module):
    """Full model: normalisation -> backbone -> 2048-d embedding."""

    def __init__(self):
        super().__init__()
        self.register_buffer("pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))
        self.backbone = ResNetIBNBackbone()
        self.heads = EmbeddingHead(2048)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``images``: float tensor N x 3 x H x W, RGB, 0..255."""
        x = (images - self.pixel_mean) / self.pixel_std
        return self.heads(self.backbone(x))

    def load_fastreid_checkpoint(self, path: str) -> tuple[list[str], list[str]]:
        """Load FastReID weights safely (tensors only, no pickled code)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state = ckpt.get("model", ckpt)
        own = self.state_dict()
        filtered = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        missing, _ = self.load_state_dict(filtered, strict=False)
        unused = [k for k in state if k not in filtered]
        return list(missing), unused
