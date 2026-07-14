# src/models/SC_DiatomNet/modules.py

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Basic Conv
# Conv -> BatchNorm -> SiLU
# ---------------------------------------------------------------------

class Conv(nn.Module):
    """Standard convolution block used in YOLO11."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        act: bool = True,
    ):
        super().__init__()

        if p is None:
            p = k // 2

        self.conv = nn.Conv2d(
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=g,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------
# Bottleneck
# ---------------------------------------------------------------------

class Bottleneck(nn.Module):

    def __init__(
        self,
        c1,
        c2,
        shortcut=True,
        expansion=0.5,
    ):
        super().__init__()

        hidden = int(c2 * expansion)

        self.cv1 = Conv(c1, hidden, 1)
        self.cv2 = Conv(hidden, c2, 3)

        self.use_shortcut = shortcut and c1 == c2

    def forward(self, x):

        y = self.cv2(self.cv1(x))

        if self.use_shortcut:
            y = y + x

        return y


# ---------------------------------------------------------------------
# C2f block (YOLO11)
# ---------------------------------------------------------------------

class C2f(nn.Module):

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=True,
        expansion=0.5,
    ):
        super().__init__()

        hidden = int(c2 * expansion)

        self.cv1 = Conv(c1, hidden * 2, 1)

        self.blocks = nn.ModuleList(
            Bottleneck(hidden, hidden, shortcut)
            for _ in range(n)
        )

        self.cv2 = Conv((2 + n) * hidden, c2, 1)

    def forward(self, x):

        y = list(self.cv1(x).chunk(2, 1))

        for block in self.blocks:
            y.append(block(y[-1]))

        return self.cv2(torch.cat(y, dim=1))


# ---------------------------------------------------------------------
# SPPF
# ---------------------------------------------------------------------

class SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        super().__init__()

        hidden = c1 // 2

        self.cv1 = Conv(c1, hidden, 1)

        self.pool = nn.MaxPool2d(
            kernel_size=k,
            stride=1,
            padding=k // 2,
        )

        self.cv2 = Conv(hidden * 4, c2, 1)

    def forward(self, x):

        x = self.cv1(x)

        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)

        return self.cv2(
            torch.cat((x, y1, y2, y3), dim=1)
        )


# ---------------------------------------------------------------------
# Channel Attention
# ---------------------------------------------------------------------

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = self.mlp(self.avg(x))
        mx = self.mlp(self.max(x))

        return self.sigmoid(avg + mx) * x


# ---------------------------------------------------------------------
# Spatial Attention
# ---------------------------------------------------------------------

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.max(x, dim=1, keepdim=True)[0]

        attention = torch.cat([avg, mx], dim=1)

        attention = self.sigmoid(self.conv(attention))

        return attention * x


# ---------------------------------------------------------------------
# CBAM
# ---------------------------------------------------------------------

class CBAM(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x):

        x = self.channel(x)
        x = self.spatial(x)

        return x

class SCDiatomNet(nn.Module):

    def __init__(self):

        self.backbone = Backbone()

        self.neck = Neck()

        self.detect = Detect()