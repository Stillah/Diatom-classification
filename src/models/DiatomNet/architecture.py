"""DiatomNet — CNN-классификатор диатомов (Inception-подобная архитектура)."""

from __future__ import annotations

import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        reduce_3x3: int,
        out_3x3: int,
        reduce_5x5: int,
        out_5x5: int,
        pool_proj: int,
    ):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(reduce_3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(reduce_5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)],
            dim=1,
        )


class DiatomNet(nn.Module):
    """Классификатор диатомов. По умолчанию — 6 видов из TARGET_CLASSES."""

    def __init__(self, num_classes: int = 6):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception1 = InceptionModule(
            in_channels=192,
            out_1x1=64,
            reduce_3x3=96,
            out_3x3=128,
            reduce_5x5=16,
            out_5x5=32,
            pool_proj=32,
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception2 = InceptionModule(
            in_channels=256,
            out_1x1=192,
            reduce_3x3=96,
            out_3x3=208,
            reduce_5x5=16,
            out_5x5=48,
            pool_proj=64,
        )
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3 = InceptionModule(
            in_channels=512,
            out_1x1=384,
            reduce_3x3=192,
            out_3x3=384,
            reduce_5x5=48,
            out_5x5=128,
            pool_proj=128,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.inception1(x))
        x = self.pool4(self.inception2(x))
        x = self.inception3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)
