from __future__ import annotations
import torch
import torch.nn as nn


class ConvAE(nn.Module):
    """
    입력: (B,3,H,W) 0~1
    출력: (B,3,H,W)
    """
    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1),  # H/2
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1),  # H/4
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),  # H/8
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, 2, 1),  # H/16
            nn.ReLU(inplace=True),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 8, base * 8, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1),  # H/8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1),  # H/4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),      # H/2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, in_ch, 4, 2, 1),         # H
            nn.Sigmoid(),  # 0~1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        z = self.bottleneck(z)
        y = self.dec(z)
        return y