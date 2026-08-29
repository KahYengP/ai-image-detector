"""Frequency branch: 2D FFT magnitude → small CNN trained from scratch.

Why this branch exists: many GAN/diffusion pipelines leave periodic upsampling
and spectral peaks that survive blur and JPEG better than raw RGB textures.
We do not use pretrained weights here — the input is a spectrum, not a photo.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rgb_to_log_fft(images_01: torch.Tensor) -> torch.Tensor:
    """Grayscale 2D FFT log-magnitude, fftshifted, per-sample standardized.

    Args:
        images_01: (B, 3, H, W) in [0, 1]
    Returns:
        (B, 1, H, W)
    """
    gray = images_01.mean(dim=1)
    spec = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
    mag = torch.log1p(spec.abs())
    # Per-image z-score so exposure / DC energy does not dominate the CNN.
    mean = mag.mean(dim=(-2, -1), keepdim=True)
    std = mag.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    mag = (mag - mean) / std
    return mag.unsqueeze(1)


class FrequencyBranch(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        # 4 conv stages, stride 2 each: 224 → 14 with AdaptiveAvgPool at the end
        # so the exact input size does not matter.
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            # GroupNorm is stable with small batches; BatchNorm was collapsing
            # frequency_score to a near-constant (~0.52) after a short run.
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, embedding_dim)
        self.score_head = nn.Linear(embedding_dim, 1)

    def forward(self, images_01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images_01: RGB in [0, 1], shape (B, 3, H, W) — same crop the CLIP branch sees
        Returns:
            embedding: (B, 128)
            logit: (B,) unnormalized frequency AIGC score
        """
        spectrum = rgb_to_log_fft(images_01)
        pooled = self.features(spectrum).flatten(1)
        embedding = self.proj(pooled)
        logit = self.score_head(embedding).squeeze(-1)
        return embedding, logit
