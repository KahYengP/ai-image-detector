"""Fusion head + full detector.

Concat(semantic 512, frequency 128) → 2-layer MLP → one logit.
Kept deliberately small: a more exotic attention fusion is unlikely to pay
for itself given limited compute, and a simple MLP is easier to train reliably.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from models.frequency_branch import FrequencyBranch
from models.semantic_branch import SemanticBranch


class FusionHead(nn.Module):
    def __init__(self, in_dim: int = 640, hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, semantic_emb: torch.Tensor, frequency_emb: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([semantic_emb, frequency_emb], dim=-1)
        return self.net(fused).squeeze(-1)


class AIGCDetector(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        self.logit_bias = float(cfg.get("thresholds", {}).get("logit_bias", 0.0))
        self.temperature = max(1e-3, float(cfg.get("thresholds", {}).get("temperature", 1.0)))
        self.use_frequency_branch = bool(model_cfg.get("use_frequency_branch", True))
        self.semantic = SemanticBranch(
            clip_name=model_cfg.get("clip_name", "openai/clip-vit-base-patch32"),
            freeze_except_last_block=bool(model_cfg.get("freeze_except_last_block", True)),
        )
        self.frequency: Optional[FrequencyBranch]
        self.fusion: Optional[FusionHead]
        if self.use_frequency_branch:
            self.frequency = FrequencyBranch(embedding_dim=128)
            self.fusion = FusionHead(
                in_dim=640,
                hidden_dim=256,
                dropout=float(model_cfg.get("dropout", 0.3)),
            )
        else:
            self.frequency = None
            self.fusion = None

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_01: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        sem_emb, sem_logit = self.semantic(pixel_values)
        freq_emb: Optional[torch.Tensor] = None
        freq_logit: Optional[torch.Tensor] = None
        if self.use_frequency_branch:
            if image_01 is None:
                raise ValueError("image_01 is required when the frequency branch is enabled.")
            freq_emb, freq_logit = self.frequency(image_01)
            logit = self.fusion(sem_emb, freq_emb)
        else:
            logit = sem_logit
        # Inference-only temperature/bias. Training sees unbiased logits.
        if self.training:
            pred_logit = logit
        else:
            pred_logit = (logit + self.logit_bias) / self.temperature
        return {
            "logit": logit,
            "pred": torch.sigmoid(pred_logit),
            "semantic_emb": sem_emb,
            "semantic_logit": sem_logit,
            "semantic_score": torch.sigmoid(sem_logit),
            "frequency_emb": freq_emb,
            "frequency_logit": freq_logit,
            "frequency_score": None if freq_logit is None else torch.sigmoid(freq_logit),
        }
