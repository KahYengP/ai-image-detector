"""Semantic branch: CLIP ViT-B/32, last transformer block unfrozen.

Why this branch exists: crop, color jitter, and resize change pixels a lot but
leave object-level content intact. A frozen-almost CLIP encoder is a cheap way
to keep that invariance without training a ViT from scratch (and it stays << 2B).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPVisionModel
from transformers.utils import logging as hf_logging

# The CLIP Hub checkpoint includes text weights; loading vision-only is expected
# to skip them. Silence that report so training logs stay readable.
hf_logging.set_verbosity_error()


class SemanticBranch(nn.Module):
    def __init__(
        self,
        clip_name: str = "openai/clip-vit-base-patch32",
        freeze_except_last_block: bool = True,
    ) -> None:
        super().__init__()
        self.vision = CLIPVisionModel.from_pretrained(clip_name)
        hidden = int(self.vision.config.hidden_size)
        # Linear projection down to a 512-d embedding (CLIP's native contrastive dim).
        self.proj = nn.Linear(hidden, 512)
        self.score_head = nn.Linear(512, 1)
        # "last_block" adapts CLIP to the training dump (can overfit CIFAR-looking
        # reals and then call ordinary photographs AI). "none" keeps CLIP frozen
        # (UnivFD-style) so natural photos stay in the real region.
        if freeze_except_last_block:
            self.freeze_except_last_block()
        else:
            self.freeze_all_clip()

    def _tower(self):
        # transformers v4 nested the ViT under `.vision_model`; v5 made CLIPVisionModel the tower itself.
        return getattr(self.vision, "vision_model", self.vision)

    def freeze_all_clip(self) -> None:
        for param in self.vision.parameters():
            param.requires_grad = False

    def freeze_except_last_block(self) -> None:
        """Keep pretrained features; only adapt the last block + our heads."""
        for param in self.vision.parameters():
            param.requires_grad = False
        tower = self._tower()
        last_block = tower.encoder.layers[-1]
        for param in last_block.parameters():
            param.requires_grad = True
        for param in tower.post_layernorm.parameters():
            param.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values: CLIP-normalized RGB, shape (B, 3, 224, 224)
        Returns:
            embedding: (B, 512)
            logit: (B,) unnormalized semantic AIGC score
        """
        outputs = self.vision(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        embedding = self.proj(pooled)
        logit = self.score_head(embedding).squeeze(-1)
        return embedding, logit
