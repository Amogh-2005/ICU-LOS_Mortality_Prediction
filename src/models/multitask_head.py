"""
multitask_head.py
-----------------
Task-specific private encoders and prediction heads.

Architecture
------------
    fused  (B, fusion_dim)
        │
        ├── LOS private encoder   [Linear → GELU → Dropout → Linear → GELU]
        │       └── Ordinal LOS head     Linear → (B, 2)
        │               logits[:, 0]  P(LOS >= Band 1)   boundary 0
        │               logits[:, 1]  P(LOS >= Band 2)   boundary 1
        │
        └── Mortality private encoder  [Linear → GELU → Dropout → Linear → GELU]
                └── Binary mortality head  Linear → (B, 1)

Each private encoder is a two-layer MLP that transforms the shared fused
representation independently for its task, so the two prediction objectives
do not compete over the same weights when learning task-specific projections.
"""

from typing import Tuple

import torch
import torch.nn as nn


class PrivateEncoder(nn.Module):
    """
    Two-layer MLP private encoder for a single task.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the incoming fused representation.
    hidden_dim : int
        Width of the first (hidden) layer.
    output_dim : int
        Width of the second layer; fed into the prediction head.
    dropout : float
        Dropout probability applied between the two linear layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  (B, input_dim)

        Returns
        -------
        Tensor  (B, output_dim)
        """
        return self.net(x)


class MultiTaskHead(nn.Module):
    """
    Dual-task prediction head with independent private encoders.

    Parameters
    ----------
    fusion_dim : int
        Dimensionality of the fused representation from
        CrossModalAttentionFusion. Read from fusion.output_dim.
    private_hidden_dim : int
        Hidden layer width inside each PrivateEncoder.
    private_output_dim : int
        Output width of each PrivateEncoder (input to the linear head).
    dropout : float
        Dropout inside each PrivateEncoder.
    """

    def __init__(
        self,
        fusion_dim: int = 128,
        private_hidden_dim: int = 64,
        private_output_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Independent private encoders — one per task
        self.los_private_encoder = PrivateEncoder(
            input_dim=fusion_dim,
            hidden_dim=private_hidden_dim,
            output_dim=private_output_dim,
            dropout=dropout,
        )
        self.mortality_private_encoder = PrivateEncoder(
            input_dim=fusion_dim,
            hidden_dim=private_hidden_dim,
            output_dim=private_output_dim,
            dropout=dropout,
        )

        # Prediction heads
        # Ordinal LOS: two boundary logits for K-1 binary decomposition
        self.los_head = nn.Linear(private_output_dim, 2)
        # Binary mortality: single scalar logit
        self.mortality_head = nn.Linear(private_output_dim, 1)

    def forward(
        self, fused: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        fused : Tensor  (B, fusion_dim)
            Output of CrossModalAttentionFusion.

        Returns
        -------
        los_logits      : Tensor  (B, 2)
            Raw boundary logits.
            [:, 0]  —  boundary 0:  P(LOS >= Band 1)
            [:, 1]  —  boundary 1:  P(LOS >= Band 2)
        mortality_logit : Tensor  (B, 1)
            Raw scalar logit for binary mortality prediction.
        """
        los_feat  = self.los_private_encoder(fused)        # (B, private_output_dim)
        mort_feat = self.mortality_private_encoder(fused)  # (B, private_output_dim)

        los_logits      = self.los_head(los_feat)          # (B, 2)
        mortality_logit = self.mortality_head(mort_feat)   # (B, 1)

        return los_logits, mortality_logit
