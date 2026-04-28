"""
losses.py
---------
Loss functions for the ICU Outcome Prediction project.

OrdinalLOSLoss
    Ordinal cross-entropy via K-1 binary decomposition.
    Accepts logits (B, 2) for two threshold boundaries and optional
    per-boundary positive class weights.

MortalityLoss
    BCEWithLogitsLoss with optional pos_weight for the binary mortality task.

CombinedMultiTaskLoss
    L = alpha * L_ordinal_LOS + (1 - alpha) * L_binary_mortality
      + band1_penalty_weight * band1_recall_penalty
      + ordinal_consistency_weight * ordinal_consistency_loss
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Ordinal LOS Loss
# ===========================================================================

class OrdinalLOSLoss(nn.Module):
    """
    Ordinal cross-entropy for 3-band LOS via K-1 binary decomposition.

    The LOS head outputs logits (B, 2):
        logits[:, 0]  →  boundary 0  — P(LOS >= Band 1)
        logits[:, 1]  →  boundary 1  — P(LOS >= Band 2)

    Each boundary is trained with an independent BCE loss weighted by
    its own per-boundary positive class weight computed from training data.

    Parameters
    ----------
    boundary_weights : Tensor (2,) or None
        pos_weight for boundary 0 and boundary 1 respectively.
        Loaded from dataset.los_boundary_weights.
    """

    def __init__(self, boundary_weights: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if boundary_weights is not None:
            self.register_buffer("boundary_weights", boundary_weights.float())
        else:
            self.boundary_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        los_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits     : Tensor (B, 2)   raw boundary logits
        los_labels : Tensor (B,)     integer LOS band in {0, 1, 2}  dtype long

        Returns
        -------
        loss : scalar Tensor
        """
        # Convert ordinal band to two binary boundary targets
        b0_target = (los_labels >= 1).float()   # Band 0  vs  {Band 1, Band 2}
        b1_target = (los_labels >= 2).float()   # {Band 0, Band 1}  vs  Band 2

        pw0 = pw1 = None
        if self.boundary_weights is not None:
            pw0 = self.boundary_weights[0].to(logits.device)
            pw1 = self.boundary_weights[1].to(logits.device)

        loss_b0 = F.binary_cross_entropy_with_logits(
            logits[:, 0], b0_target, pos_weight=pw0
        )
        loss_b1 = F.binary_cross_entropy_with_logits(
            logits[:, 1], b1_target, pos_weight=pw1
        )

        return (loss_b0 + loss_b1) / 2.0


# ===========================================================================
# Mortality Loss
# ===========================================================================

class MortalityLoss(nn.Module):
    """
    BCEWithLogitsLoss for binary in-hospital mortality prediction.

    Parameters
    ----------
    pos_weight : float or None
        Ratio of negative to positive instances in the training set.
        Loaded from dataset.mortality_pos_weight.
    """

    def __init__(self, pos_weight: Optional[float] = None) -> None:
        super().__init__()
        self._pos_weight_val = pos_weight

    def forward(
        self,
        logit: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logit  : Tensor (B, 1) or (B,)   raw scalar logit
        labels : Tensor (B,)              float binary labels {0.0, 1.0}

        Returns
        -------
        loss : scalar Tensor
        """
        logit = logit.squeeze(-1)  # (B,)
        pos_weight = None
        if self._pos_weight_val is not None:
            pos_weight = torch.tensor(
                self._pos_weight_val, dtype=torch.float32, device=logit.device
            )
        return F.binary_cross_entropy_with_logits(logit, labels, pos_weight=pos_weight)


# ===========================================================================
# Combined Multi-Task Loss
# ===========================================================================

class CombinedMultiTaskLoss(nn.Module):
    """
    L = alpha * L_ordinal_LOS + (1 - alpha) * L_binary_mortality
      + band1_penalty_weight * band1_recall_penalty(los_logits, los_labels)
      + ordinal_consistency_weight * ordinal_consistency_loss(los_logits)

    Parameters
    ----------
    alpha : float
        Weight for the LOS loss.  (1 - alpha) weights the mortality loss.
    los_boundary_weights : Tensor (2,) or None
    mortality_pos_weight : float or None
    band1_penalty_weight : float
        Coefficient for the Band 1 recall penalty term. Default 0.15.
    ordinal_consistency_weight : float
        Coefficient for the ordinal consistency penalty term. Default 0.10.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        los_boundary_weights: Optional[torch.Tensor] = None,
        mortality_pos_weight: Optional[float] = None,
        band1_penalty_weight: float = 0.15,
        ordinal_consistency_weight: float = 0.10,
    ) -> None:
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}.")

        self.alpha                     = alpha
        self.band1_penalty_weight      = band1_penalty_weight
        self.ordinal_consistency_weight = ordinal_consistency_weight
        self.los_loss_fn               = OrdinalLOSLoss(boundary_weights=los_boundary_weights)
        self.mort_loss_fn              = MortalityLoss(pos_weight=mortality_pos_weight)

    def forward(
        self,
        los_logits:       torch.Tensor,
        mortality_logit:  torch.Tensor,
        los_labels:       torch.Tensor,
        mortality_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total_loss : scalar Tensor
        los_loss   : scalar Tensor   (logged separately)
        mort_loss  : scalar Tensor   (logged separately)
        """
        los_loss  = self.los_loss_fn(los_logits, los_labels)
        mort_loss = self.mort_loss_fn(mortality_logit, mortality_labels)

        band1_pen   = band1_recall_penalty(los_logits, los_labels)
        ordinal_pen = ordinal_consistency_loss(los_logits)

        total = (
            self.alpha * los_loss
            + (1.0 - self.alpha) * mort_loss
            + self.band1_penalty_weight * band1_pen
            + self.ordinal_consistency_weight * ordinal_pen
        )
        return total, los_loss, mort_loss


# ===========================================================================
# Band 1 Recall Penalty
# ===========================================================================

def band1_recall_penalty(los_logits: torch.Tensor, los_labels: torch.Tensor) -> torch.Tensor:
    """
    Penalises the model when true Band 1 patients have high boundary-1
    logits (i.e. being pushed toward Band 2).

    Parameters
    ----------
    los_logits : Tensor (B, 2)
    los_labels : Tensor (B,)   integer LOS band in {0, 1, 2}

    Returns
    -------
    penalty : scalar Tensor
    """
    band1_mask = (los_labels == 1)
    if band1_mask.sum() == 0:
        return torch.tensor(0.0, device=los_logits.device)
    b1_logits_for_band1 = los_logits[band1_mask, 1]
    return F.relu(b1_logits_for_band1).mean()


# ===========================================================================
# Ordinal Consistency Loss
# ===========================================================================

def ordinal_consistency_loss(los_logits: torch.Tensor) -> torch.Tensor:
    """
    Penalises violations of the ordinal constraint:
    boundary 0 logit should always be >= boundary 1 logit.
    A violation means the model predicts P(>=Band2) > P(>=Band1),
    which is logically impossible in an ordinal setting.

    Parameters
    ----------
    los_logits : Tensor (B, 2)

    Returns
    -------
    penalty : scalar Tensor
    """
    b0 = los_logits[:, 0]
    b1 = los_logits[:, 1]
    violation = F.relu(b1 - b0)
    return violation.mean()
