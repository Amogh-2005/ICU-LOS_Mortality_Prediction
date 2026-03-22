"""
metrics.py
----------
Evaluation metric utilities for both prediction tasks.

ordinal_logits_to_band          (B, 2) logits → (B,) band predictions
compute_los_metrics             accuracy, macro-F1, per-class F1/acc, Band-2 AUROC
compute_mortality_metrics       AUROC, F1, accuracy, precision, recall
print_classification_report     pretty sklearn report for the LOS task
"""

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)


# ===========================================================================
# Ordinal decoding
# ===========================================================================

def ordinal_logits_to_band(
    logits: torch.Tensor,
    b0_threshold: float = 0.5,
    b1_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Convert ordinal boundary logits (B, 2) → predicted LOS band (B,).

    Predicted band = number of thresholds exceeded:
        0  → neither boundary exceeded
        1  → only boundary 0 exceeded
        2  → both boundaries exceeded

    Parameters
    ----------
    logits       : Tensor (B, 2)
    b0_threshold : float   default 0.5
    b1_threshold : float   optimised via Platt scaling after training

    Returns
    -------
    band_pred : Tensor (B,) dtype long
    """
    probs     = torch.sigmoid(logits)                          # (B, 2)
    b0_pred   = (probs[:, 0] > b0_threshold).long()
    b1_pred   = (probs[:, 1] > b1_threshold).long()
    band_pred = b0_pred + b1_pred                              # ∈ {0, 1, 2}
    return band_pred


# ===========================================================================
# LOS metrics
# ===========================================================================

def compute_los_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    band2_threshold: float = 0.5,
    b0_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    All evaluation metrics for the 3-band ordinal LOS task.

    Parameters
    ----------
    logits          : Tensor (B, 2)   raw boundary logits
    labels          : Tensor (B,)     integer LOS band {0, 1, 2}
    band2_threshold : float           decision threshold for boundary 1 (Band 2)
    b0_threshold    : float           decision threshold for boundary 0

    Returns
    -------
    dict with keys:
        los_acc, los_macro_f1,
        los_f1_band0, los_f1_band1, los_f1_band2,
        los_acc_band0, los_acc_band1, los_acc_band2,
        los_band2_auroc
    """
    probs_np  = torch.sigmoid(logits).cpu().numpy()    # (B, 2)
    labels_np = labels.cpu().numpy().astype(int)        # (B,)

    b0_pred   = (probs_np[:, 0] > b0_threshold).astype(int)
    b1_pred   = (probs_np[:, 1] > band2_threshold).astype(int)
    band_pred = b0_pred + b1_pred                       # (B,) ∈ {0, 1, 2}

    acc      = float(accuracy_score(labels_np, band_pred))
    macro_f1 = float(f1_score(labels_np, band_pred, average="macro", zero_division=0))

    per_class_f1  = f1_score(labels_np, band_pred, average=None, zero_division=0)
    per_class_acc = []
    for band in range(3):
        mask = labels_np == band
        per_class_acc.append(
            float((band_pred[mask] == band).mean()) if mask.sum() > 0 else 0.0
        )

    band2_binary = (labels_np == 2).astype(int)
    try:
        band2_auroc = float(roc_auc_score(band2_binary, probs_np[:, 1]))
    except ValueError:
        band2_auroc = float("nan")

    return {
        "los_acc":         acc,
        "los_macro_f1":    macro_f1,
        "los_f1_band0":    float(per_class_f1[0]) if len(per_class_f1) > 0 else 0.0,
        "los_f1_band1":    float(per_class_f1[1]) if len(per_class_f1) > 1 else 0.0,
        "los_f1_band2":    float(per_class_f1[2]) if len(per_class_f1) > 2 else 0.0,
        "los_acc_band0":   per_class_acc[0],
        "los_acc_band1":   per_class_acc[1],
        "los_acc_band2":   per_class_acc[2],
        "los_band2_auroc": band2_auroc,
    }


# ===========================================================================
# Mortality metrics
# ===========================================================================

def compute_mortality_metrics(
    logit: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    All evaluation metrics for the binary mortality task.

    Parameters
    ----------
    logit     : Tensor (B, 1) or (B,)   raw scalar logit
    labels    : Tensor (B,)              binary labels {0, 1}
    threshold : float

    Returns
    -------
    dict with keys: mort_auroc, mort_f1, mort_acc, mort_precision, mort_recall
    """
    probs_np  = torch.sigmoid(logit.squeeze(-1)).cpu().numpy()
    labels_np = labels.cpu().numpy().astype(int)
    preds     = (probs_np > threshold).astype(int)

    try:
        auroc = float(roc_auc_score(labels_np, probs_np))
    except ValueError:
        auroc = float("nan")

    return {
        "mort_auroc":     auroc,
        "mort_f1":        float(f1_score(labels_np, preds, zero_division=0)),
        "mort_acc":       float(accuracy_score(labels_np, preds)),
        "mort_precision": float(precision_score(labels_np, preds, zero_division=0)),
        "mort_recall":    float(recall_score(labels_np, preds, zero_division=0)),
    }


# ===========================================================================
# Utility
# ===========================================================================

def print_classification_report(
    labels_np: np.ndarray,
    band_pred: np.ndarray,
    target_names: Optional[list] = None,
) -> None:
    if target_names is None:
        target_names = ["Band 0 (<24h)", "Band 1 (24-72h)", "Band 2 (>72h)"]
    print(
        classification_report(
            labels_np, band_pred, target_names=target_names, zero_division=0
        )
    )
