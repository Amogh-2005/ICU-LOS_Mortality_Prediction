"""
tabular_encoder.py
------------------
MLP encoder for the concatenated 156-dimensional tabular feature vector.

The tabular feature vector (B, 156) is composed of:
    Lab mean / min / max summaries                24 features
    Vital sign trend slopes       (5 channels)     5 features
    Laboratory trend slopes       (8 variables)    8 features
    Composite clinical scores                      3 features
        - Shock index          (HR / SBP mean)
        - BUN-to-creatinine ratio
        - Sepsis proxy score   (lactate × WBC)
    Laboratory missingness indicators              8 features  (binary, not normalised)
    CCS diagnosis multi-hot encoding             100 features  (binary, not normalised)
    Mechanical ventilation flag                    1 feature   (binary, not normalised)
    Static features                                7 features
        - Age (continuous, normalised)
        - Gender, Ethnicity, Admission type,
          Insurance, First care unit  (embedding indices)
        - Prior ICU admission count (continuous, normalised)
    ─────────────────────────────────────────────────────────
    Total                                        156 features

The encoder is a straightforward MLP with configurable hidden dimensions,
BatchNorm1d, GELU activations, and dropout.  The `output_dim` property is
exposed so the downstream fusion layer can size its projections automatically.
"""

from typing import List, Optional

import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    """
    Multi-layer perceptron encoder for the tabular feature vector.

    Parameters
    ----------
    input_dim : int
        Total number of tabular features. Default 156.
    hidden_dims : list of int
        Width of each hidden layer, applied in order.
        Default [256, 128].
    output_dim : int
        Dimensionality of the final encoded representation.
        Default 128.
    dropout : float
        Dropout probability applied after each hidden activation.
    use_batch_norm : bool
        If True, BatchNorm1d is inserted before each activation.
        Disable for batch size 1 at inference (or switch model to eval mode).
    """

    def __init__(
        self,
        input_dim: int = 156,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 128,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self._output_dim = output_dim

        layers: List[nn.Module] = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Final projection to output_dim — no dropout after this layer
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.GELU())

        self.net = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Property used by CrossModalAttentionFusion to size its projections
    # ------------------------------------------------------------------
    @property
    def output_dim(self) -> int:
        """Dimensionality of the encoder's output vector."""
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  (B, 156)
            Concatenated tabular feature vector (z-score normalised continuous
            features; binary features passed through as-is).

        Returns
        -------
        Tensor  (B, output_dim)
        """
        return self.net(x)
