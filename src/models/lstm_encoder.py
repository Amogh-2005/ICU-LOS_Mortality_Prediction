"""
lstm_encoder.py
---------------
LSTM encoder for the vital sign time-series modality.

Input  : (B, 24, 5)  —  24 hourly bins × 5 vital sign channels
                         (heart rate, respiratory rate, SBP, DBP, SpO2)
Output : (B, output_dim)

Rather than compressing the entire sequence into the final hidden state,
a learned temporal attention mechanism computes a weighted sum over all 24
LSTM output states, allowing the model to attend selectively to the most
informative time steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Temporal Attention
# ===========================================================================

class TemporalAttention(nn.Module):
    """
    Additive (Bahdanau-style) attention over a sequence of LSTM hidden states.

    Computes a scalar score for each time step, normalises with softmax,
    and returns the weighted sum across the time dimension.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of each LSTM output vector.
        For a bidirectional LSTM this is already hidden_size * 2.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        # Single linear layer projecting each hidden state to a scalar score
        self.score_layer = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, lstm_outputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        lstm_outputs : Tensor  (B, T, hidden_size)
            All hidden states from the LSTM across T time steps.

        Returns
        -------
        context : Tensor  (B, hidden_size)
            Attention-weighted sum over the T time steps.
        """
        # (B, T, 1)
        scores = self.score_layer(lstm_outputs)
        # Normalise across the time dimension
        weights = F.softmax(scores, dim=1)          # (B, T, 1)
        # Weighted sum across T
        context = (weights * lstm_outputs).sum(dim=1)  # (B, hidden_size)
        return context


# ===========================================================================
# LSTM Encoder
# ===========================================================================

class LSTMEncoder(nn.Module):
    """
    Multi-layer LSTM followed by learned temporal attention pooling.

    Parameters
    ----------
    input_size : int
        Number of vital sign channels. Default 5.
    hidden_size : int
        Number of features in each LSTM hidden state.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability applied to the outputs of each LSTM layer
        except the last. Only active when num_layers > 1.
    bidirectional : bool
        If True the LSTM is bidirectional; output_dim = hidden_size * 2.
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Actual dimensionality of each lstm output vector
        self._output_dim = hidden_size * (2 if bidirectional else 1)

        self.attention = TemporalAttention(self._output_dim)

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
        x : Tensor  (B, 24, 5)
            Z-score normalised hourly vital sign sequences.

        Returns
        -------
        context : Tensor  (B, output_dim)
            Attention-pooled representation of the 24-step sequence.
        """
        # lstm_out : (B, T, hidden_size * num_directions)
        # _        : (h_n, c_n)  — not used; attention replaces final-state pooling
        lstm_out, _ = self.lstm(x)

        # Attention-weighted sum across all 24 time steps
        context = self.attention(lstm_out)   # (B, output_dim)
        return context
