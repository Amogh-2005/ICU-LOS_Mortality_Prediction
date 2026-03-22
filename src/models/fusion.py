"""
fusion.py
---------
Cross-modal attention fusion layer.

The LSTM vitals embedding acts as the *query* and the tabular MLP embedding
acts as both *keys* and *values*.  This allows the fused representation to
attend selectively over tabular features conditioned on the physiological
state captured by the vital sign encoder.

Data flow
---------
    vitals_emb  (B, vitals_dim)   ──► Q projection ──┐
                                                       ├── MultiheadAttention ──► residual + LN
    tabular_emb (B, tabular_dim)  ──► K, V projection ─┘
                                                              │
                                                       residual FFN + LN
                                                              │
                                                       fused  (B, fusion_dim)
"""

import torch
import torch.nn as nn


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion.

    Parameters
    ----------
    vitals_dim : int
        Dimensionality of the LSTM vitals embedding (query source).
        Read from LSTMEncoder.output_dim.
    tabular_dim : int
        Dimensionality of the tabular MLP embedding (key / value source).
        Read from TabularEncoder.output_dim.
    fusion_dim : int
        Internal and output dimensionality of the attention mechanism.
        Must be divisible by num_heads.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout applied inside MultiheadAttention and the FFN sublayer.
    ffn_expansion : int
        Width multiplier for the hidden layer of the residual FFN.
        Hidden size = fusion_dim * ffn_expansion.
    """

    def __init__(
        self,
        vitals_dim: int = 128,
        tabular_dim: int = 128,
        fusion_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        ffn_expansion: int = 2,
    ) -> None:
        super().__init__()

        if fusion_dim % num_heads != 0:
            raise ValueError(
                f"fusion_dim ({fusion_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self._output_dim = fusion_dim

        # ------------------------------------------------------------------
        # Linear projections → shared fusion_dim
        # ------------------------------------------------------------------
        self.query_proj = nn.Linear(vitals_dim,  fusion_dim)
        self.key_proj   = nn.Linear(tabular_dim, fusion_dim)
        self.value_proj = nn.Linear(tabular_dim, fusion_dim)

        # ------------------------------------------------------------------
        # Multi-head cross-attention
        # batch_first=True so tensors are (B, seq_len, embed_dim)
        # ------------------------------------------------------------------
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ------------------------------------------------------------------
        # Residual feed-forward sublayer
        # ------------------------------------------------------------------
        ffn_hidden = fusion_dim * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, fusion_dim),
        )

        self.norm1   = nn.LayerNorm(fusion_dim)
        self.norm2   = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Property read by MultimodalICUModel to size the MultiTaskHead
    # ------------------------------------------------------------------
    @property
    def output_dim(self) -> int:
        """Dimensionality of the fused output vector."""
        return self._output_dim

    def forward(
        self,
        vitals_emb: torch.Tensor,
        tabular_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        vitals_emb  : Tensor  (B, vitals_dim)
            Output of LSTMEncoder — used as the attention query.
        tabular_emb : Tensor  (B, tabular_dim)
            Output of TabularEncoder — used as attention keys and values.

        Returns
        -------
        fused : Tensor  (B, fusion_dim)
        """
        # Project inputs to fusion_dim and add a length-1 sequence dimension
        # required by MultiheadAttention: (B, 1, fusion_dim)
        Q = self.query_proj(vitals_emb).unsqueeze(1)    # (B, 1, fusion_dim)
        K = self.key_proj(tabular_emb).unsqueeze(1)     # (B, 1, fusion_dim)
        V = self.value_proj(tabular_emb).unsqueeze(1)   # (B, 1, fusion_dim)

        # Cross-attention: query from vitals, keys/values from tabular
        attn_out, _ = self.cross_attention(Q, K, V)     # (B, 1, fusion_dim)
        attn_out = attn_out.squeeze(1)                   # (B, fusion_dim)

        # Residual connection + LayerNorm (post-LN style)
        x = self.norm1(Q.squeeze(1) + self.dropout(attn_out))  # (B, fusion_dim)

        # Feed-forward sublayer with residual
        x = self.norm2(x + self.dropout(self.ffn(x)))   # (B, fusion_dim)

        return x
