"""
multimodal_model.py
-------------------
PyTorch Lightning module assembling the full multimodal ICU outcome predictor.

Full data flow
--------------
    vitals  (B, 24, 5)  ──► LSTMEncoder            ──► vitals_emb  (B, vitals_dim)
                                                               │
    tabular (B, 156)    ──► TabularEncoder          ──► tabular_emb (B, tabular_dim)
                                                               │
                              CrossModalAttentionFusion ──► fused (B, fusion_dim)
                                                               │
                              MultiTaskHead
                                ├── LOS private encoder  ──► los_logits (B, 2)
                                └── Mort private encoder ──► mortality_logit (B, 1)

Combined loss
-------------
    L = alpha * L_ordinal_LOS + (1 - alpha) * L_binary_mortality
      + 0.15 * band1_recall_penalty
      + 0.10 * ordinal_consistency_loss

where L_ordinal_LOS is the mean of two per-boundary BCE losses (K-1 decomp)
and L_binary_mortality is BCEWithLogitsLoss with pos_weight.

PyTorch Lightning compatibility
-------------------------------
Written for PL v2+.  Outputs are accumulated into instance-level lists
(_val_outputs, _test_outputs) that are cleared at the end of each epoch.
All accumulated tensors are moved to CPU immediately after each step to avoid
GPU memory fragmentation during long validation runs.

NOTE: losses.py and metrics.py (src/training/) must exist before training.
They expose:
    CombinedMultiTaskLoss  — from src.training.losses
    compute_los_metrics    — from src.training.metrics
    compute_mortality_metrics — from src.training.metrics
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import pytorch_lightning as pl

from src.models.lstm_encoder import LSTMEncoder
from src.models.tabular_encoder import TabularEncoder
from src.models.fusion import CrossModalAttentionFusion
from src.models.multitask_head import MultiTaskHead
from src.training.losses import CombinedMultiTaskLoss
from src.training.metrics import compute_los_metrics, compute_mortality_metrics


class MultimodalICUModel(pl.LightningModule):
    """
    Full multimodal ICU outcome prediction model.

    Parameters
    ----------
    config : dict
        Flat dictionary of hyperparameters.  All keys are optional;
        unset keys fall back to the defaults defined in _DEFAULTS below.
        Designed to be populated from base_config.yaml or a W&B / Optuna
        sweep config.

    Config keys
    -----------
    lstm_hidden_size        int     default 128
    lstm_num_layers         int     default 2
    lstm_dropout            float   default 0.3
    lstm_bidirectional      bool    default True

    tabular_input_dim       int     default 156
    tabular_hidden_dims     list    default [512, 256, 128]
    tabular_output_dim      int     default 256
    tabular_dropout         float   default 0.3
    tabular_use_batch_norm  bool    default True

    fusion_dim              int     default 256
    fusion_num_heads        int     default 8
    fusion_dropout          float   default 0.1
    fusion_ffn_expansion    int     default 2

    private_hidden_dim      int     default 128
    private_output_dim      int     default 128
    head_dropout            float   default 0.2

    alpha                   float   default 0.65
    los_boundary_weights    list|None   default None  (injected from dataset)
    mortality_pos_weight    float|None  default None  (injected from dataset)

    lr                      float   default 2e-4
    weight_decay            float   default 1e-4

    band2_threshold         float   default 0.5  (updated after calibration)
    mortality_threshold     float   default 0.5  (updated after calibration)
    """

    _DEFAULTS: Dict[str, Any] = {
        # LSTM encoder
        "lstm_hidden_size":       128,
        "lstm_num_layers":        2,
        "lstm_dropout":           0.3,
        "lstm_bidirectional":     True,          # FIXED: was False
        # Tabular encoder
        "tabular_input_dim":      156,
        "tabular_hidden_dims":    [512, 256, 128],  # FIXED: was [256, 128]
        "tabular_output_dim":     256,           # FIXED: was 128
        "tabular_dropout":        0.3,
        "tabular_use_batch_norm": True,
        # Fusion
        "fusion_dim":             256,           # FIXED: was 128
        "fusion_num_heads":       8,             # FIXED: was 4
        "fusion_dropout":         0.1,
        "fusion_ffn_expansion":   2,
        # Multi-task head
        "private_hidden_dim":     128,           # FIXED: was 64
        "private_output_dim":     128,           # FIXED: was 64
        "head_dropout":           0.2,
        # Loss
        "alpha":                  0.65,          # FIXED: was 0.5
        "los_boundary_weights":   None,
        "mortality_pos_weight":   None,
        # Optimiser
        "lr":                     2e-4,          # FIXED: was 1e-3
        "weight_decay":           1e-4,
        # Calibration thresholds (overwritten externally after Platt scaling)
        "band2_threshold":        0.5,
        "mortality_threshold":    0.5,
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        # Merge caller config over defaults
        cfg: Dict[str, Any] = {**self._DEFAULTS, **config}

        # Store scalars needed outside __init__
        self.alpha               = cfg["alpha"]
        self.lr                  = cfg["lr"]
        self.weight_decay        = cfg["weight_decay"]
        self.band2_threshold     = cfg["band2_threshold"]
        self.mortality_threshold = cfg["mortality_threshold"]

        # ------------------------------------------------------------------
        # Model components
        # ------------------------------------------------------------------
        self.lstm_encoder = LSTMEncoder(
            input_size=5,
            hidden_size=cfg["lstm_hidden_size"],
            num_layers=cfg["lstm_num_layers"],
            dropout=cfg["lstm_dropout"],
            bidirectional=cfg["lstm_bidirectional"],
        )

        self.tabular_encoder = TabularEncoder(
            input_dim=cfg["tabular_input_dim"],
            hidden_dims=cfg["tabular_hidden_dims"],
            output_dim=cfg["tabular_output_dim"],
            dropout=cfg["tabular_dropout"],
            use_batch_norm=cfg["tabular_use_batch_norm"],
        )

        self.fusion = CrossModalAttentionFusion(
            vitals_dim=self.lstm_encoder.output_dim,
            tabular_dim=self.tabular_encoder.output_dim,
            fusion_dim=cfg["fusion_dim"],
            num_heads=cfg["fusion_num_heads"],
            dropout=cfg["fusion_dropout"],
            ffn_expansion=cfg["fusion_ffn_expansion"],
        )

        self.head = MultiTaskHead(
            fusion_dim=self.fusion.output_dim,
            private_hidden_dim=cfg["private_hidden_dim"],
            private_output_dim=cfg["private_output_dim"],
            dropout=cfg["head_dropout"],
        )

        # ------------------------------------------------------------------
        # Loss function
        # ------------------------------------------------------------------
        bw = cfg["los_boundary_weights"]
        los_bw_tensor = (
            torch.tensor(bw, dtype=torch.float32) if bw is not None else None
        )

        self.criterion = CombinedMultiTaskLoss(
            alpha=self.alpha,
            los_boundary_weights=los_bw_tensor,
            mortality_pos_weight=cfg["mortality_pos_weight"],
        )

        # ------------------------------------------------------------------
        # Output accumulators for epoch-end metric computation (PL v2+)
        # Tensors stored here are always on CPU (.detach().cpu() in each step)
        # ------------------------------------------------------------------
        self._val_outputs:  List[Dict[str, torch.Tensor]] = []
        self._test_outputs: List[Dict[str, torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        vitals: torch.Tensor,
        tabular: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        vitals  : Tensor  (B, 24, 5)
        tabular : Tensor  (B, 156)

        Returns
        -------
        los_logits      : Tensor  (B, 2)
        mortality_logit : Tensor  (B, 1)
        """
        vitals_emb  = self.lstm_encoder(vitals)           # (B, vitals_dim)
        tabular_emb = self.tabular_encoder(tabular)       # (B, tabular_dim)
        fused       = self.fusion(vitals_emb, tabular_emb)  # (B, fusion_dim)
        los_logits, mortality_logit = self.head(fused)
        return los_logits, mortality_logit

    # ------------------------------------------------------------------
    # Shared step (used by training / val / test)
    # ------------------------------------------------------------------

    def _shared_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str,
    ) -> Dict[str, torch.Tensor]:
        vitals           = batch["vitals"]            # (B, 24, 5)
        tabular          = batch["tabular"]           # (B, 156)
        los_labels       = batch["los_label"]         # (B,) long
        mortality_labels = batch["mortality_label"]   # (B,) float32

        los_logits, mortality_logit = self(vitals, tabular)

        total_loss, los_loss, mort_loss = self.criterion(
            los_logits, mortality_logit, los_labels, mortality_labels
        )

        # Log losses — on_epoch=True so PL averages across all batches
        self.log(f"{stage}/loss",      total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/los_loss",  los_loss,   on_step=False, on_epoch=True)
        self.log(f"{stage}/mort_loss", mort_loss,  on_step=False, on_epoch=True)

        return {
            "loss":             total_loss,
            # Move to CPU immediately to avoid GPU fragmentation over many batches
            "los_logits":       los_logits.detach().cpu(),
            "mortality_logit":  mortality_logit.detach().cpu(),
            "los_labels":       los_labels.detach().cpu(),
            "mortality_labels": mortality_labels.detach().cpu(),
        }

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        out = self._shared_step(batch, "train")
        return out["loss"]   # PL uses this for backprop

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        out = self._shared_step(batch, "val")
        # Exclude the loss tensor (it stays on device for PL's internal logging)
        self._val_outputs.append({k: v for k, v in out.items() if k != "loss"})

    def on_validation_epoch_end(self) -> None:
        self._compute_and_log_epoch_metrics(self._val_outputs, stage="val")
        self._val_outputs.clear()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        out = self._shared_step(batch, "test")
        self._test_outputs.append({k: v for k, v in out.items() if k != "loss"})

    def on_test_epoch_end(self) -> None:
        self._compute_and_log_epoch_metrics(self._test_outputs, stage="test")
        self._test_outputs.clear()

    # ------------------------------------------------------------------
    # Epoch-level metric computation
    # ------------------------------------------------------------------

    def _compute_and_log_epoch_metrics(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        stage: str,
    ) -> None:
        """
        Concatenate all batch-level tensors accumulated during the epoch
        and compute AUROC, F1, per-class accuracy, etc. for both tasks.
        """
        if not outputs:
            return

        los_logits       = torch.cat([o["los_logits"]       for o in outputs], dim=0)
        mortality_logit  = torch.cat([o["mortality_logit"]  for o in outputs], dim=0)
        los_labels       = torch.cat([o["los_labels"]       for o in outputs], dim=0)
        mortality_labels = torch.cat([o["mortality_labels"] for o in outputs], dim=0)

        los_metrics  = compute_los_metrics(
            los_logits,
            los_labels,
            band2_threshold=self.band2_threshold,
        )
        mort_metrics = compute_mortality_metrics(
            mortality_logit,
            mortality_labels,
            threshold=self.mortality_threshold,
        )

        # ------------------------------------------------------------------
        # Composite validation metric — used by ModelCheckpoint and scheduler
        # composite = (Band2 AUROC + Mortality AUROC + Band1 Recall) / 3
        # Only logged on val so it doesn't pollute test reporting
        # ------------------------------------------------------------------
        if stage == "val":
            los_band2_auroc = los_metrics.get("los_band2_auroc", 0.0)
            mort_auroc      = mort_metrics.get("mort_auroc", 0.0)

            # Derive per-band probabilities from boundary logits
            probs  = torch.sigmoid(los_logits)
            p_ge1  = probs[:, 0]   # P(LOS >= Band 1)
            p_ge2  = probs[:, 1]   # P(LOS >= Band 2)

            # Predict band using current calibrated thresholds
            pred_bands = torch.zeros(len(los_labels), dtype=torch.long)
            pred_bands[p_ge1 > 0.5] = 1
            pred_bands[p_ge2 > self.band2_threshold] = 2

            true_band1_mask = (los_labels == 1)
            if true_band1_mask.sum() > 0:
                band1_recall = (pred_bands[true_band1_mask] == 1).float().mean().item()
            else:
                band1_recall = 0.0

            composite = (los_band2_auroc + mort_auroc + band1_recall) / 3.0
            self.log("val/composite",          composite,    prog_bar=True)
            self.log("val/band1_recall_epoch", band1_recall, prog_bar=True)

        for metric_name, value in los_metrics.items():
            self.log(
                f"{stage}/{metric_name}",
                value,
                prog_bar=(metric_name in {"los_macro_f1", "los_band2_auroc"}),
            )

        for metric_name, value in mort_metrics.items():
            self.log(
                f"{stage}/{metric_name}",
                value,
                prog_bar=(metric_name in {"mort_auroc", "mort_f1"}),
            )

    # ------------------------------------------------------------------
    # Optimiser and LR scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",       # FIXED: was "min" — now tracking val/composite (higher = better)
            factor=0.5,
            patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor":   "val/composite",  # FIXED: was "val/loss"
                "interval":  "epoch",
                "frequency": 1,
            },
        }
