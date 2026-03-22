import pickle
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


class ICUDataset(Dataset):
    """
    Index-addressable PyTorch Dataset for a single data split.
    Parameters
    ----------
    pkl_path : str or Path
    split : str  -- one of train / val / test
    """

    def __init__(self, pkl_path: str, split: str) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"split must be train/val/test, got {split!r}"
            )
        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            raise FileNotFoundError(f"Pickle not found: {pkl_path}")
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)

        # Resolve indices for this split
        split_key = f"{split}_idx"
        if split_key in data:
            idx = np.asarray(data[split_key], dtype=np.int64)
        elif "split" in data:
            idx = np.where(np.array(data["split"]) == split)[0].astype(np.int64)
        else:
            raise KeyError("Pickle must have <split>_idx arrays or a split string array.")

        # Core tensors
        self.vitals           = torch.tensor(data["vitals"][idx],           dtype=torch.float32)
        self.tabular          = torch.tensor(data["tabular"][idx],          dtype=torch.float32)
        self.los_labels       = torch.tensor(data["los_labels"][idx],       dtype=torch.long)
        self.mortality_labels = torch.tensor(data["mortality_labels"][idx], dtype=torch.float32)

        # sample_weights -- two possible shapes in the pickle:
        #   shape (N=15000,) : full dataset, must be subsetted with idx
        #   shape (n_split,) : already subsetted during preprocessing,
        #                      do NOT index again -- causes IndexError
        # sample_weights is only used by WeightedRandomSampler in the
        # train DataLoader. Val/test get None regardless.
        if split == "train" and "sample_weights" in data:
            sw = np.asarray(data["sample_weights"])
            # sw may be shape (N_full,) or shape (n_train,)
            if sw.shape[0] == len(idx):
                # already subsetted to train rows — use directly
                self.sample_weights = torch.tensor(sw, dtype=torch.float64)
            else:
                # full-length array — subset to train rows
                self.sample_weights = torch.tensor(sw[idx], dtype=torch.float64)
        else:
            self.sample_weights = None

        # Metadata
        self.feature_names        = data.get("feature_names",        None)
        self.vital_names          = data.get("vital_names",           None)
        self.los_boundary_weights = data.get("los_boundary_weights",  None)
        self.mortality_pos_weight = data.get("mortality_pos_weight",  None)
        self.stats                = data.get("stats",                 None)
        self.label_encoders       = data.get("label_encoders",        None)
        self.icustay_ids = data["icustay_ids"][idx] if "icustay_ids" in data else None
        self.split = split
        self._n    = len(idx)

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return {
            "vitals":          self.vitals[idx],
            "tabular":         self.tabular[idx],
            "los_label":       self.los_labels[idx],
            "mortality_label": self.mortality_labels[idx],
        }

    @property
    def n_tabular_features(self):
        return self.tabular.shape[-1]

    @property
    def n_vital_channels(self):
        return self.vitals.shape[-1]

    @property
    def n_timesteps(self):
        return self.vitals.shape[-2]
