"""
dataloader.py
-------------
Factory functions that wrap ICUDataset in PyTorch DataLoaders.

Train DataLoader
    Uses WeightedRandomSampler built from the sample_weights stored in the
    pickle file.  This oversamples minority LOS bands (Band 0 and Band 2)
    so they contribute proportionally to gradient updates each epoch.
    drop_last=True ensures every training batch is full-sized.

Val / Test DataLoaders
    Sequential sampling, no shuffling, no drop_last — all rows are seen
    exactly once per evaluation pass.
"""

from typing import Tuple

from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data_load.dataset import ICUDataset


def get_dataloader(
    pkl_path: str,
    split: str,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Build and return a DataLoader for a single split.

    Parameters
    ----------
    pkl_path : str
        Path to new2.pkl.
    split : str
        One of 'train', 'val', 'test'.
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Subprocesses for data loading (0 = load in the main process; safe for Colab).
    pin_memory : bool
        Set True when training on GPU to speed up host-to-device transfers.

    Returns
    -------
    DataLoader
    """
    dataset = ICUDataset(pkl_path=pkl_path, split=split)

    if split == "train" and dataset.sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=dataset.sample_weights,
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle   = False   # mutually exclusive with sampler
        drop_last = True    # keep all training batches full-sized
    else:
        sampler   = None
        shuffle   = False   # sequential order for reproducible val / test evaluation
        drop_last = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_all_dataloaders(
    pkl_path: str,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience wrapper — returns (train_loader, val_loader, test_loader).

    Parameters
    ----------
    pkl_path : str
        Path to new2.pkl.
    batch_size : int
    num_workers : int
    pin_memory : bool

    Returns
    -------
    Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = get_dataloader(pkl_path, "train", batch_size, num_workers, pin_memory)
    val_loader   = get_dataloader(pkl_path, "val",   batch_size, num_workers, pin_memory)
    test_loader  = get_dataloader(pkl_path, "test",  batch_size, num_workers, pin_memory)
    return train_loader, val_loader, test_loader
