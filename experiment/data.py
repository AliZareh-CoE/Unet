"""Modular data loading for neural signal translation.

This module provides flexible data loading that works with any data format
where you have two brain regions with shape (trials, channels, samples).

Supported formats:
1. Single file: (trials, total_channels, samples) - split by channel indices
2. Stacked file: (trials, 2, channels, samples) - pre-split into regions
3. Separate files: region1.npy and region2.npy

Usage:
    from data import load_data, create_dataloaders
    from config import DataConfig

    cfg = DataConfig(
        data_path="data.npy",
        region1_channels=(0, 32),
        region2_channels=(32, 64),
        sampling_rate=1000,
    )

    region1, region2, labels = load_data(cfg)
    train_loader, val_loader, test_loader = create_dataloaders(
        region1, region2, labels, cfg
    )
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from config import DataConfig


# =============================================================================
# Normalization
# =============================================================================

@dataclass
class NormStats:
    """Statistics for z-score normalization."""
    mean: np.ndarray
    std: np.ndarray

    def save(self, path: Path) -> None:
        """Save normalization stats to JSON."""
        payload = {"mean": self.mean.tolist(), "std": self.std.tolist()}
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> "NormStats":
        """Load normalization stats from JSON."""
        payload = json.loads(Path(path).read_text())
        return cls(
            mean=np.array(payload["mean"], dtype=np.float32),
            std=np.array(payload["std"], dtype=np.float32),
        )


def compute_norm_stats(data: np.ndarray, axis: Tuple[int, ...] = (0, 2)) -> NormStats:
    """Compute normalization statistics.

    Args:
        data: Input array of shape (trials, channels, samples)
        axis: Axes to compute statistics over (default: trials and samples)

    Returns:
        NormStats with mean and std arrays
    """
    mean = data.mean(axis=axis, keepdims=True).astype(np.float32)
    std = data.std(axis=axis, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)  # Prevent division by zero
    return NormStats(mean=mean, std=std)


def normalize(data: np.ndarray, stats: NormStats) -> np.ndarray:
    """Apply z-score normalization."""
    return ((data - stats.mean) / stats.std).astype(np.float32)


def denormalize(data: np.ndarray, stats: NormStats) -> np.ndarray:
    """Reverse z-score normalization."""
    return (data * stats.std + stats.mean).astype(np.float32)


# =============================================================================
# Data Loading
# =============================================================================

def load_data(
    cfg: DataConfig,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load neural data from files based on configuration.

    Handles multiple data formats:
    1. Single array (trials, total_channels, samples): Split by channel indices
    2. Stacked array (trials, 2, channels, samples): Pre-split regions
    3. Separate files: Load region1.npy and region2.npy independently

    Args:
        cfg: DataConfig with paths and channel specifications

    Returns:
        region1: Array of shape (trials, n_channels_r1, samples)
        region2: Array of shape (trials, n_channels_r2, samples)
        labels: Optional array of labels (trials,) if metadata provided
    """
    data_path = Path(cfg.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load main data file
    data = np.load(data_path).astype(np.float32)
    print(f"Loaded data: {data.shape}")

    # Parse based on format
    if cfg.data_format == "stacked" or (data.ndim == 4 and data.shape[1] == 2):
        # Format: (trials, 2, channels, samples)
        region1 = data[:, 0, :, :]
        region2 = data[:, 1, :, :]
        print(f"Stacked format: region1={region1.shape}, region2={region2.shape}")

    elif cfg.data_format == "separate" and cfg.region2_data_path:
        # Separate files for each region
        region1 = data
        region2_path = Path(cfg.region2_data_path)
        if not region2_path.exists():
            raise FileNotFoundError(f"Region2 data not found: {region2_path}")
        region2 = np.load(region2_path).astype(np.float32)
        print(f"Separate files: region1={region1.shape}, region2={region2.shape}")

    else:
        # Single array: split by channel indices
        # Handle (trials, samples, channels) format - transpose if needed
        if data.ndim == 3:
            # Check if channels are last (common format)
            if data.shape[2] < data.shape[1]:
                # Likely (trials, samples, channels) - transpose
                data = data.transpose(0, 2, 1)
                print(f"Transposed to (trials, channels, samples): {data.shape}")

        c1_start, c1_end = cfg.region1_channels
        c2_start, c2_end = cfg.region2_channels
        region1 = data[:, c1_start:c1_end, :]
        region2 = data[:, c2_start:c2_end, :]
        print(f"Split channels: region1={region1.shape}, region2={region2.shape}")

    # Load labels if metadata provided
    labels = None
    if cfg.meta_path and cfg.label_column:
        meta_path = Path(cfg.meta_path)
        if meta_path.exists():
            df = pd.read_csv(meta_path)
            if cfg.label_column in df.columns:
                # Convert string labels to integer IDs
                label_strs = df[cfg.label_column].astype(str).values
                unique_labels = sorted(set(label_strs))
                label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
                labels = np.array([label_to_id[lbl] for lbl in label_strs], dtype=np.int64)
                print(f"Loaded {len(unique_labels)} unique labels: {unique_labels}")

    return region1, region2, labels


# =============================================================================
# Train/Val/Test Splits
# =============================================================================

def create_splits(
    n_samples: int,
    labels: Optional[np.ndarray] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create train/val/test splits.

    Uses stratified splitting if labels provided, otherwise random.

    Args:
        n_samples: Total number of samples
        labels: Optional labels for stratified splitting
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed

    Returns:
        train_idx, val_idx, test_idx: Index arrays for each split
    """
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_samples)

    if labels is not None:
        # Stratified split
        test_ratio = 1.0 - train_ratio - val_ratio
        splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=1-train_ratio, random_state=seed)
        train_mask, temp_mask = next(splitter1.split(all_idx.reshape(-1, 1), labels))
        train_idx = all_idx[train_mask]
        temp_idx = all_idx[temp_mask]

        # Split remaining into val/test
        temp_labels = labels[temp_idx]
        val_frac = val_ratio / (val_ratio + test_ratio)
        splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=1-val_frac, random_state=seed+1)
        val_mask, test_mask = next(splitter2.split(temp_idx.reshape(-1, 1), temp_labels))
        val_idx = temp_idx[val_mask]
        test_idx = temp_idx[test_mask]
    else:
        # Random split
        rng.shuffle(all_idx)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        train_idx = all_idx[:n_train]
        val_idx = all_idx[n_train:n_train + n_val]
        test_idx = all_idx[n_train + n_val:]

    print(f"Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx


# =============================================================================
# PyTorch Dataset
# =============================================================================

class PairedRegionDataset(Dataset):
    """Dataset for paired neural signals from two brain regions.

    Provides (input, target) pairs where:
    - input: signals from region1 (source)
    - target: signals from region2 (target to predict)

    Supports optional labels for conditional generation.
    """

    def __init__(
        self,
        region1: np.ndarray,
        region2: np.ndarray,
        labels: Optional[np.ndarray] = None,
        normalize: bool = True,
        norm_stats_r1: Optional[NormStats] = None,
        norm_stats_r2: Optional[NormStats] = None,
    ):
        """Initialize dataset.

        Args:
            region1: Source signals (trials, channels, samples)
            region2: Target signals (trials, channels, samples)
            labels: Optional labels (trials,)
            normalize: Whether to apply z-score normalization
            norm_stats_r1: Pre-computed stats for region1 (compute if None)
            norm_stats_r2: Pre-computed stats for region2 (compute if None)
        """
        assert region1.shape[0] == region2.shape[0], "Trial count mismatch"
        assert region1.shape[2] == region2.shape[2], "Sample count mismatch"

        self.n_samples = region1.shape[0]
        self.labels = labels

        # Normalize if requested
        if normalize:
            self.norm_r1 = norm_stats_r1 or compute_norm_stats(region1)
            self.norm_r2 = norm_stats_r2 or compute_norm_stats(region2)
            self.region1 = self._normalize(region1, self.norm_r1)
            self.region2 = self._normalize(region2, self.norm_r2)
        else:
            self.norm_r1 = None
            self.norm_r2 = None
            self.region1 = region1.astype(np.float32)
            self.region2 = region2.astype(np.float32)

    def _normalize(self, data: np.ndarray, stats: NormStats) -> np.ndarray:
        """Apply normalization."""
        return ((data - stats.mean) / stats.std).astype(np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Returns:
            Dictionary with:
            - 'input': Source signal tensor (channels, samples)
            - 'target': Target signal tensor (channels, samples)
            - 'label': Label tensor if labels provided
            - 'idx': Sample index
        """
        sample = {
            "input": torch.from_numpy(self.region1[idx]),
            "target": torch.from_numpy(self.region2[idx]),
            "idx": torch.tensor(idx, dtype=torch.long),
        }
        if self.labels is not None:
            sample["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample


# =============================================================================
# DataLoader Creation
# =============================================================================

def create_dataloaders(
    region1: np.ndarray,
    region2: np.ndarray,
    labels: Optional[np.ndarray],
    cfg: DataConfig,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders.

    Args:
        region1: Source region data (trials, channels, samples)
        region2: Target region data (trials, channels, samples)
        labels: Optional labels for stratified splitting
        cfg: DataConfig with split ratios and seed
        batch_size: Batch size for training
        num_workers: Number of data loading workers

    Returns:
        train_loader, val_loader, test_loader
    """
    # Create splits
    train_idx, val_idx, test_idx = create_splits(
        n_samples=region1.shape[0],
        labels=labels,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )

    # Compute normalization stats from training set only
    norm_r1 = compute_norm_stats(region1[train_idx])
    norm_r2 = compute_norm_stats(region2[train_idx])

    # Create datasets
    train_ds = PairedRegionDataset(
        region1[train_idx], region2[train_idx],
        labels[train_idx] if labels is not None else None,
        normalize=True, norm_stats_r1=norm_r1, norm_stats_r2=norm_r2,
    )
    val_ds = PairedRegionDataset(
        region1[val_idx], region2[val_idx],
        labels[val_idx] if labels is not None else None,
        normalize=True, norm_stats_r1=norm_r1, norm_stats_r2=norm_r2,
    )
    test_ds = PairedRegionDataset(
        region1[test_idx], region2[test_idx],
        labels[test_idx] if labels is not None else None,
        normalize=True, norm_stats_r1=norm_r1, norm_stats_r2=norm_r2,
    )

    # Store norm stats for later use
    train_ds.train_idx = train_idx
    val_ds.val_idx = val_idx
    test_ds.test_idx = test_idx

    # Create dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# Utility Functions
# =============================================================================

def get_data_info(region1: np.ndarray, region2: np.ndarray) -> Dict[str, Any]:
    """Get summary information about the data.

    Args:
        region1: Source region data
        region2: Target region data

    Returns:
        Dictionary with data statistics
    """
    return {
        "n_trials": region1.shape[0],
        "region1_channels": region1.shape[1],
        "region2_channels": region2.shape[1],
        "n_samples": region1.shape[2],
        "region1_mean": float(region1.mean()),
        "region1_std": float(region1.std()),
        "region2_mean": float(region2.mean()),
        "region2_std": float(region2.std()),
    }


def prepare_data(cfg: DataConfig, batch_size: int = 8, num_workers: int = 4):
    """Convenience function to load data and create dataloaders.

    Args:
        cfg: DataConfig instance
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        train_loader, val_loader, test_loader, data_info
    """
    region1, region2, labels = load_data(cfg)
    loaders = create_dataloaders(region1, region2, labels, cfg, batch_size, num_workers)
    info = get_data_info(region1, region2)
    return (*loaders, info)
