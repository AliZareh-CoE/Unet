"""
5-Fold Cross-Validation Utilities for Nature Methods Publication.

Provides:
- Stratified 5-fold CV with holdout test set
- Data splitting utilities
- CV result aggregation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass
class CVSplit:
    """Represents a single CV fold split."""

    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray

    def __repr__(self) -> str:
        return f"CVSplit(fold={self.fold_idx}, train={len(self.train_indices)}, val={len(self.val_indices)})"


@dataclass
class DataSplits:
    """Complete data split configuration with holdout test set."""

    cv_splits: List[CVSplit]
    holdout_indices: np.ndarray
    n_samples: int
    n_folds: int = 5
    holdout_fraction: float = 0.2
    seed: int = 42

    def __repr__(self) -> str:
        return (
            f"DataSplits(n_samples={self.n_samples}, n_folds={self.n_folds}, "
            f"holdout={len(self.holdout_indices)}, cv_pool={self.n_samples - len(self.holdout_indices)})"
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save splits to JSON for reproducibility."""
        path = Path(path)
        data = {
            "n_samples": self.n_samples,
            "n_folds": self.n_folds,
            "holdout_fraction": self.holdout_fraction,
            "seed": self.seed,
            "holdout_indices": self.holdout_indices.tolist(),
            "cv_splits": [
                {
                    "fold_idx": s.fold_idx,
                    "train_indices": s.train_indices.tolist(),
                    "val_indices": s.val_indices.tolist(),
                }
                for s in self.cv_splits
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DataSplits":
        """Load splits from JSON."""
        with open(path) as f:
            data = json.load(f)

        cv_splits = [
            CVSplit(
                fold_idx=s["fold_idx"],
                train_indices=np.array(s["train_indices"]),
                val_indices=np.array(s["val_indices"]),
            )
            for s in data["cv_splits"]
        ]

        return cls(
            cv_splits=cv_splits,
            holdout_indices=np.array(data["holdout_indices"]),
            n_samples=data["n_samples"],
            n_folds=data["n_folds"],
            holdout_fraction=data["holdout_fraction"],
            seed=data["seed"],
        )


def create_data_splits(
    n_samples: int,
    n_folds: int = 5,
    holdout_fraction: float = 0.2,
    seed: int = 42,
    stratify_labels: Optional[np.ndarray] = None,
) -> DataSplits:
    """Create 5-fold CV splits with holdout test set.

    The holdout test set is NEVER used until final validation (Tier 4).
    This ensures no test set contamination during HPO.

    Args:
        n_samples: Total number of samples
        n_folds: Number of CV folds (default 5)
        holdout_fraction: Fraction for holdout test set (default 0.2)
        seed: Random seed for reproducibility
        stratify_labels: Optional labels for stratified splitting

    Returns:
        DataSplits object with all indices
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)

    if stratify_labels is not None:
        # Stratified splitting
        from sklearn.model_selection import StratifiedKFold, train_test_split

        # First split: CV pool vs holdout
        cv_indices, holdout_indices = train_test_split(
            indices,
            test_size=holdout_fraction,
            random_state=seed,
            stratify=stratify_labels,
        )
        cv_labels = stratify_labels[cv_indices]

        # Create stratified folds
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        cv_splits = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(cv_indices, cv_labels)):
            cv_splits.append(
                CVSplit(
                    fold_idx=fold_idx,
                    train_indices=cv_indices[train_idx],
                    val_indices=cv_indices[val_idx],
                )
            )
    else:
        # Random splitting
        rng.shuffle(indices)

        # Split into holdout and CV pool
        holdout_size = int(n_samples * holdout_fraction)
        holdout_indices = indices[:holdout_size]
        cv_indices = indices[holdout_size:]

        # Create k-fold splits from CV pool
        fold_size = len(cv_indices) // n_folds
        cv_splits = []

        for fold_idx in range(n_folds):
            val_start = fold_idx * fold_size
            val_end = val_start + fold_size if fold_idx < n_folds - 1 else len(cv_indices)

            val_indices = cv_indices[val_start:val_end]
            train_indices = np.concatenate([cv_indices[:val_start], cv_indices[val_end:]])

            cv_splits.append(
                CVSplit(
                    fold_idx=fold_idx,
                    train_indices=train_indices,
                    val_indices=val_indices,
                )
            )

    return DataSplits(
        cv_splits=cv_splits,
        holdout_indices=holdout_indices,
        n_samples=n_samples,
        n_folds=n_folds,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )


def get_fold_dataloaders(
    dataset: Dataset,
    split: CVSplit,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders for a single fold.

    Args:
        dataset: Full dataset
        split: CV split with indices
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Pin memory for GPU transfer

    Returns:
        (train_loader, val_loader)
    """
    train_subset = Subset(dataset, split.train_indices.tolist())
    val_subset = Subset(dataset, split.val_indices.tolist())

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def get_holdout_dataloader(
    dataset: Dataset,
    splits: DataSplits,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create dataloader for holdout test set.

    WARNING: Only call this in Tier 4 (final validation)!

    Args:
        dataset: Full dataset
        splits: DataSplits object
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Pin memory for GPU transfer

    Returns:
        Holdout test dataloader
    """
    holdout_subset = Subset(dataset, splits.holdout_indices.tolist())

    return DataLoader(
        holdout_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def iterate_folds(
    dataset: Dataset,
    splits: DataSplits,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Generator[Tuple[int, DataLoader, DataLoader], None, None]:
    """Iterate through all CV folds, yielding dataloaders.

    Args:
        dataset: Full dataset
        splits: DataSplits object
        batch_size: Batch size
        num_workers: Number of workers

    Yields:
        (fold_idx, train_loader, val_loader) for each fold
    """
    for split in splits.cv_splits:
        train_loader, val_loader = get_fold_dataloaders(
            dataset=dataset,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        yield split.fold_idx, train_loader, val_loader


@dataclass
class CVResult:
    """Result from a single CV fold."""

    fold_idx: int
    seed: int
    metrics: Dict[str, float]
    config: Optional[Dict[str, Any]] = None


@dataclass
class CVAggregatedResult:
    """Aggregated results across all folds and seeds."""

    fold_results: List[CVResult] = field(default_factory=list)
    config: Optional[Dict[str, Any]] = None

    def add_result(self, result: CVResult) -> None:
        """Add a fold result."""
        self.fold_results.append(result)

    @property
    def n_samples(self) -> int:
        """Total number of samples (folds × seeds)."""
        return len(self.fold_results)

    def get_metric_values(self, metric_name: str) -> np.ndarray:
        """Get all values for a specific metric."""
        return np.array([r.metrics.get(metric_name, np.nan) for r in self.fold_results])

    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric across all folds/seeds."""
        values = self.get_metric_values(metric_name)
        valid_values = values[~np.isnan(values)]

        if len(valid_values) == 0:
            return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "n": 0}

        return {
            "mean": float(np.mean(valid_values)),
            "std": float(np.std(valid_values, ddof=1)) if len(valid_values) > 1 else 0.0,
            "min": float(np.min(valid_values)),
            "max": float(np.max(valid_values)),
            "n": len(valid_values),
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        if not self.fold_results:
            return {}

        all_metric_names = set()
        for result in self.fold_results:
            all_metric_names.update(result.metrics.keys())

        return {name: self.get_metric_stats(name) for name in sorted(all_metric_names)}

    def summary(self) -> str:
        """Get human-readable summary."""
        stats = self.get_all_stats()
        lines = [f"CV Results ({self.n_samples} samples):"]
        for name, stat in stats.items():
            lines.append(f"  {name}: {stat['mean']:.4f} +/- {stat['std']:.4f}")
        return "\n".join(lines)


def aggregate_cv_results(
    results: List[Dict[str, float]],
    folds: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
) -> CVAggregatedResult:
    """Aggregate results from multiple CV runs.

    Args:
        results: List of metric dictionaries
        folds: Optional fold indices (for tracking)
        seeds: Optional seed values (for tracking)

    Returns:
        Aggregated result object
    """
    agg = CVAggregatedResult()

    for i, metrics in enumerate(results):
        fold_idx = folds[i] if folds else i % 5
        seed = seeds[i] if seeds else 42 + i // 5

        agg.add_result(CVResult(fold_idx=fold_idx, seed=seed, metrics=metrics))

    return agg


def run_cv_experiment(
    train_fn,
    dataset: Dataset,
    splits: DataSplits,
    n_seeds: int = 5,
    batch_size: int = 16,
    base_seed: int = 42,
    **train_kwargs,
) -> CVAggregatedResult:
    """Run a complete 5-fold CV experiment with multiple seeds.

    Args:
        train_fn: Function that takes (train_loader, val_loader, seed, **kwargs)
                  and returns Dict[str, float] of metrics
        dataset: Full dataset
        splits: DataSplits object
        n_seeds: Number of random seeds per fold
        batch_size: Batch size
        base_seed: Base seed for reproducibility
        **train_kwargs: Additional arguments to pass to train_fn

    Returns:
        Aggregated results across all folds and seeds
    """
    agg = CVAggregatedResult()

    for fold_idx, train_loader, val_loader in iterate_folds(
        dataset=dataset,
        splits=splits,
        batch_size=batch_size,
    ):
        for seed_offset in range(n_seeds):
            seed = base_seed + seed_offset

            # Run training
            metrics = train_fn(
                train_loader=train_loader,
                val_loader=val_loader,
                seed=seed,
                **train_kwargs,
            )

            agg.add_result(CVResult(fold_idx=fold_idx, seed=seed, metrics=metrics))

    return agg
