#!/usr/bin/env python
"""Diagnostic script to investigate R² gap between overall and per-session metrics.

This script analyzes the data identity between combined validation loader
and per-session validation loader to understand the metric discrepancy.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import prepare_data, create_dataloaders


def main():
    print("=" * 70)
    print("R² Gap Investigation - Data Identity Analysis")
    print("=" * 70)

    # Prepare data with explicit session split (mimicking LOSO fold 1)
    print("\n1. Loading data with separate_val_sessions=True...")
    data = prepare_data(
        split_by_session=True,
        val_sessions=["160819"],  # Single validation session (LOSO-style)
        test_sessions=[],
        no_test_set=True,
        separate_val_sessions=True,
        force_recreate_splits=True,
    )

    # Check indices
    val_idx = data["val_idx"]
    val_idx_per_session = data.get("val_idx_per_session", {})

    print(f"\n2. Checking indices...")
    print(f"   val_idx shape: {val_idx.shape}")
    print(f"   val_idx_per_session keys: {list(val_idx_per_session.keys())}")

    if "160819" in val_idx_per_session:
        per_session_idx = val_idx_per_session["160819"]
        print(f"   per_session_idx shape: {per_session_idx.shape}")

        # Check if indices are identical
        indices_match = np.array_equal(val_idx, per_session_idx)
        print(f"\n3. Index comparison:")
        print(f"   Indices are identical: {indices_match}")

        if not indices_match:
            print(f"   val_idx first 10: {val_idx[:10]}")
            print(f"   per_session_idx first 10: {per_session_idx[:10]}")

            # Check overlap
            val_set = set(val_idx.tolist())
            per_session_set = set(per_session_idx.tolist())
            overlap = len(val_set & per_session_set)
            print(f"   Overlap: {overlap}/{len(val_idx)} ({100*overlap/len(val_idx):.1f}%)")
    else:
        print(f"   WARNING: Session 160819 not found in val_idx_per_session!")

    # Create dataloaders
    print(f"\n4. Creating dataloaders (distributed=False)...")
    loaders = create_dataloaders(
        data,
        batch_size=64,
        num_workers=0,
        distributed=False,
    )

    val_loader = loaders["val"]
    val_sessions = loaders.get("val_sessions", {})

    print(f"   Combined val loader size: {len(val_loader.dataset)}")
    if "160819" in val_sessions:
        sess_loader = val_sessions["160819"]
        print(f"   Per-session loader size: {len(sess_loader.dataset)}")

    # Extract all data from both loaders
    print(f"\n5. Extracting data from loaders...")

    val_data_list = []
    for batch in val_loader:
        ob, pcx, odor, *rest = batch
        val_data_list.append((ob, pcx))

    sess_data_list = []
    if "160819" in val_sessions:
        for batch in val_sessions["160819"]:
            ob, pcx, odor, *rest = batch
            sess_data_list.append((ob, pcx))

    # Concatenate all data
    val_ob = torch.cat([b[0] for b in val_data_list])
    val_pcx = torch.cat([b[1] for b in val_data_list])
    sess_ob = torch.cat([b[0] for b in sess_data_list])
    sess_pcx = torch.cat([b[1] for b in sess_data_list])

    print(f"   Combined val: ob={val_ob.shape}, pcx={val_pcx.shape}")
    print(f"   Per-session: ob={sess_ob.shape}, pcx={sess_pcx.shape}")

    # Check if data is identical
    print(f"\n6. Data comparison:")
    ob_match = torch.allclose(val_ob, sess_ob, rtol=1e-5, atol=1e-6)
    pcx_match = torch.allclose(val_pcx, sess_pcx, rtol=1e-5, atol=1e-6)
    print(f"   OB data identical: {ob_match}")
    print(f"   PCX data identical: {pcx_match}")

    if not ob_match or not pcx_match:
        # Check element-wise differences
        ob_diff = (val_ob - sess_ob).abs()
        pcx_diff = (val_pcx - sess_pcx).abs()
        print(f"   OB max diff: {ob_diff.max().item():.6f}")
        print(f"   PCX max diff: {pcx_diff.max().item():.6f}")

        # Check if it's a reordering issue
        print(f"\n   Checking for reordering...")
        # Sort both by first element value and compare
        val_ob_sorted, val_order = val_ob.view(val_ob.shape[0], -1)[:, 0].sort()
        sess_ob_sorted, sess_order = sess_ob.view(sess_ob.shape[0], -1)[:, 0].sort()

        if torch.allclose(val_ob_sorted, sess_ob_sorted, rtol=1e-5, atol=1e-6):
            print(f"   Data is the SAME but REORDERED!")
            print(f"   val_order[:10]: {val_order[:10].tolist()}")
            print(f"   sess_order[:10]: {sess_order[:10].tolist()}")

    # Compute metrics on both datasets
    print(f"\n7. Computing metrics...")

    def compute_metrics(ob, pcx):
        """Compute correlation and R² on tensors."""
        # Flatten for overall computation
        pred_flat = ob.flatten()
        target_flat = pcx.flatten()

        # Correlation
        pred_centered = pred_flat - pred_flat.mean()
        target_centered = target_flat - target_flat.mean()
        corr = (pred_centered * target_centered).sum() / (
            torch.sqrt((pred_centered**2).sum() * (target_centered**2).sum()) + 1e-8
        )

        # R² (explained variance)
        mse = ((pred_flat - target_flat) ** 2).mean()
        var = target_flat.var()
        r2 = 1.0 - mse / (var + 1e-12)

        return corr.item(), r2.item()

    val_corr, val_r2 = compute_metrics(val_ob, val_pcx)
    sess_corr, sess_r2 = compute_metrics(sess_ob, sess_pcx)

    print(f"   Combined val: corr={val_corr:.4f}, r²={val_r2:.4f}")
    print(f"   Per-session:  corr={sess_corr:.4f}, r²={sess_r2:.4f}")

    # Now simulate per-batch averaging (like the training code does)
    print(f"\n8. Computing per-batch averaged metrics (like training code)...")

    def compute_per_batch_metrics(loader):
        """Compute metrics by averaging across batches."""
        corr_list = []
        r2_list = []

        for batch in loader:
            ob, pcx, *_ = batch

            # Correlation (batch-averaged, channel-averaged, time-series)
            pred_centered = ob - ob.mean(dim=-1, keepdim=True)
            target_centered = pcx - pcx.mean(dim=-1, keepdim=True)
            num = (pred_centered * target_centered).sum(dim=-1)
            denom = torch.sqrt((pred_centered**2).sum(dim=-1) * (target_centered**2).sum(dim=-1) + 1e-8)
            batch_corr = (num / denom).mean().item()
            corr_list.append(batch_corr)

            # R² (explained variance)
            mse = ((ob - pcx) ** 2).mean()
            var = pcx.var()
            batch_r2 = (1.0 - mse / (var + 1e-12)).item()
            r2_list.append(batch_r2)

        return np.mean(corr_list), np.mean(r2_list)

    val_corr_batch, val_r2_batch = compute_per_batch_metrics(val_loader)
    if "160819" in val_sessions:
        sess_corr_batch, sess_r2_batch = compute_per_batch_metrics(val_sessions["160819"])
    else:
        sess_corr_batch, sess_r2_batch = 0, 0

    print(f"   Combined val (batch-avg): corr={val_corr_batch:.4f}, r²={val_r2_batch:.4f}")
    print(f"   Per-session (batch-avg):  corr={sess_corr_batch:.4f}, r²={sess_r2_batch:.4f}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if ob_match and pcx_match:
        print("✓ Data is IDENTICAL between combined and per-session loaders")
        if abs(val_r2_batch - sess_r2_batch) < 0.001:
            print("✓ Metrics are consistent")
            print("  The gap you observed must be from distributed training or model state")
        else:
            print("✗ But per-batch averaging produces different results!")
            print("  This is a mathematical artifact of averaging R² across batches")
    else:
        print("✗ Data DIFFERS between combined and per-session loaders!")
        print("  This is the root cause of the metric gap")


if __name__ == "__main__":
    main()
