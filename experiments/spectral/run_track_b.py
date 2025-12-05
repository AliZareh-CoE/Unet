#!/usr/bin/env python3
"""
Track B Launcher: SpectralShift Experiments
============================================

Runs HPO for spectral correction experiments.
Stage 1 model is FROZEN - only SpectralShift parameters are optimized.

Usage:
    # Phase 1: Coarse screening (all approaches)
    python experiments/spectral/run_track_b.py --phase 1 --trials-per-approach 32 \
        --stage1-checkpoint artifacts/checkpoints/best_stage1.pt

    # Phase 2: Deep dive on top 3
    python experiments/spectral/run_track_b.py --phase 2 \
        --approaches phase_preserving adaptive_gated wavelet --trials-per-approach 96 \
        --stage1-checkpoint artifacts/checkpoints/best_stage1.pt

    # Dry-run test
    python experiments/spectral/run_track_b.py --dry-run --approach phase_preserving \
        --stage1-checkpoint artifacts/checkpoints/best_stage1.pt

    # Export results
    python experiments/spectral/run_track_b.py --export
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    import optuna
    from optuna.pruners import HyperbandPruner
except ImportError:
    print("Error: optuna not installed. Run: pip install optuna")
    sys.exit(1)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.spectral.hp_spaces import TRACK_B_APPROACHES, get_hp_space
from experiments.spectral.objective import (
    track_b_objective,
    load_frozen_stage1,
    compute_stage1_metrics,
)
from experiments.common.baseline_sampler import BaselineInjectingSampler
from experiments.common.reporting import export_study_results

from data import prepare_data, create_dataloaders


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = Path("artifacts/optuna_spectral")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    """Set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_trial_seed(base_seed: int, trial_number: int) -> int:
    """Generate deterministic seed for specific trial."""
    seed_str = f"{base_seed}_trial_{trial_number}"
    return int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**31)


# =============================================================================
# Main Runner
# =============================================================================

def run_approach(
    approach: str,
    device: torch.device,
    data_loaders,
    stage1_checkpoint: str,
    stage1_metrics: Optional[Dict[str, float]] = None,
    n_trials: int = 32,
    n_baseline: int = 20,
    phase: int = 1,
    seed: int = 42,
) -> optuna.Study:
    """Run HPO for a single spectral approach.

    Args:
        approach: Spectral approach name
        device: Target device
        data_loaders: (train_loader, val_loader)
        stage1_checkpoint: Path to frozen Stage 1 checkpoint
        stage1_metrics: Pre-computed Stage 1 metrics
        n_trials: Number of trials
        n_baseline: Number of baseline trials
        phase: HPO phase
        seed: Base random seed

    Returns:
        Optuna study
    """
    study_name = f"spectral_{approach}_phase{phase}"
    storage = f"sqlite:///{ARTIFACTS_DIR}/phase{phase}.db"

    # Create sampler
    if approach == "baseline":
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=10,
            multivariate=True,
        )
    else:
        sampler = BaselineInjectingSampler(
            n_baseline_trials=n_baseline,
        )

    # Create study
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Loaded existing study '{study_name}' with {len(study.trials)} trials")
    except:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=sampler,
            pruner=HyperbandPruner(
                min_resource=3,
                max_resource=30,
                reduction_factor=3,
            ),
        )
        print(f"Created new study '{study_name}'")

    # Define objective wrapper
    def objective(trial):
        trial_seed = get_trial_seed(seed, trial.number)
        return track_b_objective(
            trial=trial,
            approach=approach,
            data_loaders=data_loaders,
            device=device,
            stage1_checkpoint=stage1_checkpoint,
            stage1_metrics=stage1_metrics,
            num_epochs=30,
            early_stop_patience=8,
            trial_seed=trial_seed,
        )

    # Run optimization
    n_existing = len(study.trials)
    n_to_run = max(0, n_trials - n_existing)

    if n_to_run > 0:
        print(f"Running {n_to_run} trials for {approach}...")
        study.optimize(
            objective,
            n_trials=n_to_run,
            gc_after_trial=True,
            show_progress_bar=True,
        )
    else:
        print(f"Approach {approach} already has {n_existing} trials")

    return study


def run_phase1(
    device: torch.device,
    data_loaders,
    stage1_checkpoint: str,
    trials_per_approach: int = 32,
    n_baseline: int = 20,
    seed: int = 42,
) -> None:
    """Run Phase 1: Coarse screening of all spectral approaches."""
    print("=" * 60)
    print("  TRACK B - PHASE 1: Coarse Screening")
    print("=" * 60)

    # Pre-compute Stage 1 metrics (shared across all approaches)
    print("\nComputing Stage 1 baseline metrics...")
    val_loader = data_loaders["val"] if isinstance(data_loaders, dict) else data_loaders[1]
    model, _ = load_frozen_stage1(stage1_checkpoint, device)
    stage1_metrics = compute_stage1_metrics(model, val_loader, device)
    print(f"Stage 1 R²: {stage1_metrics['r2']:.4f}")
    print(f"Stage 1 PSD Error: {stage1_metrics['psd_err_db']:.2f} dB")
    del model

    for approach in TRACK_B_APPROACHES:
        print(f"\n--- Approach: {approach} ---")
        study = run_approach(
            approach=approach,
            device=device,
            data_loaders=data_loaders,
            stage1_checkpoint=stage1_checkpoint,
            stage1_metrics=stage1_metrics,
            n_trials=trials_per_approach if approach != "baseline" else n_baseline,
            n_baseline=n_baseline // 2,
            phase=1,
            seed=seed,
        )

        if study.best_trial:
            psd_imp = study.best_trial.user_attrs.get("psd_improvement_db", 0)
            r2_change = study.best_trial.user_attrs.get("r2_change", 0)
            print(f"Best score: {study.best_trial.value:.4f}")
            print(f"PSD improvement: {psd_imp:+.2f} dB")
            print(f"R² change: {r2_change:+.4f}")


def run_phase2(
    device: torch.device,
    data_loaders,
    stage1_checkpoint: str,
    approaches: List[str],
    trials_per_approach: int = 96,
    seed: int = 42,
) -> None:
    """Run Phase 2: Deep dive on selected approaches."""
    print("=" * 60)
    print("  TRACK B - PHASE 2: Deep Dive")
    print("=" * 60)
    print(f"Approaches: {approaches}")

    # Pre-compute Stage 1 metrics
    print("\nComputing Stage 1 baseline metrics...")
    val_loader = data_loaders["val"] if isinstance(data_loaders, dict) else data_loaders[1]
    model, _ = load_frozen_stage1(stage1_checkpoint, device)
    stage1_metrics = compute_stage1_metrics(model, val_loader, device)
    print(f"Stage 1 R²: {stage1_metrics['r2']:.4f}")
    print(f"Stage 1 PSD Error: {stage1_metrics['psd_err_db']:.2f} dB")
    del model

    for approach in approaches:
        print(f"\n--- Approach: {approach} ---")
        study = run_approach(
            approach=approach,
            device=device,
            data_loaders=data_loaders,
            stage1_checkpoint=stage1_checkpoint,
            stage1_metrics=stage1_metrics,
            n_trials=trials_per_approach,
            n_baseline=10,
            phase=2,
            seed=seed,
        )

        if study.best_trial:
            print(f"Best score: {study.best_trial.value:.4f}")


def run_dry_run(
    device: torch.device,
    data_loaders,
    stage1_checkpoint: str,
    approach: str = "phase_preserving",
) -> None:
    """Run dry-run test for a single approach."""
    print("=" * 60)
    print(f"  DRY-RUN TEST: {approach}")
    print("=" * 60)

    # Compute Stage 1 metrics (use dummy metrics in dry-run if checkpoint missing)
    print("\nComputing Stage 1 baseline metrics...")
    val_loader = data_loaders["val"] if isinstance(data_loaders, dict) else data_loaders[1]

    from pathlib import Path
    if not Path(stage1_checkpoint).exists():
        print(f"  Checkpoint not found, using dummy Stage 1 metrics for dry-run")
        stage1_metrics = {"r2": 0.5, "psd_err_db": 10.0}
    else:
        model, _ = load_frozen_stage1(stage1_checkpoint, device)
        stage1_metrics = compute_stage1_metrics(model, val_loader, device)
        print(f"Stage 1 R²: {stage1_metrics['r2']:.4f}")
        print(f"Stage 1 PSD Error: {stage1_metrics['psd_err_db']:.2f} dB")
        del model

    study_name = f"spectral_{approach}_dryrun"
    storage = f"sqlite:///{ARTIFACTS_DIR}/dryrun.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    def objective(trial):
        return track_b_objective(
            trial=trial,
            approach=approach,
            data_loaders=data_loaders,
            device=device,
            stage1_checkpoint=stage1_checkpoint,
            stage1_metrics=stage1_metrics,
            num_epochs=3,
            early_stop_patience=2,
            trial_seed=42,
        )

    try:
        study.optimize(objective, n_trials=1, gc_after_trial=True)
        print("\n" + "=" * 60)
        print("  DRY-RUN PASSED!")
        print("=" * 60)
        print(f"Score: {study.best_trial.value:.4f}")
        print(f"PSD improvement: {study.best_trial.user_attrs.get('psd_improvement_db', 0):+.2f} dB")
        print(f"R² change: {study.best_trial.user_attrs.get('r2_change', 0):+.4f}")
    except Exception as e:
        print("\n" + "=" * 60)
        print("  DRY-RUN FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        raise


def export_results(phase: int = 1) -> None:
    """Export study results."""
    storage = f"sqlite:///{ARTIFACTS_DIR}/phase{phase}.db"
    output_dir = ARTIFACTS_DIR / f"results_phase{phase}"
    output_dir.mkdir(exist_ok=True)

    print(f"Exporting results from phase {phase}...")

    for approach in TRACK_B_APPROACHES:
        study_name = f"spectral_{approach}_phase{phase}"
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            export_study_results(study, output_dir, study_name=approach)
        except Exception as e:
            print(f"Could not export {approach}: {e}")

    print(f"Results exported to: {output_dir}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Track B: SpectralShift HPO"
    )

    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                       help="HPO phase (1=coarse, 2=deep dive)")
    parser.add_argument("--approaches", nargs="+", default=None,
                       help="Approaches for phase 2")
    parser.add_argument("--approach", type=str, default=None,
                       help="Single approach to run")
    parser.add_argument("--trials-per-approach", type=int, default=32,
                       help="Trials per approach")
    parser.add_argument("--n-baseline", type=int, default=20,
                       help="Number of baseline trials")
    parser.add_argument("--stage1-checkpoint", type=str, required=False,
                       default="artifacts/checkpoints/best_stage1.pt",
                       help="Path to Stage 1 checkpoint")
    parser.add_argument("--gpu-id", type=int, default=0,
                       help="GPU ID to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Base random seed")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run dry-run test")
    parser.add_argument("--export", action="store_true",
                       help="Export results only")

    args = parser.parse_args()

    # Export mode
    if args.export:
        export_results(args.phase)
        return

    # Check checkpoint exists (skip in dry-run mode)
    if not args.export and not args.dry_run:
        checkpoint_path = Path(args.stage1_checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Stage 1 checkpoint not found: {checkpoint_path}")
            print("Run: python train.py --epochs 80 to create checkpoint first")
            sys.exit(1)
    elif args.dry_run:
        print("Dry-run mode: Skipping Stage 1 checkpoint check")

    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load data
    print("Loading data...")
    seed_everything(args.seed)
    data = prepare_data(seed=args.seed)
    data_loaders = create_dataloaders(
        data,
        batch_size=32,
        num_workers=4,
    )

    # Run appropriate mode
    if args.dry_run:
        approach = args.approach or "phase_preserving"
        run_dry_run(device, data_loaders, args.stage1_checkpoint, approach)

    elif args.approach:
        run_approach(
            approach=args.approach,
            device=device,
            data_loaders=data_loaders,
            stage1_checkpoint=args.stage1_checkpoint,
            n_trials=args.trials_per_approach,
            n_baseline=args.n_baseline,
            phase=args.phase,
            seed=args.seed,
        )

    elif args.phase == 1:
        run_phase1(
            device=device,
            data_loaders=data_loaders,
            stage1_checkpoint=args.stage1_checkpoint,
            trials_per_approach=args.trials_per_approach,
            n_baseline=args.n_baseline,
            seed=args.seed,
        )

    elif args.phase == 2:
        approaches = args.approaches or ["phase_preserving", "adaptive_gated", "wavelet"]
        run_phase2(
            device=device,
            data_loaders=data_loaders,
            stage1_checkpoint=args.stage1_checkpoint,
            approaches=approaches,
            trials_per_approach=args.trials_per_approach,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
