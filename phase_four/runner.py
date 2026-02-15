"""Phase 4 runner – orchestrates training, generation, and validation.

Usage
-----
# Full pipeline (all datasets):
    python -m phase_four.runner

# Training only (all datasets, 8 GPUs – one dataset per GPU):
    python -m phase_four.runner --train --n-gpus 8

# Single dataset with FSDP across 8 GPUs:
    python -m phase_four.runner --train --datasets pcx1 --n-gpus 8 --fsdp

# Generate synthetic signals only (models already trained):
    python -m phase_four.runner --generate

# Run validations only (signals already saved):
    python -m phase_four.runner --validate

# Subset of datasets:
    python -m phase_four.runner --datasets olfactory pfc_hpc
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from phase_four.config import DATASETS, Phase4Config


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Training  – invoke train.py per dataset
# ═══════════════════════════════════════════════════════════════════════════

def _build_train_args(dataset_key: str, cfg: Phase4Config) -> List[str]:
    """Build the common train.py arguments for a dataset (no launcher prefix)."""
    ds = DATASETS[dataset_key]
    # train.py uses --output-dir to set base; checkpoints go to <output-dir>/checkpoints
    output_dir = cfg.checkpoint_dir / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = cfg.output_dir / dataset_key / "train_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "train.py",
        "--dataset", ds.train_name,
        # Architecture (LOSO-optimized)
        "--arch", cfg.arch,
        "--base-channels", str(cfg.base_channels),
        "--n-downsample", str(cfg.n_downsample),
        "--attention-type", cfg.attention_type,
        "--conv-type", cfg.conv_type,
        "--activation", cfg.activation,
        "--skip-type", cfg.skip_type,
        "--cond-mode", cfg.cond_mode,
        "--conditioning", cfg.conditioning,
        "--n-heads", str(cfg.n_heads),
        # Training
        "--epochs", str(cfg.epochs),
        "--batch-size", str(cfg.batch_size),
        "--lr", str(cfg.lr),
        "--seed", str(cfg.seed),
        # Optimizer (LOSO-optimized)
        "--optimizer", cfg.optimizer,
        "--lr-schedule", cfg.lr_schedule,
        # Noise augmentation (LOSO-optimized)
        "--use-noise-augmentation",
        "--noise-gaussian-std", str(cfg.noise_gaussian_std),
        "--noise-pink-std", str(cfg.noise_pink_std),
        "--noise-channel-dropout", str(cfg.noise_channel_dropout),
        "--noise-temporal-dropout", str(cfg.noise_temporal_dropout),
        "--noise-prob", str(cfg.noise_prob),
        # Session adaptation (LOSO-optimized)
        "--use-adaptive-scaling",
        "--no-bidirectional",
        # Output
        "--checkpoint-prefix", "phase4",
        "--output-dir", str(output_dir),
        "--output-results-file", str(results_file),
        "--no-plots",
        "--quiet",
        "--split-by-session",
    ]
    if cfg.noise_pink:
        args.append("--noise-pink")
    args.extend(ds.extra_train_args)
    return args


def train_single_dataset(
    dataset_key: str,
    cfg: Phase4Config,
    gpu_id: int = 0,
) -> Path:
    """Train CondUNet on one dataset with 70/30 split, return checkpoint path."""
    ds = DATASETS[dataset_key]
    train_args = _build_train_args(dataset_key, cfg)
    cmd = [sys.executable] + train_args

    env = {
        **__import__("os").environ,
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
    }

    print(f"\n{'='*60}")
    print(f"Training: {ds.display_name} (GPU {gpu_id})")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    proc = subprocess.run(cmd, env=env, capture_output=False)
    if proc.returncode != 0:
        print(f"  WARNING: train.py exited with code {proc.returncode}")

    ckpt_path = cfg.checkpoint_dir / dataset_key / "checkpoints" / "phase4_best_model.pt"
    if ckpt_path.exists():
        print(f"  Checkpoint saved: {ckpt_path}")
    else:
        print(f"  WARNING: expected checkpoint not found at {ckpt_path}")

    return ckpt_path


def train_fsdp(
    dataset_key: str,
    cfg: Phase4Config,
    n_gpus: int = 8,
    fsdp_strategy: str = "full",
) -> Path:
    """Train one dataset with FSDP across multiple GPUs via torchrun."""
    import shutil

    ds = DATASETS[dataset_key]
    train_args = _build_train_args(dataset_key, cfg)

    # Add FSDP flags
    train_args.extend(["--fsdp", "--fsdp-strategy", fsdp_strategy])

    # Use torchrun for distributed launch
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        # Fallback to python -m torch.distributed.run
        cmd = [sys.executable, "-m", "torch.distributed.run",
               "--nproc_per_node", str(n_gpus)] + train_args
    else:
        cmd = [torchrun, "--nproc_per_node", str(n_gpus)] + train_args

    log_file = cfg.output_dir / dataset_key / "train.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"FSDP Training: {ds.display_name} ({n_gpus} GPUs, strategy={fsdp_strategy})")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_file}")
    print(f"{'='*60}")

    log_fh = open(log_file, "w")
    proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    log_fh.close()

    if proc.returncode != 0:
        print(f"  FAILED (code {proc.returncode}) — check {log_file}")
    else:
        print(f"  OK")

    ckpt_path = cfg.checkpoint_dir / dataset_key / "checkpoints" / "phase4_best_model.pt"
    if ckpt_path.exists():
        print(f"  Checkpoint saved: {ckpt_path}")
    else:
        print(f"  WARNING: expected checkpoint not found at {ckpt_path}")

    return ckpt_path


def train_all_parallel(
    cfg: Phase4Config,
    datasets: Optional[List[str]] = None,
    n_gpus: int = 8,
) -> None:
    """Launch training for all datasets in parallel across GPUs (one GPU per dataset)."""
    import os

    if datasets is None:
        datasets = cfg.datasets

    processes = []
    for i, ds_key in enumerate(datasets):
        gpu_id = i % n_gpus
        ds = DATASETS[ds_key]
        train_args = _build_train_args(ds_key, cfg)
        cmd = [sys.executable] + train_args

        log_file = cfg.output_dir / ds_key / "train.log"

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

        print(f"[GPU {gpu_id}] {ds.display_name} → {log_file}")
        log_fh = open(log_file, "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        processes.append((ds_key, proc, log_fh))

    # Wait for all
    print(f"\nWaiting for {len(processes)} training jobs...")
    for ds_key, proc, log_fh in processes:
        proc.wait()
        log_fh.close()
        status = "OK" if proc.returncode == 0 else f"FAILED (code {proc.returncode})"
        print(f"  {ds_key}: {status}")


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_all(cfg: Phase4Config, datasets: Optional[List[str]] = None):
    """Generate synthetic signals for all trained datasets."""
    import torch
    from phase_four.generate import generate_and_save

    if datasets is None:
        datasets = cfg.datasets

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ds_key in datasets:
        try:
            generate_and_save(ds_key, cfg, device)
        except FileNotFoundError as e:
            print(f"  SKIP {ds_key}: {e}")
        except Exception as e:
            print(f"  ERROR {ds_key}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_all(cfg: Phase4Config, datasets: Optional[List[str]] = None):
    """Run all five validation families on saved synthetic data."""
    from phase_four.validation.spectral import run_spectral_validation
    from phase_four.validation.cca import run_cca_validation
    from phase_four.validation.decoding import run_decoding_validation
    from phase_four.validation.pid import run_pid_validation
    from phase_four.validation.fingerprint import run_fingerprint_validation

    if datasets is None:
        datasets = cfg.datasets

    for ds_key in datasets:
        sd = cfg.get_synth_dir(ds_key)
        if not (sd / "target_test.npy").exists():
            print(f"\n  SKIP {ds_key}: no generated data at {sd}")
            continue

        ds = DATASETS[ds_key]
        fs = ds.sampling_rate
        print(f"\n{'='*60}")
        print(f"Validating: {ds.display_name}")
        print(f"{'='*60}")

        # ── Spectral ──
        if cfg.run_spectral:
            try:
                run_spectral_validation(
                    sd, fs,
                    nperseg=cfg.nperseg,
                    pac_phase_band=cfg.pac_phase_band,
                    pac_amp_band=cfg.pac_amp_band,
                    pac_n_surrogates=cfg.pac_n_surrogates,
                )
            except Exception as e:
                print(f"  Spectral failed: {e}")

        # ── CCA / DCCA ──
        if cfg.run_cca:
            try:
                run_cca_validation(
                    sd,
                    n_components=cfg.cca_n_components,
                    dcca_hidden_dim=cfg.dcca_hidden_dim,
                    dcca_epochs=cfg.dcca_epochs,
                    dcca_lr=cfg.dcca_lr,
                )
            except Exception as e:
                print(f"  CCA failed: {e}")

        # ── Decoding ──
        if cfg.run_decoding:
            try:
                run_decoding_validation(
                    sd, fs,
                    classifier_names=cfg.decode_classifiers,
                    n_cv_folds=cfg.decode_n_cv_folds,
                )
            except Exception as e:
                print(f"  Decoding failed: {e}")

        # ── PID (information decomposition) ──
        if cfg.run_pid:
            try:
                run_pid_validation(
                    sd, fs,
                    lag_ms=cfg.pid_lag_ms,
                )
            except Exception as e:
                print(f"  PID failed: {e}")

        # ── Fingerprinting (session identifiability) ──
        if cfg.run_fingerprint:
            try:
                run_fingerprint_validation(sd, fs)
            except Exception as e:
                print(f"  Fingerprinting failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4: Biological Validation of Neural Translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Stage selectors (default: run all)
    parser.add_argument("--train", action="store_true",
                        help="Run training stage only")
    parser.add_argument("--generate", action="store_true",
                        help="Run signal generation stage only")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation stage only")

    # Dataset selection
    parser.add_argument("--datasets", nargs="+", default=None,
                        choices=list(DATASETS.keys()),
                        help="Subset of datasets to run")

    # Training
    parser.add_argument("--n-gpus", type=int, default=1,
                        help="Number of GPUs for parallel training")
    parser.add_argument("--fsdp", action="store_true",
                        help="Use FSDP to shard one dataset across all GPUs (requires --datasets with 1 dataset)")
    parser.add_argument("--fsdp-strategy", type=str, default="full",
                        choices=["full", "grad_op", "no_shard", "hybrid", "hybrid_zero2"],
                        help="FSDP sharding strategy (default: full)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    # Validation
    parser.add_argument("--no-spectral", action="store_true")
    parser.add_argument("--no-cca", action="store_true")
    parser.add_argument("--no-decoding", action="store_true")
    parser.add_argument("--no-pid", action="store_true")
    parser.add_argument("--no-fingerprint", action="store_true")
    parser.add_argument("--cca-components", type=int, default=10)
    parser.add_argument("--pac-surrogates", type=int, default=200)
    parser.add_argument("--pid-lag-ms", type=int, default=50,
                        help="Temporal PID prediction lag in ms")

    # Paths
    parser.add_argument("--synth-root", type=str, default="/data/synth")
    parser.add_argument("--output-dir", type=str, default="results/phase4")

    return parser.parse_args()


def main():
    args = parse_args()

    # If no stage flags, run everything
    run_all = not (args.train or args.generate or args.validate)

    cfg = Phase4Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        n_gpus=args.n_gpus,
        synth_root=Path(args.synth_root),
        output_dir=Path(args.output_dir),
        run_spectral=not args.no_spectral,
        run_cca=not args.no_cca,
        run_decoding=not args.no_decoding,
        run_pid=not args.no_pid,
        run_fingerprint=not args.no_fingerprint,
        cca_n_components=args.cca_components,
        pac_n_surrogates=args.pac_surrogates,
        pid_lag_ms=args.pid_lag_ms,
    )

    if args.datasets:
        cfg.datasets = args.datasets

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 4: Biological Validation of Neural Translation")
    print("=" * 60)
    print(f"  Datasets     : {cfg.datasets}")
    print(f"  Synth output : {cfg.synth_root}")
    print(f"  Results      : {cfg.output_dir}")
    print(f"  GPUs         : {cfg.n_gpus}")

    # ── Train ──
    if run_all or args.train:
        print("\n\n>>> STAGE 1: Training <<<")
        if args.fsdp:
            # FSDP: shard single dataset across all GPUs
            if len(cfg.datasets) != 1:
                print(f"ERROR: --fsdp requires exactly 1 dataset via --datasets, "
                      f"got {len(cfg.datasets)}: {cfg.datasets}")
                sys.exit(1)
            train_fsdp(cfg.datasets[0], cfg,
                       n_gpus=cfg.n_gpus,
                       fsdp_strategy=args.fsdp_strategy)
        elif cfg.n_gpus > 1 and len(cfg.datasets) > 1:
            train_all_parallel(cfg, n_gpus=cfg.n_gpus)
        else:
            for ds_key in cfg.datasets:
                train_single_dataset(ds_key, cfg, gpu_id=0)

    # ── Generate ──
    if run_all or args.generate:
        print("\n\n>>> STAGE 2: Generating synthetic signals <<<")
        generate_all(cfg)

    # ── Validate ──
    if run_all or args.validate:
        print("\n\n>>> STAGE 3: Biological validation <<<")
        validate_all(cfg)

    print("\n\nPhase 4 complete.")


if __name__ == "__main__":
    main()
