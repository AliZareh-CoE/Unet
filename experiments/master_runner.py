#!/usr/bin/env python3
"""
Master Runner: Orchestrate All 6 Studies for Nature Methods
===========================================================

Two-Stage HPO Pipeline:
- STAGE 1 (OB → PCx): Studies 1-3, 5 - Find best architecture, conditioning, loss
- STAGE 2 (PCx → PCx'): Study 4 - SpectralShift using best Stage 1 model
- VALIDATION: Study 6 - Final multi-seed validation

Usage:
    # Pre-flight checks
    python experiments/master_runner.py --preflight

    # Run Stage 1 studies (architecture, conditioning, loss, classical baselines)
    python experiments/master_runner.py --stage 1 --gpus 0,1,2,3,4,5,6,7

    # Run Stage 2 (SpectralShift) after Stage 1 completes
    python experiments/master_runner.py --stage 2 --gpus 0,1,2,3

    # Run final validation
    python experiments/master_runner.py --stage validation --gpus 0,1,2,3,4,5,6,7

    # Dry-run all
    python experiments/master_runner.py --dry-run --all-studies

    # Integration test
    python experiments/master_runner.py --integration-test
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =============================================================================
# CRITICAL: Fix Python path for imports
# =============================================================================
# Get the project root directory (parent of experiments/)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import torch


# =============================================================================
# Configuration
# =============================================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Training configuration
STAGE1_CONFIG = {
    "epochs": 80,
    "patience": 8,  # Early stopping patience
    "checkpoint_top_k": 3,  # Save only top 3 checkpoints per study
}

STAGE2_CONFIG = {
    "epochs": 30,  # Shorter for SpectralShift (frozen backbone)
    "patience": 5,
    "checkpoint_top_k": 3,
}

# =============================================================================
# Study Definitions - Clearly Separated by Stage
# =============================================================================

STAGE1_STUDIES = {
    # Stage 1: OB → PCx translation (find best base model)
    1: {
        "name": "Architecture Comparison",
        "description": "Compare 8 architectures for OB→PCx translation",
        "script": "experiments/study1_architecture/run_study1.py",
        "default_trials": 100,
        "gpus_needed": 4,
        "stage": 1,
    },
    2: {
        "name": "Conditioning Mechanisms",
        "description": "Test conditioning approaches (CPC, VQ-VAE, FiLM, etc.)",
        "script": "experiments/conditioning/run_track_a.py",
        "default_trials": 80,
        "gpus_needed": 4,
        "stage": 1,
    },
    3: {
        "name": "Loss Function Ablation",
        "description": "Compare L1, MSE, Huber, spectral, probabilistic losses",
        "script": "experiments/study3_loss/run_study3.py",
        "default_trials": 80,
        "gpus_needed": 2,
        "stage": 1,
    },
    5: {
        "name": "Classical Baselines",
        "description": "Wiener, Ridge, CCA, Kalman (floor performance)",
        "script": "experiments/study5_classical/run_study5.py",
        "default_trials": 1,  # No HPO - one-shot evaluation
        "gpus_needed": 1,
        "stage": 1,
    },
}

STAGE2_STUDIES = {
    # Stage 2: PCx → PCx' (SpectralShift using best Stage 1 model)
    4: {
        "name": "SpectralShift Variants",
        "description": "Fix PSD mismatch using frozen Stage 1 backbone",
        "script": "experiments/spectral/run_track_b.py",
        "default_trials": 80,
        "gpus_needed": 2,
        "stage": 2,
        "requires": "best_stage1_model",  # Must load best model from Stage 1
    },
}

VALIDATION_STUDIES = {
    # Final Validation: Multi-seed, cross-validation, negative controls
    6: {
        "name": "Final Validation",
        "description": "Top 5 configs × 10 seeds × statistical tests",
        "script": "experiments/study6_validation/run_study6.py",
        "default_trials": 50,
        "gpus_needed": 8,
        "stage": "validation",
        "requires": "all_studies_complete",
    },
}

ABLATION_STUDIES = {
    # Component Ablation: Justify each architectural choice
    7: {
        "name": "Component Ablation",
        "description": "Ablate attention, norm, conditioning, SE, skip, depth",
        "script": "experiments/study7_ablation/run_study7.py",
        "default_trials": 35,  # 7 categories × 5 seeds (approx)
        "gpus_needed": 2,
        "stage": 1,
    },
}

# Combined for easy access
ALL_STUDIES = {**STAGE1_STUDIES, **STAGE2_STUDIES, **VALIDATION_STUDIES, **ABLATION_STUDIES}


# =============================================================================
# Pre-Flight Checks
# =============================================================================

def check_imports() -> Tuple[bool, List[str]]:
    """Verify all study imports work."""
    errors = []

    imports_to_check = [
        ("experiments.study1_architecture.architectures", "ARCHITECTURE_REGISTRY"),
        ("experiments.study3_loss.losses", "LOSS_REGISTRY"),
        ("experiments.study5_classical.baselines", "BASELINE_REGISTRY"),
        ("experiments.study6_validation.multi_seed_runner", "MultiSeedRunner"),
        ("experiments.study6_validation.statistical_tests", "StatisticalComparison"),
        ("experiments.study6_validation.negative_controls", "NegativeControlRunner"),
        ("experiments.common.validation", "validate_tensor_shape"),
        ("experiments.common.error_handling", "trial_context"),
        ("experiments.common.checkpointing", "TrialCheckpoint"),
    ]

    for module_name, attr in imports_to_check:
        try:
            # Use importlib.import_module which respects sys.path properly
            module = importlib.import_module(module_name)
            if not hasattr(module, attr):
                errors.append(f"{module_name}.{attr} not found")
        except ImportError as e:
            errors.append(f"Failed to import {module_name}: {e}")
        except Exception as e:
            errors.append(f"Error importing {module_name}: {type(e).__name__}: {e}")

    return len(errors) == 0, errors


def check_gpu_memory(min_memory_gb: float = 20.0) -> Tuple[bool, Dict[int, float]]:
    """Check GPU memory availability."""
    if not torch.cuda.is_available():
        return False, {}

    gpu_memory = {}
    all_ok = True

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024**3)
        gpu_memory[i] = total_gb
        if total_gb < min_memory_gb:
            all_ok = False

    return all_ok, gpu_memory


def check_stage1_complete() -> Tuple[bool, Optional[Dict]]:
    """Check if Stage 1 is complete and find best model."""
    best_model_path = ARTIFACTS_DIR / "best_stage1_model.json"

    if best_model_path.exists():
        with open(best_model_path) as f:
            best_model_info = json.load(f)
        return True, best_model_info

    # Check if individual study results exist
    stage1_complete = all(
        (ARTIFACTS_DIR / f"study{sid}_*").exists() or
        len(list(ARTIFACTS_DIR.glob(f"study{sid}_*"))) > 0
        for sid in STAGE1_STUDIES.keys()
    )

    return stage1_complete, None


def run_preflight_checks(verbose: bool = True) -> bool:
    """Run all pre-flight checks."""
    if verbose:
        print("=" * 60)
        print("PRE-FLIGHT CHECKS")
        print("=" * 60)

    all_passed = True

    # Check imports
    if verbose:
        print("\n1. Checking imports...")
    imports_ok, import_errors = check_imports()
    if imports_ok:
        if verbose:
            print("   ✓ All imports OK")
    else:
        if verbose:
            print("   ✗ Import errors:")
            for err in import_errors:
                print(f"      - {err}")
        all_passed = False

    # Check GPU
    if verbose:
        print("\n2. Checking GPU availability...")
    gpu_ok, gpu_memory = check_gpu_memory()
    if torch.cuda.is_available():
        if verbose:
            print(f"   ✓ {torch.cuda.device_count()} GPUs available:")
            for gpu_id, mem in gpu_memory.items():
                status = "✓" if mem >= 20 else "✗"
                print(f"      {status} GPU {gpu_id}: {mem:.1f} GB")
    else:
        if verbose:
            print("   ✗ No CUDA GPUs available")
        all_passed = False

    # Check study scripts
    if verbose:
        print("\n3. Checking study scripts...")
    for study_id, info in ALL_STUDIES.items():
        script = PROJECT_ROOT / info["script"]
        if script.exists():
            if verbose:
                print(f"   ✓ Study {study_id}: {script.name} (Stage {info['stage']})")
        else:
            if verbose:
                print(f"   ✗ Study {study_id}: {script} NOT FOUND")
            all_passed = False

    # Check Stage 1 completion (for Stage 2)
    if verbose:
        print("\n4. Checking Stage 1 status...")
    stage1_done, best_model = check_stage1_complete()
    if stage1_done:
        if verbose:
            print("   ✓ Stage 1 complete" + (f" (best: {best_model.get('name', 'unknown')})" if best_model else ""))
    else:
        if verbose:
            print("   ⚠ Stage 1 not yet complete (run --stage 1 first)")

    if verbose:
        print("\n" + "=" * 60)
        if all_passed:
            print("ALL PRE-FLIGHT CHECKS PASSED ✓")
        else:
            print("SOME CHECKS FAILED ✗ - Fix issues before proceeding")
        print("=" * 60 + "\n")

    return all_passed


# =============================================================================
# Study Execution
# =============================================================================

def run_study(
    study_id: int,
    gpu_ids: List[int],
    dry_run: bool = False,
    trials: Optional[int] = None,
    epochs: Optional[int] = None,
    patience: Optional[int] = None,
) -> Dict:
    """Run a single study."""
    study_info = ALL_STUDIES[study_id]
    script = PROJECT_ROOT / study_info["script"]

    # Determine config based on stage
    if study_info["stage"] == 1:
        config = STAGE1_CONFIG
    elif study_info["stage"] == 2:
        config = STAGE2_CONFIG
    else:
        config = STAGE1_CONFIG  # Default for validation

    # Study-specific argument support
    # Not all scripts accept all arguments
    STUDY_ARGS = {
        1: {"epochs": True, "trials": True, "patience": True},   # run_study1.py
        2: {"epochs": False, "trials": False, "patience": False}, # run_track_a.py (uses different args)
        3: {"epochs": True, "trials": True, "patience": True},   # run_study3.py
        4: {"epochs": False, "trials": False, "patience": False}, # run_track_b.py (uses different args)
        5: {"epochs": False, "trials": False, "patience": False}, # run_study5.py (classical - no training)
        6: {"epochs": True, "trials": False, "patience": False, "n_seeds": True},  # run_study6.py
        7: {"epochs": True, "trials": False, "patience": True, "seeds": True},  # run_study7.py (ablation)
    }

    supported = STUDY_ARGS.get(study_id, {"epochs": True, "trials": True, "patience": True})

    # Build command
    cmd = ["python", str(script)]

    if dry_run:
        cmd.append("--dry-run")
        if supported.get("epochs"):
            cmd.extend(["--epochs", "3"])
    else:
        if supported.get("epochs"):
            cmd.extend(["--epochs", str(epochs or config["epochs"])])

    if trials is not None and supported.get("trials"):
        cmd.extend(["--trials", str(trials)])

    # Add patience if supported by this study
    if supported.get("patience") and (patience or config.get("patience")):
        cmd.extend(["--patience", str(patience or config["patience"])])

    # Set environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    env["PYTHONPATH"] = str(PROJECT_ROOT) + ":" + env.get("PYTHONPATH", "")

    print(f"\n{'=' * 60}")
    print(f"Running Study {study_id}: {study_info['name']}")
    print(f"Stage: {study_info['stage']}")
    print(f"Description: {study_info['description']}")
    print(f"GPUs: {gpu_ids}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600 * 48,  # 48 hour timeout
            cwd=str(PROJECT_ROOT),
        )

        elapsed = time.time() - start_time

        return {
            "study_id": study_id,
            "stage": study_info["stage"],
            "status": "success" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "elapsed_seconds": elapsed,
            "stdout": result.stdout[-10000:],  # Last 10K chars
            "stderr": result.stderr[-5000:],
        }

    except subprocess.TimeoutExpired:
        return {
            "study_id": study_id,
            "stage": study_info["stage"],
            "status": "timeout",
            "elapsed_seconds": time.time() - start_time,
        }

    except Exception as e:
        return {
            "study_id": study_id,
            "stage": study_info["stage"],
            "status": "error",
            "error": str(e),
            "elapsed_seconds": time.time() - start_time,
        }


def run_stage1(
    gpu_ids: List[int],
    dry_run: bool = False,
    parallel: bool = False,
) -> List[Dict]:
    """Run all Stage 1 studies (OB → PCx)."""
    print("\n" + "=" * 60)
    print("STAGE 1: OB → PCx Translation Studies")
    print("=" * 60)
    print("Studies: Architecture, Conditioning, Loss, Classical Baselines")
    print(f"Config: {STAGE1_CONFIG['epochs']} epochs, patience {STAGE1_CONFIG['patience']}")
    print("=" * 60)

    study_ids = list(STAGE1_STUDIES.keys())
    results = []

    # Run Study 5 (Classical Baselines) first - establishes floor
    print("\n>>> Running Classical Baselines first (establishes floor)...")
    result = run_study(5, gpu_ids[:1], dry_run=dry_run)
    results.append(result)

    # Run remaining Stage 1 studies
    remaining = [sid for sid in study_ids if sid != 5]

    for study_id in remaining:
        result = run_study(
            study_id=study_id,
            gpu_ids=gpu_ids,
            dry_run=dry_run,
        )
        results.append(result)

        if result["status"] != "success" and not dry_run:
            print(f"\n⚠ Study {study_id} failed!")
            print(f"stderr: {result.get('stderr', 'N/A')[:500]}")

    # Save best Stage 1 model info
    if not dry_run:
        save_best_stage1_model(results)

    return results


def run_stage2(
    gpu_ids: List[int],
    dry_run: bool = False,
) -> List[Dict]:
    """Run Stage 2 studies (PCx → PCx' SpectralShift)."""
    print("\n" + "=" * 60)
    print("STAGE 2: PCx → PCx' SpectralShift Studies")
    print("=" * 60)

    # Check Stage 1 complete
    stage1_done, best_model = check_stage1_complete()
    if not stage1_done and not dry_run:
        print("ERROR: Stage 1 must complete before Stage 2!")
        print("Run: python experiments/master_runner.py --stage 1")
        return []

    print(f"Using best Stage 1 model: {best_model.get('name', 'default') if best_model else 'default'}")
    print(f"Config: {STAGE2_CONFIG['epochs']} epochs, patience {STAGE2_CONFIG['patience']}")
    print("=" * 60)

    results = []
    for study_id in STAGE2_STUDIES.keys():
        result = run_study(
            study_id=study_id,
            gpu_ids=gpu_ids,
            dry_run=dry_run,
        )
        results.append(result)

    return results


def run_validation(
    gpu_ids: List[int],
    dry_run: bool = False,
) -> List[Dict]:
    """Run final validation study."""
    print("\n" + "=" * 60)
    print("FINAL VALIDATION: Multi-Seed Statistical Tests")
    print("=" * 60)
    print("Top 5 configs × 10 seeds × bootstrap CIs × negative controls")
    print("=" * 60)

    results = []
    for study_id in VALIDATION_STUDIES.keys():
        result = run_study(
            study_id=study_id,
            gpu_ids=gpu_ids,
            dry_run=dry_run,
        )
        results.append(result)

    return results


def save_best_stage1_model(results: List[Dict]) -> None:
    """Save information about the best Stage 1 model."""
    # This would normally parse results and find best config
    # For now, save placeholder
    best_model_info = {
        "timestamp": datetime.now().isoformat(),
        "name": "best_stage1_model",
        "studies_completed": [r["study_id"] for r in results if r["status"] == "success"],
        "note": "Load actual best config from study databases",
    }

    output_path = ARTIFACTS_DIR / "best_stage1_model.json"
    with open(output_path, "w") as f:
        json.dump(best_model_info, f, indent=2)

    print(f"\nBest Stage 1 model info saved to: {output_path}")


# =============================================================================
# Special Modes
# =============================================================================

def run_integration_test() -> bool:
    """Run integration test (3 epochs, 1 trial per study)."""
    print("\n" + "=" * 60)
    print("INTEGRATION TEST")
    print("=" * 60)

    results = []
    gpu_ids = list(range(min(2, torch.cuda.device_count()))) or [0]

    for study_id in sorted(ALL_STUDIES.keys()):
        print(f"\nTesting Study {study_id}...")
        result = run_study(study_id, gpu_ids, dry_run=True)
        results.append(result)

        status = "✓" if result["status"] == "success" else "✗"
        print(f"  {status} Study {study_id}: {result['status']}")

        if result["status"] != "success":
            print(f"\n{'=' * 60}")
            print(f"FULL ERROR OUTPUT FOR STUDY {study_id}")
            print("=" * 60)

            if result.get("error"):
                print(f"Exception: {result['error']}")

            if result.get("returncode"):
                print(f"Return code: {result['returncode']}")

            if result.get("stderr"):
                print(f"\nSTDERR:\n{result['stderr']}")

            if result.get("stdout"):
                print(f"\nSTDOUT (last 2000 chars):\n{result['stdout'][-2000:]}")

            print("=" * 60 + "\n")

    n_passed = sum(1 for r in results if r["status"] == "success")
    print(f"\n{'=' * 60}")
    print(f"Integration Test: {n_passed}/{len(results)} passed")
    print("=" * 60)

    return n_passed == len(results)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Master Runner for Nature Methods 6-Study HPO Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
STAGE EXECUTION ORDER:
  1. --stage 1     : Run Studies 1,2,3,5 (OB→PCx translation)
  2. --stage 2     : Run Study 4 (SpectralShift, requires Stage 1)
  3. --stage validation : Run Study 6 (multi-seed validation)

RECOMMENDED WORKFLOW:
  python experiments/master_runner.py --preflight
  python experiments/master_runner.py --integration-test
  python experiments/master_runner.py --stage 1 --gpus 0,1,2,3,4,5,6,7
  python experiments/master_runner.py --stage 2 --gpus 0,1,2,3
  python experiments/master_runner.py --stage validation --gpus 0,1,2,3,4,5,6,7
        """,
    )

    # Stage selection
    parser.add_argument("--stage", type=str, choices=["1", "2", "validation", "all"],
                        help="Which stage to run")
    parser.add_argument("--all-studies", action="store_true",
                        help="Run all stages sequentially")
    parser.add_argument("--studies", type=int, nargs="+",
                        help="Run specific study IDs")

    # Modes
    parser.add_argument("--dry-run", action="store_true",
                        help="Run minimal test (3 epochs, 1 trial)")
    parser.add_argument("--integration-test", action="store_true",
                        help="Run integration test")
    parser.add_argument("--preflight", action="store_true",
                        help="Run pre-flight checks only")

    # Options
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--parallel", action="store_true",
                        help="Run studies in parallel")

    args = parser.parse_args()

    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpus.split(",")]

    # Pre-flight checks
    if args.preflight:
        run_preflight_checks()
        return

    # Integration test
    if args.integration_test:
        success = run_integration_test()
        sys.exit(0 if success else 1)

    # Run preflight before any actual execution
    if not args.dry_run:
        if not run_preflight_checks(verbose=True):
            print("\nFix pre-flight issues before running full studies")
            sys.exit(1)

    # Determine what to run
    all_results = []
    start_time = time.time()

    if args.all_studies or args.stage == "all":
        # Run everything in order
        all_results.extend(run_stage1(gpu_ids, args.dry_run))
        all_results.extend(run_stage2(gpu_ids, args.dry_run))
        all_results.extend(run_validation(gpu_ids, args.dry_run))

    elif args.stage == "1":
        all_results = run_stage1(gpu_ids, args.dry_run)

    elif args.stage == "2":
        all_results = run_stage2(gpu_ids, args.dry_run)

    elif args.stage == "validation":
        all_results = run_validation(gpu_ids, args.dry_run)

    elif args.studies:
        for study_id in args.studies:
            if study_id in ALL_STUDIES:
                result = run_study(study_id, gpu_ids, args.dry_run)
                all_results.append(result)
            else:
                print(f"Unknown study ID: {study_id}")

    else:
        parser.print_help()
        return

    # Summary
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)

    for result in all_results:
        study_id = result["study_id"]
        stage = result.get("stage", "?")
        status = "✓" if result["status"] == "success" else "✗"
        elapsed = result.get("elapsed_seconds", 0)
        print(f"  {status} Study {study_id} (Stage {stage}): {result['status']} ({elapsed/60:.1f} min)")

    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")

    # Save results
    output_file = ARTIFACTS_DIR / f"master_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "stage": args.stage,
            "dry_run": args.dry_run,
            "total_time_seconds": total_time,
            "results": all_results,
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    # Exit code
    n_failed = sum(1 for r in all_results if r["status"] != "success")
    sys.exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    main()
