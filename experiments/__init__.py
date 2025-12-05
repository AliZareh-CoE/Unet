"""
Bulletproof Tier-Based Experimental Framework for Nature Methods Publication
=============================================================================

NEW TIER SYSTEM (replaces old 6-study structure):

Tier 0 (tier0_classical/): Classical baselines - establishes floor
Tier 1 (tier1_screening/): Quick 8×6 screening - eliminates bad options
Tier 2 (tier2_factorial/): Factorial design - finds interactions (KEY TIER)
Tier 3 (tier3_ablation/): Focused ablation - removes useless components
Tier 4 (tier4_validation/): Final validation - holdout test + negative controls

SUPPORTING MODULES (kept from old structure):
- study1_architecture/architectures/: Model implementations (linear, cnn, wavenet, etc.)
- study3_loss/losses/: Loss implementations (huber, spectral, correlation, etc.)
- study5_classical/baselines/: Classical baselines (wiener, ridge, cca, kalman)
- common/: Shared utilities (CV, statistics, metrics, etc.)

MASTER RUNNER:
    python experiments/tier_runner.py           # Full pipeline
    python experiments/tier_runner.py --dry-run # Quick test
"""

from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent
ARTIFACTS_DIR = EXPERIMENTS_DIR.parent / "artifacts"

# Ensure artifact directories exist for tier system
(ARTIFACTS_DIR / "tier_system").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "tier0_classical").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "tier1_screening").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "tier2_factorial").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "tier3_ablation").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "tier4_validation").mkdir(parents=True, exist_ok=True)

# Legacy directories (for backwards compatibility)
(ARTIFACTS_DIR / "study5_classical").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "nature_methods").mkdir(parents=True, exist_ok=True)
