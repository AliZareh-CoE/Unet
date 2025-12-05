"""
Six-Study Optuna HPO Framework for Nature Methods Publication
=============================================================

Study 1 (study1_architecture/): Architecture comparison
Study 2 (conditioning/): Auto-conditioning experiments to improve Stage 1 R²
Study 3 (study3_loss/): Loss function ablation
Study 4 (spectral/): SpectralShift fixes to preserve R² during PSD correction
Study 5 (study5_classical/): Classical baselines (Wiener, Ridge, CCA, Kalman)
Study 6 (study6_validation/): Final validation (top 5 × 10 seeds × cross-validation)

All studies run independently on 8 A100 GPUs with FSDP.
"""

from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent
ARTIFACTS_DIR = EXPERIMENTS_DIR.parent / "artifacts"

# Ensure artifact directories exist for all 6 studies
(ARTIFACTS_DIR / "study1_architecture").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "optuna_conditioning").mkdir(parents=True, exist_ok=True)  # Study 2
(ARTIFACTS_DIR / "study3_loss").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "optuna_spectral").mkdir(parents=True, exist_ok=True)  # Study 4
(ARTIFACTS_DIR / "study5_classical").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "study6_validation").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "nature_methods").mkdir(parents=True, exist_ok=True)
