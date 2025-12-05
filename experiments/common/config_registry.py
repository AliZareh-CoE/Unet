"""
Configuration Registry for Tier System.

Handles passing configurations between tiers:
- Stores results from each tier
- Provides selections for next tier
- Tracks the full experiment history
- Enables gates between tiers
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# =============================================================================
# Gate Exceptions
# =============================================================================


class GateFailure(Exception):
    """Raised when a tier gate condition fails."""

    def __init__(self, tier: str, condition: str, details: str = ""):
        self.tier = tier
        self.condition = condition
        self.details = details
        message = f"Gate failed at {tier}: {condition}"
        if details:
            message += f" ({details})"
        super().__init__(message)


class RankingUnstable(Exception):
    """Raised when screening rankings don't hold on full data."""

    def __init__(self, overlap: int, expected: int):
        self.overlap = overlap
        self.expected = expected
        message = f"Screening rankings unstable: {overlap}/{expected} overlap"
        super().__init__(message)


# =============================================================================
# Tier Results
# =============================================================================


@dataclass
class ClassicalResult:
    """Result from a classical baseline method."""

    method: str
    r2_mean: float
    r2_std: float
    mae_mean: float
    mae_std: float
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tier0Result:
    """Results from Tier 0 (Classical Floor)."""

    results: List[ClassicalResult]
    best_method: str
    best_r2: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "tier": 0,
            "name": "Classical Floor",
            "best_method": self.best_method,
            "best_r2": self.best_r2,
            "all_results": [
                {"method": r.method, "r2_mean": r.r2_mean, "r2_std": r.r2_std}
                for r in self.results
            ],
            "timestamp": self.timestamp,
        }


@dataclass
class ScreeningResult:
    """Result from a single screening combination."""

    architecture: str
    loss: str
    r2: float
    mae: float
    training_time: float


@dataclass
class Tier1Result:
    """Results from Tier 1 (Quick Screening)."""

    results: List[ScreeningResult]
    top_architectures: List[str]
    top_losses: List[str]
    best_r2: float
    classical_floor: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "tier": 1,
            "name": "Quick Screening",
            "top_architectures": self.top_architectures,
            "top_losses": self.top_losses,
            "best_r2": self.best_r2,
            "classical_floor": self.classical_floor,
            "margin_over_classical": self.best_r2 - self.classical_floor,
            "timestamp": self.timestamp,
        }


@dataclass
class FactorialResult:
    """Result from a factorial combination."""

    architecture: str
    loss: str
    conditioning: str
    seed: int
    fold: int
    r2: float
    mae: float
    pearson: float
    psd_error_db: float


@dataclass
class Tier2Result:
    """Results from Tier 2 (Factorial Design)."""

    results: List[FactorialResult]
    anova_results: Dict[str, Any]
    best_combination: Tuple[str, str, str]  # (arch, loss, cond)
    best_r2_mean: float
    best_r2_std: float
    runner_up: Tuple[str, str, str]
    runner_up_r2_mean: float
    psd_error_db: float  # For SpectralShift decision
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def needs_spectral_shift(self, threshold_db: float = 3.0) -> bool:
        """Check if SpectralShift optimization is needed."""
        return self.psd_error_db > threshold_db

    def to_dict(self) -> Dict:
        return {
            "tier": 2,
            "name": "Factorial Design",
            "best_combination": {
                "architecture": self.best_combination[0],
                "loss": self.best_combination[1],
                "conditioning": self.best_combination[2],
            },
            "best_r2": f"{self.best_r2_mean:.4f} +/- {self.best_r2_std:.4f}",
            "runner_up": {
                "architecture": self.runner_up[0],
                "loss": self.runner_up[1],
                "conditioning": self.runner_up[2],
            },
            "runner_up_r2": self.runner_up_r2_mean,
            "psd_error_db": self.psd_error_db,
            "needs_spectral": self.needs_spectral_shift(),
            "anova_summary": self.anova_results,
            "timestamp": self.timestamp,
        }


@dataclass
class AblationResult:
    """Result from a single ablation test."""

    component: str
    variant: str
    r2_mean: float
    r2_std: float
    delta_from_baseline: float
    cohens_d: float
    verdict: str  # "ESSENTIAL", "KEEP", "MARGINAL", "REMOVE"


@dataclass
class Tier3Result:
    """Results from Tier 3 (Ablation)."""

    baseline_config: Tuple[str, str, str]
    baseline_r2: float
    ablations: List[AblationResult]
    components_to_remove: List[str]
    final_config: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "tier": 3,
            "name": "Ablation",
            "baseline_config": {
                "architecture": self.baseline_config[0],
                "loss": self.baseline_config[1],
                "conditioning": self.baseline_config[2],
            },
            "baseline_r2": self.baseline_r2,
            "ablations": [
                {
                    "component": a.component,
                    "variant": a.variant,
                    "delta": a.delta_from_baseline,
                    "effect_size": a.cohens_d,
                    "verdict": a.verdict,
                }
                for a in self.ablations
            ],
            "components_removed": self.components_to_remove,
            "timestamp": self.timestamp,
        }


@dataclass
class ValidationResult:
    """Result from final validation."""

    config_name: str
    r2_mean: float
    r2_std: float
    r2_ci_lower: float
    r2_ci_upper: float
    mae_mean: float
    pearson_mean: float
    psd_error_db: float
    n_seeds: int


@dataclass
class NegativeControlResult:
    """Result from a negative control test."""

    control_type: str  # "shuffled", "reversed", "permutation"
    r2: float
    expected_behavior: str
    passed: bool


@dataclass
class Tier4Result:
    """Results from Tier 4 (Final Validation)."""

    validation_results: List[ValidationResult]
    negative_controls: List[NegativeControlResult]
    winner: str
    winner_r2: float
    winner_ci: Tuple[float, float]
    classical_r2: float
    improvement_over_classical: float
    effect_size_vs_classical: float
    all_controls_passed: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "tier": 4,
            "name": "Final Validation",
            "winner": self.winner,
            "winner_r2": f"{self.winner_r2:.4f}",
            "winner_ci_95": f"[{self.winner_ci[0]:.4f}, {self.winner_ci[1]:.4f}]",
            "improvement_over_classical": f"{self.improvement_over_classical:.4f}",
            "effect_size_vs_classical": self.effect_size_vs_classical,
            "negative_controls_passed": self.all_controls_passed,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Configuration Registry
# =============================================================================


class ConfigRegistry:
    """Central registry for experiment configurations and results.

    Manages the flow of information between tiers:
    - Tier 0 → Sets classical floor
    - Tier 1 → Selects top architectures and losses
    - Tier 2 → Finds best combination, checks for interactions
    - Tier 3 → Ablates winner, removes useless components
    - Tier 4 → Final validation with proper statistics

    Usage:
        registry = ConfigRegistry()
        registry.register_tier0(results)
        registry.check_gate_tier1()  # Raises if neural doesn't beat classical
        registry.register_tier1(results)
        ...
    """

    def __init__(self, artifacts_dir: Optional[Path] = None):
        """Initialize the registry.

        Args:
            artifacts_dir: Directory to save registry state
        """
        self.artifacts_dir = artifacts_dir or Path("artifacts/tier_system")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Tier results
        self.tier0: Optional[Tier0Result] = None
        self.tier1: Optional[Tier1Result] = None
        self.tier2: Optional[Tier2Result] = None
        self.tier3: Optional[Tier3Result] = None
        self.tier4: Optional[Tier4Result] = None

        # Configuration parameters
        self.gate_margin_classical = 0.10  # Neural must beat classical by this
        self.gate_margin_screening = 0.05  # At least one neural > classical + this
        self.psd_threshold_db = 3.0  # Threshold for SpectralShift
        self.ablation_threshold_d = 0.3  # Cohen's d below this → remove component
        self.safety_margin = 0.02  # Include extra candidate if within this margin

    # =========================================================================
    # Tier 0: Classical Floor
    # =========================================================================

    def register_tier0(self, result: Tier0Result) -> None:
        """Register Tier 0 results."""
        self.tier0 = result
        self._save_state()

    @property
    def classical_floor(self) -> float:
        """Get the classical baseline R² to beat."""
        if self.tier0 is None:
            raise ValueError("Tier 0 not yet completed")
        return self.tier0.best_r2

    # =========================================================================
    # Tier 1: Quick Screening
    # =========================================================================

    def register_tier1(self, result: Tier1Result) -> None:
        """Register Tier 1 results."""
        self.tier1 = result
        self._save_state()

    def check_gate_tier1(self, dry_run: bool = False) -> None:
        """Gate: Check if screening beats classical baseline."""
        if self.tier0 is None:
            raise ValueError("Must complete Tier 0 before Tier 1")
        if self.tier1 is None:
            raise ValueError("Must complete Tier 1 before checking gate")

        margin = self.tier1.best_r2 - self.tier0.best_r2
        if margin < self.gate_margin_screening:
            msg = (
                f"Gate failed at Tier 1: Neural must beat classical by {self.gate_margin_screening} "
                f"(Best neural: {self.tier1.best_r2:.4f}, Classical: {self.tier0.best_r2:.4f}, "
                f"Margin: {margin:.4f})"
            )
            if dry_run:
                print(f"\n⚠ [DRY-RUN] Gate would fail: {msg}")
                print("  Continuing anyway for comprehensive testing...\n")
            else:
                raise GateFailure(tier="Tier 1", condition=msg, details="")

    def get_tier1_selections(self) -> Dict[str, List[str]]:
        """Get selected architectures and losses for Tier 2."""
        if self.tier1 is None:
            raise ValueError("Tier 1 not yet completed")
        return {
            "architectures": self.tier1.top_architectures,
            "losses": self.tier1.top_losses,
        }

    @staticmethod
    def select_top_k_with_safety(
        scores: Dict[str, float], k: int = 3, margin: float = 0.02
    ) -> List[str]:
        """Select top-k with safety margin.

        If (k+1)th is within margin of kth, include it.

        Args:
            scores: Dictionary of name → score
            k: Number of top items to select
            margin: Safety margin to include extras

        Returns:
            List of selected names
        """
        sorted_items = sorted(scores.items(), key=lambda x: -x[1])

        # Always keep top k
        selected = [x[0] for x in sorted_items[:k]]

        # Check if (k+1)th is within margin
        if len(sorted_items) > k:
            kth_score = sorted_items[k - 1][1]
            next_score = sorted_items[k][1]

            if kth_score - next_score < margin:
                selected.append(sorted_items[k][0])

        return selected

    # =========================================================================
    # Tier 2: Factorial Design
    # =========================================================================

    def register_tier2(self, result: Tier2Result) -> None:
        """Register Tier 2 results."""
        self.tier2 = result
        self._save_state()

    def check_gate_tier2(self, dry_run: bool = False) -> None:
        """Gate: Check factorial design results."""
        if self.tier0 is None or self.tier2 is None:
            raise ValueError("Must complete Tiers 0 and 2")

        # Must beat classical by full margin
        margin = self.tier2.best_r2_mean - self.tier0.best_r2
        if margin < self.gate_margin_classical:
            msg = (
                f"Gate failed at Tier 2: Winner must beat classical by {self.gate_margin_classical} "
                f"(Winner: {self.tier2.best_r2_mean:.4f}, Classical: {self.tier0.best_r2:.4f}, "
                f"Margin: {margin:.4f})"
            )
            if dry_run:
                print(f"\n⚠ [DRY-RUN] Gate would fail: {msg}")
                print("  Continuing anyway for comprehensive testing...\n")
            else:
                raise GateFailure(tier="Tier 2", condition=msg, details="")

    def needs_spectral_shift(self) -> bool:
        """Check if SpectralShift optimization is needed."""
        if self.tier2 is None:
            raise ValueError("Tier 2 not yet completed")
        return self.tier2.needs_spectral_shift(self.psd_threshold_db)

    def get_tier2_winner(self) -> Dict[str, str]:
        """Get winning combination from factorial design."""
        if self.tier2 is None:
            raise ValueError("Tier 2 not yet completed")
        arch, loss, cond = self.tier2.best_combination
        return {
            "architecture": arch,
            "loss": loss,
            "conditioning": cond,
        }

    def get_tier2_runner_up(self) -> Dict[str, str]:
        """Get runner-up combination for ablation comparison."""
        if self.tier2 is None:
            raise ValueError("Tier 2 not yet completed")
        arch, loss, cond = self.tier2.runner_up
        return {
            "architecture": arch,
            "loss": loss,
            "conditioning": cond,
        }

    # =========================================================================
    # Tier 3: Ablation
    # =========================================================================

    def register_tier3(self, result: Tier3Result) -> None:
        """Register Tier 3 results."""
        self.tier3 = result
        self._save_state()

    def get_components_to_remove(self) -> List[str]:
        """Get components that should be removed based on ablation."""
        if self.tier3 is None:
            raise ValueError("Tier 3 not yet completed")
        return self.tier3.components_to_remove

    def get_final_config(self) -> Dict[str, Any]:
        """Get final configuration after ablation feedback."""
        if self.tier3 is None:
            raise ValueError("Tier 3 not yet completed")
        return self.tier3.final_config

    @staticmethod
    def determine_ablation_verdict(
        delta: float, cohens_d: float, threshold_d: float = 0.3
    ) -> str:
        """Determine verdict for an ablation result.

        Args:
            delta: Change in R² (positive = component helps)
            cohens_d: Effect size
            threshold_d: Threshold for practical significance

        Returns:
            Verdict string: "ESSENTIAL", "KEEP", "MARGINAL", or "REMOVE"
        """
        d_abs = abs(cohens_d)

        if delta > 0.03 and d_abs >= 0.8:
            return "ESSENTIAL"
        elif delta > 0.01 and d_abs >= 0.5:
            return "KEEP"
        elif delta > 0.005 and d_abs >= threshold_d:
            return "MARGINAL"
        else:
            return "REMOVE"

    # =========================================================================
    # Tier 4: Final Validation
    # =========================================================================

    def register_tier4(self, result: Tier4Result) -> None:
        """Register Tier 4 results."""
        self.tier4 = result
        self._save_state()

    def check_gate_tier4(self, dry_run: bool = False) -> None:
        """Gate: Check final validation."""
        if self.tier4 is None:
            raise ValueError("Tier 4 not yet completed")

        if not self.tier4.all_controls_passed:
            failed = [c for c in self.tier4.negative_controls if not c.passed]
            msg = f"Gate failed at Tier 4: All negative controls must pass (Failed: {[c.control_type for c in failed]})"
            if dry_run:
                print(f"\n⚠ [DRY-RUN] Gate would fail: {msg}")
                print("  Continuing anyway for comprehensive testing...\n")
            else:
                raise GateFailure(tier="Tier 4", condition=msg, details="")

    # =========================================================================
    # Summary and Export
    # =========================================================================

    def get_full_summary(self) -> Dict[str, Any]:
        """Get complete summary of all tier results."""
        summary = {
            "experiment_status": self._get_status(),
            "timestamp": datetime.now().isoformat(),
        }

        if self.tier0:
            summary["tier0"] = self.tier0.to_dict()
        if self.tier1:
            summary["tier1"] = self.tier1.to_dict()
        if self.tier2:
            summary["tier2"] = self.tier2.to_dict()
        if self.tier3:
            summary["tier3"] = self.tier3.to_dict()
        if self.tier4:
            summary["tier4"] = self.tier4.to_dict()

        return summary

    def _get_status(self) -> str:
        """Get current experiment status."""
        if self.tier4 is not None:
            return "COMPLETE"
        elif self.tier3 is not None:
            return "Tier 4 pending"
        elif self.tier2 is not None:
            return "Tier 3 pending"
        elif self.tier1 is not None:
            return "Tier 2 pending"
        elif self.tier0 is not None:
            return "Tier 1 pending"
        else:
            return "Not started"

    def _save_state(self) -> None:
        """Save current state to disk."""
        state_file = self.artifacts_dir / "registry_state.json"
        with open(state_file, "w") as f:
            json.dump(self.get_full_summary(), f, indent=2, default=str)

    def load_state(self) -> None:
        """Load state from disk (for resuming)."""
        state_file = self.artifacts_dir / "registry_state.json"
        if not state_file.exists():
            return

        with open(state_file) as f:
            state = json.load(f)

        # Note: Full deserialization would require reconstructing dataclass objects
        # For now, just print what's available
        print(f"Loaded state: {state.get('experiment_status', 'unknown')}")

    def export_for_paper(self) -> str:
        """Export results formatted for Nature Methods paper."""
        lines = [
            "=" * 60,
            "NATURE METHODS RESULTS SUMMARY",
            "=" * 60,
            "",
        ]

        if self.tier0:
            lines.extend([
                "Classical Baseline:",
                f"  Best method: {self.tier0.best_method}",
                f"  R²: {self.tier0.best_r2:.4f}",
                "",
            ])

        if self.tier2:
            arch, loss, cond = self.tier2.best_combination
            lines.extend([
                "Winning Configuration:",
                f"  Architecture: {arch}",
                f"  Loss: {loss}",
                f"  Conditioning: {cond}",
                f"  R²: {self.tier2.best_r2_mean:.4f} +/- {self.tier2.best_r2_std:.4f}",
                "",
            ])

        if self.tier4:
            lines.extend([
                "Final Validation (Holdout Test Set):",
                f"  R²: {self.tier4.winner_r2:.4f}",
                f"  95% CI: [{self.tier4.winner_ci[0]:.4f}, {self.tier4.winner_ci[1]:.4f}]",
                f"  Improvement over classical: {self.tier4.improvement_over_classical:.4f}",
                f"  Effect size (Cohen's d): {self.tier4.effect_size_vs_classical:.2f}",
                f"  Negative controls: {'PASS' if self.tier4.all_controls_passed else 'FAIL'}",
                "",
            ])

        return "\n".join(lines)


# =============================================================================
# Global Registry Instance
# =============================================================================

# Singleton instance for easy access
_global_registry: Optional[ConfigRegistry] = None


def get_registry(artifacts_dir: Optional[Path] = None) -> ConfigRegistry:
    """Get or create the global config registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ConfigRegistry(artifacts_dir)
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _global_registry
    _global_registry = None
