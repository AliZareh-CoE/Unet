"""
Baseline-injecting Optuna sampler.

Ensures fair comparison by running baseline trials at the start of each study.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import optuna
from optuna.distributions import BaseDistribution
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial


class BaselineInjectingSampler(TPESampler):
    """TPE sampler that injects baseline trials at study start.

    The first `n_baseline_trials` trials use baseline/default configurations
    to establish a fair comparison point. After that, TPE takes over.

    Args:
        n_baseline_trials: Number of baseline trials to inject (default: 20)
        baseline_config: Default baseline configuration values
        *args, **kwargs: Passed to TPESampler
    """

    def __init__(
        self,
        n_baseline_trials: int = 20,
        baseline_config: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        # Set reasonable TPE defaults for HPO
        kwargs.setdefault("n_startup_trials", n_baseline_trials + 10)
        kwargs.setdefault("multivariate", True)
        kwargs.setdefault("group", True)

        super().__init__(*args, **kwargs)

        self.n_baseline_trials = n_baseline_trials
        self.baseline_config = baseline_config or {}

    def sample_independent(
        self,
        study: optuna.Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        """Sample a parameter value.

        For the first n_baseline_trials, return baseline values.
        After that, use TPE sampling.
        """
        # Count completed + running trials
        n_completed = len([
            t for t in study.trials
            if t.state in (
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.RUNNING,
                optuna.trial.TrialState.PRUNED,
            )
        ])

        # First N trials use baseline configuration
        if n_completed < self.n_baseline_trials:
            # Check if we have a baseline value for this parameter
            if param_name in self.baseline_config:
                return self.baseline_config[param_name]

            # For 'approach' or 'conditioning_type', always return baseline
            if param_name in ("approach", "conditioning_type", "spectral_method"):
                return "baseline"

            # Use distribution defaults/medians for other params
            return self._get_baseline_value(param_name, param_distribution)

        # After baseline trials, use TPE
        return super().sample_independent(study, trial, param_name, param_distribution)

    def _get_baseline_value(
        self,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        """Get a reasonable baseline value for a parameter."""
        dist_class = type(param_distribution).__name__

        if dist_class == "CategoricalDistribution":
            # Return first choice (usually the default/baseline)
            return param_distribution.choices[0]

        elif dist_class == "IntDistribution":
            # Return median
            low = param_distribution.low
            high = param_distribution.high
            return (low + high) // 2

        elif dist_class == "FloatDistribution":
            low = param_distribution.low
            high = param_distribution.high
            if param_distribution.log:
                # Geometric mean for log-scale
                import math
                return math.sqrt(low * high)
            else:
                # Arithmetic mean
                return (low + high) / 2

        # Fallback: let parent handle it
        return super().sample_independent(
            optuna.create_study(),  # Dummy study
            optuna.trial.create_trial(state=optuna.trial.TrialState.RUNNING),
            param_name,
            param_distribution,
        )


class StratifiedApproachSampler(TPESampler):
    """Sampler that ensures balanced trials across approaches.

    Cycles through approaches to ensure each gets similar trial counts.
    """

    def __init__(
        self,
        approaches: Sequence[str],
        n_baseline_per_approach: int = 4,
        *args,
        **kwargs,
    ):
        kwargs.setdefault("multivariate", True)
        kwargs.setdefault("group", True)
        super().__init__(*args, **kwargs)

        self.approaches = list(approaches)
        self.n_baseline_per_approach = n_baseline_per_approach

    def sample_independent(
        self,
        study: optuna.Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        """Sample parameter, ensuring balanced approach distribution."""

        if param_name in ("approach", "conditioning_type", "spectral_method"):
            # Count trials per approach
            approach_counts = {a: 0 for a in self.approaches}
            for t in study.trials:
                if t.state in (
                    optuna.trial.TrialState.COMPLETE,
                    optuna.trial.TrialState.RUNNING,
                ):
                    a = t.params.get(param_name)
                    if a in approach_counts:
                        approach_counts[a] += 1

            # Return approach with minimum count
            min_approach = min(approach_counts, key=approach_counts.get)
            return min_approach

        return super().sample_independent(study, trial, param_name, param_distribution)
