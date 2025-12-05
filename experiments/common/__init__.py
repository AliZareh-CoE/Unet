"""Common utilities shared across tier-based experiment system."""

from .metrics import compute_metrics, compute_psd_error_db
from .baseline_sampler import BaselineInjectingSampler
from .reporting import compute_bootstrap_ci, export_study_results

# Cross-validation utilities
from .cross_validation import (
    CVSplit,
    DataSplits,
    CVResult,
    CVAggregatedResult,
    create_data_splits,
    get_fold_dataloaders,
    get_holdout_dataloader,
    iterate_folds,
    aggregate_cv_results,
    run_cv_experiment,
)

# Statistical analysis
from .statistics import (
    cohens_d,
    cohens_d_interpretation,
    eta_squared,
    partial_eta_squared,
    one_way_anova,
    two_way_anova,
    three_way_anova,
    ANOVAResult,
    FactorialANOVAResult,
    bonferroni_correction,
    fdr_correction,
    pairwise_tests,
    bootstrap_ci,
    bootstrap_diff_ci,
    required_sample_size,
    achieved_power,
    permutation_test,
)

# Config registry for tier system
from .config_registry import (
    ConfigRegistry,
    GateFailure,
    RankingUnstable,
    Tier0Result,
    Tier1Result,
    Tier2Result,
    Tier3Result,
    Tier4Result,
    ClassicalResult,
    ScreeningResult,
    FactorialResult,
    AblationResult,
    ValidationResult,
    NegativeControlResult,
    get_registry,
    reset_registry,
)

# Validation and robustness infrastructure
from .validation import (
    validate_tensor_shape,
    validate_not_nan_inf,
    validate_finite_gradients,
    validate_model_output,
    validate_loss,
    validate_memory_available,
    estimate_model_memory,
    run_forward_pass_test,
    run_gradient_flow_test,
    count_parameters,
)
from .error_handling import (
    TrialError,
    OOMError,
    NaNError,
    ConvergenceError,
    trial_context,
    robust_objective,
    check_loss_valid,
    check_gradients_valid,
    safe_backward,
    EarlyStopping,
)
from .checkpointing import (
    TrialCheckpoint,
    BestModelSaver,
    cleanup_old_checkpoints,
)
