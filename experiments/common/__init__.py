"""Common utilities shared across all 6 studies."""

from .metrics import compute_metrics, compute_psd_error_db
from .baseline_sampler import BaselineInjectingSampler
from .reporting import compute_bootstrap_ci, export_study_results

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
