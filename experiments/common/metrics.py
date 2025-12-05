"""
Metrics computation for HPO experiments.

Shared across all 6 studies for Nature Methods publication.

Includes:
- Basic metrics (R², correlation, MAE)
- Per-band metrics for neural frequency bands
- Phase metrics (PLV, PLI)
- Phase-Amplitude Coupling (PAC)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as scipy_signal


# Neural frequency bands for PSD analysis
NEURAL_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 100),
}

# PAC frequency pairs (phase_band, amplitude_band)
PAC_PAIRS = {
    "theta_gamma": ("theta", "gamma"),
    "alpha_gamma": ("alpha", "gamma"),
    "delta_theta": ("delta", "theta"),
}


def pearson_correlation(
    pred: torch.Tensor,
    target: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Compute Pearson correlation coefficient.

    Args:
        pred: Predicted signal [B, C, T]
        target: Target signal [B, C, T]
        dim: Dimension to compute correlation over (default: time)

    Returns:
        Correlation coefficient [B, C] or scalar if reduced
    """
    pred_centered = pred - pred.mean(dim=dim, keepdim=True)
    target_centered = target - target.mean(dim=dim, keepdim=True)

    numerator = (pred_centered * target_centered).sum(dim=dim)
    denominator = torch.sqrt(
        (pred_centered ** 2).sum(dim=dim) * (target_centered ** 2).sum(dim=dim)
    )

    return numerator / (denominator + 1e-8)


def explained_variance(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute R² (explained variance ratio).

    Args:
        pred: Predicted signal [B, C, T]
        target: Target signal [B, C, T]

    Returns:
        R² value (scalar)
    """
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return 1 - ss_res / (ss_tot + 1e-8)


def compute_psd(
    signal: torch.Tensor,
    n_fft: int = 1024,
    sample_rate: float = 1000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute power spectral density using Welch's method.

    Args:
        signal: Input signal [B, C, T]
        n_fft: FFT size
        sample_rate: Sampling rate in Hz

    Returns:
        (psd, freqs): PSD values and corresponding frequencies
    """
    # Simple periodogram (can be extended to Welch's method)
    B, C, T = signal.shape

    # Zero-pad to n_fft if needed
    if T < n_fft:
        signal = F.pad(signal, (0, n_fft - T))

    # Compute FFT
    fft = torch.fft.rfft(signal, n=n_fft, dim=-1)
    psd = (fft.abs() ** 2) / n_fft

    # Frequency bins
    freqs = torch.fft.rfftfreq(n_fft, d=1/sample_rate)

    return psd, freqs


def compute_psd_error_db(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_fft: int = 1024,
    sample_rate: float = 1000.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """Compute PSD error in dB, overall and per-band.

    Args:
        pred: Predicted signal [B, C, T]
        target: Target signal [B, C, T]
        n_fft: FFT size
        sample_rate: Sampling rate in Hz
        bands: Frequency bands to compute (default: neural bands)

    Returns:
        Dictionary with 'overall' and per-band PSD errors in dB
    """
    if bands is None:
        bands = NEURAL_BANDS

    pred_psd, freqs = compute_psd(pred, n_fft, sample_rate)
    target_psd, _ = compute_psd(target, n_fft, sample_rate)

    freqs = freqs.to(pred.device)

    results = {}

    # Overall PSD error (log-space L1)
    log_pred = torch.log10(pred_psd + 1e-10)
    log_target = torch.log10(target_psd + 1e-10)
    overall_error_db = 10 * (log_pred - log_target).abs().mean().item()
    results["overall"] = overall_error_db

    # Per-band errors
    for band_name, (f_low, f_high) in bands.items():
        mask = (freqs >= f_low) & (freqs < f_high)
        if mask.sum() == 0:
            continue

        band_pred = pred_psd[..., mask]
        band_target = target_psd[..., mask]

        log_pred_band = torch.log10(band_pred + 1e-10)
        log_target_band = torch.log10(band_target + 1e-10)

        band_error_db = 10 * (log_pred_band - log_target_band).abs().mean().item()
        results[band_name] = band_error_db

    return results


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_rate: float = 1000.0,
) -> Dict[str, float]:
    """Compute all metrics for a prediction.

    Args:
        pred: Predicted signal [B, C, T]
        target: Target signal [B, C, T]
        sample_rate: Sampling rate in Hz

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # Correlation (mean across batch and channels)
    corr = pearson_correlation(pred, target, dim=-1)
    metrics["corr"] = corr.mean().item()
    metrics["corr_std"] = corr.std().item()

    # R² (explained variance)
    r2 = explained_variance(pred, target)
    metrics["r2"] = r2.item()

    # MAE
    mae = (pred - target).abs().mean()
    metrics["mae"] = mae.item()

    # PSD errors
    psd_errors = compute_psd_error_db(pred, target, sample_rate=sample_rate)
    metrics["psd_err_db"] = psd_errors["overall"]
    for band_name, error in psd_errors.items():
        if band_name != "overall":
            metrics[f"psd_err_db_{band_name}"] = error

    return metrics


def compute_batch_metrics(
    pred_list: List[torch.Tensor],
    target_list: List[torch.Tensor],
    sample_rate: float = 1000.0,
) -> Dict[str, float]:
    """Compute metrics over multiple batches.

    Args:
        pred_list: List of predictions
        target_list: List of targets
        sample_rate: Sampling rate in Hz

    Returns:
        Aggregated metrics
    """
    all_corrs = []
    all_r2s = []
    all_maes = []
    all_psd_errs = []

    for pred, target in zip(pred_list, target_list):
        corr = pearson_correlation(pred, target, dim=-1)
        all_corrs.append(corr.mean().item())

        r2 = explained_variance(pred, target)
        all_r2s.append(r2.item())

        mae = (pred - target).abs().mean()
        all_maes.append(mae.item())

        psd_err = compute_psd_error_db(pred, target, sample_rate=sample_rate)
        all_psd_errs.append(psd_err["overall"])

    return {
        "corr": np.mean(all_corrs),
        "corr_std": np.std(all_corrs),
        "r2": np.mean(all_r2s),
        "r2_std": np.std(all_r2s),
        "mae": np.mean(all_maes),
        "psd_err_db": np.mean(all_psd_errs),
    }


# ==============================================================================
# Per-Band Metrics
# ==============================================================================


def bandpass_filter(
    signal: torch.Tensor,
    f_low: float,
    f_high: float,
    sample_rate: float = 1000.0,
    order: int = 4,
) -> torch.Tensor:
    """Apply bandpass filter to signal.

    Args:
        signal: Input signal [B, C, T]
        f_low: Low cutoff frequency
        f_high: High cutoff frequency
        sample_rate: Sampling rate in Hz
        order: Filter order

    Returns:
        Filtered signal [B, C, T]
    """
    # Convert to numpy for scipy filtering
    device = signal.device
    dtype = signal.dtype
    signal_np = signal.cpu().numpy()

    # Design Butterworth filter
    nyq = sample_rate / 2
    low = f_low / nyq
    high = min(f_high / nyq, 0.99)  # Ensure below Nyquist

    if low >= high:
        return signal  # Invalid range, return original

    try:
        b, a = scipy_signal.butter(order, [low, high], btype='band')
        # Apply filter along last axis
        filtered = scipy_signal.filtfilt(b, a, signal_np, axis=-1)
    except Exception:
        return signal  # Return original if filtering fails

    return torch.from_numpy(filtered.copy()).to(device=device, dtype=dtype)


def compute_per_band_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_rate: float = 1000.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """Compute R² and correlation for each frequency band.

    Args:
        pred: Predicted signal [B, C, T]
        target: Target signal [B, C, T]
        sample_rate: Sampling rate in Hz
        bands: Frequency bands to compute

    Returns:
        Dictionary with per-band metrics
    """
    if bands is None:
        bands = NEURAL_BANDS

    metrics = {}

    for band_name, (f_low, f_high) in bands.items():
        # Filter both signals to this band
        pred_band = bandpass_filter(pred, f_low, f_high, sample_rate)
        target_band = bandpass_filter(target, f_low, f_high, sample_rate)

        # Compute correlation
        corr = pearson_correlation(pred_band, target_band, dim=-1)
        metrics[f"corr_{band_name}"] = corr.mean().item()

        # Compute R²
        r2 = explained_variance(pred_band, target_band)
        metrics[f"r2_{band_name}"] = r2.item()

    return metrics


# ==============================================================================
# Phase Metrics (PLV, PLI)
# ==============================================================================


def hilbert_transform(signal: torch.Tensor) -> torch.Tensor:
    """Compute analytic signal using Hilbert transform.

    Args:
        signal: Input signal [B, C, T]

    Returns:
        Analytic signal (complex) [B, C, T]
    """
    # Use FFT-based Hilbert transform
    N = signal.shape[-1]

    # FFT
    fft = torch.fft.fft(signal, dim=-1)

    # Create filter: h[0] = 1, h[1:N/2] = 2, h[N/2] = 1, h[N/2+1:] = 0
    h = torch.zeros(N, device=signal.device, dtype=signal.dtype)
    if N % 2 == 0:
        h[0] = 1
        h[1:N//2] = 2
        h[N//2] = 1
    else:
        h[0] = 1
        h[1:(N+1)//2] = 2

    # Apply filter and IFFT
    analytic = torch.fft.ifft(fft * h, dim=-1)

    return analytic


def compute_instantaneous_phase(signal: torch.Tensor) -> torch.Tensor:
    """Compute instantaneous phase of signal.

    Args:
        signal: Input signal [B, C, T]

    Returns:
        Phase [B, C, T] in radians
    """
    analytic = hilbert_transform(signal)
    phase = torch.angle(analytic)
    return phase


def compute_plv(
    signal1: torch.Tensor,
    signal2: torch.Tensor,
    sample_rate: float = 1000.0,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
) -> torch.Tensor:
    """Compute Phase Locking Value between two signals.

    PLV = |mean(exp(i * (phase1 - phase2)))|

    Args:
        signal1: First signal [B, C, T]
        signal2: Second signal [B, C, T]
        sample_rate: Sampling rate
        f_low: Optional low frequency for bandpass
        f_high: Optional high frequency for bandpass

    Returns:
        PLV value [B, C]
    """
    # Optionally filter signals
    if f_low is not None and f_high is not None:
        signal1 = bandpass_filter(signal1, f_low, f_high, sample_rate)
        signal2 = bandpass_filter(signal2, f_low, f_high, sample_rate)

    # Get instantaneous phases
    phase1 = compute_instantaneous_phase(signal1)
    phase2 = compute_instantaneous_phase(signal2)

    # Compute phase difference
    phase_diff = phase1 - phase2

    # PLV = |mean(exp(i * phase_diff))|
    complex_exp = torch.exp(1j * phase_diff.to(torch.complex64))
    plv = complex_exp.mean(dim=-1).abs()

    return plv.real


def compute_pli(
    signal1: torch.Tensor,
    signal2: torch.Tensor,
    sample_rate: float = 1000.0,
    f_low: Optional[float] = None,
    f_high: Optional[float] = None,
) -> torch.Tensor:
    """Compute Phase Lag Index between two signals.

    PLI = |mean(sign(phase1 - phase2))|

    More robust to volume conduction than PLV.

    Args:
        signal1: First signal [B, C, T]
        signal2: Second signal [B, C, T]
        sample_rate: Sampling rate
        f_low: Optional low frequency for bandpass
        f_high: Optional high frequency for bandpass

    Returns:
        PLI value [B, C]
    """
    # Optionally filter signals
    if f_low is not None and f_high is not None:
        signal1 = bandpass_filter(signal1, f_low, f_high, sample_rate)
        signal2 = bandpass_filter(signal2, f_low, f_high, sample_rate)

    # Get instantaneous phases
    phase1 = compute_instantaneous_phase(signal1)
    phase2 = compute_instantaneous_phase(signal2)

    # Compute phase difference
    phase_diff = phase1 - phase2

    # PLI = |mean(sign(sin(phase_diff)))|
    pli = torch.abs(torch.sign(torch.sin(phase_diff)).mean(dim=-1))

    return pli


def compute_phase_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_rate: float = 1000.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """Compute phase-based metrics between predicted and target signals.

    Args:
        pred: Predicted signal [B, C, T]
        target: Target signal [B, C, T]
        sample_rate: Sampling rate
        bands: Frequency bands for filtering

    Returns:
        Dictionary with phase metrics
    """
    if bands is None:
        bands = NEURAL_BANDS

    metrics = {}

    # Overall PLV and PLI (no filtering)
    plv_overall = compute_plv(pred, target, sample_rate)
    pli_overall = compute_pli(pred, target, sample_rate)
    metrics["plv"] = plv_overall.mean().item()
    metrics["pli"] = pli_overall.mean().item()

    # Per-band PLV
    for band_name, (f_low, f_high) in bands.items():
        plv = compute_plv(pred, target, sample_rate, f_low, f_high)
        metrics[f"plv_{band_name}"] = plv.mean().item()

    return metrics


# ==============================================================================
# Phase-Amplitude Coupling (PAC)
# ==============================================================================


def compute_pac(
    signal: torch.Tensor,
    phase_band: Tuple[float, float],
    amp_band: Tuple[float, float],
    sample_rate: float = 1000.0,
    n_bins: int = 18,
) -> torch.Tensor:
    """Compute Phase-Amplitude Coupling using Modulation Index.

    PAC measures how much the amplitude of a high-frequency signal
    is modulated by the phase of a low-frequency signal.

    Args:
        signal: Input signal [B, C, T]
        phase_band: (f_low, f_high) for phase signal
        amp_band: (f_low, f_high) for amplitude signal
        sample_rate: Sampling rate
        n_bins: Number of phase bins for MI calculation

    Returns:
        Modulation Index [B, C]
    """
    # Filter for phase signal (low freq)
    phase_signal = bandpass_filter(signal, phase_band[0], phase_band[1], sample_rate)

    # Filter for amplitude signal (high freq)
    amp_signal = bandpass_filter(signal, amp_band[0], amp_band[1], sample_rate)

    # Get phase of low-freq signal
    phase = compute_instantaneous_phase(phase_signal)

    # Get amplitude envelope of high-freq signal
    analytic_amp = hilbert_transform(amp_signal)
    amplitude = analytic_amp.abs()

    # Compute Modulation Index using phase-binned amplitude
    # Bin phases from -pi to pi
    phase_bins = torch.linspace(-np.pi, np.pi, n_bins + 1, device=signal.device)

    B, C, T = signal.shape
    mean_amp = torch.zeros(B, C, n_bins, device=signal.device)

    for i in range(n_bins):
        mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
        # Handle edge case: ensure we have some samples in each bin
        for b in range(B):
            for c in range(C):
                bin_mask = mask[b, c]
                if bin_mask.sum() > 0:
                    mean_amp[b, c, i] = amplitude[b, c][bin_mask].mean()
                else:
                    mean_amp[b, c, i] = amplitude[b, c].mean()

    # Normalize to get distribution
    p = mean_amp / (mean_amp.sum(dim=-1, keepdim=True) + 1e-10)

    # Modulation Index = (H_max - H) / H_max
    # where H = -sum(p * log(p)) is entropy
    # H_max = log(n_bins) for uniform distribution
    H = -(p * torch.log(p + 1e-10)).sum(dim=-1)
    H_max = np.log(n_bins)
    MI = (H_max - H) / H_max

    return MI


def compute_pac_preservation(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_rate: float = 1000.0,
    pac_pairs: Optional[Dict[str, Tuple[str, str]]] = None,
) -> Dict[str, float]:
    """Compute PAC preservation between predicted and target signals.

    Args:
        pred: Predicted signal [B, C, T]
        target: Target signal [B, C, T]
        sample_rate: Sampling rate
        pac_pairs: Dict mapping name to (phase_band_name, amp_band_name)

    Returns:
        Dictionary with PAC metrics
    """
    if pac_pairs is None:
        pac_pairs = PAC_PAIRS

    metrics = {}

    for pair_name, (phase_band_name, amp_band_name) in pac_pairs.items():
        phase_band = NEURAL_BANDS[phase_band_name]
        amp_band = NEURAL_BANDS[amp_band_name]

        # Compute PAC for both signals
        pac_pred = compute_pac(pred, phase_band, amp_band, sample_rate)
        pac_target = compute_pac(target, phase_band, amp_band, sample_rate)

        # PAC preservation ratio (how much of target PAC is preserved)
        # Values close to 1 are good
        pac_ratio = pac_pred / (pac_target + 1e-10)

        metrics[f"pac_{pair_name}_pred"] = pac_pred.mean().item()
        metrics[f"pac_{pair_name}_target"] = pac_target.mean().item()
        metrics[f"pac_{pair_name}_ratio"] = pac_ratio.mean().item()

        # Correlation of PAC values (preservation of coupling pattern)
        if pac_pred.numel() > 1:
            corr = pearson_correlation(
                pac_pred.flatten().unsqueeze(0).unsqueeze(0),
                pac_target.flatten().unsqueeze(0).unsqueeze(0),
            )
            metrics[f"pac_{pair_name}_corr"] = corr.mean().item()

    return metrics


# ==============================================================================
# Comprehensive Metrics
# ==============================================================================


def compute_comprehensive_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_rate: float = 1000.0,
    include_per_band: bool = True,
    include_phase: bool = True,
    include_pac: bool = True,
    subsample_for_pac: int = 4,
) -> Dict[str, float]:
    """Compute all metrics for comprehensive evaluation.

    This is the main function for Nature Methods evaluation.

    Args:
        pred: Predicted signal [B, C, T]
        target: Target signal [B, C, T]
        sample_rate: Sampling rate in Hz
        include_per_band: Include per-band R² and correlation
        include_phase: Include PLV and PLI metrics
        include_pac: Include PAC metrics
        subsample_for_pac: Subsample factor for PAC (speed optimization)

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # Basic metrics (always computed)
    basic = compute_metrics(pred, target, sample_rate)
    metrics.update(basic)

    # Per-band metrics
    if include_per_band:
        per_band = compute_per_band_metrics(pred, target, sample_rate)
        metrics.update(per_band)

    # Phase metrics (PLV, PLI)
    if include_phase:
        try:
            phase = compute_phase_metrics(pred, target, sample_rate)
            metrics.update(phase)
        except Exception:
            # Phase metrics can fail on short signals
            pass

    # PAC metrics (subsample for speed)
    if include_pac:
        try:
            # Subsample to speed up PAC computation
            if subsample_for_pac > 1:
                pred_sub = pred[..., ::subsample_for_pac]
                target_sub = target[..., ::subsample_for_pac]
                sr_sub = sample_rate / subsample_for_pac
            else:
                pred_sub = pred
                target_sub = target
                sr_sub = sample_rate

            pac = compute_pac_preservation(pred_sub, target_sub, sr_sub)
            metrics.update(pac)
        except Exception:
            # PAC can fail on very short signals
            pass

    return metrics


def validate_metrics(metrics: Dict[str, float]) -> bool:
    """Validate that metrics contain no NaN or Inf values.

    Args:
        metrics: Dictionary of metric values

    Returns:
        True if all metrics are valid
    """
    for name, value in metrics.items():
        if not np.isfinite(value):
            return False
    return True
