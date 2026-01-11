"""
DSP Metrics Module for Neural Signal Analysis
==============================================

Comprehensive signal processing metrics for evaluating neural signal translation quality.
These metrics capture both spectral and connectivity properties essential for Nature Methods publication.

Metrics included:
- Power Spectral Density (PSD)
- Phase Locking Value (PLV)
- Phase Lag Index (PLI)
- Weighted Phase Lag Index (wPLI)
- Signal-to-Noise Ratio (SNR)
- Coherence
- Cross-correlation
- Envelope correlation
- Band-specific power
- Spectral entropy
- Mutual information
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from scipy.signal import hilbert, welch, coherence as scipy_coherence
from scipy.fft import fft, fftfreq
import warnings


# ============================================================================
# Constants - Neural frequency bands (in Hz)
# ============================================================================

FREQUENCY_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'low_gamma': (30.0, 50.0),
    'high_gamma': (50.0, 100.0),
    'gamma': (30.0, 100.0),  # Combined gamma
}


# ============================================================================
# Data classes for results
# ============================================================================

@dataclass
class PSDResult:
    """Power Spectral Density result."""
    frequencies: np.ndarray
    psd: np.ndarray
    band_powers: Dict[str, float]
    total_power: float
    dominant_frequency: float
    spectral_entropy: float


@dataclass
class ConnectivityResult:
    """Connectivity metrics between two signals."""
    plv: float  # Phase Locking Value
    pli: float  # Phase Lag Index
    wpli: float  # Weighted Phase Lag Index
    coherence_mean: float  # Mean coherence across frequencies
    coherence_bands: Dict[str, float]  # Band-specific coherence
    cross_correlation_max: float
    cross_correlation_lag: int
    envelope_correlation: float


@dataclass
class SNRResult:
    """Signal-to-Noise Ratio results."""
    snr_db: float  # SNR in decibels
    snr_linear: float  # Linear SNR
    signal_power: float
    noise_power: float


@dataclass
class ComprehensiveMetrics:
    """All DSP metrics combined."""
    # Reconstruction quality
    mse: float
    rmse: float
    mae: float
    r2: float
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float

    # Spectral metrics
    psd_correlation: float  # Correlation between PSDs
    psd_rmse: float  # RMSE between normalized PSDs
    band_power_errors: Dict[str, float]  # Per-band power errors
    spectral_entropy_error: float

    # Connectivity/phase metrics
    plv: float
    pli: float
    wpli: float
    coherence_mean: float
    coherence_bands: Dict[str, float]

    # Temporal metrics
    cross_correlation_max: float
    cross_correlation_lag: int
    envelope_correlation: float

    # SNR metrics
    snr_db: float
    reconstruction_snr_db: float  # SNR of reconstruction vs original

    # Additional
    normalized_mse: float  # MSE normalized by signal variance
    explained_variance: float


# ============================================================================
# Power Spectral Density
# ============================================================================

def compute_psd(
    signal_data: np.ndarray,
    fs: float = 1000.0,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    method: str = 'welch'
) -> PSDResult:
    """
    Compute Power Spectral Density of a signal.

    Parameters
    ----------
    signal_data : np.ndarray
        1D signal array
    fs : float
        Sampling frequency in Hz
    nperseg : int
        Length of each segment for Welch's method
    noverlap : int, optional
        Overlap between segments. Default is nperseg // 2
    method : str
        'welch' or 'periodogram'

    Returns
    -------
    PSDResult
        PSD results including band powers and spectral features
    """
    signal_data = np.asarray(signal_data).flatten()

    if noverlap is None:
        noverlap = nperseg // 2

    # Compute PSD
    if method == 'welch':
        frequencies, psd = welch(signal_data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    else:
        frequencies, psd = signal.periodogram(signal_data, fs=fs)

    # Compute band powers
    band_powers = {}
    for band_name, (low, high) in FREQUENCY_BANDS.items():
        band_mask = (frequencies >= low) & (frequencies <= high)
        if np.any(band_mask):
            band_powers[band_name] = np.trapz(psd[band_mask], frequencies[band_mask])
        else:
            band_powers[band_name] = 0.0

    # Total power
    total_power = np.trapz(psd, frequencies)

    # Dominant frequency
    dominant_freq_idx = np.argmax(psd)
    dominant_frequency = frequencies[dominant_freq_idx]

    # Spectral entropy
    psd_norm = psd / (np.sum(psd) + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

    return PSDResult(
        frequencies=frequencies,
        psd=psd,
        band_powers=band_powers,
        total_power=total_power,
        dominant_frequency=dominant_frequency,
        spectral_entropy=spectral_entropy
    )


def compare_psd(
    original: np.ndarray,
    reconstructed: np.ndarray,
    fs: float = 1000.0,
    nperseg: int = 256
) -> Dict[str, float]:
    """
    Compare PSDs of original and reconstructed signals.

    Returns
    -------
    dict
        Dictionary with PSD comparison metrics
    """
    psd_orig = compute_psd(original, fs=fs, nperseg=nperseg)
    psd_recon = compute_psd(reconstructed, fs=fs, nperseg=nperseg)

    # Normalize PSDs for comparison
    psd_orig_norm = psd_orig.psd / (np.sum(psd_orig.psd) + 1e-10)
    psd_recon_norm = psd_recon.psd / (np.sum(psd_recon.psd) + 1e-10)

    # PSD correlation
    psd_corr, _ = pearsonr(psd_orig_norm, psd_recon_norm)

    # PSD RMSE
    psd_rmse = np.sqrt(np.mean((psd_orig_norm - psd_recon_norm) ** 2))

    # Band power errors (relative)
    band_power_errors = {}
    for band in FREQUENCY_BANDS.keys():
        orig_power = psd_orig.band_powers.get(band, 0)
        recon_power = psd_recon.band_powers.get(band, 0)
        if orig_power > 1e-10:
            band_power_errors[band] = abs(recon_power - orig_power) / orig_power
        else:
            band_power_errors[band] = 0.0

    # Spectral entropy error
    entropy_error = abs(psd_orig.spectral_entropy - psd_recon.spectral_entropy)

    return {
        'psd_correlation': psd_corr,
        'psd_rmse': psd_rmse,
        'band_power_errors': band_power_errors,
        'spectral_entropy_error': entropy_error,
        'original_psd': psd_orig,
        'reconstructed_psd': psd_recon
    }


# ============================================================================
# Phase-based Connectivity Metrics
# ============================================================================

def compute_analytic_signal(signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute analytic signal using Hilbert transform.

    Returns
    -------
    amplitude : np.ndarray
        Instantaneous amplitude (envelope)
    phase : np.ndarray
        Instantaneous phase
    """
    analytic = hilbert(signal_data)
    amplitude = np.abs(analytic)
    phase = np.angle(analytic)
    return amplitude, phase


def compute_plv(
    signal1: np.ndarray,
    signal2: np.ndarray,
    fs: float = 1000.0,
    band: Optional[Tuple[float, float]] = None
) -> float:
    """
    Compute Phase Locking Value between two signals.

    PLV measures the consistency of phase difference between two signals.
    PLV = 1 indicates perfect phase locking, PLV = 0 indicates no phase relationship.

    Parameters
    ----------
    signal1 : np.ndarray
        First signal
    signal2 : np.ndarray
        Second signal
    fs : float
        Sampling frequency
    band : tuple, optional
        (low, high) frequency band to filter before computing PLV

    Returns
    -------
    float
        Phase Locking Value [0, 1]
    """
    signal1 = np.asarray(signal1).flatten()
    signal2 = np.asarray(signal2).flatten()

    # Optionally bandpass filter
    if band is not None:
        low, high = band
        nyq = fs / 2
        if high < nyq:
            b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
            signal1 = signal.filtfilt(b, a, signal1)
            signal2 = signal.filtfilt(b, a, signal2)

    # Get instantaneous phases
    _, phase1 = compute_analytic_signal(signal1)
    _, phase2 = compute_analytic_signal(signal2)

    # Compute phase difference
    phase_diff = phase1 - phase2

    # PLV is the magnitude of the mean complex exponential
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))

    return float(plv)


def compute_pli(
    signal1: np.ndarray,
    signal2: np.ndarray,
    fs: float = 1000.0,
    band: Optional[Tuple[float, float]] = None
) -> float:
    """
    Compute Phase Lag Index between two signals.

    PLI is insensitive to volume conduction (zero-lag effects).
    It measures the asymmetry of phase difference distribution.

    Parameters
    ----------
    signal1 : np.ndarray
        First signal
    signal2 : np.ndarray
        Second signal
    fs : float
        Sampling frequency
    band : tuple, optional
        (low, high) frequency band to filter

    Returns
    -------
    float
        Phase Lag Index [0, 1]
    """
    signal1 = np.asarray(signal1).flatten()
    signal2 = np.asarray(signal2).flatten()

    # Optionally bandpass filter
    if band is not None:
        low, high = band
        nyq = fs / 2
        if high < nyq:
            b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
            signal1 = signal.filtfilt(b, a, signal1)
            signal2 = signal.filtfilt(b, a, signal2)

    # Get instantaneous phases
    _, phase1 = compute_analytic_signal(signal1)
    _, phase2 = compute_analytic_signal(signal2)

    # Compute phase difference
    phase_diff = phase1 - phase2

    # PLI is the mean of the sign of the imaginary part of cross-spectrum
    pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))

    return float(pli)


def compute_wpli(
    signal1: np.ndarray,
    signal2: np.ndarray,
    fs: float = 1000.0,
    band: Optional[Tuple[float, float]] = None
) -> float:
    """
    Compute Weighted Phase Lag Index between two signals.

    wPLI weights phase differences by their magnitude, reducing noise sensitivity.

    Parameters
    ----------
    signal1 : np.ndarray
        First signal
    signal2 : np.ndarray
        Second signal
    fs : float
        Sampling frequency
    band : tuple, optional
        (low, high) frequency band to filter

    Returns
    -------
    float
        Weighted Phase Lag Index [0, 1]
    """
    signal1 = np.asarray(signal1).flatten()
    signal2 = np.asarray(signal2).flatten()

    # Optionally bandpass filter
    if band is not None:
        low, high = band
        nyq = fs / 2
        if high < nyq:
            b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
            signal1 = signal.filtfilt(b, a, signal1)
            signal2 = signal.filtfilt(b, a, signal2)

    # Compute cross-spectrum
    analytic1 = hilbert(signal1)
    analytic2 = hilbert(signal2)
    cross_spectrum = analytic1 * np.conj(analytic2)

    # wPLI
    imag_cross = np.imag(cross_spectrum)
    wpli = np.abs(np.mean(imag_cross)) / np.mean(np.abs(imag_cross) + 1e-10)

    return float(wpli)


def compute_band_plv(
    signal1: np.ndarray,
    signal2: np.ndarray,
    fs: float = 1000.0
) -> Dict[str, float]:
    """
    Compute PLV for each frequency band.

    Returns
    -------
    dict
        PLV values for each frequency band
    """
    band_plv = {}
    nyq = fs / 2

    for band_name, (low, high) in FREQUENCY_BANDS.items():
        if high < nyq:
            band_plv[band_name] = compute_plv(signal1, signal2, fs=fs, band=(low, high))
        else:
            band_plv[band_name] = np.nan

    return band_plv


# ============================================================================
# Coherence
# ============================================================================

def compute_coherence(
    signal1: np.ndarray,
    signal2: np.ndarray,
    fs: float = 1000.0,
    nperseg: int = 256
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Compute magnitude-squared coherence between two signals.

    Coherence measures the linear correlation between signals at each frequency.

    Parameters
    ----------
    signal1 : np.ndarray
        First signal
    signal2 : np.ndarray
        Second signal
    fs : float
        Sampling frequency
    nperseg : int
        Segment length for coherence estimation

    Returns
    -------
    frequencies : np.ndarray
        Frequency array
    coherence : np.ndarray
        Coherence values
    band_coherence : dict
        Mean coherence for each frequency band
    """
    signal1 = np.asarray(signal1).flatten()
    signal2 = np.asarray(signal2).flatten()

    frequencies, coh = scipy_coherence(signal1, signal2, fs=fs, nperseg=nperseg)

    # Band-specific coherence
    band_coherence = {}
    for band_name, (low, high) in FREQUENCY_BANDS.items():
        band_mask = (frequencies >= low) & (frequencies <= high)
        if np.any(band_mask):
            band_coherence[band_name] = np.mean(coh[band_mask])
        else:
            band_coherence[band_name] = np.nan

    return frequencies, coh, band_coherence


# ============================================================================
# Signal-to-Noise Ratio
# ============================================================================

def compute_snr(
    signal_data: np.ndarray,
    noise_estimate: Optional[np.ndarray] = None,
    method: str = 'power'
) -> SNRResult:
    """
    Compute Signal-to-Noise Ratio.

    Parameters
    ----------
    signal_data : np.ndarray
        Signal array
    noise_estimate : np.ndarray, optional
        Known noise signal. If None, estimate from high-frequency content
    method : str
        'power' for power-based SNR, 'rms' for RMS-based SNR

    Returns
    -------
    SNRResult
        SNR results
    """
    signal_data = np.asarray(signal_data).flatten()

    if noise_estimate is not None:
        noise = np.asarray(noise_estimate).flatten()
        noise_power = np.mean(noise ** 2)
        signal_power = np.mean(signal_data ** 2)
    else:
        # Estimate noise from high-frequency content
        # Assume signal power is total minus noise
        signal_power = np.var(signal_data)
        # Use median absolute deviation as noise estimate
        noise_power = (np.median(np.abs(signal_data - np.median(signal_data))) * 1.4826) ** 2

    if noise_power < 1e-10:
        noise_power = 1e-10

    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)

    return SNRResult(
        snr_db=float(snr_db),
        snr_linear=float(snr_linear),
        signal_power=float(signal_power),
        noise_power=float(noise_power)
    )


def compute_reconstruction_snr(
    original: np.ndarray,
    reconstructed: np.ndarray
) -> float:
    """
    Compute SNR of reconstruction treating difference as noise.

    SNR = 10 * log10(var(original) / var(original - reconstructed))

    Returns
    -------
    float
        SNR in dB
    """
    original = np.asarray(original).flatten()
    reconstructed = np.asarray(reconstructed).flatten()

    signal_power = np.var(original)
    noise_power = np.var(original - reconstructed)

    if noise_power < 1e-10:
        return 100.0  # Perfect reconstruction

    snr_db = 10 * np.log10(signal_power / noise_power)
    return float(snr_db)


# ============================================================================
# Correlation Metrics
# ============================================================================

def compute_cross_correlation(
    signal1: np.ndarray,
    signal2: np.ndarray,
    max_lag: Optional[int] = None
) -> Tuple[float, int, np.ndarray]:
    """
    Compute normalized cross-correlation between two signals.

    Returns
    -------
    max_corr : float
        Maximum correlation value
    lag_at_max : int
        Lag (in samples) at maximum correlation
    correlation : np.ndarray
        Full cross-correlation array
    """
    signal1 = np.asarray(signal1).flatten()
    signal2 = np.asarray(signal2).flatten()

    # Normalize signals
    signal1 = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-10)
    signal2 = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-10)

    # Compute correlation
    correlation = np.correlate(signal1, signal2, mode='full')
    correlation = correlation / len(signal1)

    # Find max
    if max_lag is not None:
        center = len(correlation) // 2
        start = max(0, center - max_lag)
        end = min(len(correlation), center + max_lag + 1)
        correlation_subset = correlation[start:end]
        max_idx = np.argmax(np.abs(correlation_subset))
        max_corr = correlation_subset[max_idx]
        lag_at_max = max_idx - (end - start) // 2
    else:
        max_idx = np.argmax(np.abs(correlation))
        max_corr = correlation[max_idx]
        lag_at_max = max_idx - len(signal1) + 1

    return float(max_corr), int(lag_at_max), correlation


def compute_envelope_correlation(
    signal1: np.ndarray,
    signal2: np.ndarray,
    fs: float = 1000.0,
    band: Optional[Tuple[float, float]] = None
) -> float:
    """
    Compute correlation between signal envelopes.

    Envelope correlation captures amplitude coupling between signals.

    Parameters
    ----------
    signal1 : np.ndarray
        First signal
    signal2 : np.ndarray
        Second signal
    fs : float
        Sampling frequency
    band : tuple, optional
        (low, high) frequency band to filter before computing envelope

    Returns
    -------
    float
        Envelope correlation coefficient
    """
    signal1 = np.asarray(signal1).flatten()
    signal2 = np.asarray(signal2).flatten()

    # Optionally bandpass filter
    if band is not None:
        low, high = band
        nyq = fs / 2
        if high < nyq:
            b, a = signal.butter(4, [low/nyq, high/nyq], btype='band')
            signal1 = signal.filtfilt(b, a, signal1)
            signal2 = signal.filtfilt(b, a, signal2)

    # Compute envelopes
    env1, _ = compute_analytic_signal(signal1)
    env2, _ = compute_analytic_signal(signal2)

    # Correlate envelopes
    corr, _ = pearsonr(env1, env2)

    return float(corr)


# ============================================================================
# Additional Metrics
# ============================================================================

def compute_mutual_information(
    signal1: np.ndarray,
    signal2: np.ndarray,
    bins: int = 32
) -> float:
    """
    Estimate mutual information between two signals using histogram method.

    Parameters
    ----------
    signal1 : np.ndarray
        First signal
    signal2 : np.ndarray
        Second signal
    bins : int
        Number of histogram bins

    Returns
    -------
    float
        Mutual information estimate (in bits)
    """
    signal1 = np.asarray(signal1).flatten()
    signal2 = np.asarray(signal2).flatten()

    # Joint histogram
    hist_2d, _, _ = np.histogram2d(signal1, signal2, bins=bins)

    # Normalize to probability
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    # Avoid log(0)
    pxy_nz = pxy[pxy > 0]
    px_nz = px[px > 0]
    py_nz = py[py > 0]

    # Entropies
    hx = -np.sum(px_nz * np.log2(px_nz))
    hy = -np.sum(py_nz * np.log2(py_nz))
    hxy = -np.sum(pxy_nz * np.log2(pxy_nz))

    # Mutual information
    mi = hx + hy - hxy

    return float(max(0, mi))


def compute_normalized_mi(
    signal1: np.ndarray,
    signal2: np.ndarray,
    bins: int = 32
) -> float:
    """
    Compute normalized mutual information (0 to 1 range).

    NMI = MI / sqrt(H(X) * H(Y))
    """
    signal1 = np.asarray(signal1).flatten()
    signal2 = np.asarray(signal2).flatten()

    hist_2d, _, _ = np.histogram2d(signal1, signal2, bins=bins)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    px_nz = px[px > 0]
    py_nz = py[py > 0]
    pxy_nz = pxy[pxy > 0]

    hx = -np.sum(px_nz * np.log2(px_nz))
    hy = -np.sum(py_nz * np.log2(py_nz))
    hxy = -np.sum(pxy_nz * np.log2(pxy_nz))

    mi = hx + hy - hxy

    if hx * hy > 0:
        nmi = mi / np.sqrt(hx * hy)
    else:
        nmi = 0.0

    return float(max(0, min(1, nmi)))


def compute_spectral_flatness(psd: np.ndarray) -> float:
    """
    Compute spectral flatness (Wiener entropy).

    Flatness = 1 for white noise, close to 0 for tonal signals.
    """
    psd = np.asarray(psd)
    psd = psd[psd > 0]

    if len(psd) == 0:
        return 0.0

    geometric_mean = np.exp(np.mean(np.log(psd)))
    arithmetic_mean = np.mean(psd)

    if arithmetic_mean > 0:
        flatness = geometric_mean / arithmetic_mean
    else:
        flatness = 0.0

    return float(flatness)


def compute_spectral_centroid(
    frequencies: np.ndarray,
    psd: np.ndarray
) -> float:
    """
    Compute spectral centroid (center of mass of spectrum).
    """
    psd_sum = np.sum(psd)
    if psd_sum > 0:
        centroid = np.sum(frequencies * psd) / psd_sum
    else:
        centroid = 0.0
    return float(centroid)


def compute_spectral_bandwidth(
    frequencies: np.ndarray,
    psd: np.ndarray,
    centroid: Optional[float] = None
) -> float:
    """
    Compute spectral bandwidth (spread around centroid).
    """
    if centroid is None:
        centroid = compute_spectral_centroid(frequencies, psd)

    psd_sum = np.sum(psd)
    if psd_sum > 0:
        bandwidth = np.sqrt(np.sum(((frequencies - centroid) ** 2) * psd) / psd_sum)
    else:
        bandwidth = 0.0
    return float(bandwidth)


# ============================================================================
# Comprehensive Metrics Computation
# ============================================================================

def compute_all_dsp_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    fs: float = 1000.0,
    nperseg: int = 256
) -> ComprehensiveMetrics:
    """
    Compute all DSP metrics comparing original and reconstructed signals.

    This is the main function to compute comprehensive signal quality metrics.

    Parameters
    ----------
    original : np.ndarray
        Original signal
    reconstructed : np.ndarray
        Reconstructed/predicted signal
    fs : float
        Sampling frequency in Hz
    nperseg : int
        Segment length for spectral computations

    Returns
    -------
    ComprehensiveMetrics
        All metrics in a single structure
    """
    original = np.asarray(original).flatten()
    reconstructed = np.asarray(reconstructed).flatten()

    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    # Basic reconstruction metrics
    mse = float(np.mean((original - reconstructed) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(original - reconstructed)))

    # R² (coefficient of determination)
    ss_res = np.sum((original - reconstructed) ** 2)
    ss_tot = np.sum((original - np.mean(original)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))

    # Correlations
    pearson_r_val, pearson_p_val = pearsonr(original, reconstructed)
    spearman_r_val, spearman_p_val = spearmanr(original, reconstructed)

    # PSD comparison
    psd_comparison = compare_psd(original, reconstructed, fs=fs, nperseg=nperseg)

    # Phase connectivity metrics
    plv = compute_plv(original, reconstructed, fs=fs)
    pli = compute_pli(original, reconstructed, fs=fs)
    wpli = compute_wpli(original, reconstructed, fs=fs)

    # Coherence
    _, coh, band_coherence = compute_coherence(original, reconstructed, fs=fs, nperseg=nperseg)
    coherence_mean = float(np.mean(coh))

    # Cross-correlation
    max_corr, lag_at_max, _ = compute_cross_correlation(original, reconstructed, max_lag=int(fs))

    # Envelope correlation
    env_corr = compute_envelope_correlation(original, reconstructed, fs=fs)

    # SNR metrics
    snr_result = compute_snr(original)
    recon_snr = compute_reconstruction_snr(original, reconstructed)

    # Normalized MSE
    signal_var = np.var(original)
    normalized_mse = mse / (signal_var + 1e-10)

    # Explained variance
    explained_var = 1 - np.var(original - reconstructed) / (signal_var + 1e-10)

    return ComprehensiveMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
        pearson_r=float(pearson_r_val),
        pearson_p=float(pearson_p_val),
        spearman_r=float(spearman_r_val),
        spearman_p=float(spearman_p_val),
        psd_correlation=psd_comparison['psd_correlation'],
        psd_rmse=psd_comparison['psd_rmse'],
        band_power_errors=psd_comparison['band_power_errors'],
        spectral_entropy_error=psd_comparison['spectral_entropy_error'],
        plv=plv,
        pli=pli,
        wpli=wpli,
        coherence_mean=coherence_mean,
        coherence_bands=band_coherence,
        cross_correlation_max=max_corr,
        cross_correlation_lag=lag_at_max,
        envelope_correlation=env_corr,
        snr_db=snr_result.snr_db,
        reconstruction_snr_db=recon_snr,
        normalized_mse=normalized_mse,
        explained_variance=float(explained_var)
    )


def compute_connectivity_metrics(
    signal1: np.ndarray,
    signal2: np.ndarray,
    fs: float = 1000.0,
    nperseg: int = 256
) -> ConnectivityResult:
    """
    Compute all connectivity metrics between two signals.

    Use this to measure natural relationships between brain regions.

    Parameters
    ----------
    signal1 : np.ndarray
        First signal (e.g., from region A)
    signal2 : np.ndarray
        Second signal (e.g., from region B)
    fs : float
        Sampling frequency
    nperseg : int
        Segment length for spectral computations

    Returns
    -------
    ConnectivityResult
        All connectivity metrics
    """
    signal1 = np.asarray(signal1).flatten()
    signal2 = np.asarray(signal2).flatten()

    # Ensure same length
    min_len = min(len(signal1), len(signal2))
    signal1 = signal1[:min_len]
    signal2 = signal2[:min_len]

    # Phase metrics
    plv = compute_plv(signal1, signal2, fs=fs)
    pli = compute_pli(signal1, signal2, fs=fs)
    wpli = compute_wpli(signal1, signal2, fs=fs)

    # Coherence
    _, coh, band_coherence = compute_coherence(signal1, signal2, fs=fs, nperseg=nperseg)
    coherence_mean = float(np.mean(coh))

    # Cross-correlation
    max_corr, lag_at_max, _ = compute_cross_correlation(signal1, signal2, max_lag=int(fs))

    # Envelope correlation
    env_corr = compute_envelope_correlation(signal1, signal2, fs=fs)

    return ConnectivityResult(
        plv=plv,
        pli=pli,
        wpli=wpli,
        coherence_mean=coherence_mean,
        coherence_bands=band_coherence,
        cross_correlation_max=max_corr,
        cross_correlation_lag=lag_at_max,
        envelope_correlation=env_corr
    )


def compute_batch_metrics(
    original_batch: np.ndarray,
    reconstructed_batch: np.ndarray,
    fs: float = 1000.0,
    nperseg: int = 256
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Compute aggregated metrics over a batch of signals.

    Parameters
    ----------
    original_batch : np.ndarray
        Batch of original signals (batch_size, time_steps)
    reconstructed_batch : np.ndarray
        Batch of reconstructed signals (batch_size, time_steps)
    fs : float
        Sampling frequency
    nperseg : int
        Segment length

    Returns
    -------
    dict
        Aggregated metrics with mean and std
    """
    original_batch = np.atleast_2d(original_batch)
    reconstructed_batch = np.atleast_2d(reconstructed_batch)

    n_samples = min(original_batch.shape[0], reconstructed_batch.shape[0])

    # Collect all metrics
    all_metrics = []
    for i in range(n_samples):
        metrics = compute_all_dsp_metrics(
            original_batch[i],
            reconstructed_batch[i],
            fs=fs,
            nperseg=nperseg
        )
        all_metrics.append(metrics)

    # Aggregate scalar metrics
    aggregated = {}
    scalar_fields = ['mse', 'rmse', 'mae', 'r2', 'pearson_r', 'spearman_r',
                     'psd_correlation', 'psd_rmse', 'spectral_entropy_error',
                     'plv', 'pli', 'wpli', 'coherence_mean',
                     'cross_correlation_max', 'envelope_correlation',
                     'snr_db', 'reconstruction_snr_db', 'normalized_mse', 'explained_variance']

    for field in scalar_fields:
        values = [getattr(m, field) for m in all_metrics]
        aggregated[f'{field}_mean'] = float(np.mean(values))
        aggregated[f'{field}_std'] = float(np.std(values))
        aggregated[f'{field}_values'] = values

    # Aggregate band-specific metrics
    bands = list(FREQUENCY_BANDS.keys())
    for band in bands:
        # Coherence bands
        coh_values = [m.coherence_bands.get(band, np.nan) for m in all_metrics]
        coh_values = [v for v in coh_values if not np.isnan(v)]
        if coh_values:
            aggregated[f'coherence_{band}_mean'] = float(np.mean(coh_values))
            aggregated[f'coherence_{band}_std'] = float(np.std(coh_values))

        # Band power errors
        power_errors = [m.band_power_errors.get(band, np.nan) for m in all_metrics]
        power_errors = [v for v in power_errors if not np.isnan(v)]
        if power_errors:
            aggregated[f'band_power_error_{band}_mean'] = float(np.mean(power_errors))
            aggregated[f'band_power_error_{band}_std'] = float(np.std(power_errors))

    return aggregated


# ============================================================================
# Utility functions for saving/loading
# ============================================================================

def metrics_to_dict(metrics: ComprehensiveMetrics) -> Dict:
    """Convert ComprehensiveMetrics to a JSON-serializable dictionary."""
    return {
        'mse': metrics.mse,
        'rmse': metrics.rmse,
        'mae': metrics.mae,
        'r2': metrics.r2,
        'pearson_r': metrics.pearson_r,
        'pearson_p': metrics.pearson_p,
        'spearman_r': metrics.spearman_r,
        'spearman_p': metrics.spearman_p,
        'psd_correlation': metrics.psd_correlation,
        'psd_rmse': metrics.psd_rmse,
        'band_power_errors': metrics.band_power_errors,
        'spectral_entropy_error': metrics.spectral_entropy_error,
        'plv': metrics.plv,
        'pli': metrics.pli,
        'wpli': metrics.wpli,
        'coherence_mean': metrics.coherence_mean,
        'coherence_bands': metrics.coherence_bands,
        'cross_correlation_max': metrics.cross_correlation_max,
        'cross_correlation_lag': metrics.cross_correlation_lag,
        'envelope_correlation': metrics.envelope_correlation,
        'snr_db': metrics.snr_db,
        'reconstruction_snr_db': metrics.reconstruction_snr_db,
        'normalized_mse': metrics.normalized_mse,
        'explained_variance': metrics.explained_variance,
    }


def connectivity_to_dict(conn: ConnectivityResult) -> Dict:
    """Convert ConnectivityResult to a JSON-serializable dictionary."""
    return {
        'plv': conn.plv,
        'pli': conn.pli,
        'wpli': conn.wpli,
        'coherence_mean': conn.coherence_mean,
        'coherence_bands': conn.coherence_bands,
        'cross_correlation_max': conn.cross_correlation_max,
        'cross_correlation_lag': conn.cross_correlation_lag,
        'envelope_correlation': conn.envelope_correlation,
    }


__all__ = [
    # Constants
    'FREQUENCY_BANDS',
    # Data classes
    'PSDResult',
    'ConnectivityResult',
    'SNRResult',
    'ComprehensiveMetrics',
    # PSD functions
    'compute_psd',
    'compare_psd',
    # Phase connectivity
    'compute_analytic_signal',
    'compute_plv',
    'compute_pli',
    'compute_wpli',
    'compute_band_plv',
    # Coherence
    'compute_coherence',
    # SNR
    'compute_snr',
    'compute_reconstruction_snr',
    # Correlations
    'compute_cross_correlation',
    'compute_envelope_correlation',
    # Information theory
    'compute_mutual_information',
    'compute_normalized_mi',
    # Spectral features
    'compute_spectral_flatness',
    'compute_spectral_centroid',
    'compute_spectral_bandwidth',
    # Main functions
    'compute_all_dsp_metrics',
    'compute_connectivity_metrics',
    'compute_batch_metrics',
    # Utilities
    'metrics_to_dict',
    'connectivity_to_dict',
]
