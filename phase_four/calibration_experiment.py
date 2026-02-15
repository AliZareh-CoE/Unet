"""Uncertainty-aware calibration experiment for neural signal translation.

Compares three calibration strategies that operate on MC-Dropout uncertainty:
  1. Quantile Calibration  – isotonic regression on PIT values (Kuleshov et al. 2018)
  2. GP-Beta Distribution  – Gaussian-process recalibration with Beta link (Song et al. 2019)
  3. MMD Post-hoc          – minimise kernel MMD between predicted and true distributions
                             (Cui et al. 2020, adapted for post-hoc use)

All methods use MC-Dropout to obtain predictive mean + variance from the
existing trained model (no retraining needed).

Usage:
    python -m phase_four.calibration_experiment --dataset pfc_hpc --max-train 5000
    python -m phase_four.calibration_experiment --dataset olfactory --mc-samples 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

from phase_four.config import DATASETS, FREQUENCY_BANDS, Phase4Config

# ═════════════════════════════════════════════════════════════════════════════
# Stochastic Inference (MC-Dropout or Input Perturbation)
# ═════════════════════════════════════════════════════════════════════════════

def _has_active_dropout(model) -> bool:
    """Check if the model has any Dropout layers with p > 0."""
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            if m.p > 0:
                return True
    return False


def _enable_mc_dropout(model):
    """Enable dropout layers at inference time for MC-Dropout."""
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            m.train()


def stochastic_inference(
    model,
    loader,
    device,
    n_samples: int = 20,
    noise_std: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Run stochastic inference to get predictive mean and variance.

    Uses MC-Dropout if the model has active dropout layers (p > 0).
    Otherwise falls back to input perturbation: adds small Gaussian
    noise to inputs and measures output variance across passes.

    Returns dict with keys: mean, variance, target, source, labels, method.
    Shapes: [N, C, T] for mean/variance/target/source.
    """
    import torch

    model.eval()
    use_mc_dropout = _has_active_dropout(model)

    if use_mc_dropout:
        _enable_mc_dropout(model)
        method = "mc_dropout"
        print(f"  Using MC-Dropout (found active dropout layers)")
    else:
        method = "input_perturbation"
        print(f"  No active dropout (p=0) — using input perturbation (noise_std={noise_std})")

    # First pass: collect all input data, targets, labels
    all_src_batches = []
    targets, sources, labels_list = [], [], []
    for batch in loader:
        src, tgt = batch[0], batch[1]
        label = batch[2] if len(batch) > 2 else torch.zeros(src.shape[0])
        all_src_batches.append(src)
        sources.append(src.numpy())
        targets.append(tgt.numpy())
        labels_list.append(label.numpy() if isinstance(label, torch.Tensor) else np.array(label))

    all_sources = np.concatenate(sources, axis=0)
    all_targets = np.concatenate(targets, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    # Compute per-channel input std for scaling perturbation noise
    if method == "input_perturbation":
        input_std = np.std(all_sources, axis=(0, 2), keepdims=True)  # [1, C, 1]
        input_std = np.clip(input_std, 1e-8, None)
        input_std_t = torch.tensor(input_std, dtype=torch.float32, device=device)

    # Stochastic forward passes
    all_preds = []
    for s in range(n_samples):
        preds = []
        with torch.no_grad():
            for src_batch in all_src_batches:
                src = src_batch.to(device)
                if method == "input_perturbation":
                    noise = torch.randn_like(src) * input_std_t * noise_std
                    src = src + noise
                pred = model(src)
                preds.append(pred.cpu().numpy())
        all_preds.append(np.concatenate(preds, axis=0))

    stacked = np.stack(all_preds, axis=0)  # [S, N, C, T]
    mean = stacked.mean(axis=0)            # [N, C, T]
    variance = stacked.var(axis=0)         # [N, C, T]

    model.eval()  # restore full eval mode
    return {
        "mean": mean,
        "variance": variance,
        "target": all_targets,
        "source": all_sources,
        "labels": all_labels,
        "method": method,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Calibration Methods
# ═════════════════════════════════════════════════════════════════════════════

class QuantileCalibrator:
    """Quantile calibration via isotonic regression (Kuleshov et al. 2018).

    Fits a monotonic mapping from predicted CDF quantiles to empirical
    coverage, per channel.  At inference, adjusts the predicted Gaussian
    CDF so that the p-confidence interval actually covers with prob p.
    """

    def __init__(self):
        self.recalibrators: Dict[int, IsotonicRegression] = {}

    def fit(self, mu: np.ndarray, sigma: np.ndarray, y: np.ndarray):
        """Fit per-channel isotonic recalibrators.

        Args:
            mu:    predicted mean  [N, C, T]
            sigma: predicted std   [N, C, T]
            y:     ground truth    [N, C, T]
        """
        _, C, _ = mu.shape
        for c in range(C):
            mu_c = mu[:, c, :].ravel()
            sigma_c = np.clip(sigma[:, c, :].ravel(), 1e-8, None)
            y_c = y[:, c, :].ravel()

            # PIT values: F(y | mu, sigma) under Gaussian assumption
            pit = norm.cdf(y_c, loc=mu_c, scale=sigma_c)

            # Empirical coverage at discrete quantile levels
            q_levels = np.linspace(0.01, 0.99, 99)
            emp_coverage = np.array([np.mean(pit <= q) for q in q_levels])

            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(q_levels, emp_coverage)
            self.recalibrators[c] = iso

    def calibrated_interval(
        self, mu: np.ndarray, sigma: np.ndarray, confidence: float = 0.90
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return calibrated (lo, hi) interval arrays [N, C, T]."""
        N, C, T = mu.shape
        lo = np.empty_like(mu)
        hi = np.empty_like(mu)

        raw_grid = np.linspace(0.001, 0.999, 1000)
        alpha_lo = (1 - confidence) / 2
        alpha_hi = (1 + confidence) / 2

        for c in range(C):
            cal_grid = self.recalibrators[c].predict(raw_grid)
            raw_lo = np.interp(alpha_lo, cal_grid, raw_grid)
            raw_hi = np.interp(alpha_hi, cal_grid, raw_grid)

            sig_c = np.clip(sigma[:, c, :], 1e-8, None)
            lo[:, c, :] = norm.ppf(raw_lo, loc=mu[:, c, :], scale=sig_c)
            hi[:, c, :] = norm.ppf(raw_hi, loc=mu[:, c, :], scale=sig_c)

        return lo, hi

    def calibrated_std(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Return calibrated sigma such that 68.27% interval matches coverage.

        Approximates the calibrated spread as half the 68.27% interval width.
        """
        lo, hi = self.calibrated_interval(mu, sigma, confidence=0.6827)
        return (hi - lo) / 2.0


class GPBetaCalibrator:
    """Distribution calibration via GP-Beta (Song et al. 2019).

    Uses a Gaussian-process with Beta link to learn an input-dependent
    recalibration mapping.  Falls back to isotonic if netcal unavailable.
    """

    def __init__(self, n_inducing: int = 12, n_epochs: int = 256,
                 max_fit_samples: int = 10000):
        self.n_inducing = n_inducing
        self.n_epochs = n_epochs
        self.max_fit_samples = max_fit_samples
        self.calibrators: Dict[int, object] = {}
        self._use_netcal = True

    def fit(self, mu: np.ndarray, sigma: np.ndarray, y: np.ndarray):
        _, C, _ = mu.shape
        try:
            from netcal.regression import GPBeta
        except ImportError:
            print("  [GP-Beta] netcal not installed — falling back to isotonic")
            self._use_netcal = False
            self._fallback = QuantileCalibrator()
            self._fallback.fit(mu, sigma, y)
            return

        rng = np.random.default_rng(42)

        for c in range(C):
            mu_c = mu[:, c, :].ravel()
            sig_c = np.clip(sigma[:, c, :].ravel(), 1e-8, None)
            y_c = y[:, c, :].ravel()

            # Subsample to keep GP tractable
            if len(mu_c) > self.max_fit_samples:
                idx = rng.choice(len(mu_c), self.max_fit_samples, replace=False)
                mu_c, sig_c, y_c = mu_c[idx], sig_c[idx], y_c[idx]

            gp = GPBeta(
                n_inducing_points=self.n_inducing,
                n_random_samples=256,
                n_epochs=self.n_epochs,
                use_cuda=False,
            )
            try:
                gp.fit((mu_c[:, None].astype(np.float32),
                        sig_c[:, None].astype(np.float32)),
                       y_c[:, None].astype(np.float32))
                self.calibrators[c] = gp
            except Exception as e:
                print(f"  [GP-Beta] channel {c} failed: {e} — skipping")
                self.calibrators[c] = None

    def calibrated_std(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Return calibrated sigma via GP-Beta transform."""
        if not self._use_netcal:
            return self._fallback.calibrated_std(mu, sigma)

        N, C, T = mu.shape
        out = sigma.copy()
        for c in range(C):
            gp = self.calibrators.get(c)
            if gp is None:
                continue
            mu_c = mu[:, c, :].ravel()[:, None].astype(np.float32)
            sig_c = np.clip(sigma[:, c, :].ravel()[:, None], 1e-8, None).astype(np.float32)
            try:
                result = gp.transform((mu_c, sig_c))
                # netcal returns (mean, variance) tuple or similar
                if isinstance(result, tuple) and len(result) >= 2:
                    cal_var = result[1].ravel()
                    out[:, c, :] = np.sqrt(np.clip(cal_var, 1e-12, None)).reshape(N, T)
                elif hasattr(result, "var"):
                    out[:, c, :] = np.sqrt(np.clip(result.var().ravel(), 1e-12, None)).reshape(N, T)
            except Exception:
                pass  # keep original sigma
        return out


class MMDCalibrator:
    """Post-hoc MMD calibration (adapted from Cui et al. 2020).

    Learns per-channel scale/shift corrections to the predicted variance
    by minimising a kernel-MMD objective on held-out data.  This is a
    lightweight post-hoc adaptation of the training-time MMD loss.
    """

    def __init__(self, n_iters: int = 200, lr: float = 0.01,
                 bandwidth_multipliers: Optional[List[float]] = None):
        self.n_iters = n_iters
        self.lr = lr
        self.bw_mults = bandwidth_multipliers or [0.2, 0.5, 1.0, 2.0, 5.0]
        self.log_scale: Optional[np.ndarray] = None  # [C]
        self.shift: Optional[np.ndarray] = None       # [C]

    @staticmethod
    def _mmd2_rbf(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
        """Unbiased MMD² with single RBF kernel.  x, y: (n, d)."""
        n = x.shape[0]
        xx = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
        yy = np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1)
        xy = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)

        kxx = np.exp(-xx / (2 * bandwidth))
        kyy = np.exp(-yy / (2 * bandwidth))
        kxy = np.exp(-xy / (2 * bandwidth))

        mmd2 = (kxx.sum() - np.trace(kxx)) / (n * (n - 1)) \
             + (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) \
             - 2 * kxy.mean()
        return float(mmd2)

    def _mmd2_mixture(self, x: np.ndarray, y: np.ndarray) -> float:
        dists = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
        base_bw = float(np.median(dists[dists > 0])) + 1e-8
        return sum(self._mmd2_rbf(x, y, base_bw * m) for m in self.bw_mults)

    def fit(self, mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
            max_samples: int = 5000):
        """Learn per-channel variance scale+shift by minimising MMD.

        We optimise: sigma_cal = exp(log_scale) * sigma + shift
        so that samples from N(mu, sigma_cal²) are close in MMD to y.
        """
        _, C, _ = mu.shape
        self.log_scale = np.zeros(C)
        self.shift = np.zeros(C)

        rng = np.random.default_rng(42)

        for c in range(C):
            mu_c = mu[:, c, :].ravel()
            sig_c = np.clip(sigma[:, c, :].ravel(), 1e-8, None)
            y_c = y[:, c, :].ravel()

            # Subsample for speed
            if len(mu_c) > max_samples:
                idx = rng.choice(len(mu_c), max_samples, replace=False)
                mu_c, sig_c, y_c = mu_c[idx], sig_c[idx], y_c[idx]

            # Optimise log_scale and shift with numerical gradient descent
            ls, sh = 0.0, 0.0
            best_ls, best_sh, best_mmd = ls, sh, float("inf")

            for it in range(self.n_iters):
                scale = np.exp(ls)
                cal_sig = scale * sig_c + sh
                cal_sig = np.clip(cal_sig, 1e-8, None)

                # Sample from calibrated predictive distribution
                eps = rng.standard_normal(len(mu_c))
                y_hat = mu_c + cal_sig * eps

                # Subsample for MMD (to keep O(n²) tractable)
                n_mmd = min(500, len(mu_c))
                idx_s = rng.choice(len(mu_c), n_mmd, replace=False)
                mmd = self._mmd2_mixture(
                    y_c[idx_s, None], y_hat[idx_s, None]
                )

                if mmd < best_mmd:
                    best_mmd, best_ls, best_sh = mmd, ls, sh

                # Numerical gradient (central difference)
                delta = 1e-3
                for param_name in ["ls", "sh"]:
                    if param_name == "ls":
                        scale_p = np.exp(ls + delta)
                        cal_p = np.clip(scale_p * sig_c + sh, 1e-8, None)
                        y_hat_p = mu_c + cal_p * rng.standard_normal(len(mu_c))
                        mmd_p = self._mmd2_mixture(y_c[idx_s, None], y_hat_p[idx_s, None])

                        scale_m = np.exp(ls - delta)
                        cal_m = np.clip(scale_m * sig_c + sh, 1e-8, None)
                        y_hat_m = mu_c + cal_m * rng.standard_normal(len(mu_c))
                        mmd_m = self._mmd2_mixture(y_c[idx_s, None], y_hat_m[idx_s, None])
                        grad = (mmd_p - mmd_m) / (2 * delta)
                        ls -= self.lr * grad
                    else:
                        cal_p = np.clip(np.exp(ls) * sig_c + (sh + delta), 1e-8, None)
                        y_hat_p = mu_c + cal_p * rng.standard_normal(len(mu_c))
                        mmd_p = self._mmd2_mixture(y_c[idx_s, None], y_hat_p[idx_s, None])

                        cal_m = np.clip(np.exp(ls) * sig_c + (sh - delta), 1e-8, None)
                        y_hat_m = mu_c + cal_m * rng.standard_normal(len(mu_c))
                        mmd_m = self._mmd2_mixture(y_c[idx_s, None], y_hat_m[idx_s, None])
                        grad = (mmd_p - mmd_m) / (2 * delta)
                        sh -= self.lr * grad

            self.log_scale[c] = best_ls
            self.shift[c] = best_sh

    def calibrated_std(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Apply learned scale+shift to sigma."""
        out = sigma.copy()
        for c in range(sigma.shape[1]):
            scale = np.exp(self.log_scale[c])
            out[:, c, :] = np.clip(scale * sigma[:, c, :] + self.shift[c], 1e-8, None)
        return out


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation Metrics
# ═════════════════════════════════════════════════════════════════════════════

def coverage_at_confidence(
    mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, confidence: float = 0.90,
) -> float:
    """Fraction of ground-truth values inside the predicted interval."""
    z = norm.ppf((1 + confidence) / 2)
    lo = mu - z * sigma
    hi = mu + z * sigma
    return float(np.mean((y >= lo) & (y <= hi)))


def calibration_error(
    mu: np.ndarray, sigma: np.ndarray, y: np.ndarray,
    n_bins: int = 20,
) -> float:
    """Expected Calibration Error for regression (ECE-R).

    Average |expected_coverage(p) - p| across confidence levels.
    """
    levels = np.linspace(0.05, 0.95, n_bins)
    errors = []
    for p in levels:
        cov = coverage_at_confidence(mu, sigma, y, p)
        errors.append(abs(cov - p))
    return float(np.mean(errors))


def sharpness(sigma: np.ndarray) -> float:
    """Mean predicted standard deviation (lower = sharper)."""
    return float(np.mean(sigma))


def nll_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood under Gaussian predictive."""
    var = np.clip(sigma ** 2, 1e-12, None)
    nll = 0.5 * (np.log(2 * np.pi * var) + (y - mu) ** 2 / var)
    return float(np.mean(nll))


def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    """Continuous Ranked Probability Score for Gaussian predictive."""
    sig = np.clip(sigma, 1e-8, None)
    z = (y - mu) / sig
    crps = sig * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    """Global Pearson correlation."""
    return float(np.corrcoef(pred.ravel(), target.ravel())[0, 1])


def psd_correlation(pred: np.ndarray, target: np.ndarray, fs: int, nperseg: int = 1024) -> float:
    """Log-PSD Pearson correlation (averaged across channels)."""
    from scipy.signal import welch
    C = pred.shape[1]
    corrs = []
    for c in range(C):
        _, psd_p = welch(pred[:, c, :].ravel(), fs=fs, nperseg=nperseg)
        _, psd_t = welch(target[:, c, :].ravel(), fs=fs, nperseg=nperseg)
        log_p = 10 * np.log10(psd_p + 1e-20)
        log_t = 10 * np.log10(psd_t + 1e-20)
        corrs.append(float(np.corrcoef(log_p, log_t)[0, 1]))
    return float(np.mean(corrs))


def per_band_r2(pred: np.ndarray, target: np.ndarray, fs: int) -> Dict[str, float]:
    """Per-frequency-band R² between log-PSDs."""
    from scipy.signal import welch
    results = {}
    for band_name, (lo, hi) in FREQUENCY_BANDS.items():
        corrs = []
        for c in range(pred.shape[1]):
            freqs, psd_p = welch(pred[:, c, :].ravel(), fs=fs, nperseg=1024)
            _, psd_t = welch(target[:, c, :].ravel(), fs=fs, nperseg=1024)
            mask = (freqs >= lo) & (freqs <= hi)
            if mask.sum() < 2:
                continue
            lp = 10 * np.log10(psd_p[mask] + 1e-20)
            lt = 10 * np.log10(psd_t[mask] + 1e-20)
            corrs.append(float(np.corrcoef(lp, lt)[0, 1]))
        results[band_name] = float(np.mean(corrs)) if corrs else 0.0
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Data Loading (reuses logic from generate.py)
# ═════════════════════════════════════════════════════════════════════════════

def _rebuild_splits(dataset_key, ds, train_config):
    """Rebuild train/test splits from data when checkpoint lacks indices."""
    if ds.train_name == "pfc":
        from data import prepare_pfc_data
        data = prepare_pfc_data(split_by_session=True)
    elif ds.train_name == "olfactory":
        from data import prepare_data
        data = prepare_data(split_by_session=True)
    elif ds.train_name == "dandi":
        from data import prepare_dandi_data
        data = prepare_dandi_data(
            source_region=train_config.get("dandi_source_region", "amygdala"),
            target_region=train_config.get("dandi_target_region", "hippocampus"),
        )
    elif ds.train_name == "ecog":
        from data import prepare_ecog_data
        data = prepare_ecog_data(
            experiment=train_config.get("ecog_experiment", "motor_imagery"),
            source_region=train_config.get("ecog_source_region", "frontal"),
            target_region=train_config.get("ecog_target_region", "temporal"),
        )
    elif ds.train_name == "boran":
        from data import prepare_boran_data
        data = prepare_boran_data(
            source_region=train_config.get("boran_source_region", "hippocampus"),
            target_region=train_config.get("boran_target_region", "entorhinal_cortex"),
        )
    else:
        raise RuntimeError(f"Don't know how to rebuild splits for {dataset_key} ({ds.train_name})")

    train_idx = np.array(data["train_idx"])
    test_idx = np.array(data["test_idx"])
    if len(test_idx) == 0:
        test_idx = np.array(data["val_idx"])
        print(f"  test_idx empty — using val_idx ({len(test_idx)} trials) as test set")
    return train_idx, test_idx, data


def _build_loader(source, target, labels, indices, batch_size):
    """Build a DataLoader from arrays + index subset."""
    from torch.utils.data import DataLoader
    from data import PairedNeuralDataset
    ds = PairedNeuralDataset(source, target, labels, indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)


def load_train_test_loaders(dataset_key, cfg, device):
    """Load model + build train/test DataLoaders."""
    import torch
    from phase_four.generate import _load_model

    ds = DATASETS[dataset_key]
    ckpt_path = cfg.get_checkpoint_path(dataset_key)
    print(f"Loading checkpoint: {ckpt_path}")
    model, train_config, ckpt = _load_model(ckpt_path, device)
    split_info = ckpt.get("split_info", {})

    train_idx = np.array(split_info.get("train_idx", []))
    test_idx = np.array(split_info.get("test_idx", split_info.get("val_idx", [])))

    if len(train_idx) == 0 or len(test_idx) == 0:
        print("  No train/test indices in checkpoint — rebuilding splits...")
        train_idx, test_idx, data_dict = _rebuild_splits(dataset_key, ds, train_config)
    else:
        data_dict = None

    # For PCx1 use dedicated loader
    if ds.train_name == "pcx1":
        from data import create_pcx1_dataloaders, get_pcx1_session_splits
        seed = train_config.get("seed", 42)
        ws = train_config.get("pcx1_window_size", 5000)
        stride = train_config.get("pcx1_stride", 2500)
        n_val = train_config.get("pcx1_n_val", 4)
        train_sess, val_sess, _ = get_pcx1_session_splits(seed=seed, n_val=n_val, n_test=0)
        loaders = create_pcx1_dataloaders(
            train_sessions=train_sess, val_sessions=val_sess,
            window_size=ws, stride=stride, val_stride=stride,
            batch_size=cfg.batch_size, zscore_per_window=True,
            num_workers=4, separate_val_sessions=False, persistent_workers=False,
        )
        return model, loaders["train"], loaders["val"]

    # Build loaders from data arrays
    if data_dict is not None:
        reverse = "--pfc-reverse" in ds.extra_train_args
        if ds.train_name == "pfc":
            src = data_dict["ca1"] if reverse else data_dict["pfc"]
            tgt = data_dict["pfc"] if reverse else data_dict["ca1"]
            lbl = data_dict["trial_types"]
        elif ds.train_name == "olfactory":
            src, tgt, lbl = data_dict["ob"], data_dict["pcx"], data_dict["labels"]
        else:
            src, tgt, lbl = data_dict["source"], data_dict["target"], data_dict.get("labels", np.zeros(len(train_idx)))
        train_loader = _build_loader(src, tgt, lbl, train_idx, cfg.batch_size)
        test_loader = _build_loader(src, tgt, lbl, test_idx, cfg.batch_size)
    else:
        from phase_four.generate import _make_test_loader
        train_loader = _make_test_loader(dataset_key, ds, train_config, train_idx, cfg)
        test_loader = _make_test_loader(dataset_key, ds, train_config, test_idx, cfg)

    print(f"  Train: {len(train_idx)} trials, Test: {len(test_idx)} trials")
    return model, train_loader, test_loader


# ═════════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(
    dataset_key: str,
    cfg: Phase4Config,
    mc_samples: int = 20,
    max_train: Optional[int] = None,
    noise_std: float = 0.05,
):
    """Run the full calibration comparison experiment."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = DATASETS[dataset_key]
    print(f"\n{'=' * 60}")
    print(f"Calibration Experiment: {ds.display_name}")
    print(f"{'=' * 60}")

    model, train_loader, test_loader = load_train_test_loaders(dataset_key, cfg, device)

    # ── Stochastic inference (MC-Dropout or input perturbation) ───────────
    print(f"\nRunning stochastic inference ({mc_samples} forward passes)...")
    t0 = time.time()
    print("  Train split...")
    train_data = stochastic_inference(model, train_loader, device,
                                      n_samples=mc_samples, noise_std=noise_std)
    print("  Test split...")
    test_data = stochastic_inference(model, test_loader, device,
                                     n_samples=mc_samples, noise_std=noise_std)
    unc_method = test_data["method"]
    print(f"  Inference done in {time.time() - t0:.1f}s (method: {unc_method})")

    mu_train = train_data["mean"]
    sig_train = np.sqrt(np.clip(train_data["variance"], 1e-12, None))
    y_train = train_data["target"]

    mu_test = test_data["mean"]
    sig_test = np.sqrt(np.clip(test_data["variance"], 1e-12, None))
    y_test = test_data["target"]

    # Optional subsample of training data
    if max_train and mu_train.shape[0] > max_train:
        rng = np.random.default_rng(42)
        idx = rng.choice(mu_train.shape[0], max_train, replace=False)
        mu_train, sig_train, y_train = mu_train[idx], sig_train[idx], y_train[idx]

    print(f"\n  Train: {mu_train.shape}, Test: {mu_test.shape}")
    print(f"  MC std range: [{sig_test.min():.4f}, {sig_test.max():.4f}]")

    # ── Fit calibrators on TRAIN data ─────────────────────────────────────
    calibrators = {}

    print("\nFitting calibrators on train data...")

    print("  [1/3] Quantile Calibration (isotonic regression)...")
    t0 = time.time()
    qcal = QuantileCalibrator()
    qcal.fit(mu_train, sig_train, y_train)
    calibrators["quantile"] = qcal
    print(f"         done in {time.time() - t0:.1f}s")

    print("  [2/3] GP-Beta Distribution Calibration...")
    t0 = time.time()
    gpbeta = GPBetaCalibrator()
    gpbeta.fit(mu_train, sig_train, y_train)
    calibrators["gp_beta"] = gpbeta
    print(f"         done in {time.time() - t0:.1f}s")

    print("  [3/3] MMD Post-hoc Calibration...")
    t0 = time.time()
    mmd_cal = MMDCalibrator()
    mmd_cal.fit(mu_train, sig_train, y_train)
    calibrators["mmd"] = mmd_cal
    print(f"         done in {time.time() - t0:.1f}s")

    # ── Evaluate on TEST data ─────────────────────────────────────────────
    fs = ds.sampling_rate
    results = {}

    # Uncalibrated baseline
    print("\nEvaluating on test data...")
    results["uncalibrated"] = _evaluate(mu_test, sig_test, y_test, fs)

    # Calibrated variants
    for name, cal in calibrators.items():
        sig_cal = cal.calibrated_std(mu_test, sig_test)
        results[name] = _evaluate(mu_test, sig_cal, y_test, fs)

    # ── Print comparison table ────────────────────────────────────────────
    _print_table(results, fs)

    # ── Save results ──────────────────────────────────────────────────────
    out_dir = cfg.output_dir / "calibration" / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibration_results.json"

    serialisable = {}
    for k, v in results.items():
        serialisable[k] = {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                           for kk, vv in v.items()}
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def _evaluate(
    mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, fs: int,
) -> Dict[str, float]:
    """Compute all calibration + quality metrics."""
    metrics = {
        "coverage_90": coverage_at_confidence(mu, sigma, y, 0.90),
        "coverage_50": coverage_at_confidence(mu, sigma, y, 0.50),
        "ece_r": calibration_error(mu, sigma, y),
        "sharpness": sharpness(sigma),
        "nll": nll_gaussian(mu, sigma, y),
        "crps": crps_gaussian(mu, sigma, y),
        "pearson": pearson_r(mu, y),
        "psd_corr": psd_correlation(mu, y, fs),
    }
    bands = per_band_r2(mu, y, fs)
    metrics.update(bands)
    return metrics


def _print_table(results: Dict[str, Dict], fs: int):
    """Print a formatted comparison table."""
    strategies = list(results.keys())
    # Metrics to show (order matters)
    metric_keys = [
        "coverage_90", "coverage_50", "ece_r", "sharpness",
        "nll", "crps", "pearson", "psd_corr",
    ]
    band_keys = list(FREQUENCY_BANDS.keys())
    all_keys = metric_keys + band_keys

    # Header
    print(f"\n{'=' * 120}")
    print("CALIBRATION COMPARISON")
    print(f"{'=' * 120}")

    hdr = f"{'strategy':<18s}"
    for k in all_keys:
        hdr += f" {k:>12s}"
    print(hdr)
    print("-" * 120)

    # Best markers for key metrics (higher is better for coverage, pearson, psd_corr, bands;
    # lower is better for ece_r, sharpness, nll, crps)
    base = results.get("uncalibrated", {})

    for strat in strategies:
        row = results[strat]
        line = f"{strat:<18s}"
        for k in all_keys:
            val = row.get(k, 0.0)
            line += f" {val:12.4f}"
        print(line)

    # Delta vs uncalibrated
    if "uncalibrated" in results:
        print("-" * 120)
        print("Delta vs uncalibrated:")
        for strat in strategies:
            if strat == "uncalibrated":
                continue
            row = results[strat]
            line = f"  {strat:<16s}"
            for k in all_keys:
                delta = row.get(k, 0.0) - base.get(k, 0.0)
                sign = "+" if delta >= 0 else ""
                line += f" {sign}{delta:11.4f}"
            print(line)

    print(f"{'=' * 120}")

    # Interpretation help
    print("\nIdeal calibration:")
    print("  coverage_90 -> 0.90,  coverage_50 -> 0.50  (interval covers target at stated rate)")
    print("  ece_r -> 0.00  (calibration error = 0)")
    print("  sharpness -> low  (tighter intervals = more confident)")
    print("  nll -> low,  crps -> low  (better probabilistic fit)")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Uncertainty calibration experiment")
    parser.add_argument("--dataset", type=str, default="pfc_hpc",
                        choices=list(DATASETS.keys()),
                        help="Dataset key")
    parser.add_argument("--max-train", type=int, default=None,
                        help="Max training samples for calibrator fitting")
    parser.add_argument("--mc-samples", type=int, default=20,
                        help="Number of MC-Dropout forward passes")
    parser.add_argument("--noise-std", type=float, default=0.05,
                        help="Input perturbation noise std (used when model has no dropout)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="results/phase4")
    args = parser.parse_args()

    cfg = Phase4Config(batch_size=args.batch_size, output_dir=Path(args.output_dir))
    run_experiment(args.dataset, cfg, mc_samples=args.mc_samples,
                   max_train=args.max_train, noise_std=args.noise_std)


if __name__ == "__main__":
    main()
