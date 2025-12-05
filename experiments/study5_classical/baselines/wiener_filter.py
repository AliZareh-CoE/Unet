"""
Wiener Filter for Neural Signal Translation
============================================

Optimal linear filter in frequency domain that minimizes MSE.
Provides theoretical baseline for linear signal translation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class WienerFilter:
    """Wiener filter for neural signal translation.

    The Wiener filter is the optimal linear filter for minimizing MSE
    in the presence of additive noise, under the assumption of
    stationary signals.

    H(f) = S_xy(f) / S_xx(f)

    where S_xy is cross-spectral density and S_xx is power spectral density.

    Args:
        n_fft: FFT size for spectral estimation
        regularization: Regularization term to prevent division by zero
        smooth_window: Window size for spectral smoothing (0 to disable)
    """

    def __init__(
        self,
        n_fft: int = 1024,
        regularization: float = 1e-8,
        smooth_window: int = 5,
    ):
        self.n_fft = n_fft
        self.regularization = regularization
        self.smooth_window = smooth_window

        # Filter coefficients (computed during fit)
        self.H: Optional[NDArray] = None
        self._fitted = False

    def _smooth_spectrum(self, spectrum: NDArray) -> NDArray:
        """Apply moving average smoothing to spectrum."""
        if self.smooth_window <= 1:
            return spectrum

        kernel = np.ones(self.smooth_window) / self.smooth_window
        # Handle multi-dimensional spectra
        if spectrum.ndim == 1:
            return np.convolve(spectrum, kernel, mode="same")
        else:
            result = np.zeros_like(spectrum)
            for i in range(spectrum.shape[0]):
                result[i] = np.convolve(spectrum[i], kernel, mode="same")
            return result

    def fit(
        self,
        X: NDArray,
        y: NDArray,
    ) -> "WienerFilter":
        """Fit Wiener filter from input-output pairs.

        Args:
            X: Input signals [N, C_in, T] or [N, T] for single channel
            y: Target signals [N, C_out, T] or [N, T] for single channel

        Returns:
            self
        """
        # Handle shape
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        if y.ndim == 2:
            y = y[:, np.newaxis, :]

        N, C_in, T = X.shape
        _, C_out, _ = y.shape

        # Compute FFT of all signals
        X_fft = np.fft.rfft(X, n=self.n_fft, axis=-1)  # [N, C_in, F]
        y_fft = np.fft.rfft(y, n=self.n_fft, axis=-1)  # [N, C_out, F]

        n_freq = X_fft.shape[-1]

        # Initialize filter matrix [C_out, C_in, F]
        self.H = np.zeros((C_out, C_in, n_freq), dtype=np.complex128)

        for c_out in range(C_out):
            for c_in in range(C_in):
                # Cross-spectral density: E[Y(f) * X(f)*]
                S_xy = np.mean(y_fft[:, c_out, :] * np.conj(X_fft[:, c_in, :]), axis=0)

                # Auto-spectral density: E[X(f) * X(f)*]
                S_xx = np.mean(X_fft[:, c_in, :] * np.conj(X_fft[:, c_in, :]), axis=0)

                # Smooth spectra
                S_xy = self._smooth_spectrum(S_xy)
                S_xx = self._smooth_spectrum(np.real(S_xx))

                # Wiener filter
                self.H[c_out, c_in, :] = S_xy / (S_xx + self.regularization)

        self._fitted = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Apply Wiener filter to new data.

        Args:
            X: Input signals [N, C_in, T] or [N, T]

        Returns:
            Filtered signals [N, C_out, T]
        """
        if not self._fitted:
            raise RuntimeError("Filter must be fitted before prediction")

        single_channel = X.ndim == 2
        if single_channel:
            X = X[:, np.newaxis, :]

        N, C_in, T = X.shape
        C_out = self.H.shape[0]

        # FFT of input
        X_fft = np.fft.rfft(X, n=self.n_fft, axis=-1)  # [N, C_in, F]

        # Apply filter
        y_fft = np.zeros((N, C_out, X_fft.shape[-1]), dtype=np.complex128)
        for c_out in range(C_out):
            for c_in in range(C_in):
                y_fft[:, c_out, :] += self.H[c_out, c_in, :] * X_fft[:, c_in, :]

        # Inverse FFT
        y = np.fft.irfft(y_fft, n=self.n_fft, axis=-1)

        # Truncate to original length
        y = y[..., :T]

        if single_channel and C_out == 1:
            y = y[:, 0, :]

        return y

    def fit_predict(self, X: NDArray, y: NDArray) -> NDArray:
        """Fit filter and predict in one step."""
        return self.fit(X, y).predict(X)

    def get_frequency_response(self) -> Tuple[NDArray, NDArray]:
        """Get filter frequency response.

        Returns:
            frequencies: Frequency bins (normalized 0 to 0.5)
            response: Complex frequency response [C_out, C_in, F]
        """
        if not self._fitted:
            raise RuntimeError("Filter must be fitted first")

        frequencies = np.fft.rfftfreq(self.n_fft)
        return frequencies, self.H


class MultiChannelWienerFilter(WienerFilter):
    """Wiener filter with explicit multi-channel handling.

    Estimates a single MIMO (Multiple-Input Multiple-Output) filter
    that optimally maps all input channels to all output channels.

    Args:
        n_fft: FFT size
        regularization: Regularization for matrix inversion
        use_mimo: If True, jointly optimize all channels
    """

    def __init__(
        self,
        n_fft: int = 1024,
        regularization: float = 1e-6,
        use_mimo: bool = True,
    ):
        super().__init__(n_fft=n_fft, regularization=regularization)
        self.use_mimo = use_mimo

    def fit(
        self,
        X: NDArray,
        y: NDArray,
    ) -> "MultiChannelWienerFilter":
        """Fit MIMO Wiener filter.

        For MIMO, we solve: H(f) = S_xy(f) @ S_xx(f)^{-1}

        Args:
            X: Input signals [N, C_in, T]
            y: Target signals [N, C_out, T]

        Returns:
            self
        """
        if not self.use_mimo:
            return super().fit(X, y)

        # Handle shape
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        if y.ndim == 2:
            y = y[:, np.newaxis, :]

        N, C_in, T = X.shape
        _, C_out, _ = y.shape

        # Compute FFT
        X_fft = np.fft.rfft(X, n=self.n_fft, axis=-1)  # [N, C_in, F]
        y_fft = np.fft.rfft(y, n=self.n_fft, axis=-1)  # [N, C_out, F]

        n_freq = X_fft.shape[-1]

        # Initialize filter [C_out, C_in, F]
        self.H = np.zeros((C_out, C_in, n_freq), dtype=np.complex128)

        for f in range(n_freq):
            # Cross-spectral matrix S_xy [C_out, C_in]
            S_xy = np.zeros((C_out, C_in), dtype=np.complex128)
            for n in range(N):
                S_xy += np.outer(y_fft[n, :, f], np.conj(X_fft[n, :, f]))
            S_xy /= N

            # Auto-spectral matrix S_xx [C_in, C_in]
            S_xx = np.zeros((C_in, C_in), dtype=np.complex128)
            for n in range(N):
                S_xx += np.outer(X_fft[n, :, f], np.conj(X_fft[n, :, f]))
            S_xx /= N

            # Regularize and invert
            S_xx_reg = S_xx + self.regularization * np.eye(C_in)
            S_xx_inv = np.linalg.inv(S_xx_reg)

            # MIMO Wiener filter at frequency f
            self.H[:, :, f] = S_xy @ S_xx_inv

        self._fitted = True
        return self


def create_wiener_filter(
    variant: str = "standard",
    **kwargs,
) -> WienerFilter:
    """Factory function for Wiener filters.

    Args:
        variant: "standard" or "mimo"
        **kwargs: Additional arguments

    Returns:
        Wiener filter instance
    """
    if variant == "standard":
        return WienerFilter(**kwargs)
    elif variant == "mimo":
        return MultiChannelWienerFilter(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")
