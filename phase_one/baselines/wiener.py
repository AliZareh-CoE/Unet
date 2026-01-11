"""
Wiener Filter Baselines for Neural Signal Translation
======================================================

Implements optimal linear filtering in the frequency domain.
The Wiener filter minimizes MSE under stationary signal assumptions.

Theory:
    H(f) = S_xy(f) / S_xx(f)

where:
    - S_xy(f) is the cross-spectral density between input and output
    - S_xx(f) is the power spectral density of the input
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class BaselineModel(ABC):
    """Abstract base class for all baseline models."""

    @abstractmethod
    def fit(self, X: NDArray, y: NDArray) -> "BaselineModel":
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, X: NDArray) -> NDArray:
        """Generate predictions for input data."""
        pass

    def fit_predict(self, X: NDArray, y: NDArray) -> NDArray:
        """Fit model and return predictions on training data."""
        return self.fit(X, y).predict(X)


class WienerFilter(BaselineModel):
    """Single-channel Wiener filter for neural signal translation.

    Computes the optimal linear filter H(f) that minimizes MSE in frequency domain.
    Each output channel is predicted from its corresponding input channel.

    H(f) = S_xy(f) / S_xx(f)

    Args:
        n_fft: FFT size (None = auto, uses next power of 2)
        regularization: Regularization to prevent division by zero
        smooth_window: Window size for spectral smoothing (0 = no smoothing)

    Attributes:
        H: Filter coefficients [C_out, C_in, n_freq] (complex)
        n_fft: Actual FFT size used

    Example:
        >>> model = WienerFilter(regularization=1e-6)
        >>> model.fit(X_train, y_train)  # X: [N, C, T]
        >>> y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        n_fft: Optional[int] = None,
        regularization: float = 1e-6,
        smooth_window: int = 5,
    ):
        self.n_fft_init = n_fft
        self.n_fft: Optional[int] = None
        self.regularization = regularization
        self.smooth_window = smooth_window
        self.H: Optional[NDArray] = None
        self._fitted = False
        self._signal_length: Optional[int] = None

    def _smooth_spectrum(self, spectrum: NDArray) -> NDArray:
        """Apply moving average smoothing to spectrum.

        Args:
            spectrum: 1D or 2D spectrum array

        Returns:
            Smoothed spectrum of same shape
        """
        if self.smooth_window <= 1:
            return spectrum

        kernel = np.ones(self.smooth_window) / self.smooth_window

        if spectrum.ndim == 1:
            return np.convolve(spectrum, kernel, mode="same")

        # Handle multi-dimensional: smooth along last axis
        result = np.zeros_like(spectrum)
        for i in range(spectrum.shape[0]):
            result[i] = np.convolve(spectrum[i], kernel, mode="same")
        return result

    def fit(self, X: NDArray, y: NDArray) -> "WienerFilter":
        """Fit Wiener filter from input-output pairs.

        Args:
            X: Input signals [N, C_in, T] or [N, T] for single channel
            y: Target signals [N, C_out, T] or [N, T] for single channel

        Returns:
            self

        Raises:
            ValueError: If input shapes are inconsistent
        """
        # Ensure 3D shape
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        if y.ndim == 2:
            y = y[:, np.newaxis, :]

        N, C_in, T = X.shape
        _, C_out, _ = y.shape

        # Store signal length for prediction
        self._signal_length = T

        # Auto-determine FFT size (next power of 2)
        if self.n_fft_init is None:
            self.n_fft = int(2 ** np.ceil(np.log2(T)))
        else:
            self.n_fft = self.n_fft_init

        # Compute FFT of all signals
        X_fft = np.fft.rfft(X, n=self.n_fft, axis=-1)  # [N, C_in, n_freq]
        y_fft = np.fft.rfft(y, n=self.n_fft, axis=-1)  # [N, C_out, n_freq]

        n_freq = X_fft.shape[-1]

        # Initialize filter: diagonal matching (channel i -> channel i)
        self.H = np.zeros((C_out, C_in, n_freq), dtype=np.complex128)
        n_matched = min(C_out, C_in)

        for c in range(n_matched):
            # Cross-spectral density: E[Y(f) · X(f)*]
            S_xy = np.mean(y_fft[:, c, :] * np.conj(X_fft[:, c, :]), axis=0)

            # Power spectral density: E[X(f) · X(f)*] = |X(f)|²
            S_xx = np.mean(np.abs(X_fft[:, c, :]) ** 2, axis=0)

            # Apply smoothing
            S_xy = self._smooth_spectrum(S_xy)
            S_xx = self._smooth_spectrum(np.real(S_xx))

            # Wiener filter: H = S_xy / (S_xx + reg)
            self.H[c, c, :] = S_xy / (S_xx + self.regularization)

        self._fitted = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Apply Wiener filter to input data.

        Args:
            X: Input signals [N, C_in, T] or [N, T]

        Returns:
            Filtered signals [N, C_out, T]

        Raises:
            RuntimeError: If model not fitted
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        single_channel = X.ndim == 2
        if single_channel:
            X = X[:, np.newaxis, :]

        N, C_in, T = X.shape
        C_out = self.H.shape[0]

        # FFT of input
        X_fft = np.fft.rfft(X, n=self.n_fft, axis=-1)  # [N, C_in, n_freq]

        # Apply filter via matrix multiplication
        y_fft = np.zeros((N, C_out, X_fft.shape[-1]), dtype=np.complex128)
        for c_out in range(C_out):
            for c_in in range(C_in):
                if np.any(self.H[c_out, c_in, :] != 0):
                    y_fft[:, c_out, :] += self.H[c_out, c_in, :] * X_fft[:, c_in, :]

        # Inverse FFT
        y = np.fft.irfft(y_fft, n=self.n_fft, axis=-1)

        # Truncate to original length
        y = y[..., :T]

        if single_channel and C_out == 1:
            y = y[:, 0, :]

        return y.astype(np.float32)

    def get_frequency_response(self) -> Tuple[NDArray, NDArray]:
        """Get the filter frequency response.

        Returns:
            frequencies: Normalized frequencies (0 to 0.5)
            response: Complex filter coefficients [C_out, C_in, n_freq]

        Raises:
            RuntimeError: If model not fitted
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        frequencies = np.fft.rfftfreq(self.n_fft)
        return frequencies, self.H


class WienerMIMO(BaselineModel):
    """Multi-Input Multi-Output (MIMO) Wiener filter.

    Jointly optimizes all input-output channel mappings:
        H(f) = S_xy(f) @ S_xx(f)^{-1}

    where:
        - S_xy(f) is the cross-spectral matrix [C_out, C_in]
        - S_xx(f) is the auto-spectral matrix [C_in, C_in]

    Args:
        n_fft: FFT size (None = auto)
        regularization: Regularization for matrix inversion
        smooth_window: Window for spectral smoothing

    Example:
        >>> model = WienerMIMO(regularization=1e-5)
        >>> model.fit(X_train, y_train)
        >>> y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        n_fft: Optional[int] = None,
        regularization: float = 1e-5,
        smooth_window: int = 5,
    ):
        self.n_fft_init = n_fft
        self.n_fft: Optional[int] = None
        self.regularization = regularization
        self.smooth_window = smooth_window
        self.H: Optional[NDArray] = None
        self._fitted = False

    def fit(self, X: NDArray, y: NDArray) -> "WienerMIMO":
        """Fit MIMO Wiener filter using vectorized computation.

        Args:
            X: Input signals [N, C_in, T]
            y: Target signals [N, C_out, T]

        Returns:
            self
        """
        # Ensure 3D shape
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        if y.ndim == 2:
            y = y[:, np.newaxis, :]

        N, C_in, T = X.shape
        _, C_out, _ = y.shape

        # Auto FFT size
        if self.n_fft_init is None:
            self.n_fft = int(2 ** np.ceil(np.log2(T)))
        else:
            self.n_fft = self.n_fft_init

        # Compute FFT
        X_fft = np.fft.rfft(X, n=self.n_fft, axis=-1)  # [N, C_in, n_freq]
        y_fft = np.fft.rfft(y, n=self.n_fft, axis=-1)  # [N, C_out, n_freq]

        n_freq = X_fft.shape[-1]

        # Transpose for vectorized computation: [n_freq, N, C]
        X_fft_t = X_fft.transpose(2, 0, 1)  # [n_freq, N, C_in]
        y_fft_t = y_fft.transpose(2, 0, 1)  # [n_freq, N, C_out]

        # Cross-spectral matrix: S_xy[f] = E[y @ x^H] -> [n_freq, C_out, C_in]
        S_xy = np.einsum('fno,fni->foi', y_fft_t, np.conj(X_fft_t)) / N

        # Auto-spectral matrix: S_xx[f] = E[x @ x^H] -> [n_freq, C_in, C_in]
        S_xx = np.einsum('fni,fnj->fij', X_fft_t, np.conj(X_fft_t)) / N

        # Add regularization to diagonal
        eye = np.eye(C_in, dtype=np.complex128)
        S_xx_reg = S_xx + self.regularization * eye[np.newaxis, :, :]

        # Compute H = S_xy @ inv(S_xx) for all frequencies at once
        try:
            S_xx_inv = np.linalg.inv(S_xx_reg)
            H = np.einsum('foi,fij->foj', S_xy, S_xx_inv)  # [n_freq, C_out, C_in]
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if regular inverse fails
            H = np.zeros((n_freq, C_out, C_in), dtype=np.complex128)
            for f in range(n_freq):
                try:
                    H[f] = S_xy[f] @ np.linalg.pinv(S_xx_reg[f])
                except np.linalg.LinAlgError:
                    pass  # Leave as zeros

        # Store as [C_out, C_in, n_freq]
        self.H = H.transpose(1, 2, 0)
        self._fitted = True
        self._signal_length = T

        return self

    def predict(self, X: NDArray) -> NDArray:
        """Apply MIMO Wiener filter.

        Args:
            X: Input signals [N, C_in, T]

        Returns:
            Filtered signals [N, C_out, T]
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if X.ndim == 2:
            X = X[:, np.newaxis, :]

        N, C_in, T = X.shape
        C_out = self.H.shape[0]

        # FFT of input
        X_fft = np.fft.rfft(X, n=self.n_fft, axis=-1)  # [N, C_in, n_freq]

        # Apply MIMO filter: y_fft = H @ x_fft for each frequency
        # H: [C_out, C_in, n_freq], X_fft: [N, C_in, n_freq]
        # Want: y_fft[n, c_out, f] = sum_c_in H[c_out, c_in, f] * X_fft[n, c_in, f]
        y_fft = np.einsum('oif,nif->nof', self.H, X_fft)

        # Inverse FFT
        y = np.fft.irfft(y_fft, n=self.n_fft, axis=-1)
        y = y[..., :T]

        return y.astype(np.float32)

    def get_frequency_response(self) -> Tuple[NDArray, NDArray]:
        """Get MIMO frequency response."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")
        return np.fft.rfftfreq(self.n_fft), self.H
