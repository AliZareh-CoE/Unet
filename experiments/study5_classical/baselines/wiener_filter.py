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
        n_fft: Optional[int] = None,  # None = auto (use signal length)
        regularization: float = 1e-8,
        smooth_window: int = 5,
    ):
        self._n_fft_init = n_fft  # Store initial value
        self.n_fft = n_fft  # Will be set during fit if None
        self.regularization = regularization
        self.smooth_window = smooth_window

        # Filter coefficients (computed during fit)
        self.H: Optional[NDArray] = None
        self._fitted = False
        self._signal_length: Optional[int] = None  # Store original signal length

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

        # Store signal length for predict
        self._signal_length = T

        # Auto-set n_fft if not specified (use next power of 2 for efficiency)
        if self._n_fft_init is None:
            self.n_fft = int(2 ** np.ceil(np.log2(T)))
        else:
            self.n_fft = self._n_fft_init

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

        # FFT of input (use same n_fft as fit)
        X_fft = np.fft.rfft(X, n=self.n_fft, axis=-1)  # [N, C_in, F]

        # Apply filter
        y_fft = np.zeros((N, C_out, X_fft.shape[-1]), dtype=np.complex128)
        for c_out in range(C_out):
            for c_in in range(C_in):
                y_fft[:, c_out, :] += self.H[c_out, c_in, :] * X_fft[:, c_in, :]

        # Inverse FFT - use n_fft to get proper length
        y = np.fft.irfft(y_fft, n=self.n_fft, axis=-1)

        # Truncate to original signal length (from input, not fit)
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
        n_fft: Optional[int] = None,
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
        """Fit MIMO Wiener filter (vectorized for speed).

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

        # Store signal length for predict
        self._signal_length = T

        # Auto-set n_fft if not specified (use next power of 2 for efficiency)
        if self._n_fft_init is None:
            self.n_fft = int(2 ** np.ceil(np.log2(T)))
        else:
            self.n_fft = self._n_fft_init

        # Compute FFT
        X_fft = np.fft.rfft(X, n=self.n_fft, axis=-1)  # [N, C_in, F]
        y_fft = np.fft.rfft(y, n=self.n_fft, axis=-1)  # [N, C_out, F]

        n_freq = X_fft.shape[-1]

        # VECTORIZED computation (much faster than loop over frequencies)
        # Transpose for easier matrix operations: [F, N, C]
        X_fft_t = X_fft.transpose(2, 0, 1)  # [F, N, C_in]
        y_fft_t = y_fft.transpose(2, 0, 1)  # [F, N, C_out]

        # Cross-spectral matrix S_xy [F, C_out, C_in] = E[y @ x^H]
        # Using einsum: for each freq, compute outer product and average over N
        S_xy = np.einsum('fno,fni->foi', y_fft_t, np.conj(X_fft_t)) / N

        # Auto-spectral matrix S_xx [F, C_in, C_in] = E[x @ x^H]
        S_xx = np.einsum('fni,fnj->fij', X_fft_t, np.conj(X_fft_t)) / N

        # Regularize: add regularization * I to each frequency's S_xx
        eye = np.eye(C_in, dtype=np.complex128)
        S_xx_reg = S_xx + self.regularization * eye[np.newaxis, :, :]

        # Solve H @ S_xx = S_xy for each frequency using pseudoinverse (robust)
        # H [F, C_out, C_in]
        try:
            # Use solve for better numerical stability than inv
            # Solve S_xx^T @ H^T = S_xy^T -> H^T = solve(S_xx^T, S_xy^T)
            H_t = np.linalg.solve(
                S_xx_reg.transpose(0, 2, 1),  # [F, C_in, C_in]
                S_xy.transpose(0, 2, 1)       # [F, C_in, C_out]
            )  # [F, C_in, C_out]
            H = H_t.transpose(0, 2, 1)  # [F, C_out, C_in]
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if solve fails
            H = np.zeros((n_freq, C_out, C_in), dtype=np.complex128)
            for f in range(n_freq):
                H[f] = S_xy[f] @ np.linalg.pinv(S_xx_reg[f])

        # Transpose to expected shape [C_out, C_in, F]
        self.H = H.transpose(1, 2, 0)

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
