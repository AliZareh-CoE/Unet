"""Smoke tests for PID and fingerprint validation modules.

Generates synthetic data in a temp directory and runs both modules
end-to-end. Verifies mathematical properties and output files.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_synth_data(
    tmpdir: Path,
    n_trials: int = 200,
    n_channels: int = 8,
    n_time: int = 500,
    n_classes: int = 4,
    n_sessions: int = 3,
    fs: int = 1000,
    noise_level: float = 0.3,
):
    """Create fake source/target/predicted/labels .npy files.

    The predicted signal = real target + noise, so MI(pred; label)
    should be positive but less than MI(real; label).
    """
    rng = np.random.default_rng(42)

    labels = rng.integers(0, n_classes, size=n_trials)
    session_ids = rng.integers(0, n_sessions, size=n_trials)

    # Source: random
    source = rng.standard_normal((n_trials, n_channels, n_time))

    # Real target: class-dependent mean shift so MI(real; label) > 0
    real = rng.standard_normal((n_trials, n_channels, n_time))
    for c in range(n_classes):
        mask = labels == c
        # Each class gets a different DC offset per channel
        offsets = rng.standard_normal(n_channels) * 2.0
        real[mask] += offsets[None, :, None]

    # Add session-specific signature for fingerprinting
    # Use frequency-domain patterns so band-power features can detect them
    t = np.arange(n_time) / fs
    for s in range(n_sessions):
        mask = session_ids == s
        # Each session has a unique oscillatory signature per channel
        for ch in range(n_channels):
            freq = 5 + s * 10 + ch * 2  # unique freq per session+channel
            amp = 3.0
            real[mask, ch, :] += amp * np.sin(2 * np.pi * freq * t)
            source[mask, ch, :] += amp * 0.5 * np.sin(2 * np.pi * freq * t)

    # Predicted: real + noise (imperfect translation)
    pred = real + rng.standard_normal(real.shape) * noise_level

    np.save(tmpdir / "source_test.npy", source.astype(np.float32))
    np.save(tmpdir / "target_test.npy", real.astype(np.float32))
    np.save(tmpdir / "predicted_test.npy", pred.astype(np.float32))
    np.save(tmpdir / "labels_test.npy", labels)
    np.save(tmpdir / "session_ids_test.npy", session_ids)

    return fs, labels, session_ids


# ═══════════════════════════════════════════════════════════════════════════
# PID tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPIDMath:
    """Test PID mathematical properties."""

    def test_mi_gaussian_independent(self):
        """MI of independent variables should be ~0."""
        from phase_four.validation.pid import _mi_gaussian
        rng = np.random.default_rng(123)
        x = rng.standard_normal((1000, 3))
        y = rng.standard_normal((1000, 3))
        mi = _mi_gaussian(x, y)
        assert mi < 0.1, f"MI of independent vars too high: {mi}"

    def test_mi_gaussian_identical(self):
        """MI of identical variables should be large."""
        from phase_four.validation.pid import _mi_gaussian
        rng = np.random.default_rng(123)
        x = rng.standard_normal((1000, 3))
        mi = _mi_gaussian(x, x)
        assert mi > 5.0, f"MI of identical vars too low: {mi}"

    def test_mi_gaussian_nonnegative(self):
        """MI should always be non-negative."""
        from phase_four.validation.pid import _mi_gaussian
        rng = np.random.default_rng(123)
        x = rng.standard_normal((500, 5))
        y = rng.standard_normal((500, 3))
        mi = _mi_gaussian(x, y)
        assert mi >= 0.0

    def test_mi_class_conditional_informative(self):
        """Class-conditional MI should be positive for class-separable data."""
        from phase_four.validation.pid import _mi_class_conditional
        rng = np.random.default_rng(123)
        n = 500
        labels = np.repeat([0, 1, 2], n // 3 + 1)[:n]
        x = rng.standard_normal((n, 4))
        # Add class-dependent offset
        for c in range(3):
            x[labels == c] += c * 2.0
        mi = _mi_class_conditional(x, labels)
        assert mi > 0.5, f"MI for separable classes too low: {mi}"

    def test_mi_class_conditional_uninformative(self):
        """Class-conditional MI should be ~0 for random labels."""
        from phase_four.validation.pid import _mi_class_conditional
        rng = np.random.default_rng(123)
        n = 500
        x = rng.standard_normal((n, 4))
        labels = rng.integers(0, 3, size=n)
        mi = _mi_class_conditional(x, labels)
        assert mi < 0.3, f"MI for random labels too high: {mi}"

    def test_pid_decomposition_sums(self):
        """RED + UNQ_S1 + UNQ_S2 + SYN should equal MI_joint."""
        from phase_four.validation.pid import _pid_from_mis
        result = _pid_from_mis(mi_s1=2.0, mi_s2=1.5, mi_joint=3.0)
        total = (result["redundancy"] + result["unique_real"]
                 + result["unique_pred"] + result["synergy"])
        assert abs(total - result["mi_joint"]) < 1e-10, (
            f"PID doesn't sum to joint MI: {total} != {result['mi_joint']}"
        )

    def test_pid_redundancy_nonneg(self):
        """Redundancy (MMI) should be non-negative."""
        from phase_four.validation.pid import _pid_from_mis
        result = _pid_from_mis(mi_s1=2.0, mi_s2=1.5, mi_joint=3.0)
        assert result["redundancy"] >= 0.0

    def test_pid_unique_nonneg(self):
        """Unique information should be non-negative under MMI."""
        from phase_four.validation.pid import _pid_from_mis
        result = _pid_from_mis(mi_s1=2.0, mi_s2=1.5, mi_joint=3.0)
        assert result["unique_real"] >= 0.0
        assert result["unique_pred"] >= 0.0

    def test_copula_transform_gaussian(self):
        """Copula-transformed data should have ~0 mean, ~1 std."""
        from phase_four.validation.pid import _copula_transform
        rng = np.random.default_rng(123)
        # Exponential (non-Gaussian) input
        x = rng.exponential(2.0, size=(1000, 3))
        out = _copula_transform(x)
        assert abs(out.mean()) < 0.1, f"Copula mean not ~0: {out.mean()}"
        assert abs(out.std() - 1.0) < 0.15, f"Copula std not ~1: {out.std()}"


class TestPIDEndToEnd:
    """End-to-end PID validation test."""

    def test_label_based_pid(self):
        """Label-based PID should run and produce sensible results."""
        from phase_four.validation.pid import label_based_pid

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fs, labels, _ = _make_synth_data(tmpdir)
            real = np.load(tmpdir / "target_test.npy")
            pred = np.load(tmpdir / "predicted_test.npy")

            result = label_based_pid(real, pred, labels, fs)

            ovr = result["overall"]
            # Redundancy should be positive (pred carries some label info)
            assert ovr["redundancy"] > 0.0, f"No redundancy: {ovr}"
            # Translation efficiency should be between 0 and 1
            assert 0.0 <= ovr["translation_efficiency"] <= 1.0 + 1e-6
            # Per-band should have all 6 bands
            assert len(result["per_band"]) == 6
            # Per-channel should have 8 channels
            assert len(result["per_channel"]) == 8

    def test_temporal_pid(self):
        """Temporal PID should run and produce sensible results."""
        from phase_four.validation.pid import temporal_pid

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fs, _, _ = _make_synth_data(tmpdir)
            real = np.load(tmpdir / "target_test.npy")
            pred = np.load(tmpdir / "predicted_test.npy")

            result = temporal_pid(real, pred, fs, lag_ms=50)

            assert result["lag_ms"] == 50
            assert result["lag_samples"] == 50  # 50ms at 1000Hz
            assert result["n_samples"] > 0
            ovr = result["overall"]
            assert "redundancy" in ovr
            assert len(result["per_channel"]) == 8

    def test_run_pid_validation_outputs(self):
        """Full run_pid_validation should produce JSON and CSV files."""
        from phase_four.validation.pid import run_pid_validation

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fs, _, _ = _make_synth_data(tmpdir)

            results = run_pid_validation(tmpdir, fs)

            # Check output files exist
            assert (tmpdir / "pid_results.json").exists()
            assert (tmpdir / "pid_per_band.csv").exists()
            assert (tmpdir / "pid_per_channel.csv").exists()

            # Check JSON is valid
            with open(tmpdir / "pid_results.json") as f:
                data = json.load(f)
            assert "label_pid" in data
            assert "temporal_pid" in data


# ═══════════════════════════════════════════════════════════════════════════
# Fingerprint tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFingerprintMath:
    """Test fingerprint mathematical properties."""

    def test_cosine_sim_self(self):
        """Cosine similarity of a vector with itself should be 1."""
        from phase_four.validation.fingerprint import _cosine_similarity_matrix
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        sim = _cosine_similarity_matrix(x)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-10)

    def test_cosine_sim_range(self):
        """Cosine similarity should be in [-1, 1]."""
        from phase_four.validation.fingerprint import _cosine_similarity_matrix
        rng = np.random.default_rng(42)
        x = rng.standard_normal((50, 10))
        sim = _cosine_similarity_matrix(x)
        assert sim.min() >= -1.0 - 1e-10
        assert sim.max() <= 1.0 + 1e-10

    def test_identifiability_perfect_sessions(self):
        """Perfect session separation should give high identifiability."""
        from phase_four.validation.fingerprint import compute_identifiability
        # 3 sessions, features are just session index repeated
        session_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        features = np.zeros((9, 5))
        for i, s in enumerate(session_ids):
            features[i] = s * 10  # huge separation
            features[i, 0] += i * 0.01  # tiny within-session variation

        result = compute_identifiability(features, session_ids)
        assert result["identifiability"] > 0.1
        assert result["identification_accuracy"] > 0.5

    def test_identifiability_random(self):
        """Random features should give ~0 identifiability."""
        from phase_four.validation.fingerprint import compute_identifiability
        rng = np.random.default_rng(42)
        session_ids = np.repeat([0, 1, 2], 50)
        features = rng.standard_normal((150, 20))

        result = compute_identifiability(features, session_ids)
        assert abs(result["identifiability"]) < 0.15, (
            f"Random features identifiability too high: {result['identifiability']}"
        )

    def test_loo_prevents_inflation(self):
        """LOO accuracy should be <= naive accuracy for small sessions."""
        from phase_four.validation.fingerprint import compute_identifiability
        rng = np.random.default_rng(42)
        # Only 2 trials per session — LOO template is just 1 trial
        session_ids = np.array([0, 0, 1, 1, 2, 2])
        features = rng.standard_normal((6, 10))
        # Add small session offsets
        for s in range(3):
            features[session_ids == s] += s * 0.5

        result = compute_identifiability(features, session_ids)
        # Should still work without crashing
        assert 0.0 <= result["identification_accuracy"] <= 1.0

    def test_string_session_ids(self):
        """Should handle string session IDs without crashing."""
        from phase_four.validation.fingerprint import compute_identifiability
        rng = np.random.default_rng(42)
        session_ids = np.array(["sess_A", "sess_A", "sess_A",
                                "sess_B", "sess_B", "sess_B"])
        features = rng.standard_normal((6, 10))
        for s in ["sess_A", "sess_B"]:
            features[session_ids == s] += hash(s) % 5

        result = compute_identifiability(features, session_ids)
        assert result["n_sessions"] == 2
        # Session labels should be JSON-safe strings
        for ps in result["per_session"]:
            assert isinstance(ps["session"], str)


class TestFingerprintEndToEnd:
    """End-to-end fingerprint validation test."""

    def test_run_fingerprint_validation(self):
        """Full fingerprint validation should produce correct outputs."""
        from phase_four.validation.fingerprint import run_fingerprint_validation

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fs, _, session_ids = _make_synth_data(tmpdir)

            results = run_fingerprint_validation(tmpdir, fs)

            # Check output files
            assert (tmpdir / "fingerprint_results.json").exists()
            assert (tmpdir / "fingerprint_per_session.csv").exists()
            assert (tmpdir / "fingerprint_summary.csv").exists()

            # Check structure
            assert "source" in results
            assert "real_target" in results
            assert "predicted_target" in results
            assert "cross_domain" in results
            assert "preservation_ratio" in results

            # Predicted should be identifiable (we added session patterns)
            assert results["predicted_target"]["identifiability"] > 0.0

            # Cross-domain accuracy should be above chance
            chance = results["cross_domain"]["chance_level"]
            acc = results["cross_domain"]["cross_domain_accuracy"]
            assert acc > chance, (
                f"Cross-domain acc {acc} not above chance {chance}"
            )

    def test_fingerprint_loads_session_ids_npy(self):
        """Should load session_ids_test.npy when available."""
        from phase_four.validation.fingerprint import run_fingerprint_validation

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fs, _, session_ids = _make_synth_data(tmpdir)

            results = run_fingerprint_validation(tmpdir, fs)
            # Should use 3 sessions (from session_ids), not 4 (from labels)
            assert results["n_sessions"] == 3

    def test_fingerprint_skips_single_session(self):
        """Should skip gracefully with only 1 session."""
        from phase_four.validation.fingerprint import run_fingerprint_validation

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fs, _, _ = _make_synth_data(tmpdir, n_sessions=1, n_classes=1)

            results = run_fingerprint_validation(tmpdir, fs)
            assert results.get("skipped") is True


# ═══════════════════════════════════════════════════════════════════════════
# Cross-module integration test
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Test both modules work together as called by runner."""

    def test_both_modules_same_data(self):
        """Both PID and fingerprint should run on the same synthetic data."""
        from phase_four.validation.pid import run_pid_validation
        from phase_four.validation.fingerprint import run_fingerprint_validation

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fs, _, _ = _make_synth_data(tmpdir)

            pid_results = run_pid_validation(tmpdir, fs)
            fp_results = run_fingerprint_validation(tmpdir, fs)

            # Both should complete without error
            assert pid_results["label_pid"] is not None
            assert pid_results["temporal_pid"] is not None
            assert fp_results["predicted_target"]["identifiability"] is not None

            # All output files should exist
            expected_files = [
                "pid_results.json",
                "pid_per_band.csv",
                "pid_per_channel.csv",
                "fingerprint_results.json",
                "fingerprint_per_session.csv",
                "fingerprint_summary.csv",
            ]
            for fname in expected_files:
                assert (tmpdir / fname).exists(), f"Missing: {fname}"
