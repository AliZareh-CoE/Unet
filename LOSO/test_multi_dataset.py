"""Integration tests for multi-dataset LOSO support.

These tests verify that the LOSO framework works correctly across all
supported datasets: olfactory, pfc_hpc, and dandi_movie.

Run with:
    pytest LOSO/test_multi_dataset.py -v

Or individual tests:
    pytest LOSO/test_multi_dataset.py::TestDatasetConfig -v
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


class TestDatasetConfig:
    """Test dataset configuration classes."""

    def test_dataset_configs_exist(self):
        """Test that all expected dataset configs are defined."""
        from LOSO.config import DATASET_CONFIGS

        assert "olfactory" in DATASET_CONFIGS
        assert "pfc_hpc" in DATASET_CONFIGS
        assert "dandi_movie" in DATASET_CONFIGS

    def test_olfactory_config(self):
        """Test olfactory dataset configuration."""
        from LOSO.config import get_dataset_config

        config = get_dataset_config("olfactory")
        assert config.name == "olfactory"
        assert config.session_type == "session"
        assert config.in_channels == 32
        assert config.out_channels == 32
        assert config.sampling_rate == 1000
        assert config.source_region == "OB"
        assert config.target_region == "PCx"
        assert config.train_py_dataset_name == "olfactory"

    def test_pfc_hpc_config(self):
        """Test PFC/HPC dataset configuration."""
        from LOSO.config import get_dataset_config

        config = get_dataset_config("pfc_hpc")
        assert config.name == "pfc_hpc"
        assert config.session_type == "session"
        assert config.in_channels == 64
        assert config.out_channels == 32
        assert config.sampling_rate == 1250
        assert config.source_region == "PFC"
        assert config.target_region == "CA1"
        assert config.train_py_dataset_name == "pfc"

    def test_dandi_movie_config(self):
        """Test DANDI movie dataset configuration."""
        from LOSO.config import get_dataset_config

        config = get_dataset_config("dandi_movie")
        assert config.name == "dandi_movie"
        assert config.session_type == "subject"  # Critical: subjects, not sessions
        assert config.uses_sliding_window is True
        assert config.train_py_dataset_name == "dandi"

    def test_invalid_dataset_raises(self):
        """Test that invalid dataset name raises ValueError."""
        from LOSO.config import get_dataset_config

        with pytest.raises(ValueError) as exc_info:
            get_dataset_config("invalid_dataset")
        assert "Unknown dataset" in str(exc_info.value)

    def test_dataset_config_copy(self):
        """Test that get_dataset_config returns a copy."""
        from LOSO.config import get_dataset_config

        config1 = get_dataset_config("olfactory")
        config2 = get_dataset_config("olfactory")

        # Modify one
        config1.source_region = "modified"

        # Other should be unchanged
        assert config2.source_region == "OB"


class TestLOSOConfig:
    """Test LOSOConfig dataclass."""

    def test_default_config(self):
        """Test default LOSOConfig values."""
        from LOSO.config import LOSOConfig

        config = LOSOConfig()
        assert config.dataset == "olfactory"
        assert config.epochs == 80
        assert config.batch_size == 64

    def test_dandi_config(self):
        """Test LOSOConfig with DANDI options."""
        from LOSO.config import LOSOConfig

        config = LOSOConfig(
            dataset="dandi_movie",
            dandi_source_region="amygdala",
            dandi_target_region="hippocampus",
            dandi_window_size=5000,
        )
        assert config.dataset == "dandi_movie"
        assert config.dandi_source_region == "amygdala"
        assert config.dandi_target_region == "hippocampus"

    def test_pfc_config(self):
        """Test LOSOConfig with PFC options."""
        from LOSO.config import LOSOConfig

        config = LOSOConfig(
            dataset="pfc_hpc",
            pfc_resample_to_1khz=True,
            pfc_sliding_window=True,
        )
        assert config.dataset == "pfc_hpc"
        assert config.pfc_resample_to_1khz is True
        assert config.pfc_sliding_window is True

    def test_invalid_dataset_in_config(self):
        """Test that invalid dataset in LOSOConfig raises ValueError."""
        from LOSO.config import LOSOConfig

        with pytest.raises(ValueError) as exc_info:
            LOSOConfig(dataset="invalid")
        assert "Unknown dataset" in str(exc_info.value)

    def test_get_dataset_config_with_overrides(self):
        """Test that get_dataset_config applies DANDI overrides."""
        from LOSO.config import LOSOConfig

        config = LOSOConfig(
            dataset="dandi_movie",
            dandi_source_region="medial_frontal_cortex",
            dandi_target_region="amygdala",
        )
        ds_config = config.get_dataset_config()

        assert ds_config.source_region == "medial_frontal_cortex"
        assert ds_config.target_region == "amygdala"

    def test_get_session_type_label(self):
        """Test session type labels."""
        from LOSO.config import LOSOConfig

        # Session-based datasets
        config_olfactory = LOSOConfig(dataset="olfactory")
        assert config_olfactory.get_session_type_label() == "Session"

        config_pfc = LOSOConfig(dataset="pfc_hpc")
        assert config_pfc.get_session_type_label() == "Session"

        # Subject-based dataset
        config_dandi = LOSOConfig(dataset="dandi_movie")
        assert config_dandi.get_session_type_label() == "Subject"


class TestLOSOFoldResult:
    """Test LOSOFoldResult dataclass."""

    def test_fold_result_with_dataset_metadata(self):
        """Test LOSOFoldResult includes dataset metadata."""
        from LOSO.config import LOSOFoldResult

        result = LOSOFoldResult(
            fold_idx=0,
            test_session="sub-CS41",
            train_sessions=["sub-CS42", "sub-CS45"],
            val_r2=0.85,
            val_loss=0.15,
            dataset="dandi_movie",
            session_type="subject",
        )

        assert result.dataset == "dandi_movie"
        assert result.session_type == "subject"

        # Check to_dict includes metadata
        result_dict = result.to_dict()
        assert result_dict["dataset"] == "dandi_movie"
        assert result_dict["session_type"] == "subject"


class TestValidation:
    """Test validation and safety check functions."""

    def test_data_leakage_check_passes(self):
        """Test that data leakage check passes for valid configuration."""
        from LOSO.runner import check_data_leakage

        # Should not raise
        check_data_leakage(
            fold_idx=0,
            test_session="session_1",
            train_sessions=["session_2", "session_3", "session_4"],
            dataset="olfactory",
        )

    def test_data_leakage_check_fails(self):
        """Test that data leakage check fails when test in train."""
        from LOSO.runner import check_data_leakage

        with pytest.raises(RuntimeError) as exc_info:
            check_data_leakage(
                fold_idx=0,
                test_session="session_1",
                train_sessions=["session_1", "session_2", "session_3"],
                dataset="olfactory",
            )
        assert "DATA LEAKAGE" in str(exc_info.value)

    def test_data_leakage_check_empty_train(self):
        """Test that data leakage check fails with empty train."""
        from LOSO.runner import check_data_leakage

        with pytest.raises(RuntimeError) as exc_info:
            check_data_leakage(
                fold_idx=0,
                test_session="session_1",
                train_sessions=[],
                dataset="olfactory",
            )
        assert "No training" in str(exc_info.value)


class TestSessionDiscovery:
    """Test session discovery functions.

    These tests require actual data files to be present.
    They are skipped if data is not available.
    """

    @pytest.mark.skipif(
        not Path("/home/user/Unet/data/olfactory").exists(),
        reason="Olfactory data not available"
    )
    def test_olfactory_session_discovery(self):
        """Test olfactory session discovery."""
        from LOSO.runner import get_all_sessions
        from LOSO.config import LOSOConfig

        config = LOSOConfig(dataset="olfactory")
        sessions, metadata = get_all_sessions("olfactory", config)

        assert len(sessions) >= 3, "Need at least 3 sessions for LOSO"
        assert metadata["session_type"] == "session"
        assert "trials_per_session" in metadata

    @pytest.mark.skipif(
        not Path("/home/user/Unet/data/pfc").exists(),
        reason="PFC data not available"
    )
    def test_pfc_session_discovery(self):
        """Test PFC session discovery."""
        from LOSO.runner import get_all_sessions
        from LOSO.config import LOSOConfig

        config = LOSOConfig(dataset="pfc_hpc")
        sessions, metadata = get_all_sessions("pfc_hpc", config)

        assert len(sessions) >= 3, "Need at least 3 sessions for LOSO"
        assert metadata["session_type"] == "session"

    @pytest.mark.skipif(
        not Path("/home/user/Unet/data/movie").exists(),
        reason="DANDI data not available"
    )
    def test_dandi_subject_discovery(self):
        """Test DANDI subject discovery."""
        from LOSO.runner import get_all_sessions
        from LOSO.config import LOSOConfig

        config = LOSOConfig(dataset="dandi_movie")
        subjects, metadata = get_all_sessions("dandi_movie", config)

        assert len(subjects) >= 3, "Need at least 3 subjects for LOSO"
        assert metadata["session_type"] == "subject"
        assert all(s.startswith("sub-") for s in subjects), "DANDI subjects should start with 'sub-'"


class TestCLIArgumentParsing:
    """Test CLI argument parsing."""

    def test_default_args(self):
        """Test default argument values."""
        from LOSO.runner import parse_args
        import sys

        # Save original args
        original_argv = sys.argv
        sys.argv = ["runner.py"]

        try:
            args = parse_args()
            assert args.dataset == "olfactory"
            assert args.epochs == 80
            assert args.dandi_source_region == "amygdala"
            assert args.dandi_target_region == "hippocampus"
        finally:
            sys.argv = original_argv

    def test_dandi_args(self):
        """Test DANDI-specific argument parsing."""
        from LOSO.runner import parse_args
        import sys

        original_argv = sys.argv
        sys.argv = [
            "runner.py",
            "--dataset", "dandi_movie",
            "--dandi-source-region", "medial_frontal_cortex",
            "--dandi-target-region", "amygdala",
            "--dandi-window-size", "10000",
        ]

        try:
            args = parse_args()
            assert args.dataset == "dandi_movie"
            assert args.dandi_source_region == "medial_frontal_cortex"
            assert args.dandi_target_region == "amygdala"
            assert args.dandi_window_size == 10000
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
