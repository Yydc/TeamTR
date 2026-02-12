"""Unit tests for teamtr configuration."""

import os
from pathlib import Path

import pytest

# These imports should always work as they don't require heavy dependencies
from teamtr.config import (
    PROJECT_ROOT,
    DATASETS_ROOT,
    MODELS_ROOT,
    OUTPUTS_DIR,
)


class TestConfigPaths:
    """Test configuration path resolution."""

    def test_project_root_exists(self):
        """Test that project root is detected correctly."""
        assert PROJECT_ROOT.exists()
        assert (PROJECT_ROOT / "teamtr").exists()
        assert (PROJECT_ROOT / "verl").exists()

    def test_default_paths_relative_to_root(self):
        """Test that default paths are relative to project root when env vars not set."""
        # Clear environment variables
        orig_datasets = os.environ.pop("TEAMTR_DATASETS_ROOT", None)
        orig_models = os.environ.pop("TEAMTR_MODELS_ROOT", None)
        orig_outputs = os.environ.pop("TEAMTR_OUTPUTS_DIR", None)

        try:
            # Re-import to get fresh values
            import importlib
            import teamtr.config as config_module
            importlib.reload(config_module)

            # Check defaults are relative to project root
            assert str(config_module.DATASETS_ROOT).startswith(str(PROJECT_ROOT))
            assert str(config_module.MODELS_ROOT).startswith(str(PROJECT_ROOT))
            assert str(config_module.OUTPUTS_DIR).startswith(str(PROJECT_ROOT))
        finally:
            # Restore environment variables
            if orig_datasets:
                os.environ["TEAMTR_DATASETS_ROOT"] = orig_datasets
            if orig_models:
                os.environ["TEAMTR_MODELS_ROOT"] = orig_models
            if orig_outputs:
                os.environ["TEAMTR_OUTPUTS_DIR"] = orig_outputs

    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        test_datasets_path = "/tmp/test_datasets"
        os.environ["TEAMTR_DATASETS_ROOT"] = test_datasets_path

        try:
            import importlib
            import teamtr.config as config_module
            importlib.reload(config_module)

            assert str(config_module.DATASETS_ROOT) == test_datasets_path
        finally:
            os.environ.pop("TEAMTR_DATASETS_ROOT", None)
