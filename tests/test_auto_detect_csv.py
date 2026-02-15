"""Test auto-detect CSV functionality."""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from chronify import Store
from chronify.csv_utils import _should_use_auto_detect


@pytest.fixture
def csv_file() -> Generator[str, None, None]:
    """Create temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("timestamp,device,value\n")
        f.write("2020-01-01 00:00,A,100\n")
        f.write("2020-01-01 01:00,A,200\n")
        csv_path = f.name

    yield csv_path

    Path(csv_path).unlink(missing_ok=True)


@pytest.fixture
def clean_env() -> Generator[None, None, None]:
    """Ensure clean environment variable state."""
    yield
    if "CHRONIFY_AUTO_DETECT_CSV" in os.environ:
        del os.environ["CHRONIFY_AUTO_DETECT_CSV"]


def test_inspect_csv_default(csv_file: str) -> None:
    """Test CSV inspection without auto-detect."""
    store = Store()
    result = store.inspect_csv(csv_file)
    assert "error" not in result


def test_inspect_csv_with_parameter(csv_file: str) -> None:
    """Test CSV inspection with auto-detect parameter."""
    store = Store()
    result = store.inspect_csv(csv_file, auto_detect=True)
    assert "error" not in result


def test_inspect_csv_with_env_variable(csv_file: str, clean_env: None) -> None:
    """Test CSV inspection with environment variable."""
    os.environ["CHRONIFY_AUTO_DETECT_CSV"] = "true"
    store = Store()
    result = store.inspect_csv(csv_file)
    assert "error" not in result


def test_parameter_overrides_env(csv_file: str, clean_env: None) -> None:
    """Test parameter override of environment variable."""
    os.environ["CHRONIFY_AUTO_DETECT_CSV"] = "true"
    store = Store()
    result = store.inspect_csv(csv_file, auto_detect=False)
    assert "error" not in result


def test_should_use_auto_detect(clean_env: None) -> None:
    """Test auto-detect priority logic."""
    assert _should_use_auto_detect(None) is False
    assert _should_use_auto_detect(True) is True
    assert _should_use_auto_detect(False) is False

    os.environ["CHRONIFY_AUTO_DETECT_CSV"] = "true"
    assert _should_use_auto_detect(None) is True
    assert _should_use_auto_detect(False) is False
