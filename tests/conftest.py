from pathlib import Path
import sys
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning


TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
INPUT_DIR = FIXTURES_DIR / "input"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@pytest.fixture
def fixture_input_dir() -> Path:
    return INPUT_DIR


@pytest.fixture
def sentinel_b2_tile_paths(fixture_input_dir: Path) -> list[str]:
    return sorted(str(path) for path in fixture_input_dir.glob("COPERNICUS_S2_HARMONIZED_B2_lon100.112_lat25.654-*.tif"))


@pytest.fixture
def single_tile_path(sentinel_b2_tile_paths: list[str]) -> str:
    return sentinel_b2_tile_paths[0]


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    path = tmp_path / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def synthetic_t_days() -> np.ndarray:
    return np.arange(0.0, 365.0, 16.0, dtype=np.float64)


@pytest.fixture
def synthetic_t_sec(synthetic_t_days: np.ndarray) -> np.ndarray:
    return synthetic_t_days * 86400.0


@pytest.fixture
def synthetic_signal(synthetic_t_days: np.ndarray) -> np.ndarray:
    return (0.25 + 0.05 * np.sin(2.0 * np.pi * synthetic_t_days / 365.25)).astype(np.float64)
