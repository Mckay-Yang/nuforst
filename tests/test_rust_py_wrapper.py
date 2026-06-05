import json
import pytest
import numpy as np
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "rust_parity"
SYNTH_DIR = FIXTURES_DIR / "synthetic"
REAL_DIR = FIXTURES_DIR / "real" / "small_window"


def _load_synthetic_fixture(name):
    data = np.load(str(SYNTH_DIR / name / "data.npz"))
    with open(SYNTH_DIR / name / "config.json") as f:
        config = json.load(f)
    return data, config


def _load_real_fixture():
    cube = np.load(str(REAL_DIR / "inputs.npy"))
    ts = np.load(str(REAL_DIR / "timestamps.npz"))
    timestamps_days = ts["timestamps_days"]
    with open(REAL_DIR / "config.json") as f:
        config = json.load(f)
    with open(REAL_DIR / "info.json") as f:
        info = json.load(f)
    target_time = info["target_time_day"]
    return cube, timestamps_days, target_time, config


# ---------------------------------------------------------------------------
# Per-pixel parity tests — synthetic single-pixel fixtures
# ---------------------------------------------------------------------------

class TestNufrostPixelParity:
    # Tolerance table matches nufrost-core parity tests:
    #   simple_harmonic: abs < 5e-5 || rel < 5e-4
    #   gaps_outliers, step_break: abs < 1e-3 || rel < 1e-2
    TOLERANCES = {
        "simple_harmonic": (5e-5, 5e-4),
        "gaps_outliers": (1e-3, 1e-2),
        "step_break": (1e-3, 1e-2),
    }

    @pytest.mark.parametrize("fixture_name", ["simple_harmonic", "gaps_outliers", "step_break"])
    def test_parity(self, fixture_name):
        data, config = _load_synthetic_fixture(fixture_name)
        t = data["timestamps_days"].astype(np.float64)
        y = data["observations"].astype(np.float64)
        target_t = float(data["target_time_day"])
        expected = float(data["nufrost_prediction"])

        from nufrost_py import nufrost_pixel_rust

        result = nufrost_pixel_rust(
            t.tolist(), y.tolist(), target_t, json.dumps(config["config"]["nufrost"])
        )

        assert np.isfinite(result), f"result must be finite, got {result}"
        if np.isfinite(expected):
            max_abs, max_rel = self.TOLERANCES[fixture_name]
            abs_err = abs(result - expected)
            rel_err = abs_err / max(abs(expected), 1e-12)
            ok = abs_err < max_abs or rel_err < max_rel
            assert ok, (
                f"{fixture_name}: Rust={result}, Python={expected}, "
                f"abs_err={abs_err:.2e}, rel_err={rel_err:.2e}"
            )


class TestHantsPixelParity:
    @pytest.mark.parametrize("fixture_name", ["simple_harmonic", "gaps_outliers", "step_break"])
    def test_parity(self, fixture_name):
        data, config = _load_synthetic_fixture(fixture_name)
        t = data["timestamps_days"].astype(np.float64)
        y = data["observations"].astype(np.float64)
        target_t = float(data["target_time_day"])
        expected = float(data["hants_prediction"])

        from nufrost_py import hants_pixel_rust

        result = hants_pixel_rust(
            t.tolist(), y.tolist(), target_t, json.dumps(config["config"]["hants"])
        )

        assert np.isfinite(result), f"result must be finite, got {result}"
        if np.isfinite(expected):
            assert np.isclose(result, expected, rtol=1e-5, atol=1e-6), (
                f"{fixture_name}: Rust={result}, Python={expected}"
            )


class TestZhu2015PixelParity:
    @pytest.mark.parametrize("fixture_name", ["simple_harmonic", "gaps_outliers", "step_break"])
    def test_parity(self, fixture_name):
        data, config = _load_synthetic_fixture(fixture_name)
        t = data["timestamps_days"].astype(np.float64)
        y = data["observations"].astype(np.float64)
        target_t = float(data["target_time_day"])
        expected = float(data["zhu2015_prediction"])
        expected_qa = int(data["zhu2015_qa"])

        from nufrost_py import zhu2015_pixel_rust

        result, qa = zhu2015_pixel_rust(
            t.tolist(), y.tolist(), target_t, json.dumps(config["config"]["zhu2015"])
        )

        assert np.isfinite(result), f"result must be finite, got {result}"
        assert qa == expected_qa, f"QA mismatch: {qa} != {expected_qa}"
        if np.isfinite(expected):
            assert np.isclose(result, expected, rtol=5e-4, atol=1e-6), (
                f"{fixture_name}: Rust={result}, Python={expected}"
            )


# ---------------------------------------------------------------------------
# Full-raster parity test — real small_window fixture
# ---------------------------------------------------------------------------

class TestRasterParity:
    def test_nufrost_raster(self):
        cube, t_days, target_time, config = _load_real_fixture()
        expected = np.load(str(REAL_DIR / "COPERNICUS_S2_HARMONIZED_B2_lon100.112_lat25.654-0000001024-0000000000_nufrost_pred.npy"))

        from src.nufrost_py_bridge import reconstruct_nufrost_rust

        result = reconstruct_nufrost_rust(cube, t_days, float(target_time), config["nufrost"])

        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        valid = np.isfinite(expected) & np.isfinite(result)
        if valid.sum() > 0:
            diff = np.abs(result[valid] - expected[valid])
            max_diff = diff.max()
            assert max_diff < 0.1, f"NUFROST raster max diff too large: {max_diff}"

    def test_hants_raster(self):
        cube, t_days, target_time, config = _load_real_fixture()
        expected = np.load(str(REAL_DIR / "COPERNICUS_S2_HARMONIZED_B2_lon100.112_lat25.654-0000001024-0000000000_hants_pred.npy"))

        from src.nufrost_py_bridge import reconstruct_hants_rust

        result = reconstruct_hants_rust(cube, t_days, float(target_time), config["hants"])

        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        valid = np.isfinite(expected) & np.isfinite(result)
        if valid.sum() > 0:
            diff = np.abs(result[valid] - expected[valid])
            max_diff = diff.max()
            assert max_diff < 1e-4, f"HANTS raster max diff too large: {max_diff}"

    def test_zhu2015_raster(self):
        cube, t_days, target_time, config = _load_real_fixture()
        expected = np.load(str(REAL_DIR / "COPERNICUS_S2_HARMONIZED_B2_lon100.112_lat25.654-0000001024-0000000000_zhu2015_pred.npy"))
        expected_qa = np.load(str(REAL_DIR / "COPERNICUS_S2_HARMONIZED_B2_lon100.112_lat25.654-0000001024-0000000000_zhu2015_qa.npy"))

        from src.nufrost_py_bridge import reconstruct_zhu2015_rust

        result, qa = reconstruct_zhu2015_rust(cube, t_days, float(target_time), config["zhu2015"])

        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        assert qa.shape == expected_qa.shape, f"QA shape mismatch: {qa.shape} vs {expected_qa.shape}"

        valid = np.isfinite(expected) & np.isfinite(result)
        if valid.sum() > 0:
            diff = np.abs(result[valid] - expected[valid])
            max_diff = diff.max()
            assert max_diff < 0.1, f"Zhu2015 raster max diff too large: {max_diff}"

        assert np.array_equal(qa, expected_qa), "QA band mismatch"


# ---------------------------------------------------------------------------
# Oracle tests: compare Rust wrapper vs Python oracle on fixtures
# ---------------------------------------------------------------------------

class TestPythonOracleParity:
    """These tests call the Python oracle per-pixel functions directly
    and compare them against the Rust results to verify parity."""

    def test_hants_oracle(self):
        data, config = _load_synthetic_fixture("simple_harmonic")
        t = data["timestamps_days"].astype(np.float64)
        y = data["observations"].astype(np.float64)
        target_t = float(data["target_time_day"])

        from src.hants import hants_pixel
        from nufrost_py import hants_pixel_rust

        hconf = config["config"]["hants"]
        python_result = hants_pixel(
            t, y, target_t,
            nof=hconf["nof"], sf=hconf["sf"],
            valid_min=hconf.get("valid_min"), valid_max=hconf.get("valid_max"),
            fet=hconf["fet"], dod=hconf["dod"], period=hconf["period"],
        )

        rust_result = hants_pixel_rust(
            t.tolist(), y.tolist(), target_t, json.dumps(hconf)
        )

        assert np.isclose(rust_result, python_result, rtol=1e-5, atol=1e-6), (
            f"HANTS oracle: Rust={rust_result}, Python={python_result}"
        )

    def test_zhu2015_oracle(self):
        data, config = _load_synthetic_fixture("simple_harmonic")
        t = data["timestamps_days"].astype(np.float64)
        y = data["observations"].astype(np.float64)
        target_t = float(data["target_time_day"])

        from src.zhu2015 import fit_predict_pixel
        from nufrost_py import zhu2015_pixel_rust

        zconf = config["config"]["zhu2015"]
        python_result = fit_predict_pixel(t, y, target_t, lasso_alpha=zconf["lasso_alpha"])

        rust_result, rust_qa = zhu2015_pixel_rust(
            t.tolist(), y.tolist(), target_t, json.dumps(zconf)
        )

        assert np.isclose(rust_result, python_result, rtol=5e-4, atol=1e-6), (
            f"Zhu2015 oracle: Rust={rust_result}, Python={python_result}"
        )

    def test_nufrost_oracle(self):
        data, config = _load_synthetic_fixture("simple_harmonic")
        t = data["timestamps_days"].astype(np.float64)
        y = data["observations"].astype(np.float64)
        target_t = float(data["target_time_day"])

        from src.nufrost import predict_single_pixel
        from nufrost_py import nufrost_pixel_rust

        nconf = config["config"]["nufrost"]
        python_result, _ = predict_single_pixel(
            t, y, target_t,
            nufft_modes=nconf["modes"],
            eps=nconf["eps"],
            num_peaks=nconf["num_peaks"],
            power_cum=nconf["power_cum"],
            ignore_dc_hz=nconf["ignore_dc_hz"],
            frequency_selection=nconf.get("frequency_selection", "spectral"),
            preferred_periods_days=nconf.get("preferred_periods_days", ""),
            preferred_top_k=nconf.get("preferred_top_k", 4),
            spectral_top_k=nconf.get("spectral_top_k", 4),
            spectral_merge_tol=nconf.get("spectral_merge_tol", 0.15),
            refine_peaks=nconf["refine_peaks"],
            include_trend=nconf["include_trend"],
            ridge_lam=nconf["ridge_lam"],
            freq_weight=nconf["freq_weight"],
            huber_iters=nconf["huber_iters"],
            huber_delta=nconf.get("huber_delta", 1.5),
            min_obs=nconf["min_obs"],
            outlier_sigma=nconf.get("outlier_sigma", 2.0),
        )

        rust_result = nufrost_pixel_rust(
            t.tolist(), y.tolist(), target_t, json.dumps(nconf)
        )

        assert np.isfinite(rust_result) and np.isfinite(python_result)
        abs_err = abs(rust_result - python_result)
        rel_err = abs_err / max(abs(python_result), 1e-12)
        assert abs_err < 5e-5 or rel_err < 5e-4, (
            f"NUFROST oracle: Rust={rust_result}, Python={python_result}, "
            f"abs_err={abs_err:.2e}, rel_err={rel_err:.2e}"
        )
