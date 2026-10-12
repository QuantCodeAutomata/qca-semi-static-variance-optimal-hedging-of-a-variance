"""Tests for src/simulation.py."""

import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.params import RoughHestonParams
from src.kernel import compute_multifactor_coefficients
from src.simulation import generate_correlated_increments, simulate_rough_heston


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_params():
    """Small parameter set for fast tests."""
    return RoughHestonParams(H=0.1, M=100, N=20, n_factors=5)


@pytest.fixture
def cg_H01():
    return compute_multifactor_coefficients(0.1, 5)


@pytest.fixture
def cg_H05():
    return compute_multifactor_coefficients(0.5, 5)


# ---------------------------------------------------------------------------
# 1. generate_correlated_increments: output shape
# ---------------------------------------------------------------------------

def test_correlated_increments_shape():
    """generate_correlated_increments returns (dW, dB) each with shape (M, N)."""
    M, N, rho, dt = 100, 50, -0.7, 0.01
    dW, dB = generate_correlated_increments(M, N, rho, dt, seed=42)
    assert dW.shape == (M, N), f"dW shape {dW.shape}, expected ({M},{N})"
    assert dB.shape == (M, N), f"dB shape {dB.shape}, expected ({M},{N})"


# ---------------------------------------------------------------------------
# 2. generate_correlated_increments: empirical correlation ≈ rho
# ---------------------------------------------------------------------------

def test_correlated_increments_correlation():
    """Empirical correlation of dW and dB should be close to rho."""
    M, N, rho, dt = 5000, 100, -0.7, 0.01
    dW, dB = generate_correlated_increments(M, N, rho, dt, seed=42)
    emp_corr = np.corrcoef(dW.flatten(), dB.flatten())[0, 1]
    assert abs(emp_corr - rho) < 0.03, (
        f"Empirical correlation {emp_corr:.4f} too far from rho={rho}"
    )


# ---------------------------------------------------------------------------
# 3. simulate_rough_heston: path shapes
# ---------------------------------------------------------------------------

def test_simulation_shape(base_params, cg_H01):
    """X_paths shape (M, N+1), V_paths shape (M, N+1)."""
    c, gamma = cg_H01
    sim = simulate_rough_heston(base_params, c, gamma, seed=42)
    M = base_params.M
    N = base_params.N
    assert sim["X_paths"].shape == (M, N + 1)
    assert sim["V_paths"].shape == (M, N + 1)
    assert sim["sqrt_V_paths"].shape == (M, N + 1)


# ---------------------------------------------------------------------------
# 4. X_paths[:, 0] all equal X0
# ---------------------------------------------------------------------------

def test_simulation_X0(base_params, cg_H01):
    """All paths start at the initial log-price X0."""
    c, gamma = cg_H01
    sim = simulate_rough_heston(base_params, c, gamma, seed=42)
    assert np.allclose(sim["X_paths"][:, 0], base_params.X0), (
        "X_paths[:,0] should all equal X0"
    )


# ---------------------------------------------------------------------------
# 5. V_paths[:, 0] all equal V0
# ---------------------------------------------------------------------------

def test_simulation_V0(base_params, cg_H01):
    """All paths start at the initial variance V0."""
    c, gamma = cg_H01
    sim = simulate_rough_heston(base_params, c, gamma, seed=42)
    assert np.allclose(sim["V_paths"][:, 0], base_params.V0), (
        "V_paths[:,0] should all equal V0"
    )


# ---------------------------------------------------------------------------
# 6. sqrt_V_paths >= 0 (absorption floor at 0)
# ---------------------------------------------------------------------------

def test_simulation_V_nonneg(base_params, cg_H01):
    """sqrt(V_paths) should be non-negative everywhere (variance floored at 0)."""
    c, gamma = cg_H01
    sim = simulate_rough_heston(base_params, c, gamma, seed=42)
    assert (sim["sqrt_V_paths"] >= 0.0).all(), (
        "sqrt_V_paths contains negative values"
    )


# ---------------------------------------------------------------------------
# 7. exp(X_paths) > 0 everywhere (stock price always positive)
# ---------------------------------------------------------------------------

def test_simulation_price_positive(base_params, cg_H01):
    """S_t = exp(X_t) must be strictly positive."""
    c, gamma = cg_H01
    sim = simulate_rough_heston(base_params, c, gamma, seed=42)
    assert (np.exp(sim["X_paths"]) > 0.0).all(), (
        "exp(X_paths) contains non-positive values"
    )


# ---------------------------------------------------------------------------
# 8. Both H=0.5 and H=0.1 produce valid paths
# ---------------------------------------------------------------------------

def test_simulation_H05_vs_H01(cg_H01, cg_H05):
    """Both H=0.1 (rough) and H=0.5 (classical Heston) simulations run cleanly."""
    params_01 = RoughHestonParams(H=0.1, M=50, N=20, n_factors=5)
    params_05 = RoughHestonParams(H=0.5, M=50, N=20, n_factors=5)

    c01, g01 = cg_H01
    c05, g05 = cg_H05

    sim01 = simulate_rough_heston(params_01, c01, g01, seed=0)
    sim05 = simulate_rough_heston(params_05, c05, g05, seed=0)

    for key in ("X_paths", "V_paths", "sqrt_V_paths"):
        assert np.isfinite(sim01[key]).all(), f"H=0.1: {key} contains non-finite values"
        assert np.isfinite(sim05[key]).all(), f"H=0.5: {key} contains non-finite values"

    # Initial conditions match
    assert np.allclose(sim01["V_paths"][:, 0], params_01.V0)
    assert np.allclose(sim05["V_paths"][:, 0], params_05.V0)
    assert np.allclose(sim01["X_paths"][:, 0], params_01.X0)
    assert np.allclose(sim05["X_paths"][:, 0], params_05.X0)
