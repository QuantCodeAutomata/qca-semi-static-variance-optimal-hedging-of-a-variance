"""Tests for src/riccati.py."""

import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.riccati import (
    F_func,
    Q_func,
    solve_riccati_multifactor,
    solve_riccati_batch,
    compute_Khat_conv_g,
)
from src.kernel import (
    compute_multifactor_coefficients,
    compute_Khat_numerical,
)


# ---------------------------------------------------------------------------
# Shared parameter values (match paper defaults)
# ---------------------------------------------------------------------------

RHO = -0.7
SIGMA = 0.3
LAMBDA = 0.3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cg():
    """n=5 multifactor coefficients for H=0.1."""
    return compute_multifactor_coefficients(0.1, 5)


@pytest.fixture
def short_t_grid():
    """Uniform time grid [0, 1] with 21 points (N=20)."""
    return np.linspace(0.0, 1.0, 21)


@pytest.fixture
def Khat_vals(small_cg, short_t_grid):
    """Khat values on the short grid."""
    c, gamma = small_cg
    return compute_Khat_numerical(short_t_grid, c, gamma, SIGMA, LAMBDA)


# ---------------------------------------------------------------------------
# 1. F(u=0, x=0) = 0
# ---------------------------------------------------------------------------

def test_F_func_at_zero():
    """F(0, 0) must be exactly 0 by the formula 0.5*(0^2-0) + (rho*sigma*0-lambda)*0 + ...."""
    result = F_func(0.0, 0.0, RHO, SIGMA, LAMBDA)
    assert result == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# 2. F(u=1, x=0) = 0  (since 0.5*(1-1) = 0)
# ---------------------------------------------------------------------------

def test_F_func_value():
    """F(1, 0) = 0.5*(1^2 - 1) + 0 + 0 = 0."""
    result = F_func(1.0, 0.0, RHO, SIGMA, LAMBDA)
    assert result == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# 3. Q(u, x) = F(u, x) + lambda_ * x
# ---------------------------------------------------------------------------

def test_Q_func_relation():
    """Q(u, x) == F(u, x) + lambda_ * x for several (u, x) pairs."""
    test_cases = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0 + 0.5j, 0.3 + 0.1j),
        (0.5, 1.2),
    ]
    for u, x in test_cases:
        F_val = F_func(u, x, RHO, SIGMA, LAMBDA)
        Q_val = Q_func(u, x, RHO, SIGMA, LAMBDA)
        assert Q_val == pytest.approx(F_val + LAMBDA * x, abs=1e-14), (
            f"Q != F + lambda*x at u={u}, x={x}"
        )


# ---------------------------------------------------------------------------
# 4. Trivial Riccati: u=0 -> g_u(t) ≈ 0 everywhere
# ---------------------------------------------------------------------------

def test_riccati_trivial(small_cg, short_t_grid):
    """For u=0, psi_arr and g_arr should be identically 0.

    Reasoning: psi_n(0)=0, F(0,0)=0 so each Euler step adds 0,
    hence psi_n stays 0 and Q(0,0)=0 -> g_arr=0.
    """
    c, gamma = small_cg
    psi_arr, g_arr = solve_riccati_multifactor(
        0.0, short_t_grid, c, gamma, RHO, SIGMA, LAMBDA
    )
    assert np.allclose(psi_arr, 0.0, atol=1e-14), (
        f"psi_arr not zero for u=0: max|psi|={np.abs(psi_arr).max()}"
    )
    assert np.allclose(g_arr, 0.0, atol=1e-14), (
        f"g_arr not zero for u=0: max|g|={np.abs(g_arr).max()}"
    )


# ---------------------------------------------------------------------------
# 5. Output shapes: psi_arr and g_arr have shape (N+1,)
# ---------------------------------------------------------------------------

def test_riccati_shape(small_cg, short_t_grid):
    """solve_riccati_multifactor returns arrays of shape (N+1,)."""
    c, gamma = small_cg
    N_plus_1 = len(short_t_grid)
    u = 1.0 + 0.5j
    psi_arr, g_arr = solve_riccati_multifactor(
        u, short_t_grid, c, gamma, RHO, SIGMA, LAMBDA
    )
    assert psi_arr.shape == (N_plus_1,), (
        f"psi_arr shape {psi_arr.shape}, expected ({N_plus_1},)"
    )
    assert g_arr.shape == (N_plus_1,), (
        f"g_arr shape {g_arr.shape}, expected ({N_plus_1},)"
    )


# ---------------------------------------------------------------------------
# 6. compute_Khat_conv_g returns correct shape
# ---------------------------------------------------------------------------

def test_Khat_conv_g_shape(small_cg, short_t_grid, Khat_vals):
    """compute_Khat_conv_g returns an array of shape (N+1,)."""
    c, gamma = small_cg
    u = 1.0 + 0.5j
    _, g_arr = solve_riccati_multifactor(
        u, short_t_grid, c, gamma, RHO, SIGMA, LAMBDA
    )
    conv = compute_Khat_conv_g(short_t_grid, Khat_vals, g_arr)
    N_plus_1 = len(short_t_grid)
    assert conv.shape == (N_plus_1,), (
        f"Khat*g shape {conv.shape}, expected ({N_plus_1},)"
    )
    # Index 0 must be 0 by the causal convention (empty sum)
    assert conv[0] == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# 7. F(u=1, x) matches analytic formula
# ---------------------------------------------------------------------------

def test_riccati_u1():
    """For u=1: F(1, x) = (rho*sigma - lambda_)*x + 0.5*sigma^2*x^2.

    Verify the formula at a nontrivial x.
    """
    u = 1.0
    x = 0.5 + 0.2j
    F_val = F_func(u, x, RHO, SIGMA, LAMBDA)
    expected = (RHO * SIGMA - LAMBDA) * x + 0.5 * SIGMA ** 2 * x ** 2
    assert F_val == pytest.approx(expected, abs=1e-14), (
        f"F(1, x) = {F_val}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Batch solver consistency check
# ---------------------------------------------------------------------------

def test_riccati_batch_consistency(small_cg, short_t_grid):
    """solve_riccati_batch matches individual calls."""
    c, gamma = small_cg
    u_arr = np.array([0.0, 1.0 + 0.5j, 2.0 - 0.3j])
    batch = solve_riccati_batch(u_arr, short_t_grid, c, gamma, RHO, SIGMA, LAMBDA)
    for u in u_arr:
        psi_single, g_single = solve_riccati_multifactor(
            u, short_t_grid, c, gamma, RHO, SIGMA, LAMBDA
        )
        psi_batch, g_batch = batch[complex(u)]
        assert np.allclose(psi_single, psi_batch, atol=1e-14)
        assert np.allclose(g_single, g_batch, atol=1e-14)
