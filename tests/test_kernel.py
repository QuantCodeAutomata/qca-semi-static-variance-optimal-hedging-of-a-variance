"""Tests for src/kernel.py."""

import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest
from scipy.special import gamma as gamma_func

from src.kernel import (
    compute_multifactor_coefficients,
    compute_kernel_values,
    compute_multifactor_kernel_values,
    compute_Khat_numerical,
    compute_Khat_conv_1,
    compute_xi0_numerical,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cg_H01():
    """Small (n=10) multifactor coefficients for H=0.1."""
    return compute_multifactor_coefficients(0.1, 10)


@pytest.fixture
def small_cg_H03():
    """Small (n=10) multifactor coefficients for H=0.3."""
    return compute_multifactor_coefficients(0.3, 10)


@pytest.fixture
def short_t_grid():
    """Short uniform time grid [0, 1] with 21 points."""
    return np.linspace(0.0, 1.0, 21)


# ---------------------------------------------------------------------------
# 1. Multifactor coefficient checks – H = 0.1
# ---------------------------------------------------------------------------

def test_multifactor_coefficients_H01(small_cg_H01):
    c, gamma = small_cg_H01
    assert c.shape == (10,), "c should have shape (n,)"
    assert gamma.shape == (10,), "gamma should have shape (n,)"
    assert c.sum() > 0, "sum(c) must be positive"
    assert (c > 0).all(), "all weights c_i must be positive"
    assert (gamma > 0).all(), "all rates gamma_i must be positive"


# ---------------------------------------------------------------------------
# 2. Multifactor coefficient checks – H = 0.3
# ---------------------------------------------------------------------------

def test_multifactor_coefficients_H03(small_cg_H03):
    c, gamma = small_cg_H03
    assert (c > 0).all(), "all c_i > 0"
    assert (gamma > 0).all(), "all gamma_i > 0"
    assert c.sum() > 0


# ---------------------------------------------------------------------------
# 3. Exact kernel value at t = 1.0 for H = 0.1
# ---------------------------------------------------------------------------

def test_kernel_values_H01():
    """K(1.0) = 1^{H-1/2} / Gamma(H+1/2) = 1/Gamma(0.6) for H=0.1."""
    t_grid = np.array([0.0, 0.5, 1.0])
    K = compute_kernel_values(t_grid, H=0.1)
    expected = 1.0 / gamma_func(0.6)
    assert np.isclose(K[2], expected, rtol=1e-10), (
        f"K(1.0) = {K[2]}, expected {expected}"
    )
    # The singular point t=0 should be returned as 0
    assert K[0] == 0.0, "K(0) should be 0 (singularity suppressed)"


# ---------------------------------------------------------------------------
# 4. Multifactor kernel approximation accuracy
# ---------------------------------------------------------------------------

def test_multifactor_kernel_approx():
    """For H=0.1, n=200: K^n(t) ≈ K(t) within 5% for t in [0.1, 0.5, 1.0]."""
    H = 0.1
    c, gamma = compute_multifactor_coefficients(H, 200)
    t_test = np.array([0.1, 0.5, 1.0])
    Kn = compute_multifactor_kernel_values(t_test, c, gamma)
    K_exact = compute_kernel_values(t_test, H)
    rel_err = np.abs(Kn - K_exact) / np.abs(K_exact)
    assert (rel_err < 0.05).all(), (
        f"Multifactor approximation error exceeds 5%: {rel_err}"
    )


# ---------------------------------------------------------------------------
# 5. Khat numerical shape
# ---------------------------------------------------------------------------

def test_Khat_numerical_shape(small_cg_H01, short_t_grid):
    """compute_Khat_numerical returns an array of shape (N+1,)."""
    c, gamma = small_cg_H01
    sigma, lambda_ = 0.3, 0.3
    Khat = compute_Khat_numerical(short_t_grid, c, gamma, sigma, lambda_)
    assert Khat.shape == short_t_grid.shape, (
        f"Khat shape {Khat.shape} != t_grid shape {short_t_grid.shape}"
    )


# ---------------------------------------------------------------------------
# 6. Khat*1 is non-decreasing (since Khat >= 0)
# ---------------------------------------------------------------------------

def test_Khat_conv_1_monotone(small_cg_H01, short_t_grid):
    """(Khat*1)(t) = integral_0^t Khat(s) ds should be non-decreasing."""
    c, gamma = small_cg_H01
    sigma, lambda_ = 0.3, 0.3
    Khat = compute_Khat_numerical(short_t_grid, c, gamma, sigma, lambda_)
    Khat1 = compute_Khat_conv_1(short_t_grid, Khat)
    diffs = np.diff(Khat1)
    assert (diffs >= -1e-12).all(), (
        f"Khat*1 is not non-decreasing; min diff = {diffs.min():.3e}"
    )


# ---------------------------------------------------------------------------
# 7. xi0 boundary condition: xi0(0) = V0
# ---------------------------------------------------------------------------

def test_xi0_boundary(small_cg_H01, short_t_grid):
    """xi0(0) must equal V0."""
    c, gamma = small_cg_H01
    V0, theta, lambda_ = 0.02, 0.066, 0.3
    xi0 = compute_xi0_numerical(short_t_grid, V0, theta, lambda_, c, gamma)
    assert np.isclose(xi0[0], V0, rtol=1e-10), (
        f"xi0(0) = {xi0[0]}, expected V0 = {V0}"
    )


# ---------------------------------------------------------------------------
# 8. xi0 moves toward theta as t -> infinity (rough limit)
# ---------------------------------------------------------------------------

def test_xi0_limit():
    """xi0 should move toward theta over long time horizons."""
    c, gamma = compute_multifactor_coefficients(0.1, 50)
    # Long time grid so the Volterra resolvent has time to converge
    t_long = np.linspace(0.0, 20.0, 201)
    V0, theta, lambda_ = 0.02, 0.066, 0.3
    xi0 = compute_xi0_numerical(t_long, V0, theta, lambda_, c, gamma)
    assert np.isclose(xi0[0], V0, rtol=1e-6), "xi0(0) should equal V0"
    # xi0(T) should be strictly closer to theta than xi0(0) is
    dist_end = abs(xi0[-1] - theta)
    dist_start = abs(xi0[0] - theta)
    assert dist_end < dist_start, (
        f"xi0 did not move toward theta: "
        f"dist_start={dist_start:.4f}, dist_end={dist_end:.4f}"
    )
