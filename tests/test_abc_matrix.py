"""Tests for src/abc_matrix.py.

Does NOT call estimate_B_C_montecarlo (too slow for unit tests).
Tests build_contour_nodes, fourier_weight, and compute_A individually.
"""

import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.abc_matrix import build_contour_nodes, fourier_weight, compute_A
from src.kernel import (
    compute_multifactor_coefficients,
    compute_Khat_numerical,
    compute_Khat_conv_1,
    compute_xi0_numerical,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def short_grid_data():
    """Precomputed Khat1 and xi0 on a short N=20 grid for H=0.1."""
    H = 0.1
    n = 10
    sigma, lambda_ = 0.3, 0.3
    V0, theta = 0.02, 0.066
    c, gamma = compute_multifactor_coefficients(H, n)
    t_grid = np.linspace(0.0, 1.0, 21)  # N=20
    Khat_vals = compute_Khat_numerical(t_grid, c, gamma, sigma, lambda_)
    Khat1_vals = compute_Khat_conv_1(t_grid, Khat_vals)
    xi0_vals = compute_xi0_numerical(t_grid, V0, theta, lambda_, c, gamma)
    return t_grid, Khat1_vals, xi0_vals


# ---------------------------------------------------------------------------
# 1. Call contour nodes: real part == R_C, N_pts nodes
# ---------------------------------------------------------------------------

def test_contour_nodes_call():
    """build_contour_nodes with R_C=2, L=5, N=10 → 10 nodes, all Re(u)=2."""
    R_C, L, N_pts = 2.0, 5.0, 10
    nodes, dy = build_contour_nodes(R_C, L, N_pts)
    assert len(nodes) == N_pts, f"Expected {N_pts} nodes, got {len(nodes)}"
    assert np.allclose(nodes.real, R_C), (
        f"Real parts should all be {R_C}; got {nodes.real}"
    )
    assert dy == pytest.approx(2.0 * L / (N_pts - 1), rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Put contour nodes: real part == R_P
# ---------------------------------------------------------------------------

def test_contour_nodes_put():
    """Put contour nodes have real part R_P = -0.125."""
    R_P, L, N_pts = -0.125, 10.0, 20
    nodes, dy = build_contour_nodes(R_P, L, N_pts)
    assert len(nodes) == N_pts
    assert np.allclose(nodes.real, R_P), (
        f"Put contour real parts should be {R_P}; got {nodes.real}"
    )
    # Imaginary parts span [-L, L]
    assert nodes.imag.min() == pytest.approx(-L, rel=1e-12)
    assert nodes.imag.max() == pytest.approx(L, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. fourier_weight: returns a finite complex number for call contour
# ---------------------------------------------------------------------------

def test_fourier_weight_call():
    """fourier_weight at a call contour node (R=2) returns finite complex."""
    u = 2.0 + 1.0j
    K = 100.0
    w = fourier_weight(u, K)
    assert np.isfinite(w.real) and np.isfinite(w.imag), (
        f"fourier_weight returned non-finite: {w}"
    )
    assert isinstance(complex(w), complex)


# ---------------------------------------------------------------------------
# 4. fourier_weight conjugate symmetry
# ---------------------------------------------------------------------------

def test_fourier_weight_symmetry():
    """hhat(R - i*y, K) == conj(hhat(R + i*y, K)) for real K, R, y > 0."""
    R, y, K = 2.0, 3.7, 100.0
    u_plus = R + 1j * y
    u_minus = R - 1j * y
    w_plus = fourier_weight(u_plus, K)
    w_minus = fourier_weight(u_minus, K)
    assert w_minus == pytest.approx(np.conj(w_plus), abs=1e-13), (
        f"Symmetry violated: w_minus={w_minus}, conj(w_plus)={np.conj(w_plus)}"
    )


# ---------------------------------------------------------------------------
# 5. compute_A is positive for valid parameters
# ---------------------------------------------------------------------------

def test_compute_A_positive(short_grid_data):
    """compute_A must return a positive scalar for valid Khat1 and xi0."""
    t_grid, Khat1_vals, xi0_vals = short_grid_data
    rho = -0.7
    A = compute_A(t_grid, Khat1_vals, xi0_vals, rho)
    assert A > 0.0, f"A = {A} should be positive"
    assert np.isfinite(A), "A should be finite"


# ---------------------------------------------------------------------------
# 6. compute_A formula consistency
# ---------------------------------------------------------------------------

def test_compute_A_formula():
    """compute_A matches a manual trapezoidal integration of (1-rho^2)*(Khat*1)^2*xi0."""
    N = 20
    t_grid = np.linspace(0.0, 1.0, N + 1)
    dt = t_grid[1] - t_grid[0]
    rho = -0.7

    # Simple synthetic data: constant xi0, linear Khat1
    xi0_vals = np.full(N + 1, 0.05)
    Khat1_vals = np.linspace(0.0, 0.2, N + 1)  # Khat1[k] = 0.2 * k/N

    A_computed = compute_A(t_grid, Khat1_vals, xi0_vals, rho)

    # Manual formula: A = (1-rho^2) * dt * sum_{k=0}^{N-1} Khat1[N-k]^2 * xi0[k]
    factor = 1.0 - rho ** 2
    integrand = np.array([
        Khat1_vals[N - k] ** 2 * xi0_vals[k] for k in range(N)
    ])
    A_manual = float(factor * dt * integrand.sum())

    assert A_computed == pytest.approx(A_manual, rel=1e-10), (
        f"compute_A={A_computed}, manual={A_manual}"
    )
