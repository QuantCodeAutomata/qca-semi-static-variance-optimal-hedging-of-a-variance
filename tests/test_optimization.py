"""Tests for src/optimization.py.

Uses small synthetic A, B, C matrices (d=5 options) to keep tests fast.
"""

import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.optimization import (
    solve_unconstrained_subset,
    solve_longonly_subset,
    brute_force_subset_search,
    run_sparse_search,
)


# ---------------------------------------------------------------------------
# Synthetic test data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_ABC():
    """Small, well-conditioned PSD matrices for d=5 options."""
    rng = np.random.default_rng(42)
    d = 5
    G = rng.standard_normal((d, d))
    C = G @ G.T + 0.5 * np.eye(d)      # PSD with well-conditioned
    B = rng.standard_normal(d) * 0.1   # small cross-covariance
    # Ensure A > B^T C^{-1} B so epsilon^2 remains positive
    A = float(B @ np.linalg.solve(C, B)) + 0.5
    return A, B, C


# ---------------------------------------------------------------------------
# 1. Unconstrained: C = I, B = b → nu* = b
# ---------------------------------------------------------------------------

def test_unconstrained_trivial():
    """With C = identity and B = b, the optimal weights are nu* = b."""
    d = 4
    B = np.array([0.1, -0.2, 0.3, 0.05])
    C = np.eye(d)
    nu_star, B_dot_nu = solve_unconstrained_subset(B, C)
    assert np.allclose(nu_star, B, atol=1e-12), (
        f"nu* should equal B for C=I; got {nu_star}"
    )
    assert B_dot_nu == pytest.approx(float(B @ B), rel=1e-10)


# ---------------------------------------------------------------------------
# 2. Unconstrained residual formula: epsilon^2 = A - B^T C^{-1} B
# ---------------------------------------------------------------------------

def test_unconstrained_residual(synthetic_ABC):
    """epsilon^2 equals A - B^T C^{-1} B at the unconstrained optimum."""
    A, B, C = synthetic_ABC
    nu_star, B_dot_nu = solve_unconstrained_subset(B, C)
    eps2 = A - 2.0 * B_dot_nu + float(nu_star @ C @ nu_star)
    expected = A - float(B @ np.linalg.solve(C, B))
    assert eps2 == pytest.approx(expected, rel=1e-8, abs=1e-10), (
        f"eps2={eps2}, expected A - B^T C^{{-1}} B = {expected}"
    )


# ---------------------------------------------------------------------------
# 3. Long-only: all weights >= 0 (within numerical tolerance)
# ---------------------------------------------------------------------------

def test_longonly_nonneg(synthetic_ABC):
    """Long-only solver must return non-negative weights."""
    A, B, C = synthetic_ABC
    nu_star, _ = solve_longonly_subset(B, C)
    assert (nu_star >= -1e-6).all(), (
        f"Long-only weights contain negatives: {nu_star}"
    )


# ---------------------------------------------------------------------------
# 4. Long-only residual >= unconstrained residual
# ---------------------------------------------------------------------------

def test_longonly_vs_unconstrained(synthetic_ABC):
    """The long-only constraint can only increase epsilon^2."""
    A, B, C = synthetic_ABC
    nu_unc, Bnu_unc = solve_unconstrained_subset(B, C)
    eps2_unc = A - 2.0 * Bnu_unc + float(nu_unc @ C @ nu_unc)

    nu_lo, Bnu_lo = solve_longonly_subset(B, C)
    eps2_lo = A - 2.0 * Bnu_lo + float(nu_lo @ C @ nu_lo)

    assert eps2_lo >= eps2_unc - 1e-6, (
        f"Long-only eps2={eps2_lo:.6f} should be >= unconstrained eps2={eps2_unc:.6f}"
    )


# ---------------------------------------------------------------------------
# 5. Brute force m=0: residual = A, subset = [], weights = []
# ---------------------------------------------------------------------------

def test_brute_force_m0(synthetic_ABC):
    """For m=0 (no options) the residual equals A and subset is empty."""
    A, B, C = synthetic_ABC
    best_subset, best_nu, best_residual = brute_force_subset_search(A, B, C, m=0)
    assert best_subset == [], f"m=0 subset should be empty, got {best_subset}"
    assert len(best_nu) == 0, "m=0 weights should be empty"
    assert best_residual == pytest.approx(A, rel=1e-12)


# ---------------------------------------------------------------------------
# 6. Brute force m=1: runs without error and returns valid result
# ---------------------------------------------------------------------------

def test_brute_force_m1(synthetic_ABC):
    """Brute-force search for m=1 returns a valid single-option subset."""
    A, B, C = synthetic_ABC
    d = len(B)
    best_subset, best_nu, best_residual = brute_force_subset_search(A, B, C, m=1)
    assert len(best_subset) == 1, f"m=1 subset length should be 1, got {len(best_subset)}"
    assert 0 <= best_subset[0] < d
    assert best_residual <= A + 1e-10, (
        "m=1 residual should be <= A (adding a hedge can only help)"
    )


# ---------------------------------------------------------------------------
# 7. Monotone decrease: epsilon^2(0) >= epsilon^2(1) >= epsilon^2(2)
# ---------------------------------------------------------------------------

def test_monotone_decrease(synthetic_ABC):
    """Hedging error is non-increasing as more options are added."""
    A, B, C = synthetic_ABC
    results = run_sparse_search(A, B, C, d=len(B), mode="unconstrained")
    eps2_0 = results[0]["epsilon2"]
    eps2_1 = results[1]["epsilon2"]
    eps2_2 = results[2]["epsilon2"]
    assert eps2_0 >= eps2_1 - 1e-10, (
        f"eps2(0)={eps2_0:.6f} >= eps2(1)={eps2_1:.6f} required"
    )
    assert eps2_1 >= eps2_2 - 1e-10, (
        f"eps2(1)={eps2_1:.6f} >= eps2(2)={eps2_2:.6f} required"
    )
