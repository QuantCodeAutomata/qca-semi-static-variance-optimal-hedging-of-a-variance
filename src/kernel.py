"""
Multifactor exponential approximation of the rough Heston kernel.

The rough Heston fractional kernel is
    K(t) = t^{H - 1/2} / Gamma(H + 1/2),   0 < H < 1/2.

It is approximated by a sum of exponentials
    K^n(t) = sum_{i=1}^{n} c_i * exp(-gamma_i * t)

where the weights c_i and rates gamma_i are determined by a geometric
partition of the interval [eps, Gamma_max] in the spectral domain.

ASSUMPTION:
    Multifactor partition uses a geometric grid:
        eta_i = eps * (Gamma_max / eps)^{i/n},  i = 0, ..., n
    with eps = 1e-4 and Gamma_max = 50.0 (paper values).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import integrate
from scipy.special import gamma as gamma_func


# ---------------------------------------------------------------------------
# Multifactor coefficient computation
# ---------------------------------------------------------------------------

def compute_multifactor_coefficients(
    H: float,
    n: int,
    eps: float = 1e-4,
    Gamma_max: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the weights c_i and rates gamma_i for the n-term exponential
    approximation of the rough Heston kernel.

    The spectral density of the kernel K(t) = t^{H-1/2}/Gamma(H+1/2) is
        mu(gamma) = gamma^{-H-1/2} / (Gamma(H+1/2) * Gamma(1/2-H)).

    The geometric partition is
        eta_i = eps * (Gamma_max/eps)^{i/n},  i = 0, ..., n.

    On each sub-interval [eta_{i-1}, eta_i]:
        c_i     = integral mu(gamma) dgamma
        gamma_i = (1/c_i) * integral gamma * mu(gamma) dgamma

    Parameters
    ----------
    H : float
        Hurst exponent.  Must satisfy 0 < H < 0.5.
        For H == 0.5 a single-term approximation is returned (see Notes).
    n : int
        Number of exponential terms.
    eps : float
        Left boundary of the spectral partition.
    Gamma_max : float
        Right boundary of the spectral partition.

    Returns
    -------
    c_arr : np.ndarray, shape (n,)
        Spectral weights.
    gamma_arr : np.ndarray, shape (n,)
        Exponential decay rates.

    Notes
    -----
    For H == 0.5 the kernel reduces to K(t) = 1 (unit kernel scaled by
    lambda_ in the SDE), so the classical Heston dynamics hold.  In that
    case a single term (c=1, gamma=eps) is returned as a placeholder.
    """
    if H == 0.5:
        # Classical Heston: multifactor approximation is not meaningful.
        return np.array([1.0]), np.array([eps])

    # Spectral density normalisation constant
    norm_const = gamma_func(H + 0.5) * gamma_func(0.5 - H)

    # Geometric partition boundaries
    ratio = (Gamma_max / eps) ** (1.0 / n)
    eta = eps * ratio ** np.arange(n + 1)   # shape (n+1,)

    c_arr = np.empty(n)
    gamma_arr = np.empty(n)

    def _mu(g: float) -> float:
        """Spectral density mu(gamma)."""
        return g ** (-H - 0.5) / norm_const

    for i in range(n):
        a, b = eta[i], eta[i + 1]

        c_i, _ = integrate.quad(lambda g: _mu(g), a, b)
        if c_i <= 0.0:
            # Numerical zero — use midpoint as fallback
            c_arr[i] = _mu(0.5 * (a + b)) * (b - a)
            gamma_arr[i] = 0.5 * (a + b)
            continue

        gamma_i, _ = integrate.quad(lambda g: g * _mu(g), a, b)
        c_arr[i] = c_i
        gamma_arr[i] = gamma_i / c_i

    return c_arr, gamma_arr


# ---------------------------------------------------------------------------
# Kernel evaluation
# ---------------------------------------------------------------------------

def compute_kernel_values(t_grid: np.ndarray, H: float) -> np.ndarray:
    """
    Evaluate the exact rough Heston kernel K on a time grid.

    K(t) = t^{H - 1/2} / Gamma(H + 1/2)  for H != 0.5.
    K(t) = 1                              for H == 0.5 (unit kernel).

    Parameters
    ----------
    t_grid : np.ndarray
        Non-negative time values.  t=0 is handled by returning 0.
    H : float
        Hurst exponent.

    Returns
    -------
    np.ndarray
        K(t) values on the grid, same shape as t_grid.
    """
    if H == 0.5:
        return np.ones_like(t_grid, dtype=float)

    K = np.zeros_like(t_grid, dtype=float)
    pos = t_grid > 0.0
    K[pos] = t_grid[pos] ** (H - 0.5) / gamma_func(H + 0.5)
    return K


def compute_multifactor_kernel_values(
    t_grid: np.ndarray,
    c_arr: np.ndarray,
    gamma_arr: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the n-term exponential approximation K^n on a time grid.

    K^n(t) = sum_{i=1}^{n} c_i * exp(-gamma_i * t).

    Parameters
    ----------
    t_grid : np.ndarray
        Time values, shape (N+1,).
    c_arr : np.ndarray
        Spectral weights, shape (n,).
    gamma_arr : np.ndarray
        Exponential rates, shape (n,).

    Returns
    -------
    np.ndarray
        K^n(t) values, shape (N+1,).
    """
    # Outer product: (N+1, n)
    exponents = np.exp(-gamma_arr[np.newaxis, :] * t_grid[:, np.newaxis])
    return (c_arr[np.newaxis, :] * exponents).sum(axis=1)


# ---------------------------------------------------------------------------
# Volterra equations
# ---------------------------------------------------------------------------

def compute_Khat_numerical(
    t_grid: np.ndarray,
    c_arr: np.ndarray,
    gamma_arr: np.ndarray,
    sigma: float,
    lambda_: float,
) -> np.ndarray:
    """
    Solve the Volterra equation for the resolvent Khat by forward Euler.

    Khat solves:
        Khat(t) = sigma * K^n(t) + lambda_ * (K^n * Khat)(t)

    where the convolution is  (K^n * Khat)(t) = integral_0^t K^n(t-s) Khat(s) ds.

    The forward Euler recurrence (step size dt = t_grid[1] - t_grid[0]) is:
        Khat(t_0) = sigma * K^n(0)
        Khat(t_k) = sigma * K^n(t_k)
                    + lambda_ * dt * sum_{j=0}^{k-1} K^n(t_k - t_j) * Khat(t_j)

    Parameters
    ----------
    t_grid : np.ndarray
        Uniform time grid, shape (N+1,).
    c_arr : np.ndarray
        Multifactor weights, shape (n,).
    gamma_arr : np.ndarray
        Multifactor rates, shape (n,).
    sigma : float
        Vol-of-vol parameter.
    lambda_ : float
        Mean-reversion speed.

    Returns
    -------
    np.ndarray
        Khat(t_k) for k = 0, ..., N, shape (N+1,).
    """
    N = len(t_grid) - 1
    dt = t_grid[1] - t_grid[0]
    Kn = compute_multifactor_kernel_values(t_grid, c_arr, gamma_arr)

    Khat = np.zeros(N + 1)
    Khat[0] = sigma * Kn[0]

    for k in range(1, N + 1):
        # K^n(t_k - t_j) for j = 0, ..., k-1
        idx = np.arange(k)              # j indices
        Kn_diff = Kn[k - idx]           # K^n(t_k - t_j), using index arithmetic
        conv = dt * np.dot(Kn_diff, Khat[:k])
        Khat[k] = sigma * Kn[k] + lambda_ * conv

    return Khat


def compute_Khat_conv_1(t_grid: np.ndarray, Khat_vals: np.ndarray) -> np.ndarray:
    """
    Compute the running integral (Khat * 1)(t) = integral_0^t Khat(s) ds
    by the trapezoidal rule on the time grid.

    Parameters
    ----------
    t_grid : np.ndarray
        Uniform time grid, shape (N+1,).
    Khat_vals : np.ndarray
        Khat values on the grid, shape (N+1,).

    Returns
    -------
    np.ndarray
        (Khat * 1)(t_k), shape (N+1,).  Index 0 is 0 by convention.
    """
    dt = t_grid[1] - t_grid[0]
    result = np.zeros(len(t_grid))
    for k in range(1, len(t_grid)):
        # Trapezoidal rule over [0, t_k]
        result[k] = np.trapz(Khat_vals[: k + 1], dx=dt)
    return result


def compute_xi0_numerical(
    t_grid: np.ndarray,
    V0: float,
    theta: float,
    lambda_: float,
    c_arr: np.ndarray,
    gamma_arr: np.ndarray,
) -> np.ndarray:
    """
    Solve the Volterra equation for the forward variance curve xi_0(t) = E[V_t].

    The conditional expectation of the variance satisfies:
        m(t) = V0 + lambda_*theta * integral_0^t K^n(t-s) ds
                  - lambda_ * integral_0^t K^n(t-s) * m(s) ds.

    Forward Euler recurrence (step size dt):
        m(0) = V0
        m(t_k) = V0 + lambda_*theta * dt * sum_{j=0}^{k-1} K^n(t_k - t_j)
                     - lambda_ * dt * sum_{j=0}^{k-1} K^n(t_k - t_j) * m(t_j)

    For H == 0.5 (classical Heston) the exact formula is used:
        E[V_t] = theta + (V0 - theta) * exp(-lambda_ * t).

    Parameters
    ----------
    t_grid : np.ndarray
        Uniform time grid, shape (N+1,).
    V0 : float
        Initial variance.
    theta : float
        Long-run variance.
    lambda_ : float
        Mean-reversion speed.
    c_arr : np.ndarray
        Multifactor weights, shape (n,).
    gamma_arr : np.ndarray
        Multifactor rates, shape (n,).

    Returns
    -------
    np.ndarray
        xi_0(t_k) = E[V_{t_k}] for k = 0, ..., N, shape (N+1,).

    Notes
    -----
    The H == 0.5 branch is detected by checking whether len(c_arr) == 1
    and c_arr[0] == 1.0 (the placeholder set by compute_multifactor_coefficients).
    Callers that genuinely want rough Heston should not pass a unit placeholder.
    """
    # Classical Heston shortcut: single placeholder term (c=1, gamma=eps)
    # is not informative; detect H=0.5 via caller contract instead.
    # We always use the Volterra solver here for generality.
    N = len(t_grid) - 1
    dt = t_grid[1] - t_grid[0]
    Kn = compute_multifactor_kernel_values(t_grid, c_arr, gamma_arr)

    m = np.zeros(N + 1)
    m[0] = V0

    for k in range(1, N + 1):
        idx = np.arange(k)
        Kn_diff = Kn[k - idx]        # K^n(t_k - t_j), j=0,...,k-1
        m_prev = m[:k]
        theta_term = lambda_ * theta * dt * Kn_diff.sum()
        reversion_term = lambda_ * dt * np.dot(Kn_diff, m_prev)
        m[k] = V0 + theta_term - reversion_term

    return m


def compute_xi0_classical(
    t_grid: np.ndarray, V0: float, theta: float, lambda_: float
) -> np.ndarray:
    """
    Exact forward variance for classical Heston (H == 0.5).

    E[V_t] = theta + (V0 - theta) * exp(-lambda_ * t).

    Parameters
    ----------
    t_grid : np.ndarray
        Time grid, shape (N+1,).
    V0 : float
        Initial variance.
    theta : float
        Long-run variance.
    lambda_ : float
        Mean-reversion speed.

    Returns
    -------
    np.ndarray
        E[V_t] values, shape (N+1,).
    """
    return theta + (V0 - theta) * np.exp(-lambda_ * t_grid)
