"""
Riccati-Volterra equation solver for the rough Heston characteristic function.

The Laplace-Fourier transform of the log-price is
    E[exp(u X_T)] = exp(u X_0 + g_u(T))

where g_u solves the multifactor Riccati-Volterra system:

    psi_u^{n,i}(t) solves the ODE
        d/dt psi_u^{n,i} = -gamma_i * psi_u^{n,i} + F(u, psi_u^n),  psi_u^{n,i}(0) = 0
    psi_u^n(t) = sum_i c_i * psi_u^{n,i}(t)
    g_u(t)     = integral_0^t Q(u, psi_u^n(s)) ds  (accumulated via Q each step)

where
    F(u, x) = 0.5*(u^2 - u) + (rho*sigma*u - lambda_)*x + 0.5*sigma^2*x^2
    Q(u, x) = F(u, x) + lambda_*x
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Building blocks F and Q
# ---------------------------------------------------------------------------

def F_func(
    u: complex,
    x: complex,
    rho: float,
    sigma: float,
    lambda_: float,
) -> complex:
    """
    Evaluate the nonlinear Riccati driver F(u, x).

    F(u, x) = 0.5*(u^2 - u) + (rho*sigma*u - lambda_)*x + 0.5*sigma^2*x^2.

    Parameters
    ----------
    u : complex
        Fourier/Laplace argument.
    x : complex
        Current value of psi_u^n.
    rho : float
        Correlation coefficient.
    sigma : float
        Vol-of-vol.
    lambda_ : float
        Mean-reversion speed.

    Returns
    -------
    complex
        F(u, x).
    """
    return (
        0.5 * (u * u - u)
        + (rho * sigma * u - lambda_) * x
        + 0.5 * sigma ** 2 * x * x
    )


def Q_func(
    u: complex,
    x: complex,
    rho: float,
    sigma: float,
    lambda_: float,
) -> complex:
    """
    Evaluate Q(u, x) = F(u, x) + lambda_ * x.

    This is the integrand for g_u:
        g_u(t) = integral_0^t Q(u, psi_u^n(s)) ds.

    Parameters
    ----------
    u : complex
        Fourier/Laplace argument.
    x : complex
        Current value of psi_u^n.
    rho : float
        Correlation coefficient.
    sigma : float
        Vol-of-vol.
    lambda_ : float
        Mean-reversion speed.

    Returns
    -------
    complex
        Q(u, x).
    """
    return F_func(u, x, rho, sigma, lambda_) + lambda_ * x


# ---------------------------------------------------------------------------
# Single-u solver
# ---------------------------------------------------------------------------

def solve_riccati_multifactor(
    u: complex,
    t_grid: np.ndarray,
    c_arr: np.ndarray,
    gamma_arr: np.ndarray,
    rho: float,
    sigma: float,
    lambda_: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the multifactor Riccati-Volterra ODE system for a single (complex) u.

    The ODE system for i = 0, ..., n-1 is
        d/dt psi^{n,i} = -gamma_i * psi^{n,i} + F(u, psi^n),  psi^{n,i}(0) = 0
        psi^n = sum_i c_i * psi^{n,i}

    solved by the explicit Euler scheme:
        psi^{n,i}_{k+1} = psi^{n,i}_k + dt * (-gamma_i * psi^{n,i}_k + F(u, psi^n_k))

    and
        g_u(t_k) = Q(u, psi^n_k)

    Parameters
    ----------
    u : complex
        Fourier/Laplace argument.
    t_grid : np.ndarray
        Uniform time grid [0, T], shape (N+1,).
    c_arr : np.ndarray
        Multifactor weights, shape (n,).
    gamma_arr : np.ndarray
        Multifactor rates, shape (n,).
    rho : float
        Correlation coefficient.
    sigma : float
        Vol-of-vol.
    lambda_ : float
        Mean-reversion speed.

    Returns
    -------
    psi_arr : np.ndarray, shape (N+1,), dtype complex
        psi_u^n(t_k) for each time step.
    g_arr : np.ndarray, shape (N+1,), dtype complex
        g_u(t_k) = Q(u, psi_u^n(t_k)) for each time step.

    Notes
    -----
    The result is complex-valued; take the real part only where the paper
    asserts a real result (e.g., when u is purely imaginary for characteristic
    function evaluation).
    """
    N = len(t_grid) - 1
    dt = t_grid[1] - t_grid[0]
    n = len(c_arr)

    # Per-factor state vectors (complex)
    psi_factors = np.zeros(n, dtype=complex)   # psi^{n,i}_k

    psi_arr = np.zeros(N + 1, dtype=complex)
    g_arr = np.zeros(N + 1, dtype=complex)

    # t_0 = 0: all psi = 0
    psi_n = 0.0 + 0.0j
    psi_arr[0] = psi_n
    g_arr[0] = Q_func(u, psi_n, rho, sigma, lambda_)

    for k in range(N):
        F_val = F_func(u, psi_n, rho, sigma, lambda_)
        # Euler step for each factor
        psi_factors += dt * (-gamma_arr * psi_factors + F_val)
        # Aggregate
        psi_n = np.dot(c_arr, psi_factors)
        psi_arr[k + 1] = psi_n
        g_arr[k + 1] = Q_func(u, psi_n, rho, sigma, lambda_)

    return psi_arr, g_arr


# ---------------------------------------------------------------------------
# Batch solver
# ---------------------------------------------------------------------------

def solve_riccati_batch(
    u_arr: np.ndarray,
    t_grid: np.ndarray,
    c_arr: np.ndarray,
    gamma_arr: np.ndarray,
    rho: float,
    sigma: float,
    lambda_: float,
) -> Dict[complex, Tuple[np.ndarray, np.ndarray]]:
    """
    Solve the multifactor Riccati-Volterra system for each u in u_arr.

    Parameters
    ----------
    u_arr : np.ndarray
        Array of (complex) Fourier/Laplace arguments.
    t_grid : np.ndarray
        Uniform time grid, shape (N+1,).
    c_arr : np.ndarray
        Multifactor weights, shape (n,).
    gamma_arr : np.ndarray
        Multifactor rates, shape (n,).
    rho : float
        Correlation coefficient.
    sigma : float
        Vol-of-vol.
    lambda_ : float
        Mean-reversion speed.

    Returns
    -------
    dict
        Mapping {u: (psi_arr, g_arr)} where psi_arr and g_arr each have
        shape (N+1,) and dtype complex.
    """
    results: Dict[complex, Tuple[np.ndarray, np.ndarray]] = {}
    for u in u_arr:
        psi, g = solve_riccati_multifactor(
            u, t_grid, c_arr, gamma_arr, rho, sigma, lambda_
        )
        results[complex(u)] = (psi, g)
    return results


# ---------------------------------------------------------------------------
# Convolution (Khat * g_u)
# ---------------------------------------------------------------------------

def compute_Khat_conv_g(
    t_grid: np.ndarray,
    Khat_vals: np.ndarray,
    g_vals: np.ndarray,
) -> np.ndarray:
    """
    Compute the causal convolution (Khat * g_u)(t_k) by a forward Riemann sum.

    (Khat * g_u)(t_k) = integral_0^{t_k} Khat(t_k - s) * g_u(s) ds
                      ≈ dt * sum_{j=0}^{k-1} Khat(t_k - t_j) * g_u(t_j)

    Parameters
    ----------
    t_grid : np.ndarray
        Uniform time grid, shape (N+1,).
    Khat_vals : np.ndarray
        Khat(t_k) values, real-valued, shape (N+1,).
    g_vals : np.ndarray
        g_u(t_k) values, complex-valued, shape (N+1,).

    Returns
    -------
    np.ndarray
        (Khat * g_u)(t_k), complex-valued, shape (N+1,).
        Index 0 equals 0 by convention (empty sum).

    Notes
    -----
    Using a forward (left-endpoint) Riemann sum keeps the convolution causal:
    the value at t_k uses only g_u at times t_j < t_k.
    """
    N = len(t_grid) - 1
    dt = t_grid[1] - t_grid[0]
    result = np.zeros(N + 1, dtype=complex)

    for k in range(1, N + 1):
        # Khat(t_k - t_j) for j = 0, ..., k-1
        Khat_rev = Khat_vals[k:0:-1][:k]   # Khat[k], Khat[k-1], ..., Khat[1]
        result[k] = dt * np.dot(Khat_rev, g_vals[:k])

    return result
