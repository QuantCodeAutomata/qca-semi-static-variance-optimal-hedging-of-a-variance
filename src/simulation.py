"""
Rough Heston simulation engine.

Simulates M paths of the rough Heston model using the multifactor Euler scheme.

Model dynamics:
    dX_t = -0.5 * V_t dt + sqrt(V_t) dW_t
    V_t^n = V0_term(t) + sum_i c_i * V_t^{n,i}
    dV_t^{n,i} = (-gamma_i * V_t^{n,i} - lambda_ * V_t^n) dt
                 + sigma * sqrt(V_t^n) dB_t
    d<W,B>_t = rho dt

where
    V0_term(t_k) = V0 + lambda_*theta * sum_i (c_i/gamma_i)*(1 - exp(-gamma_i*t_k))

is the deterministic drift contribution precomputed on the time grid.

ASSUMPTION:
    Maturity T = 1.0 (not stated in the paper; chosen as standard annual maturity).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .params import RoughHestonParams


# ---------------------------------------------------------------------------
# Correlated Brownian increment generator
# ---------------------------------------------------------------------------

def generate_correlated_increments(
    M: int,
    N: int,
    rho: float,
    dt: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate M*N pairs of correlated Brownian increments (dW, dB).

    dZ[:,:,0] and dZ[:,:,1] are i.i.d. N(0,1); then
        dW[:,k] = sqrt(dt) * dZ[:,k,0]
        dB[:,k] = sqrt(dt) * (rho * dZ[:,k,0] + sqrt(1-rho^2) * dZ[:,k,1])

    Parameters
    ----------
    M : int
        Number of Monte Carlo paths.
    N : int
        Number of time steps.
    rho : float
        Correlation parameter.
    dt : float
        Time step size.
    seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    dW : np.ndarray, shape (M, N)
        Brownian increment for the log-price SDE.
    dB : np.ndarray, shape (M, N)
        Correlated Brownian increment for the variance SDE.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((M, N, 2))
    sqrt_dt = np.sqrt(dt)
    dW = sqrt_dt * Z[:, :, 0]
    dB = sqrt_dt * (rho * Z[:, :, 0] + np.sqrt(1.0 - rho ** 2) * Z[:, :, 1])
    return dW, dB


# ---------------------------------------------------------------------------
# Deterministic V0 contribution
# ---------------------------------------------------------------------------

def _compute_V0_term(
    t_grid: np.ndarray,
    V0: float,
    theta: float,
    lambda_: float,
    c_arr: np.ndarray,
    gamma_arr: np.ndarray,
) -> np.ndarray:
    """
    Precompute the deterministic part of V^n:

        V0_term(t) = V0 + lambda_*theta * sum_i (c_i/gamma_i)*(1 - exp(-gamma_i*t))

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
        V0_term(t_k), shape (N+1,).
    """
    # Outer product: (N+1, n)
    exp_mat = np.exp(-gamma_arr[np.newaxis, :] * t_grid[:, np.newaxis])
    # (c_i / gamma_i) * (1 - exp(-gamma_i * t))
    integrals = (c_arr / gamma_arr)[np.newaxis, :] * (1.0 - exp_mat)
    return V0 + lambda_ * theta * integrals.sum(axis=1)


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def simulate_rough_heston(
    params: RoughHestonParams,
    c_arr: np.ndarray,
    gamma_arr: np.ndarray,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Simulate M paths of the rough Heston model using the multifactor Euler scheme.

    For each time step k = 0, ..., N-1:
        1.  Assemble V_k^n = V0_term[k] + sum_i c_i * V_factors[:,i]
        2.  sqrt_V_k = sqrt(max(0, V_k^n))  elementwise
        3.  Update each variance factor:
                V_factors[:,i] += dt * (-gamma_i * V_factors[:,i] - lambda_ * V_k^n)
                                  + sigma * sqrt_V_k * dB[:,k]
        4.  Update log-price:
                X += -0.5 * V_k^n * dt + sqrt_V_k * dW[:,k]

    The classical Heston branch (H == 0.5) uses a direct Euler discretisation
    of dV = lambda_*(theta - V) dt + sigma*sqrt(V) dB without the multifactor
    decomposition.

    Parameters
    ----------
    params : RoughHestonParams
        Full model and numerical parameter set.
    c_arr : np.ndarray
        Multifactor weights from :func:`kernel.compute_multifactor_coefficients`,
        shape (n,).
    gamma_arr : np.ndarray
        Multifactor rates, shape (n,).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        ``X_paths``        : np.ndarray, shape (M, N+1) – log-price paths
        ``V_paths``        : np.ndarray, shape (M, N+1) – variance paths
        ``V_factors_paths``: np.ndarray, shape (M, N+1, n) – factor paths
        ``dW_paths``       : np.ndarray, shape (M, N) – price BM increments
        ``dB_paths``       : np.ndarray, shape (M, N) – variance BM increments
        ``sqrt_V_paths``   : np.ndarray, shape (M, N+1) – sqrt(V) paths
        ``t_grid``         : np.ndarray, shape (N+1,) – time grid
        ``V0_term``        : np.ndarray, shape (N+1,) – deterministic drift array

    Notes
    -----
    Variance floored at zero (``max(0, V_t^n)``) to prevent NaN from square roots.
    """
    M = params.M
    N = params.N
    T = params.T
    dt = params.dt
    t_grid = params.t_grid
    V0 = params.V0
    X0 = params.X0
    theta = params.theta
    lambda_ = params.lambda_
    sigma = params.sigma
    rho = params.rho
    H = params.H
    n = len(c_arr)

    dW, dB = generate_correlated_increments(M, N, rho, dt, seed=seed)

    X_paths = np.empty((M, N + 1))
    V_paths = np.empty((M, N + 1))
    sqrt_V_paths = np.empty((M, N + 1))
    V_factors_paths = np.empty((M, N + 1, n))

    X_paths[:, 0] = X0
    sqrt_V_paths[:, 0] = np.sqrt(V0)

    if H == 0.5:
        # Classical Heston: direct Euler on V without multifactor
        V_paths[:, 0] = V0
        V_factors_paths[:, 0, :] = 0.0
        for k in range(N):
            V_k = V_paths[:, k]
            sqrt_V_k = np.sqrt(np.maximum(V_k, 0.0))
            sqrt_V_paths[:, k] = sqrt_V_k
            # Variance update
            V_next = V_k + lambda_ * (theta - V_k) * dt + sigma * sqrt_V_k * dB[:, k]
            V_paths[:, k + 1] = V_next
            sqrt_V_paths[:, k + 1] = np.sqrt(np.maximum(V_next, 0.0))
            # Log-price update
            X_paths[:, k + 1] = X_paths[:, k] - 0.5 * V_k * dt + sqrt_V_k * dW[:, k]
        V_factors_paths[:] = 0.0
        V0_term = np.full(N + 1, V0)  # placeholder
    else:
        # Rough Heston: multifactor Euler
        V0_term = _compute_V0_term(t_grid, V0, theta, lambda_, c_arr, gamma_arr)

        V_factors = np.zeros((M, n))          # V_t^{n,i}, current step
        V_factors_paths[:, 0, :] = 0.0

        # V at t=0: V0_term[0] + sum_i c_i * V_factors[0] = V0
        V_curr = np.full(M, V0_term[0])
        V_paths[:, 0] = V_curr
        sqrt_V_paths[:, 0] = np.sqrt(np.maximum(V_curr, 0.0))

        for k in range(N):
            sqrt_V_k = np.sqrt(np.maximum(V_curr, 0.0))
            sqrt_V_paths[:, k] = sqrt_V_k

            # Update each variance factor: shape (M, n)
            # dV_factors[:,i] = (-gamma_i * V_factors[:,i] - lambda_ * V_curr) * dt
            #                   + sigma * sqrt_V_k * dB[:,k]
            dB_k = dB[:, k]                              # (M,)
            V_factors += (
                dt * (-gamma_arr[np.newaxis, :] * V_factors
                      - lambda_ * V_curr[:, np.newaxis])
                + sigma * sqrt_V_k[:, np.newaxis] * dB_k[:, np.newaxis]
            )
            # Reassemble V^n at t_{k+1}
            V_next = V0_term[k + 1] + (c_arr[np.newaxis, :] * V_factors).sum(axis=1)

            V_factors_paths[:, k + 1, :] = V_factors
            V_paths[:, k + 1] = V_next

            # Log-price update uses V and sqrt_V at t_k
            X_paths[:, k + 1] = (
                X_paths[:, k]
                - 0.5 * V_curr * dt
                + sqrt_V_k * dW[:, k]
            )
            V_curr = V_next

        # Fill sqrt_V for the last time step
        sqrt_V_paths[:, N] = np.sqrt(np.maximum(V_curr, 0.0))

    return {
        "X_paths": X_paths,
        "V_paths": V_paths,
        "V_factors_paths": V_factors_paths,
        "dW_paths": dW,
        "dB_paths": dB,
        "sqrt_V_paths": sqrt_V_paths,
        "t_grid": t_grid,
        "V0_term": V0_term,
    }
