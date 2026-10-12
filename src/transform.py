"""
Conditional transform (Htilde) computation for the rough Heston model.

The conditional transform is defined as
    Htilde_t^u = E[exp(u X_T) | F_t] = exp(u X_t + Phi_t^u)

where the exponent update is
    Phi_t^u = integral_t^T g_u(T - s) * xi_t(s) ds.

Under the multifactor approximation the update equation along a simulated path is
    dPhi_t^u = -g_u(T-t) * V_t dt + sigma * sqrt(V_t) * dB_t * beta_u(t)

with
    beta_u(t) = sum_i c_i * Psi_u^i(t)

and Psi_u^i satisfies the backward ODE
    d/dt Psi_u^i(t) = gamma_i * Psi_u^i(t) - g_u(T - t),   Psi_u^i(T) = 0.

Numerically the backward ODE is solved by a simple Euler scheme marching from
t = T down to t = 0.

The initial value of Phi is
    Phi_0^u(t) = integral_t^T g_u(T - r) * xi_0(r) dr

which accumulates the forward variance curve and is used to initialise the
path-by-path update.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Backward ODE for Psi (beta integrand)
# ---------------------------------------------------------------------------

def compute_Psi_backward(
    t_grid: np.ndarray,
    g_vals: np.ndarray,
    gamma_arr: np.ndarray,
) -> np.ndarray:
    """
    Solve the backward Euler ODE for Psi_u^i on the time grid.

    ODE (time-reversed for numerical stability):
        d/dt Psi_u^i(t) = gamma_i * Psi_u^i(t) - g_u(T - t)

    Terminal condition: Psi_u^i(T) = 0.

    Backward Euler step from t_{k+1} to t_k:
        Psi_u^i(t_k) = Psi_u^i(t_{k+1}) - dt * (gamma_i * Psi_u^i(t_{k+1})
                                                   - g_u(T - t_{k+1}))

    Parameters
    ----------
    t_grid : np.ndarray
        Uniform time grid [0, T], shape (N+1,).
    g_vals : np.ndarray
        g_u(t_k) values on t_grid, shape (N+1,), dtype complex.
        Note: g_u is evaluated forward in time; g_u(T - t_{k+1}) corresponds
        to g_vals[N - (k+1)] = g_vals[N-k-1].
    gamma_arr : np.ndarray
        Multifactor rates, shape (n,).

    Returns
    -------
    np.ndarray, shape (N+1, n), dtype complex
        Psi_u^i(t_k) for all k and all factors i.
    """
    N = len(t_grid) - 1
    dt = t_grid[1] - t_grid[0]
    n = len(gamma_arr)

    Psi = np.zeros((N + 1, n), dtype=complex)
    # Terminal condition: Psi[N, :] = 0 (already set)

    for k in range(N - 1, -1, -1):
        # g_u(T - t_{k+1}) = g_u evaluated at time argument T - t_{k+1}
        # Since t_{k+1} = (k+1)*dt and g_vals is indexed forward from 0,
        # g_u(T - t_{k+1}) = g_vals[N - (k+1)] = g_vals[N - k - 1]
        g_arg = g_vals[N - k - 1]
        Psi[k, :] = Psi[k + 1, :] - dt * (gamma_arr * Psi[k + 1, :] - g_arg)

    return Psi


def compute_beta(
    t_grid: np.ndarray,
    g_vals: np.ndarray,
    c_arr: np.ndarray,
    gamma_arr: np.ndarray,
) -> np.ndarray:
    """
    Compute beta_u(t_k) = sum_i c_i * Psi_u^i(t_k) on the time grid.

    Parameters
    ----------
    t_grid : np.ndarray
        Uniform time grid [0, T], shape (N+1,).
    g_vals : np.ndarray
        g_u(t_k), shape (N+1,), dtype complex.
    c_arr : np.ndarray
        Multifactor weights, shape (n,).
    gamma_arr : np.ndarray
        Multifactor rates, shape (n,).

    Returns
    -------
    np.ndarray, shape (N+1,), dtype complex
        beta_u(t_k).
    """
    Psi = compute_Psi_backward(t_grid, g_vals, gamma_arr)
    return (c_arr[np.newaxis, :] * Psi).sum(axis=1)


# ---------------------------------------------------------------------------
# Initial Phi from forward variance
# ---------------------------------------------------------------------------

def compute_Phi0(
    t_grid: np.ndarray,
    g_vals: np.ndarray,
    xi0_vals: np.ndarray,
) -> np.ndarray:
    """
    Compute the deterministic initial value of Phi_t^u.

    Phi_0^u(t_k) = integral_{t_k}^T g_u(T - r) * xi_0(r) dr

    Solved by accumulating a right Riemann sum backward:
        Phi_0^u(T) = 0
        Phi_0^u(t_k) = Phi_0^u(t_{k+1}) + g_u(T - t_k) * xi_0(t_k) * dt

    Here g_u(T - t_k) is g_vals indexed at the mirror point, i.e. g_vals[N-k].

    Parameters
    ----------
    t_grid : np.ndarray
        Uniform time grid [0, T], shape (N+1,).
    g_vals : np.ndarray
        g_u(t_k), shape (N+1,), dtype complex.  (Forward-time index.)
    xi0_vals : np.ndarray
        Forward variance xi_0(t_k), shape (N+1,), real.

    Returns
    -------
    np.ndarray, shape (N+1,), dtype complex
        Phi_0^u(t_k).  Phi_0^u(T) == 0.
    """
    N = len(t_grid) - 1
    dt = t_grid[1] - t_grid[0]
    Phi0 = np.zeros(N + 1, dtype=complex)

    for k in range(N - 1, -1, -1):
        # g_u(T - t_k) = g_vals[N - k]
        Phi0[k] = Phi0[k + 1] + g_vals[N - k] * xi0_vals[k] * dt

    return Phi0


# ---------------------------------------------------------------------------
# Path-by-path Phi update
# ---------------------------------------------------------------------------

def update_Phi_paths(
    Phi_curr: np.ndarray,
    k: int,
    dt: float,
    V_paths: np.ndarray,
    dB_paths: np.ndarray,
    g_vals: np.ndarray,
    beta_vals: np.ndarray,
    sqrt_V_paths: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Euler update of Phi along all M paths from t_k to t_{k+1}.

    Update rule:
        Phi_{k+1} = Phi_k
                    - g_u(T - t_k) * V_t_k * dt
                    + sigma * sqrt_V_t_k * dB_k * beta_u(t_k)

    Parameters
    ----------
    Phi_curr : np.ndarray, shape (M,), dtype complex
        Current Phi values across all paths.
    k : int
        Current time index (Phi_curr corresponds to t_k).
    dt : float
        Time step size.
    V_paths : np.ndarray, shape (M, N+1)
        Simulated variance paths.
    dB_paths : np.ndarray, shape (M, N)
        Variance Brownian increments (pre-scaled by sqrt(dt)).
    g_vals : np.ndarray, shape (N+1,), dtype complex
        g_u values in forward time.  g_u(T - t_k) = g_vals[N - k].
    beta_vals : np.ndarray, shape (N+1,), dtype complex
        beta_u(t_k) precomputed by :func:`compute_beta`.
    sqrt_V_paths : np.ndarray, shape (M, N+1)
        sqrt(V_t_k) values across paths.
    sigma : float
        Vol-of-vol parameter.

    Returns
    -------
    np.ndarray, shape (M,), dtype complex
        Updated Phi at t_{k+1}.
    """
    N = V_paths.shape[1] - 1
    g_k = g_vals[N - k]          # g_u(T - t_k)
    beta_k = beta_vals[k]         # beta_u(t_k)

    Phi_next = (
        Phi_curr
        - g_k * V_paths[:, k] * dt
        + sigma * sqrt_V_paths[:, k] * dB_paths[:, k] * beta_k
    )
    return Phi_next


# ---------------------------------------------------------------------------
# Conditional transform
# ---------------------------------------------------------------------------

def compute_Htilde(
    u: complex,
    X_paths_k: np.ndarray,
    Phi_k: np.ndarray,
) -> np.ndarray:
    """
    Compute the conditional transform Htilde_t^u on all paths at time t_k.

    Htilde_t^u = exp(u * X_t + Phi_t^u).

    Parameters
    ----------
    u : complex
        Fourier/Laplace argument.
    X_paths_k : np.ndarray, shape (M,)
        Log-price at time t_k across all paths.
    Phi_k : np.ndarray, shape (M,), dtype complex
        Phi_t_k^u across all paths.

    Returns
    -------
    np.ndarray, shape (M,), dtype complex
        Htilde values.
    """
    return np.exp(u * X_paths_k + Phi_k)
