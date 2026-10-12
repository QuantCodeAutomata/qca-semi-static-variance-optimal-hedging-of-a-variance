"""
Performance metrics and diagnostic utilities for the hedging experiment.

Functions cover:
    - Effectiveness metrics Delta(m) and eta(m)
    - Validation of the ABC matrices (symmetry, PSD, positivity)
    - Path-by-path computation of the optimal dynamic hedge ratio
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import numpy.linalg as la

from .abc_matrix import build_contour_nodes, fourier_weight
from .transform import compute_Htilde
from .params import RoughHestonParams


# ---------------------------------------------------------------------------
# Effectiveness metrics
# ---------------------------------------------------------------------------

def compute_effectiveness(
    epsilon2_dict: Dict[int, Optional[float]],
    A: float,
) -> Dict[int, Dict[str, Optional[float]]]:
    """
    Compute Delta(m) and eta(m) from the epsilon^2 results dictionary.

    Delta(m) = (1 - epsilon^2(m) / A) * 100   [% variance reduction vs unhedged]
    eta(m)   = (A - epsilon^2(m)) / (A - epsilon^2(d)) * 100
               [% of maximum attainable variance reduction]

    Parameters
    ----------
    epsilon2_dict : dict
        Mapping m -> epsilon^2(m).  Missing or None values are propagated as None.
    A : float
        Unhedged hedging error (epsilon^2(0) = A).

    Returns
    -------
    dict
        {m: {'Delta': float | None, 'eta': float | None}}.
    """
    d = max(epsilon2_dict.keys())
    eps_full = epsilon2_dict.get(d)
    denominator = (A - eps_full) if (eps_full is not None and abs(A - eps_full) > 1e-15) else 1.0

    metrics: Dict[int, Dict[str, Optional[float]]] = {}
    for m, eps2 in epsilon2_dict.items():
        if eps2 is None:
            metrics[m] = {"Delta": None, "eta": None}
            continue
        Delta_m = (1.0 - eps2 / A) * 100.0 if A > 0 else 0.0
        eta_m = (A - eps2) / denominator * 100.0 if eps_full is not None else None
        metrics[m] = {"Delta": Delta_m, "eta": eta_m}

    return metrics


# ---------------------------------------------------------------------------
# ABC matrix validation
# ---------------------------------------------------------------------------

def validate_ABC(
    A: float,
    B: np.ndarray,
    C: np.ndarray,
) -> Dict[str, float]:
    """
    Run diagnostic checks on the matrices A, B, C.

    Checks performed:
        - A > 0 (positive definite scalar)
        - C symmetric (symmetry error relative to Frobenius norm)
        - C PSD (minimum eigenvalue of symmetrised C)
        - B in a sensible range (max absolute value)

    Parameters
    ----------
    A : float
        Scalar hedging error of the unhedged position.
    B : np.ndarray, shape (d,)
        Cross-covariance vector.
    C : np.ndarray, shape (d, d)
        Option covariance matrix.

    Returns
    -------
    dict
        Diagnostic values:
            ``A_positive``      : bool – whether A > 0
            ``symmetry_error``  : float – ||C - C^T||_F / ||C||_F
            ``min_eigenvalue``  : float – minimum eigenvalue of 0.5*(C+C^T)
            ``max_B_abs``       : float – max |B[j]|
            ``C_frobenius``     : float – Frobenius norm of C
    """
    C_sym = 0.5 * (C + C.T)
    C_frob = float(la.norm(C, "fro"))
    sym_err = float(la.norm(C - C.T, "fro")) / (C_frob + 1e-30)
    eigvals = la.eigvalsh(C_sym)
    min_eig = float(eigvals.min())

    return {
        "A_positive": bool(A > 0),
        "symmetry_error": sym_err,
        "min_eigenvalue": min_eig,
        "max_B_abs": float(np.abs(B).max()),
        "C_frobenius": C_frob,
    }


# ---------------------------------------------------------------------------
# Dynamic hedge ratio
# ---------------------------------------------------------------------------

def compute_dynamic_hedge(
    t_grid: np.ndarray,
    sim_data: Dict[str, np.ndarray],
    riccati_data: Dict[int, Dict],
    nu_star: np.ndarray,
    params: RoughHestonParams,
    Khat1_vals: np.ndarray,
    Khat_g_data: Dict[int, np.ndarray],
    contour_data: Dict[int, Dict],
) -> np.ndarray:
    """
    Compute the variance-optimal dynamic hedge ratio vartheta_t^* pathwise.

    The optimal dynamic hedge in the stock is
        vartheta_t^* = vartheta_t^0 - sum_j nu_j^* * vartheta_t^j

    where:
        vartheta_t^0 = rho * (Khat*1)(T-t) / S_t
        vartheta_t^j = (1/S_t) * Re[ integral_{contour_j}
                             Htilde_t^u * (u + rho*(Khat*g_u)(T-t)) * hhat_j(u) du ]

    Parameters
    ----------
    t_grid : np.ndarray
        Uniform time grid [0, T], shape (N+1,).
    sim_data : dict
        Simulation output (``X_paths``, ``V_paths``, ``dB_paths``,
        ``sqrt_V_paths``).
    riccati_data : dict
        option_index -> dict with keys ``u_nodes``, ``g_arr``,
        ``Khat_g_vals``, ``beta_vals``, ``Phi0_vals``.
    nu_star : np.ndarray, shape (d,) or shape (m,) for a subset
        Optimal static weights.  Non-participating options have nu=0.
    params : RoughHestonParams
        Model and numerical parameters.
    Khat1_vals : np.ndarray, shape (N+1,)
        (Khat*1)(t) on [0, T].
    Khat_g_data : dict
        option_index -> np.ndarray shape (N_u_j, N+1) – (Khat*g_u)(T-t).
    contour_data : dict
        option_index -> dict with keys ``dy``, ``K``, ``is_call``.

    Returns
    -------
    np.ndarray, shape (M, N+1)
        Optimal delta hedge ratio paths (in units of 1/S_t already divided out).

    Notes
    -----
    The output at t_k is vartheta_{t_k}^* in the *number of shares*.
    """
    X_paths = sim_data["X_paths"]       # (M, N+1)
    V_paths = sim_data["V_paths"]       # (M, N+1)
    dB_paths = sim_data["dB_paths"]     # (M, N)
    sqrt_V_paths = sim_data["sqrt_V_paths"]  # (M, N+1)

    M = params.M
    N = params.N
    dt = params.dt
    sigma = params.sigma
    rho = params.rho
    d = len(nu_star)

    # Initialise Phi paths for each option and contour node
    Phi_curr: List[np.ndarray] = []
    for j in range(d):
        rd_j = riccati_data[j]
        N_u_j = len(rd_j["u_nodes"])
        Phi0_j = rd_j["Phi0_vals"]   # (N_u_j, N+1)
        Phi_curr.append(
            np.tile(Phi0_j[:, 0], (M, 1)).T.astype(complex)  # (N_u_j, M)
        )

    hedge_paths = np.zeros((M, N + 1))

    for k in range(N + 1):
        S_k = np.exp(X_paths[:, k])    # (M,)

        # Unhedged dynamic component: rho * (Khat*1)(T-t_k) / S_k
        Khat1_k = Khat1_vals[N - k]
        vartheta0_k = rho * Khat1_k / S_k

        # Static option contribution
        opt_contrib = np.zeros(M, dtype=float)
        for j in range(d):
            if abs(nu_star[j]) < 1e-15:
                continue
            rd_j = riccati_data[j]
            u_nodes_j = rd_j["u_nodes"]
            N_u_j = len(u_nodes_j)
            Khat_g_j = Khat_g_data[j]   # (N_u_j, N+1)
            dy_j = contour_data[j]["dy"]
            K_j = contour_data[j]["K"]

            integrand_sum = np.zeros(M, dtype=complex)
            for l in range(N_u_j):
                u_l = u_nodes_j[l]
                fw_l = fourier_weight(u_l, K_j)
                Kg_l = Khat_g_j[l, N - k]
                Phi_l = Phi_curr[j][l, :]
                H_l = compute_Htilde(u_l, X_paths[:, k], Phi_l)
                integrand_sum += H_l * (u_l + rho * Kg_l) * fw_l * dy_j

            opt_contrib += nu_star[j] * (integrand_sum.real / S_k)

        hedge_paths[:, k] = vartheta0_k - opt_contrib

        # Advance Phi if not at terminal time
        if k < N:
            for j in range(d):
                if abs(nu_star[j]) < 1e-15:
                    continue
                rd_j = riccati_data[j]
                u_nodes_j = rd_j["u_nodes"]
                N_u_j = len(u_nodes_j)
                beta_j = rd_j["beta_vals"]
                g_j = rd_j["g_arr"]

                from .transform import update_Phi_paths
                for l in range(N_u_j):
                    Phi_curr[j][l, :] = update_Phi_paths(
                        Phi_curr=Phi_curr[j][l, :],
                        k=k,
                        dt=dt,
                        V_paths=V_paths,
                        dB_paths=dB_paths,
                        g_vals=g_j[l, :],
                        beta_vals=beta_j[l, :],
                        sqrt_V_paths=sqrt_V_paths,
                        sigma=sigma,
                    )

    return hedge_paths
