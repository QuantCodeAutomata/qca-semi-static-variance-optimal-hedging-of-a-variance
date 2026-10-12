"""
Monte Carlo estimation of the quadratic hedging matrices A, B, C.

Background
----------
The variance-optimal hedging error for a variance swap hedged by a stock (dynamic)
plus a portfolio of European options (static) is

    epsilon^2(nu) = A - 2 * nu^T B + nu^T C nu

where:
    A   = variance of the unhedged variance swap (scalar)
    B   = cross-covariance vector between the variance swap and the options (d-vector)
    C   = covariance matrix of the options (d x d matrix)

All three quantities involve integrals over time and over a Fourier/Laplace
contour.  The price of option j (put or call with strike K_j) enters through
the Fourier weight:
    hhat_j(u) = K_j^{1-u} / (u*(u-1)) / (2*pi)

and the contour integral is approximated by a discrete sum over N_C (or N_P)
quadrature nodes on the contour {R + i*y : y in [-L, L]}.

Matrix formulas (time-integrated, then MC-averaged):
    A     = (1-rho^2) * integral_0^T (Khat*1)^2(T-t) * E[V_t] dt
    B[j]  = (1-rho^2) * integral_0^T (Khat*1)(T-t)
               * E[V_t * integral_contour (Khat*g_u)(T-t) * Htilde_t^u * hhat_j(u) du] dt
    C[i,j]= (1-rho^2) * integral_0^T
               E[V_t * integral_ci (Khat*g_ui)(T-t) * Htilde_t^{ui} * hhat_i(ui) dui
                      * integral_cj (Khat*g_uj)(T-t) * Htilde_t^{uj} * hhat_j(uj) duj] dt

The contour integrals are discretised as Riemann sums and the time integral as
a forward Euler sum.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .params import RoughHestonParams
from .transform import compute_Phi0, compute_beta, update_Phi_paths, compute_Htilde


# ---------------------------------------------------------------------------
# Contour helpers
# ---------------------------------------------------------------------------

def build_contour_nodes(
    R: float,
    L: float,
    N_pts: int,
) -> Tuple[np.ndarray, float]:
    """
    Build a uniform contour in the complex plane {R + i*y : y in [-L, L]}.

    Parameters
    ----------
    R : float
        Real part of the contour.
    L : float
        Half-length in the imaginary direction.
    N_pts : int
        Number of quadrature points (including both endpoints).

    Returns
    -------
    u_nodes : np.ndarray, shape (N_pts,), dtype complex
        Contour nodes.
    dy : float
        Spacing between imaginary parts.
    """
    y = np.linspace(-L, L, N_pts)
    dy = y[1] - y[0] if N_pts > 1 else 2.0 * L
    u_nodes = R + 1j * y
    return u_nodes, dy


def fourier_weight(u: complex, K: float) -> complex:
    """
    Compute the Fourier weight for call/put option pricing via contour integral.

    hhat(u, K) = K^{1-u} / (u * (u-1)) / (2*pi)

    Parameters
    ----------
    u : complex
        Contour point.
    K : float
        Strike price.

    Returns
    -------
    complex
        The Fourier weight hhat(u, K).
    """
    return K ** (1.0 - u) / (u * (u - 1.0)) / (2.0 * np.pi)


# ---------------------------------------------------------------------------
# Scalar A
# ---------------------------------------------------------------------------

def compute_A(
    t_grid: np.ndarray,
    Khat1_vals: np.ndarray,
    xi0_vals: np.ndarray,
    rho: float,
) -> float:
    """
    Compute the scalar A by a trapezoidal quadrature over the time grid.

    A = (1-rho^2) * integral_0^T (Khat*1)^2(T-t) * E[V_t] dt

    Here (Khat*1)(T-t) at time index k is Khat1_vals[N-k] where N = len(t_grid)-1,
    and E[V_t] is approximated by xi0_vals.

    Parameters
    ----------
    t_grid : np.ndarray
        Uniform time grid [0, T], shape (N+1,).
    Khat1_vals : np.ndarray
        (Khat*1)(t) on [0, T], shape (N+1,).  Index k corresponds to t_k.
    xi0_vals : np.ndarray
        Forward variance curve xi_0(t_k) = E[V_{t_k}], shape (N+1,).
    rho : float
        Correlation parameter.

    Returns
    -------
    float
        The scalar A.

    Notes
    -----
    (Khat*1)(T - t_k) at step k uses Khat1_vals[N-k].
    """
    N = len(t_grid) - 1
    dt = t_grid[1] - t_grid[0]
    factor = 1.0 - rho ** 2

    # At each time k in [0, N], the integrand is Khat1(T-t_k)^2 * xi0(t_k).
    # We integrate over k = 0 to N-1 (i.e. exclude T itself since the hedge
    # can act up to but not including t=T).
    integrand = np.array([
        Khat1_vals[N - k] ** 2 * xi0_vals[k]
        for k in range(N)
    ])
    return float(factor * dt * integrand.sum())


# ---------------------------------------------------------------------------
# Main MC estimation function
# ---------------------------------------------------------------------------

def estimate_B_C_montecarlo(
    sim_data: Dict[str, np.ndarray],
    riccati_data: Dict[int, Dict],
    contour_data: Dict[int, Dict],
    params: RoughHestonParams,
    t_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the vectors B and matrix C by Monte Carlo simulation.

    Parameters
    ----------
    sim_data : dict
        Output of :func:`simulation.simulate_rough_heston`.  Expected keys:
        ``X_paths``, ``V_paths``, ``dB_paths``, ``sqrt_V_paths``.
    riccati_data : dict
        Mapping option_index -> dict with keys:
            ``u_nodes``    : contour nodes, shape (N_u,)
            ``g_arr``      : g_u(t) for each node, shape (N_u, N+1), dtype complex
            ``Khat_g_vals``: (Khat*g_u)(T-t) for each node, shape (N_u, N+1), dtype complex
            ``beta_vals``  : beta_u(t) for each node, shape (N_u, N+1), dtype complex
            ``Phi0_vals``  : Phi_0^u(t) for each node, shape (N_u, N+1), dtype complex
    contour_data : dict
        Mapping option_index -> dict with keys:
            ``dy``         : contour spacing
            ``K``          : strike
            ``is_call``    : bool
    params : RoughHestonParams
        Model and numerical parameters.
    t_grid : np.ndarray
        Uniform time grid [0, T], shape (N+1,).

    Returns
    -------
    B_vec : np.ndarray, shape (d,)
        Real-valued cross-covariance vector.
    C_mat : np.ndarray, shape (d, d)
        Real-valued covariance matrix.

    Notes
    -----
    The update of Phi is done path-by-path and contour-by-contour using
    :func:`transform.update_Phi_paths`.  For each (option, contour node) pair
    the conditional transform Htilde is computed and the Fourier-weighted sum
    accumulated.  B and C are assembled from these sums.
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
    d = params.d
    factor = 1.0 - rho ** 2

    # ------------------------------------------------------------------ init
    B_vec = np.zeros(d, dtype=float)
    C_mat = np.zeros((d, d), dtype=float)

    # Phi for each option j and each contour node l: shape (d, N_u_max, M)
    # We store a list of arrays (d elements), each (N_u_j, M)
    Phi_curr: List[np.ndarray] = []
    for j in range(d):
        N_u_j = len(riccati_data[j]["u_nodes"])
        Phi0_j = riccati_data[j]["Phi0_vals"]   # (N_u_j, N+1), complex
        # Initialise each path at t=0 with Phi0(0)
        Phi_curr.append(
            np.tile(Phi0_j[:, 0], (M, 1)).T.astype(complex)  # (N_u_j, M)
        )

    # ------------------------------------------------------------------ time loop
    for k in range(N):
        V_k = V_paths[:, k]          # (M,)
        sqrt_V_k = sqrt_V_paths[:, k]  # (M,)

        # --- compute the Fourier-weighted Htilde integral for each option j
        # Z_j[m] = sum_l  (Khat*g_{u_l})(T-t_k) * Htilde_t_k^{u_l}(m) * fw(u_l, K_j) * dy_j
        Z: List[np.ndarray] = []     # each entry shape (M,), complex

        for j in range(d):
            rd_j = riccati_data[j]
            cd_j = contour_data[j]
            u_nodes_j = rd_j["u_nodes"]          # (N_u_j,)
            Khat_g_j = rd_j["Khat_g_vals"]       # (N_u_j, N+1)
            beta_j = rd_j["beta_vals"]            # (N_u_j, N+1)
            g_j = rd_j["g_arr"]                  # (N_u_j, N+1)
            dy_j = cd_j["dy"]
            K_j = cd_j["K"]
            N_u_j = len(u_nodes_j)

            # For each contour node l compute Htilde_k^{u_l} and weight
            Z_j = np.zeros(M, dtype=complex)
            for l in range(N_u_j):
                u_l = u_nodes_j[l]
                fw_l = fourier_weight(u_l, K_j)
                Kg_l_k = Khat_g_j[l, N - k]     # (Khat*g_{u_l})(T-t_k)  scalar
                Phi_l = Phi_curr[j][l, :]        # (M,)
                Htilde_l = compute_Htilde(u_l, X_paths[:, k], Phi_l)
                Z_j += Kg_l_k * Htilde_l * fw_l * dy_j

            Z.append(Z_j)

        # ---- accumulate B and C contributions at this time step
        for j in range(d):
            rd_j = riccati_data[j]
            # Khat1(T-t_k) is stored per option (same for all j, but kept here for flexibility)
            Khat1_k = rd_j["Khat1_at_Ttk"][k]   # scalar: (Khat*1)(T-t_k)

            # B[j] += (1-rho^2)*dt * mean_m[ V_k[m] * Khat1(T-t_k) * Re(Z_j[m]) ]
            B_vec[j] += factor * dt * Khat1_k * np.mean(V_k * Z[j].real)

            for i in range(d):
                # C[i,j] += (1-rho^2)*dt * mean_m[ V_k[m] * Re(Z_i[m]) * Re(Z_j[m]) ]
                C_mat[i, j] += factor * dt * np.mean(V_k * Z[i].real * Z[j].real)

        # --- update Phi for all options and all contour nodes
        for j in range(d):
            rd_j = riccati_data[j]
            u_nodes_j = rd_j["u_nodes"]
            beta_j = rd_j["beta_vals"]
            g_j = rd_j["g_arr"]
            N_u_j = len(u_nodes_j)

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

    return B_vec, C_mat


# ---------------------------------------------------------------------------
# Convenience: build riccati_data dict from pre-computed arrays
# ---------------------------------------------------------------------------

def build_riccati_data(
    option_indices: List[int],
    u_nodes_list: List[np.ndarray],
    g_list: List[np.ndarray],
    Khat_g_list: List[np.ndarray],
    beta_list: List[np.ndarray],
    Phi0_list: List[np.ndarray],
    Khat1_vals: np.ndarray,
    t_grid: np.ndarray,
) -> Dict[int, Dict]:
    """
    Assemble the ``riccati_data`` dict expected by :func:`estimate_B_C_montecarlo`.

    Parameters
    ----------
    option_indices : list of int
        Integer indices 0, ..., d-1 for each option.
    u_nodes_list : list of np.ndarray
        Contour nodes for each option, each shape (N_u_j,).
    g_list : list of np.ndarray
        g_u values for each option, each shape (N_u_j, N+1).
    Khat_g_list : list of np.ndarray
        (Khat*g_u)(T-t) for each option, each shape (N_u_j, N+1).
    beta_list : list of np.ndarray
        beta_u(t) for each option, each shape (N_u_j, N+1).
    Phi0_list : list of np.ndarray
        Phi_0^u(t) for each option, each shape (N_u_j, N+1).
    Khat1_vals : np.ndarray
        (Khat*1)(t) on the full grid [0, T], shape (N+1,).
        Used to extract (Khat*1)(T-t_k) = Khat1_vals[N-k].
    t_grid : np.ndarray
        Time grid, shape (N+1,).

    Returns
    -------
    dict
        riccati_data mapping option index -> parameter dict.
    """
    N = len(t_grid) - 1
    riccati_data: Dict[int, Dict] = {}
    for idx, j in enumerate(option_indices):
        # (Khat*1)(T - t_k) for k = 0, ..., N-1
        Khat1_at_Ttk = np.array([Khat1_vals[N - k] for k in range(N)])
        riccati_data[j] = {
            "u_nodes": u_nodes_list[idx],
            "g_arr": g_list[idx],
            "Khat_g_vals": Khat_g_list[idx],
            "beta_vals": beta_list[idx],
            "Phi0_vals": Phi0_list[idx],
            "Khat1_at_Ttk": Khat1_at_Ttk,
        }
    return riccati_data


def build_contour_data(
    option_indices: List[int],
    all_strikes: List[float],
    n_puts: int,
    params: RoughHestonParams,
) -> Dict[int, Dict]:
    """
    Build the ``contour_data`` dict expected by :func:`estimate_B_C_montecarlo`.

    Puts (j < n_puts) use the put contour; calls (j >= n_puts) use the call contour.

    Parameters
    ----------
    option_indices : list of int
        Indices 0, ..., d-1.
    all_strikes : list of float
        Strike prices in order [put_strikes..., call_strikes...].
    n_puts : int
        Number of put options (so indices 0,...,n_puts-1 are puts).
    params : RoughHestonParams
        Parameter set containing contour specifications.

    Returns
    -------
    dict
        contour_data mapping option index -> dict with keys ``dy``, ``K``, ``is_call``.
    """
    _, dy_call = build_contour_nodes(params.R_C, params.L_C, params.N_C)
    _, dy_put = build_contour_nodes(params.R_P, params.L_P, params.N_P)

    contour_data: Dict[int, Dict] = {}
    for j in option_indices:
        is_call = j >= n_puts
        contour_data[j] = {
            "dy": dy_call if is_call else dy_put,
            "K": float(all_strikes[j]),
            "is_call": is_call,
        }
    return contour_data
