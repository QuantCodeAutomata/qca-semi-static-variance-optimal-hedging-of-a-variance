"""
Shared pipeline helpers for all four experiments.

All expensive pre-computation is isolated here so the experiment scripts stay
readable.  Every experiment calls _run_one_H(...) which returns a flat dict
of all relevant arrays and scalars ready for optimisation and plotting.
"""

import os

import numpy as np

from src.params import RoughHestonParams
from src.kernel import (
    compute_multifactor_coefficients,
    compute_Khat_numerical,
    compute_Khat_conv_1,
    compute_xi0_numerical,
)
from src.riccati import solve_riccati_multifactor, compute_Khat_conv_g
from src.transform import compute_beta, compute_Phi0
from src.simulation import simulate_rough_heston
from src.abc_matrix import (
    build_contour_nodes,
    compute_A,
    estimate_B_C_montecarlo,
)
from src.optimization import brute_force_subset_search

# ---------------------------------------------------------------------------
# Experiment default settings (reduced from paper for tractable run-time)
# ---------------------------------------------------------------------------
# Paper settings: M=10_000, N=250, n_factors=500, N_C=N_P=100, L_C=L_P=150
# Reduced settings below allow the full pipeline to complete in under 30s per H.
EXP_DEFAULTS = dict(
    M=2000,
    N=100,
    n_factors=100,
    N_C=10,   # paper: 100
    L_C=30.0,  # paper: 150.0
    N_P=10,   # paper: 100
    L_P=30.0,  # paper: 150.0
)

# Maximum subset size for brute-force search.
# Unconstrained uses a fast linear solve (~0.1 ms/subset):
#   C(21,6) ≈ 54 K subsets → ~6 s.  Paper reports m=0..10.
M_MAX_BRUTE = 6
# Long-only uses CVXPY (~3.5 ms/subset — order-of-magnitude slower):
#   C(21,4) ≈ 6 K subsets → ~21 s.  Set cap at 4 to stay ≤ 90s/H.
M_MAX_BRUTE_LO = 4


def make_params(H: float, **overrides) -> RoughHestonParams:
    """Return experiment params for Hurst exponent H with optional overrides."""
    kwargs = {**EXP_DEFAULTS, **overrides, "H": H}
    return RoughHestonParams(**kwargs)


def _build_riccati_contour(params: RoughHestonParams, c, gamma,
                            Khat_vals, Khat1_vals, xi0_vals):
    """Pre-compute Riccati solutions and assemble riccati_data / contour_data."""
    t_grid = params.t_grid
    N = params.N
    n_puts = len(params.put_strikes)

    call_nodes, dy_c = build_contour_nodes(params.R_C, params.L_C, params.N_C)
    put_nodes, dy_p = build_contour_nodes(params.R_P, params.L_P, params.N_P)

    # Solve Riccati ODEs once per unique contour node
    riccati_cache: dict = {}
    for u in list(call_nodes) + list(put_nodes):
        if u in riccati_cache:
            continue
        psi, g = solve_riccati_multifactor(
            u, t_grid, c, gamma, params.rho, params.sigma, params.lambda_
        )
        Kg = compute_Khat_conv_g(t_grid, Khat_vals, g)
        beta = compute_beta(t_grid, g, c, gamma)
        Phi0 = compute_Phi0(t_grid, g, xi0_vals)
        riccati_cache[u] = (psi, g, Kg, beta, Phi0)

    # Assemble per-option arrays
    Khat1_at_Ttk = np.array([Khat1_vals[N - k] for k in range(N)])
    riccati_data: dict = {}
    contour_data: dict = {}

    for j in range(params.d):
        is_call = j >= n_puts
        nodes = call_nodes if is_call else put_nodes
        dy = dy_c if is_call else dy_p
        N_u = len(nodes)

        g_arr = np.zeros((N_u, N + 1), dtype=complex)
        Kg_arr = np.zeros((N_u, N + 1), dtype=complex)
        beta_arr = np.zeros((N_u, N + 1), dtype=complex)
        Phi0_arr = np.zeros((N_u, N + 1), dtype=complex)

        for l_idx, u in enumerate(nodes):
            _, g, Kg, beta, Phi0 = riccati_cache[u]
            g_arr[l_idx] = g
            Kg_arr[l_idx] = Kg
            beta_arr[l_idx] = beta
            Phi0_arr[l_idx] = Phi0

        riccati_data[j] = {
            "u_nodes": nodes,
            "g_arr": g_arr,
            "Khat_g_vals": Kg_arr,
            "beta_vals": beta_arr,
            "Phi0_vals": Phi0_arr,
            "Khat1_at_Ttk": Khat1_at_Ttk,
        }
        contour_data[j] = {
            "dy": dy,
            "K": float(params.all_strikes[j]),
            "is_call": is_call,
        }

    return riccati_data, contour_data, call_nodes, put_nodes


def run_one_H(H: float, seed: int = 42, mode: str = "unconstrained",
              m_max: int = M_MAX_BRUTE, verbose: bool = True,
              **param_overrides) -> dict:
    """
    Run the complete pipeline for one Hurst exponent H.

    Steps
    -----
    1. Build params and multifactor kernel coefficients.
    2. Compute Khat, Khat*1, xi0 on the time grid.
    3. Solve Riccati ODEs for all contour nodes.
    4. Simulate M rough Heston paths.
    5. Estimate A, B, C by Monte Carlo.
    6. Brute-force subset search for m = 0, ..., m_max.

    Returns
    -------
    dict with keys:
        params, c, gamma, t_grid, Khat_vals, Khat1_vals, xi0_vals,
        riccati_data, contour_data, sim_data,
        A, B, C, search_results
    """
    params = make_params(H, **param_overrides)
    t_grid = params.t_grid
    N = params.N
    m_max = min(m_max, params.d)

    if verbose:
        print(f"  [H={H}] Params: M={params.M}, N={N}, n={params.n_factors}, "
              f"N_C={params.N_C}, d={params.d}")

    # -- Kernel
    c, gamma = compute_multifactor_coefficients(params.H, params.n_factors)
    Khat_vals = compute_Khat_numerical(t_grid, c, gamma, params.sigma, params.lambda_)
    Khat1_vals = compute_Khat_conv_1(t_grid, Khat_vals)
    xi0_vals = compute_xi0_numerical(
        t_grid, params.V0, params.theta, params.lambda_, c, gamma
    )

    # -- Riccati + contour data
    riccati_data, contour_data, call_nodes, put_nodes = _build_riccati_contour(
        params, c, gamma, Khat_vals, Khat1_vals, xi0_vals
    )
    if verbose:
        n_unique = len(call_nodes) + len(put_nodes)
        print(f"  [H={H}] Riccati ODEs solved for {n_unique} unique contour nodes.")

    # -- Simulation
    sim_data = simulate_rough_heston(params, c, gamma, seed=seed)
    if verbose:
        print(f"  [H={H}] Simulated {params.M} paths × {N} steps.")

    # -- MC estimation of A, B, C
    A = compute_A(t_grid, Khat1_vals, xi0_vals, params.rho)
    B, C = estimate_B_C_montecarlo(
        sim_data, riccati_data, contour_data, params, t_grid
    )
    if verbose:
        print(f"  [H={H}] A={A:.4e}, B range=[{B.min():.3e}, {B.max():.3e}], "
              f"|Im(C)|_max={np.abs(np.imag(C)).max():.2e}")

    # -- Compute full-portfolio result for eta denominator
    from src.optimization import solve_unconstrained_subset, solve_longonly_subset

    if mode == "unconstrained":
        nu_full, Bnu_full = solve_unconstrained_subset(B, C)
    else:
        nu_full, Bnu_full = solve_longonly_subset(B, C)
    eps2_full = float(A - 2.0 * Bnu_full + nu_full @ C @ nu_full)
    denom = A - eps2_full if abs(A - eps2_full) > 1e-15 else 1.0

    # -- Brute-force subset search for m = 0, ..., m_max
    search_results: dict = {}
    search_results[0] = {
        "subset": [],
        "nu": np.array([]),
        "epsilon2": A,
        "Delta": 0.0,
        "eta": 0.0,
    }
    search_results[params.d] = {
        "subset": list(range(params.d)),
        "nu": nu_full,
        "epsilon2": eps2_full,
        "Delta": (1.0 - eps2_full / A) * 100.0 if A > 0 else 0.0,
        "eta": 100.0,
    }

    for m in range(1, m_max + 1):
        best_subset, best_nu, eps2_m = brute_force_subset_search(
            A, B, C, m=m, mode=mode
        )
        Delta_m = (1.0 - eps2_m / A) * 100.0 if A > 0 else 0.0
        eta_m = (A - eps2_m) / denom * 100.0
        search_results[m] = {
            "subset": best_subset,
            "nu": best_nu,
            "epsilon2": eps2_m,
            "Delta": Delta_m,
            "eta": eta_m,
        }
        if verbose:
            print(f"  [H={H}] m={m}: eps2={eps2_m:.4e}, Delta={Delta_m:.2f}%, eta={eta_m:.2f}%")

    return {
        "H": H,
        "params": params,
        "c": c,
        "gamma": gamma,
        "t_grid": t_grid,
        "Khat_vals": Khat_vals,
        "Khat1_vals": Khat1_vals,
        "xi0_vals": xi0_vals,
        "riccati_data": riccati_data,
        "contour_data": contour_data,
        "call_nodes": call_nodes,
        "put_nodes": put_nodes,
        "sim_data": sim_data,
        "A": A,
        "B": B,
        "C": C,
        "search_results": search_results,
        "mode": mode,
        "seed": seed,
        "m_max": m_max,
    }


def ensure_results_dir() -> str:
    """Create and return the path to the results/ directory."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(path, exist_ok=True)
    return path
