"""
Subset selection and quadratic optimization for semi-static hedging.

The hedging problem reduces to minimising the quadratic hedging error:
    epsilon^2(nu) = A - 2 * nu^T B_S + nu^T C_S nu

over a subset S of candidate options and a weight vector nu in R^|S|.

Two modes are supported:
    - 'unconstrained': nu* = C_S^{-1} B_S   (unrestricted positions)
    - 'longonly'     : nu* = argmin_{nu >= 0} nu^T C_S nu - 2 B_S^T nu

The brute-force search enumerates all C(d, m) subsets of size m and selects
the one with the smallest residual hedging error epsilon^2.

For large d the search is feasible only for small m; the limit ``m <= 12``
is applied when d >= 15 unless the user overrides it (see :func:`run_sparse_search`).
"""

from __future__ import annotations

import itertools
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.linalg as la

try:
    import cvxpy as cp
    _CVXPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CVXPY_AVAILABLE = False

from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Unconstrained QP
# ---------------------------------------------------------------------------

def solve_unconstrained_subset(
    B_S: np.ndarray,
    C_S: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Solve the unconstrained variance-optimal QP for a given option subset S.

    The optimal weights are
        nu* = C_S^{-1} B_S

    and the minimum hedging error is
        epsilon^2 = A - B_S^T nu*

    (the caller subtracts B_S^T nu* from A to obtain epsilon^2).

    Parameters
    ----------
    B_S : np.ndarray, shape (m,)
        Sub-vector of B for the chosen subset.
    C_S : np.ndarray, shape (m, m)
        Sub-matrix of C for the chosen subset.

    Returns
    -------
    nu_star : np.ndarray, shape (m,)
        Optimal unconstrained weights.
    B_dot_nu : float
        B_S @ nu_star  (equals A - epsilon^2 when A is known).

    Notes
    -----
    If the condition number of C_S exceeds 1e10 the Moore-Penrose pseudo-inverse
    is used instead of a direct solve to handle near-singular matrices.
    """
    cond = np.linalg.cond(C_S)
    if cond > 1e10:
        nu_star = la.pinv(C_S) @ B_S
    else:
        try:
            nu_star = la.solve(C_S, B_S)
        except la.LinAlgError:
            nu_star = la.pinv(C_S) @ B_S

    B_dot_nu = float(B_S @ nu_star)
    return nu_star, B_dot_nu


# ---------------------------------------------------------------------------
# Long-only QP
# ---------------------------------------------------------------------------

def solve_longonly_subset(
    B_S: np.ndarray,
    C_S: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Solve the long-only variance-optimal QP for a given option subset S.

    Minimise  nu^T C_S nu - 2 B_S^T nu  subject to  nu >= 0.

    Solved via cvxpy (preferred) with a scipy.optimize fallback.

    Parameters
    ----------
    B_S : np.ndarray, shape (m,)
        Sub-vector of B.
    C_S : np.ndarray, shape (m, m)
        Sub-matrix of C.

    Returns
    -------
    nu_star : np.ndarray, shape (m,)
        Optimal long-only weights.
    B_dot_nu : float
        B_S @ nu_star.
    """
    m = len(B_S)
    # Symmetrise and add a tiny ridge for numerical PSD
    C_sym = 0.5 * (C_S + C_S.T) + 1e-10 * np.eye(m)

    if _CVXPY_AVAILABLE:
        try:
            x = cp.Variable(m)
            objective = cp.Minimize(cp.quad_form(x, C_sym) - 2.0 * B_S @ x)
            constraints = [x >= 0]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.CLARABEL, verbose=False)

            if prob.status in ("optimal", "optimal_inaccurate") and x.value is not None:
                nu_star = np.maximum(x.value, 0.0)
                return nu_star, float(B_S @ nu_star)
            # Fall through to scipy if cvxpy fails
        except Exception:
            pass

    # Scipy fallback: L-BFGS-B with bounds
    def objective_scipy(nu: np.ndarray) -> float:
        return float(nu @ C_sym @ nu - 2.0 * B_S @ nu)

    def gradient_scipy(nu: np.ndarray) -> np.ndarray:
        return 2.0 * C_sym @ nu - 2.0 * B_S

    bounds = [(0.0, None)] * m
    x0 = np.maximum(la.pinv(C_sym) @ B_S, 0.0)
    result = minimize(
        objective_scipy,
        x0,
        jac=gradient_scipy,
        method="L-BFGS-B",
        bounds=bounds,
    )
    nu_star = np.maximum(result.x, 0.0)
    return nu_star, float(B_S @ nu_star)


# ---------------------------------------------------------------------------
# Brute-force subset enumeration
# ---------------------------------------------------------------------------

def brute_force_subset_search(
    A: float,
    B: np.ndarray,
    C: np.ndarray,
    m: int,
    mode: str = "unconstrained",
    n_jobs: int = 1,
) -> Tuple[List[int], np.ndarray, float]:
    """
    Enumerate all C(d, m) subsets and return the one minimising epsilon^2.

    The hedging error for a subset S and weights nu is:
        epsilon^2(S, nu) = A - 2 * nu^T B_S + nu^T C_S nu.

    At the optimum nu*:
        epsilon^2(S, nu*) = A - B_S^T nu*   [unconstrained]

    Parameters
    ----------
    A : float
        Scalar hedging error without any option hedge.
    B : np.ndarray, shape (d,)
        Full cross-covariance vector.
    C : np.ndarray, shape (d, d)
        Full option covariance matrix.
    m : int
        Subset size.
    mode : {'unconstrained', 'longonly'}
        Optimisation mode.
    n_jobs : int
        Reserved for future parallelism; currently ignored.

    Returns
    -------
    best_subset : list of int
        Indices (into the full d-vector) of the best subset.
    best_nu : np.ndarray, shape (m,)
        Optimal weights for the best subset.
    best_residual : float
        Minimum epsilon^2 achieved.
    """
    d = len(B)
    if m == 0:
        return [], np.array([]), A

    _solver = (
        solve_unconstrained_subset if mode == "unconstrained"
        else solve_longonly_subset
    )

    best_subset: List[int] = list(range(m))
    best_nu: np.ndarray = np.zeros(m)
    best_residual: float = np.inf

    for subset in itertools.combinations(range(d), m):
        idx = list(subset)
        B_S = B[idx]
        C_S = C[np.ix_(idx, idx)]

        nu, B_dot_nu = _solver(B_S, C_S)
        eps2 = A - 2.0 * B_dot_nu + float(nu @ C_S @ nu)

        if eps2 < best_residual:
            best_residual = eps2
            best_subset = idx
            best_nu = nu

    return best_subset, best_nu, best_residual


# ---------------------------------------------------------------------------
# Sparse search across all subset sizes
# ---------------------------------------------------------------------------

def run_sparse_search(
    A: float,
    B: np.ndarray,
    C: np.ndarray,
    d: int = 21,
    mode: str = "unconstrained",
) -> Dict[int, Dict]:
    """
    Run the brute-force subset search for m = 0, 1, ..., d and collect results.

    For each subset size m, :func:`brute_force_subset_search` is called.
    The effectiveness metrics are:
        epsilon^2(m)  = minimum hedging error using m options
        Delta(m)      = (1 - epsilon^2(m) / A) * 100  (% variance reduced vs unhedged)
        eta(m)        = (A - epsilon^2(m)) / (A - epsilon^2(d)) * 100
                        (% of maximum attainable variance reduction)

    .. warning::
        For d >= 15 the search is limited to m <= 12 by default to keep
        the enumeration tractable (C(21,12) ≈ 300K).  For m > 12 with d = 21
        the brute-force search is skipped and results are marked as None.
        Override by calling :func:`brute_force_subset_search` directly.

    Parameters
    ----------
    A : float
        Unhedged variance (m=0 baseline).
    B : np.ndarray, shape (d,)
        Cross-covariance vector.
    C : np.ndarray, shape (d, d)
        Option covariance matrix.
    d : int
        Total number of candidate options.
    mode : {'unconstrained', 'longonly'}
        Optimisation mode.

    Returns
    -------
    dict
        {m: {'subset': list[int], 'nu': np.ndarray, 'epsilon2': float,
             'Delta': float, 'eta': float}}
        for m = 0, ..., d.  For skipped m the entry is None.
    """
    SEARCH_LIMIT = 12   # Maximum m for brute-force when d >= 15

    results: Dict[int, Optional[Dict]] = {}

    # m = 0 baseline
    results[0] = {
        "subset": [],
        "nu": np.array([]),
        "epsilon2": A,
        "Delta": 0.0,
        "eta": 0.0,
    }

    # m = d: full portfolio (closed-form)
    nu_full, B_dot_nu_full = (
        solve_unconstrained_subset(B, C)
        if mode == "unconstrained"
        else solve_longonly_subset(B, C)
    )
    eps2_full = A - 2.0 * B_dot_nu_full + float(nu_full @ C @ nu_full)
    denominator = A - eps2_full if abs(A - eps2_full) > 1e-15 else 1.0

    results[d] = {
        "subset": list(range(d)),
        "nu": nu_full,
        "epsilon2": eps2_full,
        "Delta": (1.0 - eps2_full / A) * 100.0 if A > 0 else 0.0,
        "eta": 100.0,
    }

    # m = 1, ..., d-1
    for m in range(1, d):
        if d >= 15 and m > SEARCH_LIMIT:
            warnings.warn(
                f"Skipping brute-force search for m={m} > {SEARCH_LIMIT} "
                f"(d={d} >= 15).  C({d},{m}) is too large for exhaustive enumeration.",
                RuntimeWarning,
                stacklevel=2,
            )
            results[m] = None
            continue

        best_subset, best_nu, eps2_m = brute_force_subset_search(
            A, B, C, m, mode=mode
        )
        Delta_m = (1.0 - eps2_m / A) * 100.0 if A > 0 else 0.0
        eta_m = (A - eps2_m) / denominator * 100.0

        results[m] = {
            "subset": best_subset,
            "nu": best_nu,
            "epsilon2": eps2_m,
            "Delta": Delta_m,
            "eta": eta_m,
        }

    return results
