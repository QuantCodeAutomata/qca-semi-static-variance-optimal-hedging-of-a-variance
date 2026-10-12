"""
Model parameters and numerical settings for the rough Heston semi-static hedging project.

Assumptions:
- Maturity T = 1.0  (not stated in the paper; chosen as standard annual maturity)
- Multifactor partition uses a geometric grid from eps=1e-4 to Gamma_max=50.0 with n intervals
- Paper values for M, N, n_factors are reduced here for testing
"""

from __future__ import annotations

import dataclasses
from typing import List

import numpy as np


@dataclasses.dataclass
class RoughHestonParams:
    """
    Full parameter set for the rough Heston model and numerical experiment.

    Model SDE (multifactor approximation):
        dX_t = -0.5 * V_t dt + sqrt(V_t) dW_t
        V_t  = V0 + lambda_*theta * int_0^t K(t-s) ds
               - lambda_ * int_0^t K(t-s) V_s ds
               + sigma * int_0^t K(t-s) sqrt(V_s) dB_s
        d<W,B>_t = rho dt
        K(t) = t^{H-1/2} / Gamma(H+1/2)

    Attributes
    ----------
    H : float
        Hurst exponent (0 < H < 1/2 for rough; H = 1/2 reduces to classical Heston).
    lambda_ : float
        Mean-reversion speed.
    rho : float
        Correlation between price and variance Brownian motions.
    sigma : float
        Vol-of-vol (volatility of variance).
    V0 : float
        Initial variance.
    X0 : float
        Initial log-price.
    theta : float
        Long-run variance.
    T : float
        Maturity in years.  ASSUMPTION: T = 1.0 (not stated in paper).
    M : int
        Number of Monte Carlo paths.  Paper uses 10^4; reduced to 2000 for testing.
    N : int
        Number of time steps.  Paper uses 250; reduced to 100 for testing.
    n_factors : int
        Number of exponential terms in the multifactor kernel approximation.
        Paper uses 500; reduced to 100 for testing.
    eps : float
        Left endpoint of the geometric partition for multifactor coefficients.
    Gamma_max : float
        Right endpoint of the geometric partition for multifactor coefficients.
    N_C : int
        Number of quadrature nodes on the call contour.  Paper uses 100.
    L_C : float
        Half-length of the call contour in the imaginary direction.  Paper uses 150.
    R_C : float
        Real part of the call contour.
    N_P : int
        Number of quadrature nodes on the put contour.  Paper uses 100.
    L_P : float
        Half-length of the put contour in the imaginary direction.  Paper uses 150.
    R_P : float
        Real part of the put contour.
    put_strikes : List[int]
        Put strikes (in-the-money puts: K = 50, 55, ..., 95).
    call_strikes : List[int]
        Call strikes (at-/out-of-the-money calls: K = 100, 105, ..., 150).
    """

    # ------------------------------------------------------------------ model
    H: float = 0.1
    lambda_: float = 0.3
    rho: float = -0.7
    sigma: float = 0.3
    V0: float = 0.02
    X0: float = np.log(100.0)
    theta: float = 0.066

    # ------------------------------------------------------------ maturity
    # ASSUMPTION: T = 1.0 (not stated in the paper; standard annual maturity)
    T: float = 1.0

    # --------------------------------------------------- numerical settings
    # Paper uses M=10^4, N=250, n_factors=500; reduced below for testing
    M: int = 2000
    N: int = 100
    n_factors: int = 100

    # ----------------------------------------- multifactor kernel partition
    eps: float = 1e-4
    Gamma_max: float = 50.0

    # ------------------------------------------------------ contour settings
    N_C: int = 50
    L_C: float = 50.0
    R_C: float = 2.0
    N_P: int = 50
    L_P: float = 50.0
    R_P: float = -0.125

    # -------------------------------------------------- candidate option grid
    # Puts: K = 50, 55, ..., 95  (10 puts)
    # Calls: K = 100, 105, ..., 150  (11 calls)
    # Total d = 21 instruments
    put_strikes: List[int] = dataclasses.field(
        default_factory=lambda: list(range(50, 100, 5))
    )
    call_strikes: List[int] = dataclasses.field(
        default_factory=lambda: list(range(100, 155, 5))
    )

    # ---------------------------------------------------------------- derived
    @property
    def dt(self) -> float:
        """Euler time step."""
        return self.T / self.N

    @property
    def t_grid(self) -> np.ndarray:
        """Uniform time grid from 0 to T with N+1 points."""
        return np.linspace(0.0, self.T, self.N + 1)

    @property
    def all_strikes(self) -> List[int]:
        """Concatenated list of put strikes then call strikes."""
        return self.put_strikes + self.call_strikes

    @property
    def d(self) -> int:
        """Total number of candidate options."""
        return len(self.put_strikes) + len(self.call_strikes)


def get_default_params(H: float = 0.1) -> RoughHestonParams:
    """
    Return a :class:`RoughHestonParams` instance with default values from the paper.

    Parameters
    ----------
    H : float
        Hurst exponent.  Defaults to 0.1 (rough regime).

    Returns
    -------
    RoughHestonParams
        Parameter dataclass ready for use.

    Notes
    -----
    Monte Carlo sample size (M=2000), time steps (N=100), and multifactor
    terms (n_factors=100) are reduced from the paper's values (10^4, 250, 500)
    to enable fast testing.  Increase these for publication-quality results.
    """
    return RoughHestonParams(H=H)
