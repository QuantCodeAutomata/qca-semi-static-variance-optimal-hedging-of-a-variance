# Rough Heston Semi-Static Variance-Optimal Hedging — Project Notes

## Overview
Numerical implementation of the core experiment from a paper on semi-static
hedging of variance swaps under rough Heston.  All source files live in
`src/`.

## Module Map
| File | Purpose |
|------|---------|
| `src/params.py` | `RoughHestonParams` dataclass + `get_default_params(H)` |
| `src/kernel.py` | Fractional kernel, multifactor coefficients, Khat, xi0 via Volterra |
| `src/riccati.py` | Riccati-Volterra ODE system for g_u (characteristic function exponent) |
| `src/simulation.py` | Multifactor Euler MC for rough Heston paths |
| `src/transform.py` | Conditional transform Htilde, backward Psi ODE, Phi update |
| `src/abc_matrix.py` | MC estimation of quadratic hedging matrices A, B, C |
| `src/optimization.py` | Unconstrained/long-only QP, brute-force subset search |
| `src/metrics.py` | Effectiveness metrics, ABC validation, dynamic hedge paths |
| `src/plotting.py` | All matplotlib visualisation (non-interactive Agg backend) |

## Key Assumptions (not stated in paper)
- **T = 1.0**: Maturity chosen as standard annual horizon.
- **Geometric partition**: `eta_i = eps*(Gamma_max/eps)^(i/n)` with `eps=1e-4`, `Gamma_max=50.0`.
- **Paper vs test sizes**: Paper uses M=10^4, N=250, n_factors=500; defaults reduced to
  M=2000, N=100, n_factors=100 for feasibility.  Contour points similarly halved.

## Model Equations
- `dX_t = -½V_t dt + √V_t dW_t`
- `V_t^n = V0_term(t) + Σ_i c_i V_t^{n,i}`
- `dV_t^{n,i} = (-γ_i V_t^{n,i} - λ V_t^n) dt + σ √V_t^n dB_t`
- `d<W,B>_t = ρ dt`
- `K(t) = t^{H-½} / Γ(H+½)`

## Optimization
- `ε²(ν) = A - 2ν^T B_S + ν^T C_S ν`
- Brute-force limited to `m ≤ 12` when `d ≥ 15` (C(21,12)≈293k).
- cvxpy (CLARABEL solver) used for long-only QP; scipy L-BFGS-B as fallback.

## Dependencies
All present in the container: `numpy`, `scipy`, `matplotlib`, `cvxpy`.

## Experiment Scripts (`exp/`)
| File | Purpose |
|------|---------|
| `exp/_pipeline.py` | Shared `run_one_H()` helper; constants `M_MAX_BRUTE=6`, `M_MAX_BRUTE_LO=4` |
| `exp/exp1_unconstrained.py` | Unconstrained sparse search, all H, generates `exp1_*.{pkl,csv,png}` |
| `exp/exp2_long_only.py` | Long-only sparse search + comparison; `exp2_*` outputs |
| `exp/exp3_dynamic_hedge.py` | Dynamic hedge paths + relative differences; `exp3_*` outputs |
| `exp/exp4_validation.py` | Structural checks + M/n sensitivity; `exp4_*` outputs + `RESULTS.md` |
| `exp/run_all.py` | Sequential subprocess runner for all four experiments |

## Test Suite (`tests/`)  — 37 tests, 0.68s
| File | Tests |
|------|-------|
| `test_kernel.py` | 8: multifactor coefficients, exact K(1.0), kernel approx 5%, Khat shape, Khat*1 monotone, xi0(0)=V0, xi0 converges |
| `test_riccati.py` | 8: F(0,0)=0, F(1,0)=0, Q=F+λx, trivial Riccati u=0, shape, Khat*g shape, F(1,x) formula, batch consistency |
| `test_simulation.py` | 8: shape, empirical ρ, X0/V0 init, V non-neg, price positive, H=0.5 vs H=0.1 |
| `test_optimization.py` | 7: C=I→ν*=B, residual formula, long-only ≥0, lo≥unc, m=0→A, m=1 valid, monotone |
| `test_abc_matrix.py` | 6: call/put contour nodes, fourier_weight finite+symmetric, compute_A positive+formula |

## Performance Notes
- **Long-only CVXPY overhead**: ~3.5 ms/subset → C(21,4)=5985 subsets ≈ 21s per H.
  Cap `M_MAX_BRUTE_LO=4`; unconstrained uses linear solve (~0.1 ms/subset).
- **Plotting signatures**: `plot_epsilon2_vs_m(results_by_H, save_path)` where
  `results_by_H = {H: {m: info_dict}}` (NOT the raw `all_results[H]` dict).
- **`plot_relative_difference`**: takes `t_grid, rel_diff_01, rel_diff_03` as three
  separate positional args (not a dict).

## Running smoke tests
```bash
cd /workspace/project
python -m pytest tests/ -v                  # 37 tests, ~0.7s
python exp/exp1_unconstrained.py            # ~80s; saves to results/
python exp/run_all.py                       # all four experiments sequentially
```
