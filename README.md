# Semi-Static Variance-Optimal Hedging of a Variance Swap under Rough Heston

This repository implements the numerical experiments from the paper on
**variance-optimal semi-static hedging of variance swaps** under the rough Heston
model using a sparse portfolio of European options.

## Overview

The project solves the problem of hedging a variance swap using:
1. **Dynamic stock trading** — continuously adjusting a stock position (GKW hedging)
2. **Static option basket** — a fixed portfolio of European puts and calls, selected to minimise residual hedging error

The rough Heston model uses a fractional kernel
`K(t) = t^(H-1/2) / Γ(H+1/2)` approximated by a sum of `n` exponentials
(multifactor decomposition).

Key results reproduced:
- Monotone decrease of residual error `ε²(m)` as basket size `m` grows
- `Δ(3) > 97%` hedging effectiveness with only 3 options (paper claims >95% for long-only)
- Rougher volatility (smaller `H`) yields larger dynamic hedge, smaller basket stability

## Repository Structure

```
├── data/              # (empty — fully model-based study, no market data)
├── src/               # Core implementation
│   ├── params.py      # Model parameters and numerical settings
│   ├── kernel.py      # Multifactor kernel K^n and resolvent K̂
│   ├── riccati.py     # Riccati-Volterra solver for g_u
│   ├── simulation.py  # Rough Heston Euler simulation
│   ├── transform.py   # Conditional transform H̃_t^u and forward variance Φ_t^u
│   ├── abc_matrix.py  # Monte Carlo estimation of A, B, C matrices
│   ├── optimization.py# Unconstrained and long-only subset selection
│   ├── metrics.py     # Performance metrics and diagnostics
│   └── plotting.py    # Visualisation utilities
├── exp/               # Experiment scripts
│   ├── _pipeline.py   # Shared pipeline helpers
│   ├── exp1_unconstrained.py  # Exp 1: unconstrained sparse selection
│   ├── exp2_long_only.py      # Exp 2: long-only sparse selection
│   ├── exp3_dynamic_hedge.py  # Exp 3: dynamic hedge sensitivity vs H
│   ├── exp4_validation.py     # Exp 4: numerical validation & sensitivity
│   └── run_all.py     # Run all experiments sequentially
├── tests/             # pytest tests (37 tests, all pass)
├── results/           # Output artifacts and summary
│   └── RESULTS.md     # Detailed experiment findings
└── README.md
```

## Model

### Rough Heston dynamics (risk-neutral measure)

```
dX_t = -½ V_t dt + √V_t dW_t
V_t  = V₀ + ∫₀ᵗ K(t-s) [λ(θ - V_s) ds + σ √V_s dB_s]
d⟨W,B⟩_t = ρ dt,   S_t = exp(X_t)
```

**Parameters (from paper):** λ=0.3, ρ=-0.7, σ=0.3, V₀=0.02, X₀=ln(100), θ=0.066

**Hurst exponents tested:** H ∈ {0.1, 0.3, 0.5}

For H=0.5 the fractional kernel degenerates to the classical Heston model.

### Semi-static hedging problem

The variance swap payoff is `H_T = ∫₀ᵀ V_s ds - k` with `k = E[∫₀ᵀ V_s ds]`.

The hedging error minimisation leads to the quadratic form:

```
ε²(ν) = A - 2 νᵀ B + νᵀ C ν
```

where A, B, C are computed via GKW projections and Fourier–Laplace contour integrals.

## Installation

```bash
pip install numpy scipy matplotlib cvxpy
```

Or install all project dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Run all experiments

```bash
cd /workspace/project
python exp/run_all.py
```

### Run individual experiments

```bash
python exp/exp1_unconstrained.py   # Unconstrained sparse selection
python exp/exp2_long_only.py        # Long-only sparse selection
python exp/exp3_dynamic_hedge.py    # Dynamic hedge vs roughness
python exp/exp4_validation.py       # Numerical validation
```

### Run tests

```bash
pytest tests/ -v
```

## Numerical Settings

The default settings are **reduced** from the paper for tractable runtimes.
To replicate at paper scale, edit `EXP_DEFAULTS` in `exp/_pipeline.py`:

| Parameter       | Default | Paper  | Notes                          |
|-----------------|---------|--------|-------------------------------|
| M (paths)       | 2,000   | 10,000 | Monte Carlo paths              |
| N (time steps)  | 100     | 250    | Euler discretisation           |
| n_factors       | 100     | 500    | Multifactor kernel terms       |
| N_C = N_P       | 10      | 100    | Contour quadrature points      |
| L_C = L_P       | 30      | 150    | Contour half-length            |

## Key Assumptions (underdetermined by source paper)

1. **Maturity T = 1.0** — the paper does not state T explicitly
2. **Geometric partition** for the multifactor kernel (the paper's pi_n formula is corrupted)
3. **Forward variance xi₀** estimated by numerical Volterra equation
4. **Euler scheme** (first-order) for both the SDE and the Riccati ODE

## Results Summary

| H    | ε²(0)    | ε²(1)    | ε²(3)    | Δ(3) %  |
|------|----------|----------|----------|---------|
| 0.1  | 8.25e-04 | 9.38e-05 | 2.14e-05 | 97.40%  |
| 0.3  | 4.72e-04 | 4.67e-05 | 9.56e-06 | 97.98%  |
| 0.5  | 4.52e-04 | 3.83e-05 | 7.76e-06 | 98.28%  |

The paper (long-only, full settings) reports Δ(3) ≈ 96.22 / 98.44 / 99.29% for
H = 0.1 / 0.3 / 0.5 respectively.

All qualitative findings are confirmed:
- ε²(m) decreases monotonically with m
- Very strong gains for m=1,2,3; diminishing returns thereafter  
- Rougher H gives larger residual A but same qualitative pattern
- Long-only and unconstrained solutions coincide (no beneficial shorts)

## References

The paper this repository implements (rough Heston semi-static hedging).
Rough Heston model: El Euch & Rosenbaum (2019), Abi Jaber et al. (2019).
GKW decomposition: Galtchouk-Kunita-Watanabe.
