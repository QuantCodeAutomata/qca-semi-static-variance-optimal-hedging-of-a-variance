#!/usr/bin/env python
"""
Experiment 3: Dynamic hedge sensitivity across Hurst exponents.

We use the same random increments (matched seed) for all three H values so
that the hedge paths differ only because of the model parameter H, not because
of Monte Carlo noise.  For each H we take the m=3 long-only optimal basket
from Experiment 2 (or recompute it) and compute the full path-by-path optimal
dynamic hedge ratio

    vartheta_t^* = vartheta_t^0 - sum_j nu_j^* * vartheta_t^j

where vartheta_t^0 is the unhedged Heston delta and vartheta_t^j is the
contribution from option j via the conditional characteristic function.

We then report:
  - Path-wise hedge ratio trajectories for a representative subset of paths.
  - Relative differences of vartheta^{H} vs the H=0.5 benchmark:
      delta_rel(t, m) = (vartheta_t^{H=0.1} - vartheta_t^{H=0.5}) / |vartheta_t^{H=0.5}|
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np

from exp._pipeline import run_one_H, ensure_results_dir, M_MAX_BRUTE
from src.metrics import compute_dynamic_hedge
from src.plotting import plot_dynamic_hedge_paths, plot_relative_difference


def main():
    H_values = [0.1, 0.3, 0.5]
    results_dir = ensure_results_dir()
    m_target = min(3, M_MAX_BRUTE)   # m=3 long-only basket

    print("=" * 60)
    print("Experiment 3: Dynamic hedge sensitivity (matched paths)")
    print("=" * 60)
    print(f"Using m={m_target} long-only basket, seed=42, H in {H_values}")
    print()

    hedge_paths_by_H: dict = {}
    all_results: dict = {}

    for H in H_values:
        print(f"\n--- H = {H} ---")
        try:
            # Long-only run (same seed ensures matched paths via generate_correlated_increments)
            res = run_one_H(H, seed=42, mode="longonly",
                            m_max=m_target, verbose=True)
            all_results[H] = res
            sr = res["search_results"]

            if m_target not in sr or sr[m_target] is None:
                print(f"  m={m_target} result not available; skipping hedge computation.")
                continue

            nu_star_m = sr[m_target]["nu"]     # weights for chosen m-subset
            best_subset = sr[m_target]["subset"]

            # Build full-length nu_star (zero for non-selected options)
            d = res["params"].d
            nu_full = np.zeros(d)
            nu_full[best_subset] = nu_star_m

            # Prepare Khat_g_data as expected by compute_dynamic_hedge
            Khat_g_data = {
                j: res["riccati_data"][j]["Khat_g_vals"]
                for j in range(d)
            }

            print(f"  [H={H}] Computing hedge paths for {res['params'].M} paths...")
            hedge = compute_dynamic_hedge(
                t_grid=res["t_grid"],
                sim_data=res["sim_data"],
                riccati_data=res["riccati_data"],
                nu_star=nu_full,
                params=res["params"],
                Khat1_vals=res["Khat1_vals"],
                Khat_g_data=Khat_g_data,
                contour_data=res["contour_data"],
            )
            hedge_paths_by_H[H] = hedge
            print(f"  [H={H}] Hedge path shape: {hedge.shape}")
            print(f"  [H={H}] Hedge range: [{hedge.min():.4f}, {hedge.max():.4f}]")

        except Exception as exc:
            print(f"  ERROR for H={H}: {exc}")
            import traceback
            traceback.print_exc()
            continue

    # ------------------------------------------------------- relative differences
    print("\n")
    print("=" * 60)
    print("Relative differences vs H=0.5 benchmark")
    print("=" * 60)

    rel_diff_by_H: dict = {}
    if 0.5 in hedge_paths_by_H:
        benchmark = hedge_paths_by_H[0.5]
        denom = np.abs(benchmark)
        denom_safe = np.where(denom > 1e-10, denom, 1e-10)

        for H in [0.1, 0.3]:
            if H not in hedge_paths_by_H:
                continue
            rel_diff = (hedge_paths_by_H[H] - benchmark) / denom_safe
            rel_diff_by_H[H] = rel_diff
            print(f"H={H} vs H=0.5:")
            print(f"  Mean |rel diff|:  {np.abs(rel_diff).mean():.4f}")
            print(f"  Median |rel diff|:{np.median(np.abs(rel_diff)):.4f}")
            print(f"  90th pct:         {np.percentile(np.abs(rel_diff), 90):.4f}")
            print(f"  Range:            [{rel_diff.min():.4f}, {rel_diff.max():.4f}]")
    else:
        print("H=0.5 benchmark not available.")

    # ------------------------------------------------------------------ plots
    # plot_dynamic_hedge_paths: {H: np.ndarray (M, N+1)}
    # plot_relative_difference: t_grid, rel_diff_01, rel_diff_03 (two arrays)
    try:
        if hedge_paths_by_H:
            t_grid_ref = all_results[next(iter(hedge_paths_by_H))]["t_grid"]
            n_paths_plot = min(10, next(iter(hedge_paths_by_H.values())).shape[0])
            plot_dynamic_hedge_paths(
                t_grid=t_grid_ref,
                hedge_paths_by_H=hedge_paths_by_H,
                n_paths=n_paths_plot,
                save_path=os.path.join(results_dir, "exp3_hedge_paths.png"),
            )

        if 0.1 in rel_diff_by_H and 0.3 in rel_diff_by_H:
            t_grid_ref = all_results[0.1]["t_grid"]
            # plot_relative_difference expects mean-across-paths scalars per time step
            rd01 = np.abs(rel_diff_by_H[0.1]).mean(axis=0)  # (N+1,)
            rd03 = np.abs(rel_diff_by_H[0.3]).mean(axis=0)
            plot_relative_difference(
                t_grid=t_grid_ref,
                rel_diff_01=rd01,
                rel_diff_03=rd03,
                save_path=os.path.join(results_dir, "exp3_rel_diff.png"),
            )
        print(f"Saved plots to {results_dir}/exp3_*.png")
    except Exception as exc:
        print(f"  Plot generation failed (non-fatal): {exc}")

    # ------------------------------------------------------------------ save
    pkl_path = os.path.join(results_dir, "exp3_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "hedge_paths_by_H": hedge_paths_by_H,
                "rel_diff_by_H": rel_diff_by_H,
                "m_target": m_target,
                "H_values": H_values,
                "params_by_H": {H: all_results[H]["params"]
                                for H in all_results},
            },
            f,
        )
    print(f"\nSaved results to {pkl_path}")
    print("\nExperiment 3 complete.")
    return hedge_paths_by_H, rel_diff_by_H


if __name__ == "__main__":
    main()
