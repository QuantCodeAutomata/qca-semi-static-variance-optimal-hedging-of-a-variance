#!/usr/bin/env python
"""
Experiment 1: Unconstrained sparse option selection.

For each H in [0.1, 0.3, 0.5] we:
  1. Build params with the reduced (but documented) numerical settings.
  2. Compute multifactor kernel coefficients (c, gamma).
  3. Compute Khat, Khat*1, xi0 on the time grid.
  4. Solve Riccati-Volterra ODEs for every contour node.
  5. Simulate M rough Heston paths (seed=42 for reproducibility).
  6. Estimate A, B, C by Monte Carlo.
  7. Run unconstrained brute-force subset search for m = 0..m_max.
  8. Save results and plots.

Reduced settings (set in exp/_pipeline.py):
  M=2000, N=100, n_factors=100, N_C=N_P=10, L_C=L_P=30, m_max=6.

Paper settings (for publication-quality results):
  M=10_000, N=250, n_factors=500, N_C=N_P=100, L_C=L_P=150, m_max=10.
  Override via: run_one_H(H, M=10000, N=250, n_factors=500, N_C=100, L_C=150,
                          N_P=100, L_P=150, m_max=10)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np

from exp._pipeline import run_one_H, ensure_results_dir, M_MAX_BRUTE
from src.plotting import plot_epsilon2_vs_m, plot_Delta_vs_m, plot_eta_vs_m


def main():
    H_values = [0.1, 0.3, 0.5]
    results_dir = ensure_results_dir()
    all_results: dict = {}

    print("=" * 60)
    print("Experiment 1: Unconstrained sparse option selection")
    print("=" * 60)
    print(f"H values: {H_values}")
    print(f"Brute-force search up to m = {M_MAX_BRUTE}")
    print()

    for H in H_values:
        print(f"\n--- H = {H} ---")
        try:
            res = run_one_H(H, seed=42, mode="unconstrained",
                            m_max=M_MAX_BRUTE, verbose=True)
            all_results[H] = res
        except Exception as exc:
            print(f"  ERROR for H={H}: {exc}")
            import traceback
            traceback.print_exc()
            continue

    # ------------------------------------------------------------------ summary
    print("\n")
    print("=" * 60)
    print("Summary table: epsilon^2(m) and Delta(m) [%]")
    print("=" * 60)
    header = f"{'m':>4}" + "".join(
        f"  {'eps2(H='+str(H)+')':>14}  {'Delta(H='+str(H)+')':>14}"
        for H in H_values
    )
    print(header)
    print("-" * len(header))

    m_vals = list(range(0, M_MAX_BRUTE + 1))
    for m in m_vals:
        row = f"{m:>4}"
        for H in H_values:
            if H not in all_results:
                row += f"  {'N/A':>14}  {'N/A':>14}"
                continue
            sr = all_results[H]["search_results"]
            if m in sr and sr[m] is not None:
                eps2 = sr[m]["epsilon2"]
                delta = sr[m]["Delta"]
                row += f"  {eps2:>14.4e}  {delta:>13.2f}%"
            else:
                row += f"  {'N/A':>14}  {'N/A':>14}"
        print(row)

    # ---------------------------------------------------------------- save data
    pkl_path = os.path.join(results_dir, "exp1_results.pkl")
    with open(pkl_path, "wb") as f:
        # Strip non-picklable sim arrays before saving to reduce file size
        save_dict = {}
        for H, res in all_results.items():
            save_dict[H] = {
                k: v for k, v in res.items()
                if k not in ("sim_data",)
            }
        pickle.dump(save_dict, f)
    print(f"\nSaved results to {pkl_path}")

    # ---------------------------------------------------------------- CSV
    csv_rows = ["H,m,epsilon2,Delta,eta,best_subset"]
    for H in H_values:
        if H not in all_results:
            continue
        for m, info in all_results[H]["search_results"].items():
            if info is None:
                continue
            subset_str = "|".join(map(str, info["subset"]))
            csv_rows.append(
                f"{H},{m},{info['epsilon2']:.6e},"
                f"{info['Delta']:.4f},{info['eta']:.4f},{subset_str}"
            )
    csv_path = os.path.join(results_dir, "exp1_summary.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_rows) + "\n")
    print(f"Saved CSV to {csv_path}")

    # ------------------------------------------------------------------ plots
    # Reformat to {H: {m: info_dict}} as expected by the plotting functions
    plot_data = {H: all_results[H]["search_results"] for H in all_results}
    try:
        plot_epsilon2_vs_m(
            plot_data,
            save_path=os.path.join(results_dir, "exp1_epsilon2_vs_m.png"),
        )
        plot_Delta_vs_m(
            plot_data,
            save_path=os.path.join(results_dir, "exp1_Delta_vs_m.png"),
        )
        plot_eta_vs_m(
            plot_data,
            save_path=os.path.join(results_dir, "exp1_eta_vs_m.png"),
        )
        print(f"Saved plots to {results_dir}/exp1_*.png")
    except Exception as exc:
        print(f"  Plot generation failed (non-fatal): {exc}")

    print("\nExperiment 1 complete.")
    return all_results


if __name__ == "__main__":
    main()
