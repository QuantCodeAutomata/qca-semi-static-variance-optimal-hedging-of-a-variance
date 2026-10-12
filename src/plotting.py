"""
Visualization utilities for the rough Heston semi-static hedging experiment.

All plots use matplotlib only.  Figures are saved to the ``results/`` directory
when a ``save_path`` is provided.

Functions
---------
plot_epsilon2_vs_m          Line plot of epsilon^2(m) for each Hurst exponent H.
plot_Delta_vs_m             Delta(m) effectiveness curves.
plot_eta_vs_m               eta(m) normalised effectiveness curves.
plot_weight_vs_strike       Bar chart of optimal weights nu_j vs strike.
plot_dynamic_hedge_paths    Sample paths of the optimal dynamic hedge ratio.
plot_relative_difference    Relative difference R_t^H between H values.
plot_sensitivity_table      Heatmap of sensitivity analysis results.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend suitable for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STYLE = {
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
}


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save figure to ``save_path`` (creating parent directories as needed)."""
    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _make_fig(**kwargs) -> tuple:
    """Create a figure with the project style applied."""
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(**kwargs)
    return fig, ax


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_epsilon2_vs_m(
    results_by_H: Dict[float, Dict[int, Optional[Dict]]],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot epsilon^2(m) against the number of options m for each Hurst exponent H.

    Parameters
    ----------
    results_by_H : dict
        {H: {m: {'epsilon2': float, ...}, ...}}.
        None entries (skipped brute-force searches) are omitted.
    save_path : str, optional
        File path to save the figure.  If None the figure is closed without saving.
    """
    fig, ax = _make_fig(figsize=(7, 5))
    for H, res in sorted(results_by_H.items()):
        ms = sorted(k for k, v in res.items() if v is not None)
        eps2 = [res[m]["epsilon2"] for m in ms]
        ax.plot(ms, eps2, marker="o", markersize=4, label=f"H = {H}")

    ax.set_xlabel("Number of options m")
    ax.set_ylabel(r"$\varepsilon^2(m)$")
    ax.set_title(r"Hedging error $\varepsilon^2$ vs number of static options")
    ax.legend(title="Hurst exponent")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_Delta_vs_m(
    results_by_H: Dict[float, Dict[int, Optional[Dict]]],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot Delta(m) = (1 - epsilon^2(m)/A)*100 against m for each H.

    Delta(m) is the percentage of variance eliminated relative to the
    purely dynamic hedge.

    Parameters
    ----------
    results_by_H : dict
        {H: {m: {'Delta': float, ...}, ...}}.
    save_path : str, optional
        Save path.
    """
    fig, ax = _make_fig(figsize=(7, 5))
    for H, res in sorted(results_by_H.items()):
        ms = sorted(k for k, v in res.items() if v is not None and v.get("Delta") is not None)
        deltas = [res[m]["Delta"] for m in ms]
        ax.plot(ms, deltas, marker="s", markersize=4, label=f"H = {H}")

    ax.set_xlabel("Number of options m")
    ax.set_ylabel(r"$\Delta(m)$ [%]")
    ax.set_title(r"Effectiveness $\Delta(m)$: variance reduction relative to dynamic-only")
    ax.legend(title="Hurst exponent")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_eta_vs_m(
    results_by_H: Dict[float, Dict[int, Optional[Dict]]],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot eta(m) = (A - epsilon^2(m))/(A - epsilon^2(d))*100 against m.

    eta(m) measures the fraction of the maximum attainable variance reduction
    achieved by using m options.

    Parameters
    ----------
    results_by_H : dict
        {H: {m: {'eta': float, ...}, ...}}.
    save_path : str, optional
        Save path.
    """
    fig, ax = _make_fig(figsize=(7, 5))
    for H, res in sorted(results_by_H.items()):
        ms = sorted(k for k, v in res.items() if v is not None and v.get("eta") is not None)
        etas = [res[m]["eta"] for m in ms]
        ax.plot(ms, etas, marker="^", markersize=4, label=f"H = {H}")

    ax.set_xlabel("Number of options m")
    ax.set_ylabel(r"$\eta(m)$ [%]")
    ax.set_title(r"Normalised effectiveness $\eta(m)$")
    ax.legend(title="Hurst exponent")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_weight_vs_strike(
    nu_star: np.ndarray,
    strikes: List[float],
    m: int,
    H: float,
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart (or log-log scatter) of the optimal static weights nu_j vs strike K_j.

    Parameters
    ----------
    nu_star : np.ndarray, shape (d,) or (m,)
        Optimal weights (zero for non-selected options in a subset).
    strikes : list of float
        Strike prices in the same order as nu_star.
    m : int
        Number of non-zero options in the portfolio (for the title).
    H : float
        Hurst exponent (for the title).
    save_path : str, optional
        Save path.
    """
    fig, ax = _make_fig(figsize=(8, 4))
    colors = ["steelblue" if nu > 0 else "salmon" for nu in nu_star]
    ax.bar(range(len(strikes)), nu_star, color=colors, edgecolor="white")
    ax.set_xticks(range(len(strikes)))
    ax.set_xticklabels([str(int(k)) for k in strikes], rotation=45, ha="right")
    ax.set_xlabel("Strike K")
    ax.set_ylabel(r"Optimal weight $\nu_j^*$")
    ax.set_title(f"Optimal static weights (H = {H}, m = {m})")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_dynamic_hedge_paths(
    t_grid: np.ndarray,
    hedge_paths_by_H: Dict[float, np.ndarray],
    n_paths: int = 5,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot sample paths of the optimal dynamic hedge ratio vartheta_t^*.

    Parameters
    ----------
    t_grid : np.ndarray
        Time grid [0, T], shape (N+1,).
    hedge_paths_by_H : dict
        {H: np.ndarray of shape (M, N+1)} – hedge paths for each H.
    n_paths : int
        Number of sample paths to display per H.
    save_path : str, optional
        Save path.
    """
    n_H = len(hedge_paths_by_H)
    fig, axes = plt.subplots(1, n_H, figsize=(5 * n_H, 4), sharey=False)
    if n_H == 1:
        axes = [axes]

    for ax, (H, paths) in zip(axes, sorted(hedge_paths_by_H.items())):
        for p in range(min(n_paths, paths.shape[0])):
            ax.plot(t_grid, paths[p, :], linewidth=0.8, alpha=0.8)
        ax.set_xlabel("t")
        ax.set_ylabel(r"$\vartheta_t^*$")
        ax.set_title(f"H = {H}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Optimal dynamic hedge ratio paths", y=1.02)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_relative_difference(
    t_grid: np.ndarray,
    rel_diff_01: np.ndarray,
    rel_diff_03: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the relative difference R_t^H between the rough and classical Heston
    dynamic hedge ratios for two Hurst values.

    R_t^H = |vartheta_t^{rough} - vartheta_t^{classical}| / |vartheta_t^{classical}|

    Parameters
    ----------
    t_grid : np.ndarray
        Time grid [0, T], shape (N+1,).
    rel_diff_01 : np.ndarray
        Pathwise or mean relative difference for H = 0.1, shape (N+1,).
    rel_diff_03 : np.ndarray
        Pathwise or mean relative difference for H = 0.3, shape (N+1,).
    save_path : str, optional
        Save path.
    """
    fig, ax = _make_fig(figsize=(7, 4))
    ax.plot(t_grid, rel_diff_01, label="H = 0.1", linewidth=1.5)
    ax.plot(t_grid, rel_diff_03, label="H = 0.3", linewidth=1.5, linestyle="--")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$R_t^H$")
    ax.set_title("Relative difference in dynamic hedge ratio vs classical Heston")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_sensitivity_table(
    sensitivity_df,
    save_path: Optional[str] = None,
) -> None:
    """
    Heatmap of sensitivity analysis results.

    Parameters
    ----------
    sensitivity_df : pandas.DataFrame or np.ndarray
        Table of sensitivity values.  Rows and columns should be labelled
        (use a DataFrame for automatic axis labels).
    save_path : str, optional
        Save path.
    """
    try:
        import pandas as pd
        is_df = isinstance(sensitivity_df, pd.DataFrame)
    except ImportError:
        is_df = False

    data = sensitivity_df.values if is_df else np.asarray(sensitivity_df)
    row_labels = list(sensitivity_df.index) if is_df else [str(i) for i in range(data.shape[0])]
    col_labels = list(sensitivity_df.columns) if is_df else [str(j) for j in range(data.shape[1])]

    fig, ax = _make_fig(figsize=(max(6, len(col_labels) * 0.9), max(4, len(row_labels) * 0.6)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Sensitivity analysis")

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    _save_or_show(fig, save_path)
