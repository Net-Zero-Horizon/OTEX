# -*- coding: utf-8 -*-
"""
Visualization functions for OTEC uncertainty analysis.

Provides plotting functions for:
- Histograms with statistics
- Tornado diagrams
- Sobol sensitivity bar charts
- Scatter matrices
"""

from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .uncertainty import UncertaintyResults
from .sensitivity import SobolResults, TornadoResults


def plot_histogram(
    results: UncertaintyResults,
    output: str = 'lcoe',
    ax: Optional[Axes] = None,
    bins: int = 50,
    show_stats: bool = True,
    color: str = 'steelblue',
    alpha: float = 0.7
) -> Axes:
    """
    Plot histogram of output variable with statistics.

    Args:
        results: UncertaintyResults from Monte Carlo analysis
        output: Output variable to plot ('lcoe', 'net_power', 'capex', 'opex')
        ax: Matplotlib axes (creates new figure if None)
        bins: Number of histogram bins
        show_stats: Whether to show mean and percentile lines
        color: Histogram color
        alpha: Histogram transparency

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    values = getattr(results, output)
    valid = values[~np.isnan(values)]

    # Plot histogram
    ax.hist(valid, bins=bins, color=color, alpha=alpha, edgecolor='white')

    # Labels
    labels = {
        'lcoe': 'LCOE (ct/kWh)',
        'net_power': 'Net Power (kW)',
        'capex': 'CAPEX ($)',
        'opex': 'OPEX ($/year)'
    }
    ax.set_xlabel(labels.get(output, output))
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of {output.upper()}')

    if show_stats:
        mean = np.mean(valid)
        p5 = np.percentile(valid, 5)
        p95 = np.percentile(valid, 95)

        ax.axvline(mean, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean:.2f}')
        ax.axvline(p5, color='orange', linestyle='--', linewidth=1.5, label=f'P5: {p5:.2f}')
        ax.axvline(p95, color='orange', linestyle='--', linewidth=1.5, label=f'P95: {p95:.2f}')
        ax.legend()

    return ax


def plot_tornado(
    results: TornadoResults,
    ax: Optional[Axes] = None,
    top_n: int = 10,
    color_low: str = '#3498db',
    color_high: str = '#e74c3c',
    show_baseline: bool = True
) -> Axes:
    """
    Plot tornado diagram showing parameter sensitivity.

    Args:
        results: TornadoResults from tornado analysis
        ax: Matplotlib axes (creates new figure if None)
        top_n: Number of top parameters to show
        color_low: Color for low-side bars
        color_high: Color for high-side bars
        show_baseline: Whether to show baseline value line

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Get ranking and limit to top_n
    ranking = results.get_ranking()[:top_n]
    param_names = [r[0] for r in ranking]

    # Find indices of ranked parameters
    indices = [results.parameter_names.index(name) for name in param_names]

    low_vals = results.low_values[indices]
    high_vals = results.high_values[indices]
    baseline = results.baseline

    # Reverse for top-to-bottom display
    param_names = param_names[::-1]
    low_vals = low_vals[::-1]
    high_vals = high_vals[::-1]

    y_pos = np.arange(len(param_names))

    # Calculate deltas from baseline
    low_delta = low_vals - baseline
    high_delta = high_vals - baseline

    # Plot bars
    ax.barh(y_pos, low_delta, height=0.6, color=color_low, label='Low value')
    ax.barh(y_pos, high_delta, height=0.6, color=color_high, label='High value', left=0)

    if show_baseline:
        ax.axvline(0, color='black', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names)
    ax.set_xlabel(f'Change in {results.output_name.upper()} from baseline ({baseline:.2f})')
    ax.set_title('Tornado Diagram: Parameter Sensitivity')
    ax.legend(loc='best')

    # Add grid
    ax.grid(axis='x', alpha=0.3)

    return ax


def plot_sobol_indices(
    results: SobolResults,
    ax: Optional[Axes] = None,
    top_n: int = 10,
    show_S1: bool = True,
    show_ST: bool = True,
    color_S1: str = '#2ecc71',
    color_ST: str = '#9b59b6'
) -> Axes:
    """
    Plot Sobol sensitivity indices as horizontal bars.

    Args:
        results: SobolResults from Sobol analysis
        ax: Matplotlib axes (creates new figure if None)
        top_n: Number of top parameters to show
        show_S1: Whether to show first-order indices
        show_ST: Whether to show total-order indices
        color_S1: Color for S1 bars
        color_ST: Color for ST bars

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Sort by ST
    ranking = results.get_ranking('ST')[:top_n]
    param_names = [r[0] for r in ranking]
    indices = [results.parameter_names.index(name) for name in param_names]

    S1 = results.S1[indices]
    ST = results.ST[indices]

    # Reverse for top-to-bottom display
    param_names = param_names[::-1]
    S1 = S1[::-1]
    ST = ST[::-1]

    y_pos = np.arange(len(param_names))
    bar_height = 0.35

    if show_S1 and show_ST:
        ax.barh(y_pos - bar_height/2, S1, height=bar_height, color=color_S1, label='S1 (First-order)')
        ax.barh(y_pos + bar_height/2, ST, height=bar_height, color=color_ST, label='ST (Total)')
    elif show_S1:
        ax.barh(y_pos, S1, height=0.6, color=color_S1, label='S1 (First-order)')
    else:
        ax.barh(y_pos, ST, height=0.6, color=color_ST, label='ST (Total)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names)
    ax.set_xlabel('Sensitivity Index')
    ax.set_title(f'Sobol Sensitivity Indices for {results.output_name.upper()}')
    ax.legend(loc='best')
    ax.set_xlim(0, 1.05)
    ax.grid(axis='x', alpha=0.3)

    return ax


def plot_scatter_matrix(
    results: UncertaintyResults,
    output: str = 'lcoe',
    params: Optional[List[str]] = None,
    max_params: int = 5,
    figsize: Tuple[int, int] = (12, 12),
    alpha: float = 0.3,
    s: int = 10
) -> Figure:
    """
    Plot scatter matrix of parameters vs output.

    Args:
        results: UncertaintyResults from Monte Carlo analysis
        output: Output variable to plot against
        params: List of parameter names to include (uses top by correlation if None)
        max_params: Maximum number of parameters to show
        figsize: Figure size
        alpha: Point transparency
        s: Point size

    Returns:
        Matplotlib figure
    """
    from scipy.stats import spearmanr

    output_values = getattr(results, output)
    valid = ~np.isnan(output_values)
    samples = results.samples[valid]
    out_valid = output_values[valid]

    param_names = results.parameter_names

    # Select parameters
    if params is None:
        # Compute correlations and select top
        correlations = []
        for i, name in enumerate(param_names):
            corr, _ = spearmanr(samples[:, i], out_valid)
            correlations.append((name, i, abs(corr)))
        correlations.sort(key=lambda x: x[2], reverse=True)
        selected = correlations[:max_params]
        params = [s[0] for s in selected]
        indices = [s[1] for s in selected]
    else:
        indices = [param_names.index(p) for p in params]

    n_params = len(params)
    fig, axes = plt.subplots(n_params, 1, figsize=figsize, squeeze=False)

    for i, (param, idx) in enumerate(zip(params, indices)):
        ax = axes[i, 0]
        ax.scatter(samples[:, idx], out_valid, alpha=alpha, s=s)
        ax.set_xlabel(param)
        ax.set_ylabel(output.upper())

        # Add correlation
        corr, _ = spearmanr(samples[:, idx], out_valid)
        ax.set_title(f'{param} vs {output.upper()} (r = {corr:.3f})')

    plt.tight_layout()
    return fig


def plot_cumulative_distribution(
    results: UncertaintyResults,
    output: str = 'lcoe',
    ax: Optional[Axes] = None,
    color: str = 'steelblue',
    show_percentiles: bool = True
) -> Axes:
    """
    Plot cumulative distribution function (CDF) of output.

    Args:
        results: UncertaintyResults from Monte Carlo analysis
        output: Output variable to plot
        ax: Matplotlib axes (creates new figure if None)
        color: Line color
        show_percentiles: Whether to show P10, P50, P90 lines

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    values = getattr(results, output)
    valid = np.sort(values[~np.isnan(values)])
    n = len(valid)
    probs = np.arange(1, n + 1) / n

    ax.plot(valid, probs, color=color, linewidth=2)

    if show_percentiles:
        p10 = np.percentile(valid, 10)
        p50 = np.percentile(valid, 50)
        p90 = np.percentile(valid, 90)

        for p, label in [(p10, 'P10'), (p50, 'P50'), (p90, 'P90')]:
            ax.axvline(p, color='gray', linestyle='--', alpha=0.7)
            ax.annotate(f'{label}: {p:.2f}', xy=(p, 0.5), rotation=90,
                       va='bottom', ha='right', fontsize=9)

    labels = {
        'lcoe': 'LCOE (ct/kWh)',
        'net_power': 'Net Power (kW)',
        'capex': 'CAPEX ($)',
        'opex': 'OPEX ($/year)'
    }
    ax.set_xlabel(labels.get(output, output))
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'Cumulative Distribution of {output.upper()}')
    ax.grid(alpha=0.3)

    return ax


def plot_comparison(
    results_list: List[UncertaintyResults],
    labels: List[str],
    output: str = 'lcoe',
    ax: Optional[Axes] = None,
    kind: str = 'boxplot'
) -> Axes:
    """
    Compare distributions from multiple analyses.

    Args:
        results_list: List of UncertaintyResults to compare
        labels: Labels for each result set
        output: Output variable to compare
        ax: Matplotlib axes (creates new figure if None)
        kind: 'boxplot' or 'violin'

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    for results in results_list:
        values = getattr(results, output)
        valid = values[~np.isnan(values)]
        data.append(valid)

    if kind == 'boxplot':
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    else:
        parts = ax.violinplot(data, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)

    label_map = {
        'lcoe': 'LCOE (ct/kWh)',
        'net_power': 'Net Power (kW)',
        'capex': 'CAPEX ($)',
        'opex': 'OPEX ($/year)'
    }
    ax.set_ylabel(label_map.get(output, output))
    ax.set_title(f'Comparison of {output.upper()} Distributions')
    ax.grid(axis='y', alpha=0.3)

    return ax


def create_summary_figure(
    mc_results: UncertaintyResults,
    tornado_results: TornadoResults,
    sobol_results: Optional[SobolResults] = None,
    output: str = 'lcoe',
    figsize: Tuple[int, int] = (16, 12)
) -> Figure:
    """
    Create a summary figure with multiple plots.

    Args:
        mc_results: Monte Carlo results
        tornado_results: Tornado analysis results
        sobol_results: Sobol analysis results (optional)
        output: Output variable
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    n_plots = 3 if sobol_results is not None else 2
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Histogram
    plot_histogram(mc_results, output=output, ax=axes[0, 0])

    # CDF
    plot_cumulative_distribution(mc_results, output=output, ax=axes[0, 1])

    # Tornado
    plot_tornado(tornado_results, ax=axes[1, 0])

    # Sobol or empty
    if sobol_results is not None:
        plot_sobol_indices(sobol_results, ax=axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, 'Sobol analysis not available',
                        ha='center', va='center', fontsize=12)
        axes[1, 1].axis('off')

    plt.tight_layout()
    return fig
