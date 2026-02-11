# -*- coding: utf-8 -*-
"""
OTEX Uncertainty and Sensitivity Analysis Module.

This module provides tools for:
- Monte Carlo uncertainty propagation with Latin Hypercube Sampling
- Global sensitivity analysis using Sobol indices
- Local sensitivity analysis using Tornado diagrams
- Visualization of analysis results

Example:
    >>> from otex.analysis import (
    ...     MonteCarloAnalysis, UncertaintyConfig,
    ...     SobolAnalysis, TornadoAnalysis,
    ...     plot_histogram, plot_tornado
    ... )
    >>>
    >>> # Monte Carlo analysis
    >>> config = UncertaintyConfig(n_samples=1000, seed=42)
    >>> mc = MonteCarloAnalysis(T_WW=28.0, T_CW=5.0, config=config)
    >>> results = mc.run()
    >>> stats = results.compute_statistics()
    >>> print(f"LCOE: {stats['lcoe']['lcoe_mean']:.2f} ct/kWh")
    >>>
    >>> # Tornado analysis
    >>> tornado = TornadoAnalysis(T_WW=28.0, T_CW=5.0)
    >>> tornado_results = tornado.run()
    >>> plot_tornado(tornado_results)
    >>>
    >>> # Sobol analysis (requires SALib)
    >>> sobol = SobolAnalysis(T_WW=28.0, T_CW=5.0, n_samples=512)
    >>> sobol_results = sobol.run()
    >>> for name, val in sobol_results.get_ranking()[:5]:
    ...     print(f"{name}: ST={val:.3f}")
"""

# Distributions and configuration
from .distributions import (
    UncertainParameter,
    UncertaintyConfig,
    get_default_parameters,
)

# Monte Carlo analysis
from .uncertainty import (
    MonteCarloAnalysis,
    UncertaintyResults,
)

# Sensitivity analysis
from .sensitivity import (
    SobolAnalysis,
    SobolResults,
    TornadoAnalysis,
    TornadoResults,
)

# Visualization
from .visualization import (
    plot_histogram,
    plot_tornado,
    plot_sobol_indices,
    plot_scatter_matrix,
    plot_cumulative_distribution,
    plot_comparison,
    create_summary_figure,
)

__all__ = [
    # Distributions
    "UncertainParameter",
    "UncertaintyConfig",
    "get_default_parameters",
    # Monte Carlo
    "MonteCarloAnalysis",
    "UncertaintyResults",
    # Sensitivity
    "SobolAnalysis",
    "SobolResults",
    "TornadoAnalysis",
    "TornadoResults",
    # Visualization
    "plot_histogram",
    "plot_tornado",
    "plot_sobol_indices",
    "plot_scatter_matrix",
    "plot_cumulative_distribution",
    "plot_comparison",
    "create_summary_figure",
]
