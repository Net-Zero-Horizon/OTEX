# -*- coding: utf-8 -*-
"""
Export utilities for OTEX uncertainty and sensitivity analysis results.

Generates a structured set of files suitable for deep statistical analysis
and advanced visualisations:

    output_dir/
    ├── metadata.json        — run configuration and reproducibility info
    ├── samples.csv          — raw MC samples: all parameter values + all outputs
    ├── statistics.csv       — full descriptive statistics per output variable
    ├── correlations.csv     — Spearman rank correlations (inputs → outputs)
    ├── parameters.csv       — uncertain parameter definitions
    ├── tornado.csv          — OAT sensitivity (only if TornadoResults provided)
    └── sobol.csv            — Sobol variance-based indices (only if SobolResults provided)

Typical usage
-------------
    from otex.analysis import MonteCarloAnalysis, TornadoAnalysis, SobolAnalysis
    from otex.analysis.export import export_analysis

    mc_results      = MonteCarloAnalysis(T_WW=28, T_CW=5).run()
    tornado_results = TornadoAnalysis(T_WW=28, T_CW=5).run()
    sobol_results   = SobolAnalysis(T_WW=28, T_CW=5).run()

    export_analysis(
        output_dir='results/run_01',
        mc_results=mc_results,
        tornado_results=tornado_results,
        sobol_results=sobol_results,
        metadata={'T_WW': 28, 'T_CW': 5, 'cost_level': 'low_cost'},
    )
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Lazy imports to avoid circular dependencies
# (UncertaintyResults, SobolResults, TornadoResults are type-checked only)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .uncertainty import UncertaintyResults
    from .sensitivity import SobolResults, TornadoResults
    from .distributions import UncertaintyConfig


# ---------------------------------------------------------------------------
# Column labels for human-readable output
# ---------------------------------------------------------------------------

_OUTPUT_UNITS = {
    'lcoe':      'ct/kWh',
    'net_power': 'kW',
    'capex':     'USD',
    'opex':      'USD/yr',
}

_OUTPUT_LABELS = {
    'lcoe':      'LCOE',
    'net_power': 'Net Power',
    'capex':     'CAPEX',
    'opex':      'OPEX',
}


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------

def export_analysis(
    output_dir: str | Path,
    mc_results: Optional['UncertaintyResults'] = None,
    tornado_results: Optional['TornadoResults'] = None,
    sobol_results: Optional['SobolResults'] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Export a complete uncertainty analysis bundle to ``output_dir``.

    All files that can be generated from the provided results are written.
    At least one of ``mc_results``, ``tornado_results``, or ``sobol_results``
    must be supplied.

    Parameters
    ----------
    output_dir : str or Path
        Directory where files are written (created if it does not exist).
    mc_results : UncertaintyResults, optional
        Output of ``MonteCarloAnalysis.run()``.  Produces *samples.csv*,
        *statistics.csv*, *correlations.csv*, and *parameters.csv*.
    tornado_results : TornadoResults, optional
        Output of ``TornadoAnalysis.run()``.  Produces *tornado.csv*.
    sobol_results : SobolResults, optional
        Output of ``SobolAnalysis.run()``.  Produces *sobol.csv*.
    metadata : dict, optional
        Extra key/value pairs merged into *metadata.json* (e.g. ``T_WW``,
        ``T_CW``, ``cost_level``, ``p_gross``).

    Returns
    -------
    Path
        Absolute path to ``output_dir``.

    Raises
    ------
    ValueError
        If no results are supplied.
    """
    if mc_results is None and tornado_results is None and sobol_results is None:
        raise ValueError("At least one of mc_results, tornado_results, or sobol_results must be provided.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: List[str] = []

    # --- metadata.json ---
    meta = _build_metadata(mc_results, tornado_results, sobol_results, metadata or {})
    _write_json(meta, output_dir / 'metadata.json')
    written.append('metadata.json')

    # --- Monte Carlo derived files ---
    if mc_results is not None:
        samples_df = make_samples_df(mc_results)
        samples_df.to_csv(output_dir / 'samples.csv', index=False)
        written.append('samples.csv')

        stats_df = make_statistics_df(mc_results)
        stats_df.to_csv(output_dir / 'statistics.csv', index=False)
        written.append('statistics.csv')

        corr_df = make_correlations_df(mc_results)
        corr_df.to_csv(output_dir / 'correlations.csv', index=False)
        written.append('correlations.csv')

        if mc_results.config is not None:
            param_df = make_parameters_df(mc_results.config)
            param_df.to_csv(output_dir / 'parameters.csv', index=False)
            written.append('parameters.csv')

    # --- Tornado ---
    if tornado_results is not None:
        config = mc_results.config if mc_results is not None else None
        tornado_df = make_tornado_df(tornado_results, config=config)
        tornado_df.to_csv(output_dir / 'tornado.csv', index=False)
        written.append('tornado.csv')

    # --- Sobol ---
    if sobol_results is not None:
        config = mc_results.config if mc_results is not None else None
        sobol_df = make_sobol_df(sobol_results, config=config)
        sobol_df.to_csv(output_dir / 'sobol.csv', index=False)
        written.append('sobol.csv')

    print(f"Exported {len(written)} files to {output_dir.resolve()}:")
    for name in written:
        print(f"  {name}")

    return output_dir.resolve()


# ---------------------------------------------------------------------------
# Individual DataFrame builders (public — usable independently)
# ---------------------------------------------------------------------------

def make_samples_df(results: 'UncertaintyResults') -> pd.DataFrame:
    """
    Build a tidy DataFrame with all Monte Carlo samples and outputs.

    Columns
    -------
    sample_id : int
        Zero-based index of each simulation run.
    <parameter names> : float
        Sampled value for each uncertain parameter.
    lcoe : float
        Levelised cost of energy [ct/kWh].
    net_power : float
        Net electrical power output [kW].
    capex : float
        Total capital expenditure [USD].
    opex : float
        Annual operating expenditure [USD/yr].
    valid : bool
        True if the simulation converged successfully.

    Parameters
    ----------
    results : UncertaintyResults

    Returns
    -------
    pd.DataFrame
    """
    n = results.samples.shape[0]

    data: Dict[str, Any] = {'sample_id': np.arange(n)}

    for i, name in enumerate(results.parameter_names):
        data[name] = results.samples[:, i]

    data['lcoe']      = results.lcoe
    data['net_power'] = results.net_power
    data['capex']     = results.capex
    data['opex']      = results.opex if len(results.opex) == n else np.full(n, np.nan)
    data['valid']     = ~np.isnan(results.lcoe)

    return pd.DataFrame(data)


def make_statistics_df(results: 'UncertaintyResults') -> pd.DataFrame:
    """
    Build a descriptive-statistics DataFrame for all output variables.

    Columns
    -------
    output, unit, n_valid, n_invalid,
    mean, std, cv, skewness, kurtosis,
    min, p5, p10, p25, median, p75, p90, p95, max

    Parameters
    ----------
    results : UncertaintyResults

    Returns
    -------
    pd.DataFrame
    """
    outputs = {
        'lcoe':      results.lcoe,
        'net_power': results.net_power,
        'capex':     results.capex,
        'opex':      results.opex if len(results.opex) > 0 else np.array([]),
    }

    rows = []
    for name, values in outputs.items():
        if len(values) == 0:
            continue
        valid = values[~np.isnan(values)]
        n_valid   = len(valid)
        n_invalid = len(values) - n_valid

        if n_valid == 0:
            rows.append({'output': name, 'unit': _OUTPUT_UNITS.get(name, ''), 'n_valid': 0, 'n_invalid': n_invalid})
            continue

        rows.append({
            'output':    name,
            'label':     _OUTPUT_LABELS.get(name, name),
            'unit':      _OUTPUT_UNITS.get(name, ''),
            'n_valid':   n_valid,
            'n_invalid': n_invalid,
            'mean':      np.mean(valid),
            'std':       np.std(valid),
            'cv':        np.std(valid) / np.mean(valid) if np.mean(valid) != 0 else np.nan,
            'skewness':  float(scipy_stats.skew(valid)),
            'kurtosis':  float(scipy_stats.kurtosis(valid)),   # excess kurtosis
            'min':       np.min(valid),
            'p5':        np.percentile(valid, 5),
            'p10':       np.percentile(valid, 10),
            'p25':       np.percentile(valid, 25),
            'median':    np.median(valid),
            'p75':       np.percentile(valid, 75),
            'p90':       np.percentile(valid, 90),
            'p95':       np.percentile(valid, 95),
            'max':       np.max(valid),
        })

    return pd.DataFrame(rows)


def make_correlations_df(results: 'UncertaintyResults') -> pd.DataFrame:
    """
    Compute Spearman rank correlations between every uncertain parameter
    and every output variable, including p-values.

    Columns
    -------
    parameter : str
        Parameter name.
    category : str
        Parameter category (from UncertaintyConfig if available).
    <output>_rho : float
        Spearman ρ with that output.
    <output>_pvalue : float
        Two-tailed p-value for that correlation.
    lcoe_rank : int
        Rank by |lcoe_rho| (1 = most correlated).

    Parameters
    ----------
    results : UncertaintyResults

    Returns
    -------
    pd.DataFrame
    """
    outputs = {
        'lcoe':      results.lcoe,
        'net_power': results.net_power,
        'capex':     results.capex,
        'opex':      results.opex if len(results.opex) > 0 else None,
    }

    # Build category lookup from config if available
    category_map: Dict[str, str] = {}
    if results.config is not None:
        category_map = {p.name: p.category for p in results.config.parameters}

    rows = []
    for i, param_name in enumerate(results.parameter_names):
        row: Dict[str, Any] = {
            'parameter': param_name,
            'category':  category_map.get(param_name, 'unknown'),
        }
        param_values = results.samples[:, i]

        for out_name, out_values in outputs.items():
            if out_values is None or len(out_values) == 0:
                row[f'{out_name}_rho']    = np.nan
                row[f'{out_name}_pvalue'] = np.nan
                continue

            # Only use rows where both values are valid
            mask = ~np.isnan(param_values) & ~np.isnan(out_values)
            if mask.sum() < 3:
                row[f'{out_name}_rho']    = np.nan
                row[f'{out_name}_pvalue'] = np.nan
                continue

            rho, pval = scipy_stats.spearmanr(param_values[mask], out_values[mask])
            row[f'{out_name}_rho']    = float(rho)
            row[f'{out_name}_pvalue'] = float(pval)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Add rank by |lcoe_rho|
    if 'lcoe_rho' in df.columns:
        df['lcoe_rank'] = df['lcoe_rho'].abs().rank(ascending=False, method='min').astype(int)
        df = df.sort_values('lcoe_rank').reset_index(drop=True)

    return df


def make_parameters_df(config: 'UncertaintyConfig') -> pd.DataFrame:
    """
    Build a DataFrame describing the uncertain parameters used in the analysis.

    Columns
    -------
    name, category, nominal, distribution, bound_low, bound_high

    Parameters
    ----------
    config : UncertaintyConfig

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for p in config.parameters:
        rows.append({
            'name':         p.name,
            'category':     p.category,
            'nominal':      p.nominal,
            'distribution': p.distribution,
            'bound_low':    p.bounds[0],
            'bound_high':   p.bounds[1],
        })
    return pd.DataFrame(rows)


def make_tornado_df(
    results: 'TornadoResults',
    config: Optional['UncertaintyConfig'] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame with the full one-at-a-time (OAT) sensitivity results.

    Columns
    -------
    rank : int
        Rank by absolute swing (1 = most sensitive).
    parameter : str
    category : str
        Parameter category if config is provided, else 'unknown'.
    nominal : float
        Nominal parameter value.
    low_bound, high_bound : float
        Parameter values used in the OAT runs.
    output_baseline : float
        Output value at all-nominal conditions.
    output_at_low, output_at_high : float
        Output values when the parameter is set to its low/high bound.
    swing_abs : float
        Absolute swing: ``output_at_high - output_at_low``.
    swing_pct : float
        Swing as percentage of baseline: ``swing_abs / |baseline| × 100``.
    output_name : str
        Name of the output variable analysed.

    Parameters
    ----------
    results : TornadoResults
    config : UncertaintyConfig, optional
        If provided, category and nominal values are read from here.

    Returns
    -------
    pd.DataFrame
    """
    category_map: Dict[str, str] = {}
    nominal_map:  Dict[str, float] = {}
    bounds_map:   Dict[str, tuple] = {}

    if config is not None:
        for p in config.parameters:
            category_map[p.name] = p.category
            nominal_map[p.name]  = p.nominal
            if p.distribution == 'uniform':
                bounds_map[p.name] = (p.bounds[0], p.bounds[1])
            elif p.distribution == 'normal':
                mean, std = p.bounds
                bounds_map[p.name] = (mean - 2 * std, mean + 2 * std)
            else:
                bounds_map[p.name] = (p.bounds[0], p.bounds[1])

    baseline = results.baseline
    rows = []
    for name, low_val, high_val, swing in zip(
        results.parameter_names,
        results.low_values,
        results.high_values,
        results.swings,
    ):
        low_b, high_b = bounds_map.get(name, (np.nan, np.nan))
        rows.append({
            'parameter':       name,
            'category':        category_map.get(name, 'unknown'),
            'nominal':         nominal_map.get(name, np.nan),
            'low_bound':       low_b,
            'high_bound':      high_b,
            'output_baseline': baseline,
            'output_at_low':   low_val,
            'output_at_high':  high_val,
            'swing_abs':       swing,
            'swing_pct':       swing / abs(baseline) * 100 if baseline != 0 else np.nan,
            'output_name':     results.output_name,
        })

    df = pd.DataFrame(rows)
    df['rank'] = df['swing_abs'].abs().rank(ascending=False, method='min').astype(int)
    df = df.sort_values('rank')[['rank', 'parameter', 'category', 'nominal',
                                  'low_bound', 'high_bound', 'output_baseline',
                                  'output_at_low', 'output_at_high',
                                  'swing_abs', 'swing_pct', 'output_name']]
    return df.reset_index(drop=True)


def make_sobol_df(
    results: 'SobolResults',
    config: Optional['UncertaintyConfig'] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame with the Sobol variance-based sensitivity indices.

    Columns
    -------
    ST_rank : int
        Rank by total-order index ST (1 = most influential).
    parameter : str
    category : str
    S1 : float
        First-order index (direct effect).
    S1_conf : float
        Bootstrap confidence interval for S1.
    ST : float
        Total-order index (direct + interaction effects).
    ST_conf : float
        Bootstrap confidence interval for ST.
    interactions : float
        Estimated interaction contribution: ``ST - S1``.
    S1_pct : float
        S1 as percentage of total explained variance.
    ST_pct : float
        ST as percentage of total ST sum (relative importance).
    output_name : str

    Parameters
    ----------
    results : SobolResults
    config : UncertaintyConfig, optional
        If provided, adds ``category`` column.

    Returns
    -------
    pd.DataFrame
    """
    category_map: Dict[str, str] = {}
    if config is not None:
        category_map = {p.name: p.category for p in config.parameters}

    n = len(results.parameter_names)
    S1_conf = results.S1_conf if len(results.S1_conf) == n else np.full(n, np.nan)
    ST_conf = results.ST_conf if len(results.ST_conf) == n else np.full(n, np.nan)

    st_sum = np.sum(np.abs(results.ST)) or 1.0
    s1_sum = np.sum(np.abs(results.S1)) or 1.0

    rows = []
    for i, name in enumerate(results.parameter_names):
        rows.append({
            'parameter':    name,
            'category':     category_map.get(name, 'unknown'),
            'S1':           float(results.S1[i]),
            'S1_conf':      float(S1_conf[i]),
            'ST':           float(results.ST[i]),
            'ST_conf':      float(ST_conf[i]),
            'interactions': float(results.ST[i] - results.S1[i]),
            'S1_pct':       float(results.S1[i] / s1_sum * 100),
            'ST_pct':       float(results.ST[i] / st_sum * 100),
            'output_name':  results.output_name,
        })

    df = pd.DataFrame(rows)
    df['ST_rank'] = df['ST'].abs().rank(ascending=False, method='min').astype(int)
    df['S1_rank'] = df['S1'].abs().rank(ascending=False, method='min').astype(int)
    df = df.sort_values('ST_rank')[['ST_rank', 'S1_rank', 'parameter', 'category',
                                     'S1', 'S1_conf', 'ST', 'ST_conf',
                                     'interactions', 'S1_pct', 'ST_pct', 'output_name']]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

def _build_metadata(
    mc_results:      Optional['UncertaintyResults'],
    tornado_results: Optional['TornadoResults'],
    sobol_results:   Optional['SobolResults'],
    extra:           Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the metadata dictionary."""
    try:
        from otex import __version__ as otex_version
    except ImportError:
        otex_version = 'unknown'

    analyses: List[str] = []
    if mc_results is not None:
        analyses.append('monte_carlo')
    if tornado_results is not None:
        analyses.append('tornado')
    if sobol_results is not None:
        analyses.append('sobol')

    meta: Dict[str, Any] = {
        'timestamp':       datetime.now(timezone.utc).isoformat(),
        'otex_version':    otex_version,
        'analyses':        analyses,
    }

    # Monte Carlo metadata
    if mc_results is not None:
        valid_mask   = ~np.isnan(mc_results.lcoe)
        n_valid      = int(np.sum(valid_mask))
        n_invalid    = int(np.sum(~valid_mask))
        config       = mc_results.config

        meta['monte_carlo'] = {
            'n_samples':       len(mc_results.lcoe),
            'n_valid':         n_valid,
            'n_invalid':       n_invalid,
            'seed':            config.seed if config else None,
            'sampling_method': 'Latin Hypercube Sampling',
            'n_parameters':    len(mc_results.parameter_names),
            'parameter_names': mc_results.parameter_names,
        }

    # Tornado metadata
    if tornado_results is not None:
        meta['tornado'] = {
            'n_parameters': len(tornado_results.parameter_names),
            'output':       tornado_results.output_name,
            'baseline':     tornado_results.baseline,
        }

    # Sobol metadata
    if sobol_results is not None:
        meta['sobol'] = {
            'n_parameters': len(sobol_results.parameter_names),
            'output':       sobol_results.output_name,
        }

    # Merge user-supplied extra metadata
    meta.update(extra)
    return meta


def _write_json(data: Dict[str, Any], path: Path) -> None:
    """Write a dictionary as indented JSON, converting numpy types."""
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2, default=_convert)
