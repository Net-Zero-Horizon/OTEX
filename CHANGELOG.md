# Changelog

All notable changes to OTEX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Performance
- ``otex.data.cmems._extract_year_data`` rewritten to drop the
  ``np.hstack``-per-match pattern that was O(M²) in memory ops.
  Matches are accumulated into Python lists and concatenated once
  at the end; T_water is now sliced with a single fancy-index
  operation instead of one slice per cell. **6× faster** on a
  200×200 grid with ~2000 matched sites (1.76 s → 0.30 s),
  scales linearly with grid size.

### Added
- ``otex.plant.off_design_analysis`` now exposes optional
  parallelism for the design-sweep loop via the joblib backend.
  Default remains serial (``OTEX_OFF_DESIGN_NJOBS=1``) — empirical
  benchmark on a 4-year Jamaica run showed threading n_jobs=-1
  *increased* wall-clock time (1.84 → 2.15 min) because the
  vectorised CoolProp PropsSI added in 0.2.0 already saturates
  the work and the outer joblib layer creates oversubscription.
  The infrastructure is left in place as opt-in for workloads
  with much larger configuration sweeps or site counts.
- End-to-end regression test
  ``TestRegionalPipelineE2E::test_run_regional_analysis_full_pipeline``
  exercises the full ``run_regional_analysis`` path with synthetic
  CMEMS NetCDFs. Runs in ~5 seconds, asserts CSV schema, multi-year
  columns, output_dir routing, and per-(site, year) row counts.

### Changed
- ``scripts/regional_batch.py::run_region`` is now a thin wrapper
  around ``otex.regional.run_regional_analysis``. Removed ~90 LOC
  of duplicated pipeline code, plus two pre-existing bugs that
  would have surfaced in non-default usage:
  - ``parent_dir = os.getcwd() + 'Data_Results/'`` (missing path
    separator).
  - ``pd.read_csv('CMEMS_points_with_properties.csv', ...)``
    referencing the bundled CSV that was deleted in 0.2.0.
- ``scripts/regional_analysis.py`` deleted entirely — was a stale
  duplicate of ``otex.regional.run_regional_analysis`` with a
  docstring still referencing the removed bundled CSV. Use
  ``from otex.regional import run_regional_analysis`` directly.
- ``joblib>=1.2`` added as a core dependency (was already a
  transitive dep via pandas/scikit-learn; pin made explicit).

### Fixed
- ``otex.data.cmems.download_data`` ignored its ``new_path`` argument
  and always wrote NetCDFs to ``Data_Results/<region>/`` relative to
  the current working directory. Both ``copernicusmarine.subset``
  call sites now forward ``new_path`` directly, and the directory is
  created up front. Worked by accident in the default
  ``regional.py`` flow because that path matched; broke any caller
  using a custom output directory (HPC scratch dirs, batch scripts
  with ``--output-dir``, tests). Regression test added.

## [0.2.0] - Multi-year simulations + on-demand site catalog

### Added

#### On-demand region & site catalog (replaces bundled CSVs)
- New :mod:`otex.data.regions` — resolve country/territory by name or ISO
  3166-1 alpha-2/alpha-3 against Natural Earth admin-0 boundaries.
  Multi-part regions that cross the antimeridian (e.g. Fiji) yield
  multiple :class:`BBox` parts automatically.
- New :mod:`otex.data.bathymetry` — fetch ETOPO 2022 1-arcmin
  bathymetry subsets via NOAA NCEI THREDDS OPeNDAP, cached locally
  per-bbox in ``~/.otex/cache/bathymetry/``.
- New :mod:`otex.data.coastline` — distance-to-shore via Natural Earth
  1:50m coastlines, densified to ~2 km point cloud, KDTree on the
  unit-sphere. ±1 km accurate, sub-millisecond per query after warmup.
- New :func:`otex.data.build_sites` — builds the OTEC site DataFrame
  on demand for a region / bbox / polygon, with caching at
  ``~/.otex/cache/sites/<key>.parquet``.
- New :mod:`otex.data.demand` — multi-source electricity demand
  lookup. Tries Our World in Data first (bulk CSV, ~9 MB cached
  once locally), falls back to the World Bank Open Data API
  (``EG.USE.ELEC.KH.PC × SP.POP.TOTL``). Both sources are free and
  require no authentication. Used by ``regional_batch.py`` for
  demand-driven plant sizing.

#### Multi-year simulations
- **Multi-year simulations.** `DataConfig` now accepts `year_start` and
  `year_end` (inclusive). The thermal/oceanographic pipeline reads N years
  of NetCDFs, concatenates them along the time axis, and propagates a
  continuous `DatetimeIndex` through the rest of the analysis.
  - CLI: `otex-regional --year-start 2020 --year-end 2022`.
  - Loader validates that the SAME sites appear in every year of the
    configured range; mismatches raise a clear error pointing at the bad
    year so users know which NetCDF to re-download.
- **NPV-based LCOE** (`otex.economics.lcoe_npv`). Replaces the legacy CRF
  annualisation when `n_years > 1` with explicit per-year discounted
  cashflows. Inputs from the simulated window are extrapolated cyclically
  to the project lifetime.
- **Configurable degradation models** (`DegradationConfig`):
  `constant`, `logistic`, and `step`.
- **Configurable OPEX escalation** (`OpexEscalationConfig`):
  `flat`, `fixed_rate`, and `indexed`.
- New helper module `otex.economics.timeseries` with
  `aggregate_p_net_by_year()` and `annual_energy_kwh()` (leap-year aware).
- Multi-year aware regional outputs:
  - `OTEC_sites_*.csv` gains `LCOE_legacy`, `AEP_min`, `AEP_p50`,
    `AEP_max`, and `AEP_std` columns when `n_years > 1`.
  - New per-(site, year) CSV `OTEC_sites_yearly_*.csv` for diagnostic
    plotting.
- HYCOM backend gains the same multi-year loop; runs that span multiple
  experiment epochs (e.g. reanalysis ↔ analysis) resolve the right
  experiment per year.

### Removed
- **All bundled CSV catalogs.** The wheel ships no static data
  catalog any more:
  - `CMEMS_points_with_properties.csv` (9.7 MB site grid) → replaced
    by :func:`otex.data.build_sites` (ETOPO 2022 + Natural Earth).
  - `download_ranges_per_region.csv` (region bboxes) → replaced by
    :func:`otex.data.regions.resolve_region` (Natural Earth admin-0).
  - `country_demand.csv` (electricity demand by country) → replaced
    by :func:`otex.data.demand.fetch_demand_TWh` (OWID + World Bank).

### Changed
- :func:`otex.data.load_sites` now requires a ``region`` argument
  (the legacy zero-arg call returned a 218 k-row global DataFrame
  built from the now-deleted CSV). Pass a country name / ISO code
  and optionally ``min_depth`` / ``max_depth``.
- ``run_regional_analysis`` calls ``load_sites(region, ...)`` with
  the depth filter from ``DepthLimits`` instead of slicing a bundled
  global catalog.
- ``cmems.download_data`` and ``hycom.download_data`` resolve region
  bboxes via Natural Earth at runtime; they no longer read the
  deleted bbox CSV.
- Computed ``water_depth`` and ``dist_shore`` will differ slightly
  from the legacy CSV values (~5-10 % typical) because the
  underlying bathymetry source changed (ETOPO 2022 instead of the
  unspecified source used to build the original CSV). Downstream
  LCOE values for the same site may shift on the order of 1-2 %
  for that reason alone.
- **Minimum Python version raised to 3.10.** Python 3.9 reached
  end-of-life in October 2025 and started showing numpy/PyTables
  binary-compatibility issues on the 0.2.0 dependency set.
- `time_resolution` default in `DataConfig` is now `'24h'` (lowercase) to
  match pandas ≥ 2.2 frequency aliases. Previously this raised
  `ValueError: Invalid frequency: H` on modern pandas installs.
- Cache filenames produced by `data_processing()` and the off-design
  time-series writer use the multi-year label (`2020-2022`) when
  applicable. Single-year filenames are unchanged.
- `otex/analysis/visualization.py` axis labels for OPEX changed from
  `'OPEX ($/year)'` to `'OPEX ($/yr, lifetime-avg)'` to reflect that
  multi-year OPEX is the lifetime-discounted average.

### Deprecated
- `DataConfig.year` and the `year=` parameter on
  `parameters_and_constants()` / `get_default_config()` /
  `run_regional_analysis()`. They still work but emit a
  `DeprecationWarning`. Use `year_start`/`year_end` instead.
- `lcoe_time_series()` without a `timestamp` argument (legacy CRF path).

### Notes
- **Single-year LCOE is unchanged.** The NPV path is only triggered when
  `n_years > 1`, so existing single-year case studies produce the same
  numbers as 0.1.x.

## [0.1.4.2] - 2026-05-04

### Fixed
- The bundled reference CSVs were ignored by `.gitignore` (`*.csv`),
  so they were never tracked in git and consequently absent from CI
  builds even after the `package-data` rule was corrected in 0.1.4.1.
  The wheel now actually contains them.

## [0.1.4.1] - 2026-05-04

### Fixed
- Bundled CSV data files (`download_ranges_per_region.csv`,
  `CMEMS_points_with_properties.csv`) were missing from every wheel
  since 0.1.0 because the `package-data` rule was declared on the
  `otex` package instead of the `otex.data` sub-package. Loading the
  region database from a pip-installed copy now works.

## [0.1.4] - 2026-05-04

### Fixed
- Kalina and Uehara turbines no longer violate the Carnot limit
  (entropy-balance replaces the prior latent-heat approximation).
- Antoine coefficients for the NH3-H2O mixture properties replaced
  with NIST log10(P_bar) form.
- Rankine Open Q_evap (was inflated by the inherited base-class
  formula) and Hybrid heat-flow direction corrected.

### Changed
- Uehara rewritten as a faithful single-loop separator / 2-stage
  turbine / regenerator / absorber topology (Uehara & Ikegami, 1990).
- Kalina rewritten as faithful KCS-11 with exact mass and energy
  balances at the separator and recuperator.
- Mixture cycles now use the basic-solution bubble point for
  p_evap / p_cond instead of the pure-NH3 saturation pressure.

## [0.1.3] - 2026-05-01

### Added
- **Site-screening (siting) layers** (`otex.data.siting`)
  - Protected areas exclusion via WDPA (April 2026 snapshot, IUCN I-IV strict
    categories, configurable buffer)
  - Shipping-lane exclusion via World Bank Global Vessel Density (P95 cutoff,
    configurable buffer)
  - Seismic hazard cost multiplier via GEM Global Seismic Hazard Map
    (PGA at 475-yr return period, applied to CAPEX)
  - Tropical-cyclone cost multiplier via NOAA IBTrACS v04 (track frequency
    within 100 km, applied to OPEX)
  - On-demand download of all four global datasets to `~/.otex/siting_cache/`
    with auto-extraction of zipped distributions; subsequent runs reuse cache
  - `SitingConfig` dataclass for tuning (buffers, percentile cutoff, weights,
    normalisation references); fully off by default for backward compatibility
  - `scripts/build_siting_layers.py` CLI to pre-compute the enriched site CSV
  - Lazy import of geopandas/rasterio so the core install stays lean; install
    geospatial extras with `pip install otex[siting]`
  - Wiring in `otex.regional.run_regional_analysis` to apply hard exclusions
    and stuff per-site hazard arrays into the cost pipeline; multipliers
    applied in `otex.economics.costs.capex_opex_lcoe`
- HYCOM data source support (`otex.data.hycom`)
  - Download ocean temperature data via OPeNDAP (no authentication required)
  - Reanalysis (GLBv0.08/expt_53.X, 1994–2015) and analysis (GLBy0.08/expt_93.0, 2019–2024) experiments
  - 40 standard depth levels, 0.08° spatial resolution
  - Automatic experiment selection based on year
  - CMEMS-compatible NetCDF output (drop-in replacement)
  - `data_source='HYCOM'` parameter in `run_regional_analysis()`
- Uncertainty analysis module (`otex.analysis`)
  - Monte Carlo analysis with Latin Hypercube Sampling
  - Sobol global sensitivity analysis (requires SALib)
  - Tornado diagram analysis
  - Visualization functions for all analysis types
- CLI script for uncertainty analysis (`scripts/uncertainty_analysis.py`)
- Comprehensive documentation
  - Installation guide with CMEMS setup
  - Quick start tutorial
  - Regional analysis tutorial
  - Uncertainty analysis tutorial
  - API reference structure
- CONTRIBUTING.md with development guidelines
- This CHANGELOG.md

### Changed
- Enhanced README.md with badges, better structure, and examples

## [0.1.0] - 2024-01-30

### Added
- Initial release of OTEX
- Core thermodynamic cycle models
  - Rankine closed cycle
  - Rankine open cycle
  - Rankine hybrid cycle
  - Kalina cycle
  - Uehara cycle
- Working fluid support
  - Ammonia (polynomial and CoolProp)
  - R134a, R245fa, Propane, Isobutane (CoolProp)
- Plant sizing module
  - Component sizing (turbine, heat exchangers, pumps)
  - Seawater pipe design
  - Off-design performance analysis
- Economic analysis
  - CAPEX calculation by component
  - OPEX estimation
  - LCOE calculation
  - Onshore vs offshore cost models
- Data integration
  - CMEMS oceanographic data download
  - NetCDF processing
  - Multi-depth temperature profiles
- Configuration management
  - Centralized configuration with dataclasses
  - Legacy compatibility layer
- Regional analysis script
- Global analysis script
- Test suite with pytest

### Based On
- Original pyOTEC methodology by Langer et al. (2023)

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2024-01-30 | Initial release |

---

## Upgrade Guide

### Upgrading to 0.2.0 (when released)

The uncertainty analysis module is fully backwards compatible. No breaking changes.

To use the new features:

```python
# New imports
from otex.analysis import (
    MonteCarloAnalysis,
    UncertaintyConfig,
    TornadoAnalysis,
    SobolAnalysis,
    plot_histogram,
    plot_tornado
)
```

Install optional dependency for Sobol analysis:
```bash
pip install SALib>=1.4.0
```

---

[Unreleased]: https://github.com/msotocalvo/OTEX/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/msotocalvo/OTEX/releases/tag/v0.2.0
[0.1.0]: https://github.com/msotocalvo/OTEX/releases/tag/v0.1.0
