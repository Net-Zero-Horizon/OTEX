# Changelog

All notable changes to OTEX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-07-22

### Changed
- **``build_sites`` now pre-filters candidates against the CMEMS grid
  mask by default.** Previously, ``data.sites.build_sites`` returned
  every GEBCO ocean cell in the region's bbox that met the depth
  window. On regions with complex bank/channel bathymetry (Bahamas,
  Maldives, Marshall Islands…) up to ~93 % of those candidates landed
  in CMEMS cells that GLORYS12v1 marks as land or above the model
  seafloor, and ``data_processing`` silently dropped them as NaN. The
  reported site count for Bahamas at defaults was ``290`` — pyOTEC
  reports ~1342 for the same domain because its bundled
  ``CMEMS_points_with_properties.csv`` is CMEMS-grid-aligned by
  construction. This asymmetry was reported by a user comparing the
  two frameworks.

  ``build_sites`` now downloads a single-time, single-depth CMEMS
  ``thetao`` snapshot at ``cmems_verify_depth`` (default ``1062.44 m``,
  matching ``SeawaterPipes.cw_inlet_length`` and pyOTEC's
  ``length_CW_inlet``), extracts its boolean valid-cell mask, and
  discards GEBCO candidates that fall on CMEMS-invalid cells. The
  mask is cached under ``$OTEX_CACHE_DIR/cmems_mask/`` so subsequent
  (and offline) calls are free.

  For Bahamas at defaults the pool now returns ``2618`` sites —
  the remaining gap vs pyOTEC (~1342) is explained by OTEX's wider
  admin-0 bbox with a 2° offshore buffer.

  Opt out with ``build_sites(..., cmems_verify=False)`` (matches the
  0.4.x behaviour). The graceful fallback still handles missing
  ``copernicusmarine`` / CMEMS credentials with a ``RuntimeWarning``
  and returns the un-filtered pool.

### Added
- ``otex.data.cmems_mask`` — new module exposing
  ``cmems_valid_mask(west, east, south, north, depth_m, refresh=False)``
  and ``filter_sites_by_cmems_mask(sites, ...)``. Both cache the
  downloaded snapshot per ``(bbox, depth)`` tuple under
  ``$OTEX_CACHE_DIR/cmems_mask/`` for offline reuse on HPC compute
  nodes.
- ``build_sites`` gained ``cmems_verify: bool = True`` and
  ``cmems_verify_depth: float = 1062.44`` keyword arguments. Both are
  part of the on-disk cache key so verified and unverified pools are
  cached independently.

## [0.4.1] - 2026-07-08

### Fixed
- **Rankine Hybrid cycle produced identical results to Rankine Closed.**
  ``core.cycles.RankineHybridCycle`` computed a full 8-state hybrid
  (closed primary + open flash secondary) via
  ``calculate_cycle_states``, but ``plant.utils.enthalpies_entropies``
  routed the hybrid through the same ``h_1..h_4``-only branch as a plain
  closed Rankine (states 5-8 of the flash-steam secondary were silently
  dropped). Every downstream sizing/LCOE call therefore reduced to the
  closed-cycle result, and the two cycles emitted byte-different h5
  files with numerically identical ``LCOE_nom`` and ``p_net_nom``.

### Added
- ``Efficiencies.hybrid_secondary_boost`` (default ``0.10``) — the
  design-time power uplift attributed to the flash-steam secondary
  turbine (midpoint of the 8-15 % range documented in
  ``RankineHybridCycle``). Exposed in the parameters dict as
  ``inputs['hybrid_secondary_boost']`` so uncertainty and sensitivity
  studies can sweep it.
- ``plant.utils.enthalpies_entropies`` now has an explicit ``hybrid``
  branch that tags the returned enthalpies with
  ``cycle_type='rankine_hybrid'``.
- ``plant.sizing.otec_sizing`` applies ``_hybrid_boost`` **only** to
  ``p_gross`` inside the ``p_net`` formula — the primary evaporator,
  condenser and pump sizing use the un-boosted rating, so CAPEX stays
  identical to the corresponding closed Rankine. The hybrid advantage
  therefore materialises purely as ~10-15 % more electrical output at
  the same capital cost, and LCOE drops by roughly the boost factor.
- ``scripts/regenerate_paper_figure_2.py`` and
  ``scripts/regenerate_paper_figure_3.py`` — reproducible scripts that
  rebuild the case-study figures from
  ``Data_Results/Cuba/Time_series_data_Cuba_*.h5``.

## [0.4.0] - 2026-07-08

### Changed
- **Licence: MIT → Mozilla Public Licence v. 2.0 (MPL-2.0).** Resolves a
  licence-consistency issue raised by SoftwareX editorial review:
  portions of OTEX (six files under ``otex/plant/``, ``otex/economics/``
  and ``otex/data/``) derive from Langer et al.'s pyOTEC
  (https://github.com/JKALanger/pyOTEC), which is licensed under
  EUPL-1.2 and requires derivative works to adopt a compatible licence.
  MPL-2.0 is explicitly listed as EUPL-1.2-compatible in the EUPL Art. 5
  Appendix, and preserves file-level integration into Apache-2.0
  downstreams (e.g. the ESFEX energy-systems framework). See
  ``NOTICE`` for full attribution and licence-compatibility rationale.
- ``LICENSE`` file added (canonical MPL-2.0 text, was missing).
- ``NOTICE`` file added, listing pyOTEC-derived files and reproducing
  the required attribution.
- ``README.md``, ``pyproject.toml`` (both the ``license`` field and the
  OSI classifier) updated to declare MPL-2.0.

### Fixed
- **PyPI project links** now point to the canonical repository. The
  ``[project.urls]`` table in ``pyproject.toml`` previously listed
  ``github.com/otex-dev/otex`` (which does not exist — PyPI users
  clicking *Repository* / *Issues* landed on a 404). The Homepage,
  Repository, Issues and (new) Changelog links now point to
  ``github.com/Net-Zero-Horizon/OTEX``. README CI/coverage badges and
  the ``git clone`` snippet, previously pinned to the personal fork
  ``msotocalvo/OTEX``, are also aligned to the canonical org repo.
- **CMIP6 delta cache-key alignment** (bundled from post-0.3.1
  work-in-progress). ``run_regional_analysis`` now derives its climate
  bbox from ``build_sites(studied_region, lat_max)`` so it matches the
  key ``hpc/warm_caches.py`` writes on the internet-capable login node.
  Without this, HPC compute nodes fell through to a (failing) network
  zarr fetch. Also adds ``zarr>=2.10,<3`` and ``gcsfs>=2023.1`` to core
  dependencies, since ``climate.ensemble_delta`` needs them to read
  Pangeo's Zarr archive.

## [0.3.1] - 2026-05-14

### Fixed
- Uncertainty / sensitivity analysis silently hard-coded the
  thermodynamic configuration to ``rankine_closed`` + ``ammonia`` +
  ``offshore``. Calling :class:`MonteCarloAnalysis`,
  :class:`TornadoAnalysis` or :class:`SobolAnalysis` against any
  other cycle (Kalina, Uehara, Open / Hybrid Rankine), working
  fluid, or onshore installation produced LCOE uncertainty bounds
  for the *wrong* plant — the per-simulation worker rebuilt
  ``inputs`` via ``parameters_and_constants(p_gross=..., cost_level=...)``
  with no other arguments, so the default cycle / fluid / install
  came back regardless of what the user requested.
  ``cycle_type``, ``fluid_type``, ``installation_type`` and
  ``use_coolprop`` are now first-class parameters on every
  uncertainty / sensitivity class and are forwarded into the
  per-simulation arg tuple consumed by ``_run_single_simulation``.

## [0.3.0] - 2026-05-14

### Added
- **Mode Inverso — per-site formal design optimisation.** A second
  operation mode parallel to the existing forward pipeline. Instead
  of asking *"what's the LCOE if I install P_gross MW?"*, the new
  ``otex-regional-optimal`` command answers *"what plant design
  minimises LCOE at each site?"*. The optimiser solves a 4-D
  continuous NLP per site over ``x = (p_gross, dT_WW, dT_CW,
  depth_CW)`` via L-BFGS-B on a normalised [0,1] cube, with a
  quadratic penalty method on both physical constraints (pinch
  points, pipe diameter, parasitic ratio, bathymetric fit, ΔT
  margin) and **user-supplied exogenous caps** (max AEP, CAPEX,
  p_net, p_gross, or parasitic ratio) — at least one user cap is
  required for an interior optimum because OTEX's modelled cost
  function is monotone in p_gross. Categorical choices (cycle,
  fluid, installation type) stay exogenous. The optimiser uses a
  warm-start that respects the active user cap to converge 3-10×
  faster than from the box centre. New module ``otex.optimization``
  (``DesignVector``, ``SiteContext``, ``UserConstraints``,
  ``evaluate``, ``optimize_site``, ``run_regional_optimization``).
  Tutorial: ``docs/tutorials/optimal_sizing.md``. End-to-end smoke
  on Jamaica 2020-2023 with ``--max-p-gross-MW 120 --no-coolprop``:
  242/242 sites feasible in 87 s (0.36 s/site), median LCOE 22.63
  ¢/kWh vs 24.97 in forward mode @ 100 MW fixed (−8.6 %).
- **CMIP6 climate-scenario support** (delta-method downscaling).
  ``run_regional_analysis`` accepts a non-historical scenario
  (``ssp126``/``ssp245``/``ssp370``/``ssp585``) plus a target year;
  OTEX pulls thetao deltas from a 3-model GCM ensemble
  (``MPI-ESM1-2-LR``, ``EC-Earth3``, ``CanESM5``) via the public
  Pangeo CMIP6 Zarr archive on Google Cloud, interpolates them onto
  each site's coordinates, and adds them to both the warm and cold
  CMEMS time series before the off-design analysis runs. Output
  files include a ``_<scenario>_<target_year>`` suffix; default
  ``historical`` is a no-op equivalent to the 0.2.0 pipeline.
  - New module :mod:`otex.data.climate` (``fetch_thetao_mean``,
    ``compute_delta_field``, ``ensemble_delta``, ``delta_at_points``).
  - New :class:`otex.config.ClimateConfig` and three CLI flags:
    ``--climate-scenario``, ``--climate-year``, ``--climate-models``.
  - Per-(model, scenario, period, depth, bbox) Pangeo slices are
    cached as Parquet under ``~/.otex/cache/cmip6/`` so subsequent
    runs are sub-millisecond.
  - Smoke test on Jamaica 2020-2023 under SSP2-4.5 / 2050: median
    LCOE drops 5.6% vs the historical baseline because surface
    warming (+1.0 °C) outpaces 1062 m warming (+0.3 °C), widening
    ΔT and raising AEP by ~1.8%.

### Performance
- ``otex.data.cmems._extract_year_data`` rewritten to drop the
  ``np.hstack``-per-match pattern that was O(M²) in memory ops.
  Matches are accumulated into Python lists and concatenated once
  at the end; T_water is now sliced with a single fancy-index
  operation instead of one slice per cell. **6× faster** on a
  200×200 grid with ~2000 matched sites (1.76 s → 0.30 s),
  scales linearly with grid size.

### Added
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

[Unreleased]: https://github.com/msotocalvo/OTEX/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/msotocalvo/OTEX/releases/tag/v0.3.1
[0.3.0]: https://github.com/msotocalvo/OTEX/releases/tag/v0.3.0
[0.2.0]: https://github.com/msotocalvo/OTEX/releases/tag/v0.2.0
[0.1.0]: https://github.com/msotocalvo/OTEX/releases/tag/v0.1.0
