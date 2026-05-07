# -*- coding: utf-8 -*-
"""
OTEX Regional Analysis

Generate spatially and temporally resolved power generation profiles
for any technically feasible OTEC system size and region.

This module provides the same functionality as scripts/regional_analysis.py
but is accessible as part of the installed otex package.

Example usage::

    from otex.regional import run_regional_analysis

    otec_plants, sites_df = run_regional_analysis(
        studied_region='Jamaica',
        p_gross=-50000,
        cost_level='low_cost',
        year=2020,
        cycle_type='rankine_closed',
        fluid_type='ammonia',
    )
"""

import os
import time
import platform

import numpy as np
import pandas as pd

from otex.config import parameters_and_constants
from otex.data.cmems import data_processing, load_temperatures
from otex.data.resources import load_sites
from otex.plant.off_design_analysis import off_design_analysis


def run_regional_analysis(
    studied_region,
    p_gross=-100000,
    cost_level='low_cost',
    year=None,
    year_start=None,
    year_end=None,
    cycle_type='rankine_closed',
    fluid_type='ammonia',
    use_coolprop=True,
    output_dir=None,
    data_source='CMEMS',
    climate_scenario='historical',
    climate_year=None,
    climate_models=None,
):
    """
    Run regional OTEC analysis for a specified region.

    Downloads oceanographic data (if not cached), sizes OTEC plants
    at each feasible site, performs off-design analysis, and calculates
    LCOE for the entire region.

    Args:
        studied_region: Region name (e.g., 'Jamaica', 'Philippines').
            Must match an entry in the bundled regions database.
            Use ``otex.data.load_regions()`` to list available regions.
        p_gross: Gross power output in kW (negative, e.g., -100000 for 100 MW).
        cost_level: Cost assumption, either ``'low_cost'`` or ``'high_cost'``.
        year: Single calendar year (deprecated; use year_start/year_end).
        year_start: First simulated year (inclusive). Defaults to 2020 if
            neither year nor year_start is provided.
        year_end: Last simulated year (inclusive). Defaults to year_start.
        cycle_type: Thermodynamic cycle. One of ``'rankine_closed'``,
            ``'rankine_open'``, ``'rankine_hybrid'``, ``'kalina'``, ``'uehara'``.
        fluid_type: Working fluid. One of ``'ammonia'``, ``'r134a'``,
            ``'r245fa'``, ``'propane'``, ``'isobutane'``.
        use_coolprop: Whether to use CoolProp for fluid properties (default True).
        output_dir: Directory for results. Defaults to ``./Data_Results/``.
        data_source: Data source, ``'CMEMS'`` or ``'HYCOM'`` (default ``'CMEMS'``).

    Returns:
        tuple: ``(otec_plants, sites_df)`` where ``otec_plants`` is a dict
        with plant performance arrays and ``sites_df`` is a DataFrame
        with per-site results.
    """
    start = time.time()

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "Data_Results")

    if year is None and year_start is None:
        year_start = 2020

    inputs = parameters_and_constants(
        p_gross=p_gross,
        cost_level=cost_level,
        data=data_source,
        fluid_type=fluid_type,
        cycle_type=cycle_type,
        use_coolprop=use_coolprop,
        year=year,
        year_start=year_start,
        year_end=year_end,
        climate_scenario=climate_scenario,
        climate_year=climate_year,
        climate_models=climate_models,
    )
    year_str = inputs['year_label']
    climate_suffix = (
        f"_{inputs['climate_label']}" if inputs.get('climate_enabled') else ''
    )

    region_dir = os.path.join(output_dir, studied_region.replace(" ", "_"))
    run_dir = os.path.join(
        region_dir,
        f"{studied_region}_{year_str}{climate_suffix}_{-p_gross/1000}_MW_{cost_level}".replace(" ", "_"),
    )
    os.makedirs(run_dir, exist_ok=True)

    depth_WW = inputs['length_WW_inlet']
    depth_CW = inputs['length_CW_inlet']

    if inputs['data'] == 'HYCOM':
        from otex.data.hycom import download_data
    else:
        from otex.data.cmems import download_data

    files = download_data(cost_level, inputs, studied_region, region_dir + os.sep)

    print('\n++ Processing seawater temperature data ++\n')

    # Sites are now built on demand for the requested region (since
    # 0.2.0). The legacy bundled CSV is gone; the depth filter is applied
    # by build_sites itself, but we re-apply the inputs-derived bounds
    # here for full back-compat with downstream filtering.
    sites_df = load_sites(
        studied_region,
        # inputs['min_depth'] / ['max_depth'] are stored as negative
        # elevations, so we flip the sign for build_sites' positive-
        # depth API.
        min_depth=abs(inputs['min_depth']),
        max_depth=abs(inputs['max_depth']),
    )
    sites_df = sites_df[
        (sites_df['water_depth'] <= inputs['min_depth'])
        & (sites_df['water_depth'] >= inputs['max_depth'])
    ]
    sites_df = sites_df.sort_values(by=['longitude', 'latitude'], ascending=True)

    # Siting layers: protected areas (hard exclusion), AIS density (hard above
    # P95, soft penalty below), seismic + cyclone hazards (cost multipliers
    # applied later in economics/costs.py).
    siting_active = (
        inputs.get('siting_enable_mpa_filter')
        or inputs.get('siting_enable_ais_filter')
        or inputs.get('siting_enable_hazard_costs')
    )
    if siting_active:
        from otex.data.siting import enrich_sites
        sites_df = enrich_sites(
            sites_df,
            mpa_buffer_km=inputs.get('siting_mpa_buffer_km', 5.0),
            ais_buffer_km=inputs.get('siting_ais_buffer_km', 5.0),
            cache_dir=inputs.get('siting_cache_dir'),
            refresh=inputs.get('siting_refresh', False),
        )
        n0 = len(sites_df)
        if inputs.get('siting_enable_mpa_filter'):
            sites_df = sites_df[~sites_df['in_mpa_strict']]
        if inputs.get('siting_enable_ais_filter'):
            sites_df = sites_df[
                sites_df['ais_density_pct'] < inputs.get('siting_ais_exclusion_pct', 95.0)
            ]
        print(f'  [siting] retained {len(sites_df)}/{n0} sites after exclusions.')

    h5_file_WW = os.path.join(
        run_dir, f'T_{round(depth_WW, 0)}m_{year_str}_{studied_region}.h5'.replace(" ", "_")
    )
    h5_file_CW = os.path.join(
        run_dir, f'T_{round(depth_CW, 0)}m_{year_str}_{studied_region}.h5'.replace(" ", "_")
    )

    if os.path.isfile(h5_file_CW):
        T_CW_profiles, T_CW_design, coordinates_CW, id_sites, timestamp, inputs, nan_columns_CW = load_temperatures(h5_file_CW, inputs)
        print(f'{h5_file_CW} already exists. No processing necessary.')
    else:
        T_CW_profiles, T_CW_design, coordinates_CW, id_sites, timestamp, inputs, nan_columns_CW = data_processing(
            files[len(files) // 2:], sites_df, inputs, studied_region, run_dir + os.sep, 'CW'
        )

    if os.path.isfile(h5_file_WW):
        T_WW_profiles, T_WW_design, coordinates_WW, id_sites, timestamp, inputs, nan_columns_WW = load_temperatures(h5_file_WW, inputs)
        print(f'{h5_file_WW} already exists. No processing necessary.')
    else:
        T_WW_profiles, T_WW_design, coordinates_WW, id_sites, timestamp, inputs, nan_columns_WW = data_processing(
            files[:len(files) // 2], sites_df, inputs, studied_region, run_dir + os.sep, 'WW', nan_columns_CW
        )

    # ── CMIP6 climate-scenario delta (0.3.0+) ──────────────────────────
    # When a non-historical scenario is configured, add the per-site
    # ensemble-mean thetao anomaly to both the warm and cold time
    # series and to the design temperatures. The shift is constant in
    # time per site (delta-method assumes the seasonal pattern is
    # preserved) but varies across sites with the GCM resolution.
    if inputs.get('climate_enabled'):
        from otex.data.climate import ensemble_delta, delta_at_points
        from otex.data.regions import BBox

        cfg = inputs['climate_config']
        # Bbox covering the site cloud, padded by 1° so GCM cells at
        # the edge are interpolated cleanly.
        site_lons = coordinates_CW[:, 0]
        site_lats = coordinates_CW[:, 1]
        clim_bbox = BBox(
            north=float(site_lats.max()) + 1.0,
            south=float(site_lats.min()) - 1.0,
            east=float(site_lons.max()) + 1.0,
            west=float(site_lons.min()) - 1.0,
        )

        for label, depth_m, T_profiles, T_design in (
            ('WW', depth_WW, T_WW_profiles, T_WW_design),
            ('CW', depth_CW, T_CW_profiles, T_CW_design),
        ):
            print(f'  [climate] computing ensemble Δ at {depth_m:.0f} m for '
                  f'{cfg.scenario}/{cfg.target_year} ...', flush=True)
            ens = ensemble_delta(
                scenario=cfg.scenario,
                target_year=cfg.target_year,
                depth_m=depth_m,
                bbox=clim_bbox,
                models=cfg.models,
                baseline_period=(cfg.baseline_start, cfg.baseline_end),
                future_window_years=cfg.future_window_years,
            )
            d = delta_at_points(site_lons, site_lats, ens)
            print(f'    {label}: mean Δ = {np.nanmean(d):+.2f} °C '
                  f'± {np.nanstd(d):.2f} (across sites)')
            T_profiles += d[None, :]
            T_design += d[None, :]
        # Note: T_*_profiles / T_*_design were modified in place above.

    # Stuff per-site siting attributes into inputs aligned with id_sites so
    # economics/costs.py can apply hazard multipliers. Always populated (zeros
    # when siting is disabled) to keep the cost path uniform.
    ids_flat = np.squeeze(id_sites).astype(np.int64).ravel()
    if siting_active and len(ids_flat):
        attr_lookup = sites_df.set_index('id')
        def _col(name):
            if name in attr_lookup.columns:
                return attr_lookup.loc[ids_flat, name].to_numpy(dtype=np.float64).reshape(1, -1)
            return np.zeros((1, len(ids_flat)), dtype=np.float64)
        inputs['ais_density_pct'] = _col('ais_density_pct')
        inputs['pga_475'] = _col('pga_475')
        inputs['cyclone_freq_per_yr'] = _col('cyclone_freq_per_yr')
    else:
        inputs['ais_density_pct'] = np.zeros((1, len(ids_flat)), dtype=np.float64)
        inputs['pga_475'] = np.zeros((1, len(ids_flat)), dtype=np.float64)
        inputs['cyclone_freq_per_yr'] = np.zeros((1, len(ids_flat)), dtype=np.float64)

    otec_plants, capex_opex_comparison = off_design_analysis(
        T_WW_design, T_CW_design, T_WW_profiles, T_CW_profiles,
        inputs, coordinates_CW, timestamp, studied_region, run_dir + os.sep, cost_level,
    )

    # ── Multi-year aggregations ────────────────────────────────────────
    # Always compute per-year mean power and annual energy. For multi-year
    # runs we additionally recompute LCOE via the NPV formulation so that
    # degradation, leap years, and OPEX escalation are reflected. Single-
    # year runs keep the legacy LCOE from off_design_analysis.
    from otex.economics.timeseries import (
        aggregate_p_net_by_year, annual_energy_kwh,
    )

    p_net_by_year, sim_years = aggregate_p_net_by_year(
        otec_plants['p_net'], timestamp
    )
    annual_energy_MWh = (
        annual_energy_kwh(
            p_net_by_year, sim_years, inputs['availability_factor']
        ) / 1000.0
    )  # shape (n_years, n_sites)

    if inputs['n_years'] > 1:
        from otex.economics.costs import lcoe_npv
        # Preserve the legacy single-year LCOE for transparency/comparison.
        otec_plants['LCOE_legacy'] = otec_plants['LCOE']
        otec_plants['LCOE'] = lcoe_npv(
            otec_plants, inputs, p_net_by_year, sim_years
        )
        otec_plants['p_net_by_year'] = p_net_by_year
        otec_plants['annual_energy_MWh'] = annual_energy_MWh
        otec_plants['simulated_years'] = sim_years

    sites = pd.DataFrame()
    sites.index = np.squeeze(id_sites)
    sites['longitude'] = coordinates_CW[:, 0]
    sites['latitude'] = coordinates_CW[:, 1]
    sites['p_net_nom'] = -otec_plants['p_net_nom'].T / 1000
    # AEP: lifetime-average annual energy (MWh). For multi-year runs this
    # is the mean of per-year energies; for single-year runs it equals the
    # year's energy. Leap years are accounted for via annual_energy_MWh.
    sites['AEP'] = annual_energy_MWh.mean(axis=0)
    sites['CAPEX'] = otec_plants['CAPEX'].T / 1_000_000
    sites['LCOE'] = otec_plants['LCOE'].T
    if inputs['n_years'] > 1:
        sites['LCOE_legacy'] = otec_plants['LCOE_legacy'].T
        # Inter-annual variability of AEP (MWh).
        sites['AEP_min']  = annual_energy_MWh.min(axis=0)
        sites['AEP_p50']  = np.median(annual_energy_MWh, axis=0)
        sites['AEP_max']  = annual_energy_MWh.max(axis=0)
        sites['AEP_std']  = annual_energy_MWh.std(axis=0, ddof=0)
    sites['Configuration'] = otec_plants['Configuration'].T
    sites['T_WW_min'] = T_WW_design[0, :]
    sites['T_WW_med'] = T_WW_design[1, :]
    sites['T_WW_max'] = T_WW_design[2, :]
    sites['T_CW_min'] = T_CW_design[2, :]
    sites['T_CW_med'] = T_CW_design[1, :]
    sites['T_CW_max'] = T_CW_design[0, :]
    sites = sites.dropna(axis='rows')

    p_net_profile = pd.DataFrame(
        np.mean(otec_plants['p_net'], axis=1), columns=['p_net'], index=timestamp
    )

    p_gross_val = inputs['p_gross']
    sites.to_csv(
        os.path.join(run_dir, f'OTEC_sites_{studied_region}_{year_str}{climate_suffix}_{-p_gross_val/1000}_MW_{cost_level}.csv'.replace(" ", "_")),
        index=True, index_label='id', float_format='%.3f', sep=';',
    )
    p_net_profile.to_csv(
        os.path.join(run_dir, f'net_power_profiles_per_day_{studied_region}_{year_str}{climate_suffix}_{-p_gross_val/1000}_MW_{cost_level}.csv'.replace(" ", "_")),
        index=True, sep=';',
    )

    # Multi-year per-year breakdown: one row per (site, year). Only emitted
    # when the run actually spans more than one year.
    if inputs['n_years'] > 1:
        valid_ids = sites.index.to_numpy()
        valid_mask = np.isin(np.squeeze(id_sites).astype(np.int64), valid_ids)
        per_year_rows = []
        for yi, year in enumerate(sim_years):
            for si, site_id in enumerate(np.squeeze(id_sites).astype(np.int64)):
                if not valid_mask[si]:
                    continue
                per_year_rows.append({
                    'id': int(site_id),
                    'year': int(year),
                    'p_net_mean_kW': -p_net_by_year[yi, si],
                    'AEP_MWh': annual_energy_MWh[yi, si],
                })
        per_year_df = pd.DataFrame(per_year_rows)
        per_year_df.to_csv(
            os.path.join(
                run_dir,
                f'OTEC_sites_yearly_{studied_region}_{year_str}{climate_suffix}_{-p_gross_val/1000}_MW_{cost_level}.csv'.replace(" ", "_"),
            ),
            index=False, float_format='%.3f', sep=';',
        )

    end = time.time()
    print(f'Total runtime: {(end - start) / 60:.2f} minutes.')

    return otec_plants, sites


def main():
    """CLI entry point for ``otex-regional`` command."""
    import argparse

    parser = argparse.ArgumentParser(
        description='OTEX Regional Analysis - Generate spatially resolved OTEC power profiles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  otex-regional Philippines
  otex-regional Philippines --power -100000 --year 2021
  otex-regional Philippines --year-start 2020 --year-end 2022
  otex-regional Jamaica --cycle kalina --cost high_cost
  otex-regional Jamaica --data-source HYCOM --year 2020
  otex-regional Hawaii --cycle rankine_closed --fluid r134a
        ''',
    )

    parser.add_argument('region', nargs='?', default=None,
                        help='Region to analyze (use otex.data.load_regions() for list)')
    parser.add_argument('--power', '-p', type=int, default=-100000,
                        help='Gross power output in kW (negative, default: -100000)')
    parser.add_argument('--cost', '-c', choices=['low_cost', 'high_cost'], default='low_cost',
                        help='Cost level (default: low_cost)')
    parser.add_argument('--year', '-y', type=int, default=None,
                        help='Single year for analysis (deprecated; use --year-start/--year-end)')
    parser.add_argument('--year-start', type=int, default=None,
                        help='First simulated year, inclusive (defaults to 2020)')
    parser.add_argument('--year-end', type=int, default=None,
                        help='Last simulated year, inclusive (defaults to year-start)')
    parser.add_argument('--cycle', choices=['rankine_closed', 'rankine_open', 'rankine_hybrid', 'kalina', 'uehara'],
                        default='rankine_closed',
                        help='Thermodynamic cycle (default: rankine_closed)')
    parser.add_argument('--fluid', choices=['ammonia', 'r134a', 'r245fa', 'propane', 'isobutane'],
                        default='ammonia',
                        help='Working fluid (default: ammonia)')
    parser.add_argument('--data-source', '-d', choices=['CMEMS', 'HYCOM'], default='CMEMS',
                        help='Oceanographic data source (default: CMEMS)')
    parser.add_argument('--no-coolprop', action='store_true',
                        help='Disable CoolProp (use polynomial correlations)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: ./Data_Results/)')
    # CMIP6 climate-scenario flags (0.3.0+).
    parser.add_argument('--climate-scenario',
                        choices=['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585'],
                        default='historical',
                        help='CMIP6 scenario for delta-method downscaling '
                             '(default: historical = no delta).')
    parser.add_argument('--climate-year', type=int, default=None,
                        help='Target year for climate delta '
                             '(e.g. 2050). Required when --climate-scenario != historical.')
    parser.add_argument('--climate-models', nargs='+', default=None,
                        help='CMIP6 GCM names to ensemble '
                             '(default: MPI-ESM1-2-LR EC-Earth3 CanESM5).')

    args = parser.parse_args()
    if args.climate_scenario != 'historical' and args.climate_year is None:
        parser.error('--climate-year is required when --climate-scenario != historical')

    if args.region is None:
        print('++ Setting up seawater temperature data download ++\n')
        args.region = input('Enter the region to be analysed: ')

    if args.year is None and args.year_start is None:
        # Preserve historical default for unflagged invocations.
        args.year_start = 2020
    year_label = (
        str(args.year) if args.year is not None
        else (f'{args.year_start}-{args.year_end}' if args.year_end and args.year_end != args.year_start
              else str(args.year_start))
    )

    print(f'\n++ OTEX Regional Analysis ++')
    print(f'Region: {args.region}')
    print(f'Power: {args.power} kW ({-args.power/1000:.1f} MW)')
    print(f'Years: {year_label}')
    print(f'Cycle: {args.cycle}')
    print(f'Fluid: {args.fluid}')
    print(f'Cost level: {args.cost}')
    print(f'Data source: {args.data_source}')
    print(f'CoolProp: {not args.no_coolprop}')
    if args.climate_scenario != 'historical':
        models = args.climate_models or 'default ensemble'
        print(f'Climate: {args.climate_scenario} @ {args.climate_year} ({models})')
    print()

    run_regional_analysis(
        studied_region=args.region,
        p_gross=args.power,
        cost_level=args.cost,
        year=args.year,
        year_start=args.year_start,
        year_end=args.year_end,
        cycle_type=args.cycle,
        fluid_type=args.fluid,
        use_coolprop=not args.no_coolprop,
        output_dir=args.output_dir,
        data_source=args.data_source,
        climate_scenario=args.climate_scenario,
        climate_year=args.climate_year,
        climate_models=args.climate_models,
    )


if __name__ == "__main__":
    main()
