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
from otex.data.cmems import download_data, data_processing, load_temperatures
from otex.data.resources import load_sites
from otex.plant.off_design_analysis import off_design_analysis


def run_regional_analysis(
    studied_region,
    p_gross=-136000,
    cost_level='low_cost',
    year=2020,
    cycle_type='rankine_closed',
    fluid_type='ammonia',
    use_coolprop=True,
    output_dir=None,
):
    """
    Run regional OTEC analysis for a specified region.

    Downloads CMEMS oceanographic data (if not cached), sizes OTEC plants
    at each feasible site, performs off-design analysis, and calculates
    LCOE for the entire region.

    Args:
        studied_region: Region name (e.g., 'Jamaica', 'Philippines').
            Must match an entry in the bundled regions database.
            Use ``otex.data.load_regions()`` to list available regions.
        p_gross: Gross power output in kW (negative, e.g., -136000 for 136 MW).
        cost_level: Cost assumption, either ``'low_cost'`` or ``'high_cost'``.
        year: Year for analysis (default: 2020).
        cycle_type: Thermodynamic cycle. One of ``'rankine_closed'``,
            ``'rankine_open'``, ``'rankine_hybrid'``, ``'kalina'``, ``'uehara'``.
        fluid_type: Working fluid. One of ``'ammonia'``, ``'r134a'``,
            ``'r245fa'``, ``'propane'``, ``'isobutane'``.
        use_coolprop: Whether to use CoolProp for fluid properties (default True).
        output_dir: Directory for results. Defaults to ``./Data_Results/``.

    Returns:
        tuple: ``(otec_plants, sites_df)`` where ``otec_plants`` is a dict
        with plant performance arrays and ``sites_df`` is a DataFrame
        with per-site results.
    """
    start = time.time()

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "Data_Results")

    inputs = parameters_and_constants(
        p_gross=p_gross,
        cost_level=cost_level,
        data='CMEMS',
        fluid_type=fluid_type,
        cycle_type=cycle_type,
        use_coolprop=use_coolprop,
        year=year,
    )
    year_str = str(year)

    region_dir = os.path.join(output_dir, studied_region.replace(" ", "_"))
    run_dir = os.path.join(
        region_dir,
        f"{studied_region}_{year_str}_{-p_gross/1000}_MW_{cost_level}".replace(" ", "_"),
    )
    os.makedirs(run_dir, exist_ok=True)

    depth_WW = inputs['length_WW_inlet']
    depth_CW = inputs['length_CW_inlet']

    files = download_data(cost_level, inputs, studied_region, region_dir + os.sep)

    print('\n++ Processing seawater temperature data ++\n')

    sites_df = load_sites()
    sites_df = sites_df[
        (sites_df['region'] == studied_region)
        & (sites_df['water_depth'] <= inputs['min_depth'])
        & (sites_df['water_depth'] >= inputs['max_depth'])
    ]
    sites_df = sites_df.sort_values(by=['longitude', 'latitude'], ascending=True)

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

    otec_plants, capex_opex_comparison = off_design_analysis(
        T_WW_design, T_CW_design, T_WW_profiles, T_CW_profiles,
        inputs, coordinates_CW, timestamp, studied_region, run_dir + os.sep, cost_level,
    )

    sites = pd.DataFrame()
    sites.index = np.squeeze(id_sites)
    sites['longitude'] = coordinates_CW[:, 0]
    sites['latitude'] = coordinates_CW[:, 1]
    sites['p_net_nom'] = -otec_plants['p_net_nom'].T / 1000
    sites['AEP'] = -np.nanmean(otec_plants['p_net'], axis=0) * 8760 / 1_000_000
    sites['CAPEX'] = otec_plants['CAPEX'].T / 1_000_000
    sites['LCOE'] = otec_plants['LCOE'].T
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
        os.path.join(run_dir, f'OTEC_sites_{studied_region}_{year_str}_{-p_gross_val/1000}_MW_{cost_level}.csv'.replace(" ", "_")),
        index=True, index_label='id', float_format='%.3f', sep=';',
    )
    p_net_profile.to_csv(
        os.path.join(run_dir, f'net_power_profiles_per_day_{studied_region}_{year_str}_{-p_gross_val/1000}_MW_{cost_level}.csv'.replace(" ", "_")),
        index=True, sep=';',
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
  otex-regional Philippines --power -136000 --year 2021
  otex-regional Jamaica --cycle kalina --cost high_cost
  otex-regional Hawaii --cycle rankine_closed --fluid r134a
        ''',
    )

    parser.add_argument('region', nargs='?', default=None,
                        help='Region to analyze (use otex.data.load_regions() for list)')
    parser.add_argument('--power', '-p', type=int, default=-136000,
                        help='Gross power output in kW (negative, default: -136000)')
    parser.add_argument('--cost', '-c', choices=['low_cost', 'high_cost'], default='low_cost',
                        help='Cost level (default: low_cost)')
    parser.add_argument('--year', '-y', type=int, default=2020,
                        help='Year for analysis (default: 2020)')
    parser.add_argument('--cycle', choices=['rankine_closed', 'rankine_open', 'rankine_hybrid', 'kalina', 'uehara'],
                        default='rankine_closed',
                        help='Thermodynamic cycle (default: rankine_closed)')
    parser.add_argument('--fluid', choices=['ammonia', 'r134a', 'r245fa', 'propane', 'isobutane'],
                        default='ammonia',
                        help='Working fluid (default: ammonia)')
    parser.add_argument('--no-coolprop', action='store_true',
                        help='Disable CoolProp (use polynomial correlations)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: ./Data_Results/)')

    args = parser.parse_args()

    if args.region is None:
        print('++ Setting up seawater temperature data download ++\n')
        args.region = input('Enter the region to be analysed: ')

    print(f'\n++ OTEX Regional Analysis ++')
    print(f'Region: {args.region}')
    print(f'Power: {args.power} kW ({-args.power/1000:.1f} MW)')
    print(f'Year: {args.year}')
    print(f'Cycle: {args.cycle}')
    print(f'Fluid: {args.fluid}')
    print(f'Cost level: {args.cost}')
    print(f'CoolProp: {not args.no_coolprop}\n')

    run_regional_analysis(
        studied_region=args.region,
        p_gross=args.power,
        cost_level=args.cost,
        year=args.year,
        cycle_type=args.cycle,
        fluid_type=args.fluid,
        use_coolprop=not args.no_coolprop,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
