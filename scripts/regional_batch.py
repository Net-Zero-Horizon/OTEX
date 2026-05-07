# -*- coding: utf-8 -*-
"""
OTEX Regional Batch Analysis
Run OTEC analysis for multiple regions sequentially.

This script is a thin batch driver around
``otex.regional.run_regional_analysis``. The single-region pipeline
lives in the package; this module only adds the iteration loop and
optional demand-driven plant sizing.
"""

import os
import time

import numpy as np
import pandas as pd

from otex.regional import run_regional_analysis


def run_region(
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
):
    """
    Run OTEC analysis for a single region.

    Thin wrapper around :func:`otex.regional.run_regional_analysis`.
    Kept for backward compatibility with code that imports this script.

    Args:
        studied_region: Region name
        p_gross: Gross power output in kW (negative)
        cost_level: 'low_cost' or 'high_cost'
        year: Single year (deprecated; use year_start/year_end).
        year_start: First simulated year, inclusive (default 2020).
        year_end: Last simulated year, inclusive (default year_start).
        cycle_type: Thermodynamic cycle type
        fluid_type: Working fluid type
        use_coolprop: Whether to use CoolProp
        output_dir: Output base directory (default: ./Data_Results/)

    Returns:
        tuple: (otec_plants dict, sites DataFrame)
    """
    return run_regional_analysis(
        studied_region=studied_region,
        p_gross=p_gross,
        cost_level=cost_level,
        year=year,
        year_start=year_start,
        year_end=year_end,
        cycle_type=cycle_type,
        fluid_type=fluid_type,
        use_coolprop=use_coolprop,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='OTEX Regional Batch Analysis - Run OTEC analysis for multiple regions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python regional_batch.py
  python regional_batch.py --year 2021 --cost high_cost
  python regional_batch.py --year-start 2020 --year-end 2022
  python regional_batch.py --cycle kalina --regions Philippines Jamaica Hawaii
  python regional_batch.py --max-power -50000
        '''
    )

    parser.add_argument('--regions', nargs='+', default=None,
                        help='Specific regions to analyze (default: all from CSV)')
    parser.add_argument('--max-power', type=int, default=-100000,
                        help='Maximum gross power in kW (caps demand-based sizing, default: -100000)')
    parser.add_argument('--cost', '-c', choices=['low_cost', 'high_cost'], default='low_cost',
                        help='Cost level (default: low_cost)')
    parser.add_argument('--year', '-y', type=int, default=None,
                        help='Single year (deprecated; use --year-start/--year-end)')
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
    parser.add_argument('--no-coolprop', action='store_true',
                        help='Disable CoolProp (use polynomial correlations)')

    args = parser.parse_args()

    if args.year is None and args.year_start is None:
        args.year_start = 2020
    year_label = (
        str(args.year) if args.year is not None
        else (f'{args.year_start}-{args.year_end}'
              if args.year_end and args.year_end != args.year_start
              else str(args.year_start))
    )

    print(f'\n++ OTEX Regional Batch Analysis ++')
    print(f'Years: {year_label}')
    print(f'Cycle: {args.cycle}')
    print(f'Fluid: {args.fluid}')
    print(f'Cost level: {args.cost}')
    print(f'Max power: {args.max_power} kW')
    print(f'CoolProp: {not args.no_coolprop}\n')

    # Resolve the list of regions to process. With --regions, the user
    # supplies an explicit list; otherwise we iterate every Natural Earth
    # admin-0 entry and let demand-based sizing decide which to skip.
    if args.regions:
        regions = args.regions
    else:
        from otex.data import list_regions
        regions = list_regions()

    print(f'Processing {len(regions)} regions...\n')

    # Demand is resolved on-demand (since 0.2.0) via OTEX's multi-source
    # provider. The first call within a process downloads OWID's bulk
    # CSV (~9 MB, cached locally) so subsequent lookups are sub-ms.
    from otex.data.demand import fetch_demand_TWh
    from otex.data.regions import resolve_region

    for index, region in enumerate(regions):
        studied_region = region

        # Determine power based on demand (if available)
        if args.regions:
            # Explicit regions: use max-power.
            p_gross = args.max_power
        else:
            try:
                resolved = resolve_region(studied_region)
            except ValueError:
                print(f'Skipping {studied_region}: not found in Natural Earth')
                continue
            twh, src, yr = fetch_demand_TWh(resolved.iso_a3) if resolved.iso_a3 else (None, None, None)
            if twh is None or twh == 0:
                print(f'Skipping {studied_region}: no demand data from any provider')
                continue
            # Convert TWh/yr → average kW (negative = output convention).
            avg_kw = -twh * 1.0e9 / 8760.0
            if avg_kw < args.max_power:
                p_gross = args.max_power
            else:
                p_gross = int(avg_kw)
            print(f'  [{studied_region}] demand={twh:.2f} TWh ({src}, {yr}) → P_gross={p_gross} kW')

        print(f'\n=== {region} (P_gross={p_gross} kW) ===')
        run_region(
            studied_region=studied_region,
            p_gross=p_gross,
            cost_level=args.cost,
            year=args.year,
            year_start=args.year_start,
            year_end=args.year_end,
            cycle_type=args.cycle,
            fluid_type=args.fluid,
            use_coolprop=not args.no_coolprop
        )
