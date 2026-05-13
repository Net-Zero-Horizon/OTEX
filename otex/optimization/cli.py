# -*- coding: utf-8 -*-
"""``otex-regional-optimal`` CLI.

A second mode of operation parallel to ``otex-regional``: instead of
asking *"what's the LCOE if I install p_gross MW at each site?"*, this
command answers *"what plant design minimises LCOE at each site?"*

The categorical configuration (cycle, fluid, installation type) is
specified exogenously — to compare alternatives, run the command
several times with each combination and merge the resulting CSVs.
"""

from __future__ import annotations

import argparse
import os
import sys

from .design_vector import Bounds, DEFAULT_BOUNDS
from .optimize import run_regional_optimization
from .user_constraints import UserConstraints


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='otex-regional-optimal',
        description='Per-site OTEC design optimisation (Mode Inverso). '
                    'Minimises LCOE over (p_gross, dT_WW, dT_CW, depth_CW) '
                    'subject to physical/technical constraints.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  otex-regional-optimal Jamaica
  otex-regional-optimal Jamaica --year-start 2020 --year-end 2023 --cycle kalina
  otex-regional-optimal Cuba --p-max-MW 150
        ''',
    )

    parser.add_argument('region', nargs='?', default=None,
                        help='Region name or ISO code (Natural Earth).')
    parser.add_argument('--cost', '-c',
                        choices=['low_cost', 'high_cost'],
                        default='low_cost')
    parser.add_argument('--year', '-y', type=int, default=None,
                        help='Single year (deprecated; use --year-start/--year-end).')
    parser.add_argument('--year-start', type=int, default=None)
    parser.add_argument('--year-end', type=int, default=None)
    parser.add_argument('--cycle',
                        choices=['rankine_closed', 'rankine_open',
                                 'rankine_hybrid', 'kalina', 'uehara'],
                        default='rankine_closed')
    parser.add_argument('--fluid',
                        choices=['ammonia', 'r134a', 'r245fa',
                                 'propane', 'isobutane'],
                        default='ammonia')
    parser.add_argument('--data-source', '-d',
                        choices=['CMEMS', 'HYCOM'], default='CMEMS')
    parser.add_argument('--no-coolprop', action='store_true')
    parser.add_argument('--output-dir', default=None)

    # Bounds on the design vector. Defaults match DEFAULT_BOUNDS.
    parser.add_argument('--p-min-MW', type=float, default=1.0,
                        help='Lower bound on plant size (MW). Default 1.')
    parser.add_argument('--p-max-MW', type=float, default=500.0,
                        help='Upper bound on plant size (MW). Default 500. '
                             'OTEX cost correlations are calibrated near '
                             '100 MW — set this to your maximum credible '
                             'plant size to avoid extrapolation.')
    parser.add_argument('--dT-min', type=float, default=1.0,
                        help='Lower bound on dT_WW, dT_CW (°C).')
    parser.add_argument('--dT-max', type=float, default=6.0,
                        help='Upper bound on dT_WW, dT_CW (°C).')
    parser.add_argument('--depth-min-m', type=float, default=600.0,
                        help='Lower bound on cold-water intake depth (m).')
    parser.add_argument('--depth-max-m', type=float, default=3000.0,
                        help='Upper bound on cold-water intake depth (m).')
    # Exogenous user constraints (path B). Set at least one of these
    # to produce a genuine interior optimum; without them OTEX's
    # internal physics has no hard upper limit on p_gross within the
    # plausible plant-size range, and the solver walks to the upper
    # box bound.
    user_grp = parser.add_argument_group(
        'user constraints',
        'Optional exogenous caps imposed by the decision-maker. '
        'At least one is recommended.',
    )
    user_grp.add_argument('--max-aep-MWh', type=float, default=None,
                          help='Max annual energy production per plant (MWh/yr).')
    user_grp.add_argument('--max-p-net-MW', type=float, default=None,
                          help='Max net delivered power (MW).')
    user_grp.add_argument('--max-capex-MUSD', type=float, default=None,
                          help='Max total CAPEX (millions USD).')
    user_grp.add_argument('--max-p-gross-MW', type=float, default=None,
                          help='Max gross plant size (MW). Hard constraint vs '
                               'the --p-max-MW box bound.')
    user_grp.add_argument('--max-parasitic-ratio', type=float, default=None,
                          help='Max P_pump/|P_gross| (e.g. 0.30).')

    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args(argv)

    if args.region is None:
        args.region = input('Region: ').strip()

    # Resolve year handling consistent with the forward CLI.
    year_start = args.year_start
    year_end = args.year_end
    if args.year is not None and year_start is None:
        year_start = args.year
        year_end = args.year
    if year_start is None:
        year_start = 2020
    if year_end is None:
        year_end = year_start

    bounds = Bounds(
        p_gross=(-args.p_max_MW * 1000.0, -args.p_min_MW * 1000.0),
        dT_WW=(args.dT_min, args.dT_max),
        dT_CW=(args.dT_min, args.dT_max),
        depth_CW=(args.depth_min_m, args.depth_max_m),
    )

    user_cons = UserConstraints(
        max_aep_MWh=args.max_aep_MWh,
        max_p_net_MW=args.max_p_net_MW,
        max_capex_MUSD=args.max_capex_MUSD,
        max_p_gross_MW=args.max_p_gross_MW,
        max_parasitic_ratio=args.max_parasitic_ratio,
    )
    if not user_cons.any_active:
        print('\nWARNING: no --max-* constraint set. The LCOE function is '
              'monotonically decreasing in p_gross over the bounded range, so '
              'the optimiser will simply hit the upper box bound. Set at least '
              'one user constraint (max-aep-MWh, max-p-net-MW, max-capex-MUSD, '
              'max-p-gross-MW, or max-parasitic-ratio) for a meaningful '
              'interior optimum.\n')

    print('\n++ OTEX Regional Optimization (Mode Inverso) ++')
    print(f'Region: {args.region}')
    print(f'Years: {year_start}-{year_end}')
    print(f'Cycle: {args.cycle} / Fluid: {args.fluid}')
    print(f'Cost level: {args.cost}')
    print(f'Power bounds: {args.p_min_MW:.0f}-{args.p_max_MW:.0f} MW')
    print(f'dT bounds: {args.dT_min:.1f}-{args.dT_max:.1f} °C')
    print(f'Depth bounds: {args.depth_min_m:.0f}-{args.depth_max_m:.0f} m\n')

    df = run_regional_optimization(
        studied_region=args.region,
        cost_level=args.cost,
        year_start=year_start, year_end=year_end,
        cycle_type=args.cycle, fluid_type=args.fluid,
        use_coolprop=not args.no_coolprop,
        data_source=args.data_source,
        bounds=bounds,
        user_constraints=user_cons,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    if args.output_dir is None:
        out_root = os.path.join(os.getcwd(), 'Data_Results')
    else:
        out_root = args.output_dir
    region_dir = os.path.join(out_root, args.region.replace(' ', '_'))
    year_label = (f'{year_start}-{year_end}' if year_end != year_start
                  else str(year_start))
    fname = f'OTEC_sites_optimal_{args.region}_{year_label}_{args.cost}.csv'
    out_path = os.path.join(region_dir, fname.replace(' ', '_'))
    df.to_csv(out_path, sep=';', index=False, float_format='%.4f')
    print(f'\nWrote {len(df)} optimised sites → {out_path}')

    # Headline summary
    feas = df[df['feasible']]
    if len(feas) > 0:
        best_idx = feas['lcoe_min'].idxmin()
        best = feas.loc[best_idx]
        print(f'\nBest site: ({best["longitude"]:.3f}, {best["latitude"]:.3f}) '
              f'→ LCOE {best["lcoe_min"]:.2f} ¢/kWh at '
              f'p_gross = {best["p_gross_opt_MW"]:.1f} MW '
              f'(dT_WW {best["dT_WW_opt"]:.1f}, dT_CW {best["dT_CW_opt"]:.1f}, '
              f'depth {best["depth_CW_opt"]:.0f} m)')
        print(f'Feasible sites: {len(feas)}/{len(df)} '
              f'({100*len(feas)/len(df):.0f}%)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
