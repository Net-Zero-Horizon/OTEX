# -*- coding: utf-8 -*-
"""
OTEX Regional Batch Analysis
Run OTEC analysis for multiple regions sequentially.

@author: OTEX Development Team
"""

import os
import time
import pandas as pd
import numpy as np
import platform

from otex.config import parameters_and_constants
from otex.plant.off_design_analysis import off_design_analysis
from otex.data.cmems import download_data, data_processing, load_temperatures


def run_region(
    studied_region,
    p_gross=-100000,
    cost_level='low_cost',
    year=None,
    year_start=None,
    year_end=None,
    cycle_type='rankine_closed',
    fluid_type='ammonia',
    use_coolprop=True
):
    """
    Run OTEC analysis for a single region.

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

    Returns:
        tuple: (otec_plants dict, sites_df DataFrame)
    """
    if year is None and year_start is None:
        year_start = 2020
    start = time.time()
    parent_dir = os.getcwd() + 'Data_Results/'

    if platform.system() == 'Windows':
        new_path = os.path.join(parent_dir,f'{studied_region}\\'.replace(" ","_"))
    else :
        new_path = os.path.join(parent_dir,f'{studied_region}/'.replace(" ","_"))

    if os.path.isdir(new_path):
        pass
    else:
        os.mkdir(new_path)

    inputs = parameters_and_constants(
        p_gross=p_gross,
        cost_level=cost_level,
        data='CMEMS',
        fluid_type=fluid_type,
        cycle_type=cycle_type,
        use_coolprop=use_coolprop,
        year=year,
        year_start=year_start,
        year_end=year_end,
    )
    year_str = inputs['year_label']
    
    # if os.path.isfile(new_path+f'net_power_profiles_{studied_region}_{year}__{-p_gross/1000}_MW_{cost_level}.csv'.replace(" ","_")):
    #     print(f'{studied_region} already analysed.')
    # else:
  
        
    depth_WW = inputs['length_WW_inlet']
    depth_CW = inputs['length_CW_inlet']
      
    files = download_data(cost_level,inputs,studied_region,new_path)
    
    print('\n++ Processing seawater temperature data ++\n')    
    
    sites_df = pd.read_csv('CMEMS_points_with_properties.csv',delimiter=';',encoding='latin-1')
    sites_df = sites_df[(sites_df['region']==studied_region) & (sites_df['water_depth'] <= inputs['min_depth']) & (sites_df['water_depth'] >= inputs['max_depth'])]   
    sites_df = sites_df.sort_values(by=['longitude','latitude'],ascending=True)
    
    h5_file_WW = os.path.join(new_path, f'T_{round(depth_WW,0)}m_{year_str}_{studied_region}.h5'.replace(" ","_"))
    h5_file_CW = os.path.join(new_path, f'T_{round(depth_CW,0)}m_{year_str}_{studied_region}.h5'.replace(" ","_"))
    
    if os.path.isfile(h5_file_CW):
        T_CW_profiles, T_CW_design, coordinates_CW, id_sites, timestamp, inputs, nan_columns_CW = load_temperatures(h5_file_CW, inputs)
        print(f'{h5_file_CW} already exist. No processing necessary.')
    else:
        T_CW_profiles, T_CW_design, coordinates_CW, id_sites, timestamp, inputs, nan_columns_CW = data_processing(files[int(len(files)/2):int(len(files))],sites_df,inputs,studied_region,new_path,'CW')
    
    if os.path.isfile(h5_file_WW):
        T_WW_profiles, T_WW_design, coordinates_WW, id_sites, timestamp, inputs, nan_columns_WW = load_temperatures(h5_file_WW, inputs)
        print(f'{h5_file_WW} already exist. No processing necessary.')
    else:
        T_WW_profiles, T_WW_design, coordinates_WW, id_sites, timestamp, inputs, nan_columns_WW = data_processing(files[0:int(len(files)/2)],sites_df,inputs,studied_region,new_path,'WW',nan_columns_CW)
         
    otec_plants = off_design_analysis(T_WW_design,T_CW_design,T_WW_profiles,T_CW_profiles,inputs,coordinates_CW,timestamp,studied_region,new_path,cost_level)  
    
    sites = pd.DataFrame()
    sites.index = np.squeeze(id_sites)
    sites['longitude'] = coordinates_CW[:,0]
    sites['latitude'] = coordinates_CW[:,1]
    sites['p_net_nom'] = -otec_plants['p_net_nom'].T/1000

    # Multi-year-aware AEP and LCOE: per-year aggregation, NPV LCOE when
    # n_years > 1, legacy single-year behaviour otherwise.
    from otex.economics.timeseries import aggregate_p_net_by_year, annual_energy_kwh
    p_net_by_year, sim_years = aggregate_p_net_by_year(otec_plants['p_net'], timestamp)
    annual_energy_MWh = annual_energy_kwh(
        p_net_by_year, sim_years, inputs['availability_factor']
    ) / 1000.0
    sites['AEP'] = annual_energy_MWh.mean(axis=0)
    if inputs['n_years'] > 1:
        from otex.economics.costs import lcoe_npv
        otec_plants['LCOE_legacy'] = otec_plants['LCOE']
        otec_plants['LCOE'] = lcoe_npv(otec_plants, inputs, p_net_by_year, sim_years)

    sites['CAPEX'] = otec_plants['CAPEX'].T/1000000
    sites['LCOE'] = otec_plants['LCOE'].T
    if inputs['n_years'] > 1:
        sites['LCOE_legacy'] = otec_plants['LCOE_legacy'].T
        sites['AEP_min'] = annual_energy_MWh.min(axis=0)
        sites['AEP_p50'] = np.median(annual_energy_MWh, axis=0)
        sites['AEP_max'] = annual_energy_MWh.max(axis=0)
        sites['AEP_std'] = annual_energy_MWh.std(axis=0, ddof=0)
    sites['Configuration'] = otec_plants['Configuration'].T
    sites['T_WW_min'] = T_WW_design[0,:]
    sites['T_WW_med'] = T_WW_design[1,:]
    sites['T_WW_max'] = T_WW_design[2,:]
    sites['T_CW_min'] = T_CW_design[2,:]
    sites['T_CW_med'] = T_CW_design[1,:]
    sites['T_CW_max'] = T_CW_design[0,:]
    
    sites = sites.dropna(axis='rows')

    p_net_profile = pd.DataFrame(np.mean(otec_plants['p_net'],axis=1),columns=['p_net'],index=timestamp)
    
    p_gross = inputs['p_gross']
    
    sites.to_csv(new_path + f'OTEC_sites_{studied_region}_{year_str}_{-p_gross/1000}_MW_{cost_level}.csv'.replace(" ","_"),index=True, index_label='id',float_format='%.3f')
    p_net_profile.to_csv(new_path + f'net_power_profiles_{studied_region}_{year_str}__{-p_gross/1000}_MW_{cost_level}.csv'.replace(" ","_"),index=True)
    
    end = time.time()
    print('Total runtime: ' + str(round((end-start)/60,2)) + ' minutes.')
    
    return otec_plants, sites_df

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
