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
    p_gross=-136000,
    cost_level='low_cost',
    year=2020,
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
        year: Year for analysis
        cycle_type: Thermodynamic cycle type
        fluid_type: Working fluid type
        use_coolprop: Whether to use CoolProp

    Returns:
        tuple: (otec_plants dict, sites_df DataFrame)
    """
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
        year=year
    )
    year_str = str(year)  
    
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
    sites['AEP'] = -np.mean(otec_plants['p_net'],axis=0)*8760/1000000
    sites['CAPEX'] = otec_plants['CAPEX'].T/1000000
    sites['LCOE'] = otec_plants['LCOE'].T
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
  python regional_batch.py --cycle kalina --regions Philippines Jamaica Hawaii
  python regional_batch.py --max-power -50000
        '''
    )

    parser.add_argument('--regions', nargs='+', default=None,
                        help='Specific regions to analyze (default: all from CSV)')
    parser.add_argument('--max-power', type=int, default=-136000,
                        help='Maximum gross power in kW (caps demand-based sizing, default: -136000)')
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
    parser.add_argument('--csv', default='HYCOM_download_ranges_per_region.csv',
                        help='CSV file with regions and demand data')

    args = parser.parse_args()

    print(f'\n++ OTEX Regional Batch Analysis ++')
    print(f'Year: {args.year}')
    print(f'Cycle: {args.cycle}')
    print(f'Fluid: {args.fluid}')
    print(f'Cost level: {args.cost}')
    print(f'Max power: {args.max_power} kW')
    print(f'CoolProp: {not args.no_coolprop}\n')

    # Load regions from CSV
    unique_regions = pd.read_csv(args.csv, delimiter=';', encoding='latin-1').drop_duplicates(subset=['region'])

    if args.regions:
        # Filter to specified regions
        regions = args.regions
        demand = [None] * len(regions)  # No demand data for explicit regions
    else:
        regions = list(unique_regions['region'])
        demand = list(unique_regions['demand'])

    print(f'Processing {len(regions)} regions...\n')

    for index, region in enumerate(regions):
        studied_region = region

        # Determine power based on demand (if available)
        if args.regions:
            # Explicit regions: use max-power
            p_gross = args.max_power
        elif np.isnan(demand[index]) or demand[index] == 0:
            print(f'Skipping {region}: no demand data')
            continue
        elif -demand[index] * 1000000000 / 8760 < args.max_power:
            p_gross = args.max_power
        else:
            p_gross = int(-demand[index] * 1000000000 / 8760)

        print(f'\n=== {region} (P_gross={p_gross} kW) ===')
        run_region(
            studied_region=studied_region,
            p_gross=p_gross,
            cost_level=args.cost,
            year=args.year,
            cycle_type=args.cycle,
            fluid_type=args.fluid,
            use_coolprop=not args.no_coolprop
        )
