#%%
# -*- coding: utf-8 -*-
"""
OTEX Regional Analysis

Generate spatially and temporally resolved power generation profiles
for any technically feasible OTEC system size and region.

Please refer to the paper "The global economic potential of ocean thermal
energy conversion" by Langer et al. (2023) for further details.

@author: OTEX Development Team
"""

import os
import time
import pandas as pd
import numpy as np
import platform

from otex.config import parameters_and_constants
from otex.data.cmems import download_data, data_processing, load_temperatures
from otex.plant.off_design_analysis import off_design_analysis




## Main function for regional OTEC analysis. Inputs: studied_region, gross power output, and cost level.
## Please scroll down to the bottom for further details on the inputs.

def run_regional_analysis(
    studied_region,
    p_gross=-136000,
    cost_level='low_cost',
    year=2020,
    cycle_type='rankine_closed',
    fluid_type='ammonia',
    use_coolprop=True
):
    """
    Run regional OTEC analysis for a specified region.

    Args:
        studied_region: Region name (must match entries in download_ranges_per_region.csv)
        p_gross: Gross power output in kW (negative, e.g., -136000 for 136 MW)
        cost_level: 'low_cost' or 'high_cost'
        year: Year for analysis (default: 2020)
        cycle_type: Thermodynamic cycle ('rankine_closed', 'rankine_open', 'kalina', 'uehara')
        fluid_type: Working fluid ('ammonia', 'r134a', 'r245fa', 'propane', 'isobutane')
        use_coolprop: Whether to use CoolProp for fluid properties

    Returns:
        tuple: (otec_plants dict, sites_df DataFrame)
    """
    start = time.time()
    parent_dir = os.getcwd() + '/Data_Results/'
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
    
    if platform.system() == 'Windows':
        dl_path = os.path.join(parent_dir,f'{studied_region}\\'.replace(" ","_"))
        new_path = dl_path + f'{studied_region}_{year_str}_{-p_gross/1000}_MW_{cost_level}\\'.replace(" ","_")
    else :
        dl_path = os.path.join(parent_dir,f'{studied_region}/'.replace(" ","_"))
        new_path = dl_path+ f'{studied_region}_{year_str}_{-p_gross/1000}_MW_{cost_level}/'.replace(" ","_")
    
    if os.path.isdir(new_path):
        pass
    else:
        os.makedirs(new_path)
        

        
    depth_WW = inputs['length_WW_inlet']
    depth_CW = inputs['length_CW_inlet']
      
    files = download_data(cost_level,inputs,studied_region,dl_path)
    
    print('\n++ Processing seawater temperature data ++\n')   
    
    from otex.data.resources import load_sites
    sites_df = load_sites()
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
         
    otec_plants,capex_opex_comparison = off_design_analysis(T_WW_design,T_CW_design,T_WW_profiles,T_CW_profiles,inputs,coordinates_CW,timestamp,studied_region,new_path,cost_level)  
    
    sites = pd.DataFrame()
    sites.index = np.squeeze(id_sites)
    sites['longitude'] = coordinates_CW[:,0]
    sites['latitude'] = coordinates_CW[:,1]
    sites['p_net_nom'] = -otec_plants['p_net_nom'].T/1000
    sites['AEP'] = -np.nanmean(otec_plants['p_net'],axis=0)*8760/1000000
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
   
    sites.to_csv(new_path + f'OTEC_sites_{studied_region}_{year_str}_{-p_gross/1000}_MW_{cost_level}.csv'.replace(" ","_"),index=True, index_label='id',float_format='%.3f',sep=';')
    p_net_profile.to_csv(new_path + f'net_power_profiles_per_day_{studied_region}_{year_str}_{-p_gross/1000}_MW_{cost_level}.csv'.replace(" ","_"),index=True,sep=';')
    
    ## Further analysis, credit: Lucas Vatinel
    ## functions CWP_details and eco_details have not been fully validated, use with caution!
        
    # pipe = pd.DataFrame()
    # pipe['d_pipes_CW']=otec_plants['d_pipes_CW']
    # pipe['num_pipes_CW']=otec_plants['num_pipes_CW']
    # pipe['m_pipes_CW']=otec_plants['m_pipes_CW']
    # pipe['A_pipes_CW']=otec_plants['A_pipes_CW']
    
    # pipe.to_csv(new_path + f'CWP_details_{studied_region}_{year}_{-p_gross/1000}_MW_{cost_level}.csv'.replace(" ","_"),index=True, index_label='id',float_format='%.3f',sep=';')   
   
    # p_net_per_location=np.mean(otec_plants['p_net'],axis=0)
    # pnet_lon_lat = np.array([coordinates_CW[:,0],coordinates_CW[:,1],p_net_per_location])
    # column_names = ['longitude','latitude','p_net']
    
    # # saves all the pnet profiles at all the locations ? here only the mean per location
    # all_pnet_df=pd.DataFrame(np.transpose(pnet_lon_lat))
    # all_pnet_df.columns=column_names
    
    # all_pnet_df.to_csv(new_path + f'net_power_profiles_per_location_{studied_region}_{year}_{-p_gross/1000}_MW_{cost_level}.csv'.replace(" ","_"),index=False,sep=';') 
    
    # cost_dict,best_LCOE_index = cp.plot_capex_opex(new_path,capex_opex_comparison,sites,p_gross,studied_region)
    # #enregistrer ce résultat afin qu'on puisse l'utiliser pour comparer les LCOE etc pour différentes puissances ou différentes hypothèses de calculs (ex: épaisseur de tuyau)
    # eco = pd.DataFrame.from_dict(cost_dict)
    # # print(eco)
    # eco.to_csv(new_path + f'eco_details_{studied_region}_{year}_pos_{best_LCOE_index}_index_{-p_gross/1000}_MW_{cost_level}.csv'.replace(" ","_"),index=True, index_label='Configuration',float_format='%.3f',sep=';')
        
    # co.extract_costs_at_study_location(sites,capex_opex_comparison,user_lon=55.25,user_lat=-20.833)s
    
    end = time.time()
    print('Total runtime: ' + str(round((end-start)/60,2)) + ' minutes.')
    
    return otec_plants,sites_df # add cost_dict and capex_opex_comparison here if further analysis above is conducted

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='OTEX Regional Analysis - Generate spatially resolved OTEC power profiles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python regional_analysis.py Philippines
  python regional_analysis.py Philippines --power -136000 --year 2021
  python regional_analysis.py Jamaica --cycle kalina --cost high_cost
  python regional_analysis.py Hawaii --cycle rankine_closed --fluid r134a
        '''
    )

    parser.add_argument('region', nargs='?', default=None,
                        help='Region to analyze (from download_ranges_per_region.csv)')
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

    args = parser.parse_args()

    # Interactive fallback if no region specified
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

    print("If you are asked for Copernicus credentials, link your PC with your account.")
    print("See README.md for details.\n")

    otec_plants, sites = run_regional_analysis(
        studied_region=args.region,
        p_gross=args.power,
        cost_level=args.cost,
        year=args.year,
        cycle_type=args.cycle,
        fluid_type=args.fluid,
        use_coolprop=not args.no_coolprop
    )


