# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 08:49:22 2023

@author: jkalanger
"""

import numpy as np

# NumPy 2.0 compatibility fix for pickled data
# Older HDF5 files may have been pickled with numpy.core (pre-2.0)
# but NumPy 2.0+ uses numpy._core
import sys
if not hasattr(np, 'core'):
    np.core = np._core

import pandas as pd
import os
import time

from .sizing import otec_sizing
from ..economics.costs import capex_opex_lcoe
from .operation import otec_operation

def safe_hdf_write(df, filepath, key, mode='a', max_retries=30, retry_delay=1.0):
    """
    Safely write to HDF5 file with retry logic for parallel execution

    Uses exponential backoff with jitter to handle concurrent access from multiple processes.

    Args:
        df: DataFrame to write
        filepath: Path to HDF5 file
        key: Key name in HDF5 file
        mode: Write mode ('a' for append, 'w' for write)
        max_retries: Maximum number of retry attempts (default: 30)
        retry_delay: Base delay between retries in seconds (default: 1.0)
    """
    import random

    for attempt in range(max_retries):
        try:
            df.to_hdf(filepath, key=key, mode=mode)
            if attempt > 0:
                print(f"  Success writing to HDF5 after {attempt + 1} attempts")
            return True
        except (OSError, BlockingIOError, PermissionError) as e:
            if attempt < max_retries - 1:
                # File is locked by another process, wait and retry
                # Exponential backoff: 1s, 2s, 4s, 8s, ... up to max 30s
                wait_time = min(retry_delay * (2 ** attempt), 30.0)
                # Add jitter (±20%) to prevent thundering herd
                jitter = random.uniform(0.8, 1.2)
                actual_wait = wait_time * jitter

                # Only show message every 5 attempts to reduce spam
                if attempt % 5 == 0 or attempt > 15:
                    print(f"  HDF5 file locked (attempt {attempt + 1}/{max_retries}), waiting {actual_wait:.1f}s...")

                time.sleep(actual_wait)
            else:
                # Final attempt failed
                print(f"ERROR: Failed to write to HDF5 after {max_retries} attempts ({max_retries * retry_delay:.0f}s)")
                print(f"  File: {filepath}")
                print(f"  Error: {e}")
                print(f"  This region's HDF5 data will be missing (but pickle/CSV results are preserved)")
                return False
        except Exception as e:
            # Other unexpected errors
            print(f"ERROR: Unexpected error writing to HDF5: {e}")
            print(f"  File: {filepath}")
            return False
    return False

def on_design_analysis(T_WW_in,T_CW_in,inputs,cost_level='low_cost'):
    
    # inputs = parameters_and_constants(cost_level)
   
    del_T_WW_min, \
    del_T_CW_min, \
    del_T_WW_max, \
    del_T_CW_max, \
    interval_WW, \
    interval_CW = inputs['del_T_for_looping']
    
    if T_WW_in.ndim == 0:
        lcoe_matrix_nominal = np.empty([int((del_T_WW_max-del_T_WW_min)/interval_WW+1),int((del_T_CW_max-del_T_CW_min)/interval_CW+1)],dtype=np.float64)
    else: 
        lcoe_matrix_nominal = np.empty([int((del_T_WW_max-del_T_WW_min)/interval_WW+1),int((del_T_CW_max-del_T_CW_min)/interval_CW+1),np.shape(T_WW_in)[0]],dtype=np.float64)
    
    for i in range(del_T_CW_min,(del_T_CW_max+interval_CW),interval_CW):
        for j in range(del_T_WW_min,del_T_WW_max+interval_WW,interval_WW):  
            del_T_CW = i/10     # delta between inlet and outlet warm seawater temperature in degree Celsius
            del_T_WW = j/10     # delta between inlet and outlet cold seawater temperature in degree Celsius          
   
   ## Calculate system and unpack results here         
   
            otec_plant_nominal = otec_sizing(T_WW_in,
                                    T_CW_in,
                                    del_T_WW,
                                    del_T_CW,
                                    inputs,
                                    cost_level)
            
            _, CAPEX, OPEX, LCOE_nom = capex_opex_lcoe(
                otec_plant_nominal, inputs, cost_level)

            otec_plant_nominal['CAPEX'] = CAPEX
            otec_plant_nominal['OPEX'] = OPEX
            otec_plant_nominal['LCOE_nom'] = LCOE_nom
            
            if T_WW_in.ndim == 0:          
                lcoe_matrix_nominal[int((i-del_T_CW_min)/interval_CW)][int((j-del_T_WW_min)/interval_WW)] = LCOE_nom
                lcoe_matrix_nominal = np.nan_to_num(lcoe_matrix_nominal,nan=10000) # replace NaN with unreasonably high value
                del_T_pair = divmod(lcoe_matrix_nominal.argmin(),lcoe_matrix_nominal.shape[1])
                del_T_CW = (del_T_pair[0] * interval_CW + 20)/10
                del_T_WW = (del_T_pair[1] * interval_CW + 20)/10
            else:
                lcoe_matrix_nominal[int((i-del_T_CW_min)/interval_CW)][int((j-del_T_WW_min)/interval_WW)][:] = LCOE_nom
                lcoe_matrix_nominal = np.nan_to_num(lcoe_matrix_nominal,nan=10000) # replace NaN with unreasonably high value
                del_T_CW = ( np.argmin(np.min(lcoe_matrix_nominal,axis=1),axis=0) * interval_CW + 20)/10
                del_T_WW = ( np.argmin(np.min(lcoe_matrix_nominal,axis=0),axis=0) * interval_WW + 20)/10
       
    
    # It would be more elegant to not re-calculate the plants, but I don't know how to make it better.
    
    otec_plant_nominal_lowest_lcoe = otec_sizing(T_WW_in,
                                        T_CW_in,
                                        del_T_WW,
                                        del_T_CW,
                                        inputs,
                                        cost_level)
    
    all_CAPEX_OPEX,CAPEX,OPEX,LCOE_nom = capex_opex_lcoe(otec_plant_nominal_lowest_lcoe,                                              
                                inputs,
                                cost_level)
    
    
    
    otec_plant_nominal_lowest_lcoe['CAPEX'] = CAPEX
    otec_plant_nominal_lowest_lcoe['OPEX'] = OPEX
    otec_plant_nominal_lowest_lcoe['LCOE_nom'] = LCOE_nom

    
    return otec_plant_nominal_lowest_lcoe,all_CAPEX_OPEX

def off_design_analysis(T_WW_design,T_CW_design,T_WW_profiles,T_CW_profiles,inputs,coordinates,timestamp,studied_region,new_path,cost_level='low_cost',verbose=False):

    if verbose:
        print('\n++ Initiate off-design analysis ++\n')

    # Validate that design temperature arrays are not empty
    T_WW_design_arr = np.atleast_1d(T_WW_design)
    T_CW_design_arr = np.atleast_1d(T_CW_design)

    if T_WW_design_arr.size == 0 or T_CW_design_arr.size == 0:
        error_msg = (f"Cannot perform off-design analysis: "
                    f"T_WW_design has {T_WW_design_arr.size} elements, "
                    f"T_CW_design has {T_CW_design_arr.size} elements. "
                    f"At least one temperature dataset is empty after data processing. "
                    f"This should have been caught earlier in the pipeline.")
        print(f"ERROR: {error_msg}")
        raise ValueError(error_msg)

    results_matrix = {}
    
    if T_WW_design.ndim == 1:
        lcoe_matrix = np.empty((len(T_WW_design),len(T_CW_design)),dtype=np.float64) 
    else:    
        lcoe_matrix = np.empty((len(T_WW_design),len(T_CW_design),np.shape(T_WW_profiles)[1]),dtype=np.float64)
    
    CAPEX_OPEX_for_comparison = []
    for index_cw,t_cw_design in enumerate(T_CW_design):
        for index_ww,t_ww_design in enumerate(T_WW_design):
            
            # print(f'Configuration {index_ww + index_cw*3 + 1}')
            
            otec_plant_nominal_lowest_lcoe,all_CAPEX_OPEX = on_design_analysis(t_ww_design,t_cw_design,inputs,cost_level)          
            otec_plant_off_design = otec_operation(otec_plant_nominal_lowest_lcoe,T_WW_profiles,T_CW_profiles,inputs)
            
            otec_plant_off_design.update(otec_plant_nominal_lowest_lcoe)
            
            # otec_plant_off_design['Configuration'] = index_ww + index_cw*3 + 1
            
            
            results_matrix[index_ww + index_cw*3 + 1] = otec_plant_off_design
            
            if T_WW_design.ndim == 1:
                lcoe_matrix[index_cw][index_ww]  = otec_plant_off_design['LCOE']
            else:    
                lcoe_matrix[index_cw][index_ww][:]  = otec_plant_off_design['LCOE']
                
            CAPEX_OPEX_for_comparison.append([all_CAPEX_OPEX])
    
    lcoe_matrix = np.nan_to_num(lcoe_matrix,nan=10000) # replace NaN with unreasonably high value
              
    index_CW_lowest_LCOE = np.argmin(np.min(lcoe_matrix,axis=1),axis=0)
    index_WW_lowest_LCOE = np.argmin(np.min(lcoe_matrix,axis=0),axis=0)
    
    configuration_lowest_LCOE = (index_WW_lowest_LCOE + index_CW_lowest_LCOE*3 + 1).T
    
    if T_WW_design.ndim == 1:
        otec_plant_lowest_lcoe = results_matrix[configuration_lowest_LCOE]
    else:
    
        # Here we make a dummy dictionary which we will overwrite with the values of the best off-design plants.
        # We use configuration 1 as default because most plants return that configuration as the one with lowest LCOE

        otec_plant_lowest_lcoe = results_matrix[1]
        
        for index, plant in enumerate(index_WW_lowest_LCOE):
            if configuration_lowest_LCOE[index] == 1:
                continue
            else:
                for key in otec_plant_lowest_lcoe:
                    source_value = results_matrix[configuration_lowest_LCOE[index]][key]
                    dest_value = otec_plant_lowest_lcoe[key]

                    # Check dimensionality of both source and destination
                    if np.ndim(source_value) == 0:
                        # Scalar source value
                        if np.ndim(dest_value) == 0:
                            # Both are scalars - skip (keep original from config 1)
                            continue
                        else:
                            # Source is scalar, dest is array - assign to index
                            otec_plant_lowest_lcoe[key][index] = source_value
                    elif np.ndim(source_value) == 1:
                        # 1D array source
                        if np.ndim(dest_value) == 0:
                            # Can't assign 1D to scalar - skip
                            continue
                        else:
                            # 1D array - index it
                            otec_plant_lowest_lcoe[key][index] = source_value[index]
                    else:
                        # 2D or higher source
                        if np.ndim(dest_value) < 2:
                            # Can't assign 2D slice to lower-dim - skip
                            continue
                        else:
                            # 2D or higher - slice it
                            otec_plant_lowest_lcoe[key][:,index] = source_value[:,index]
        
    otec_plant_lowest_lcoe['Configuration'] = configuration_lowest_LCOE
    
    net_power_df = pd.DataFrame(np.round(otec_plant_lowest_lcoe['p_net']/otec_plant_lowest_lcoe['p_gross_nom'],3))
    net_power_df.columns = [str(val[0]) + '_' + str(val[1]) for idx,val in enumerate(coordinates)]

    net_power_df['Time'] = timestamp
    net_power_df = net_power_df.set_index('Time')
    
    date_start = inputs['date_start']
    p_gross = inputs['p_gross']

    # Generate unique filename including configuration info to avoid parallel write conflicts
    # MUST include: region, cycle, fluid, installation_type to ensure uniqueness
    # Use config strings stored in inputs (guaranteed to be unique per configuration)
    cycle_type = inputs.get('config_cycle_type', 'unknown_cycle')
    fluid_type = inputs.get('config_fluid_type', 'unknown_fluid')
    installation_type = inputs.get('installation_type', 'unknown_installation')

    # Sanitize strings for filesystem compatibility
    # Remove parentheses, spaces, and other problematic characters
    # IMPORTANT: Do NOT sanitize path separators (/ or \)
    def sanitize_filename_component(s):
        """
        Remove or replace characters that are problematic in filename components.
        Does NOT touch path separators - only use on individual filename parts, not full paths.
        """
        s = s.replace('(', '').replace(')', '')  # Remove parentheses
        s = s.replace(' ', '_')  # Spaces to underscores
        s = s.replace(':', '_')  # Colons to underscores
        s = s.replace('*', '_')  # Asterisks to underscores
        s = s.replace('?', '_')  # Question marks to underscores
        s = s.replace('"', '')   # Remove quotes
        s = s.replace('<', '_')  # Less than to underscore
        s = s.replace('>', '_')  # Greater than to underscore
        s = s.replace('|', '_')  # Pipe to underscore
        return s

    # Sanitize individual components (NOT the full path)
    cycle_safe = sanitize_filename_component(cycle_type)
    fluid_safe = sanitize_filename_component(fluid_type)
    region_safe = sanitize_filename_component(studied_region)
    install_safe = sanitize_filename_component(installation_type)

    # Create unique filename with configuration INCLUDING installation_type
    # NOTE: new_path already contains the directory path with slashes - don't sanitize it!
    # Format: Time_series_data_{region}_{cycle}_{fluid}_{installation}_{year}_{power}_MW_{cost}.h5
    config_str = f"{cycle_safe}_{fluid_safe}_{install_safe}"
    filename = f'Time_series_data_{region_safe}_{config_str}_{date_start[0:4]}_{-p_gross/1000}_MW_{cost_level}.h5'
    filepath = new_path + filename

    # Write time series data with retry logic for parallel execution
    # NOTE: Each configuration now writes to its own file, eliminating parallel write conflicts
    for key,value in otec_plant_lowest_lcoe.items():
        # Convert to numpy array if not already, and handle scalars
        value_array = np.asarray(value)
        if value_array.ndim == 0:
            # Scalar value - reshape to 1x1 array
            value_array = value_array.reshape(1, 1)
        elif value_array.ndim == 1:
            # 1D array - reshape to 1xN
            value_array = value_array.reshape(1, -1)
        df_to_write = pd.DataFrame(np.round(value_array, 2), columns=net_power_df.columns)
        safe_hdf_write(df_to_write, filepath, key=f'{key}', mode='a')

    if verbose:
        print('\nTime series data successfully exported as h5 file.\n\nEnd of script.')
    
    return otec_plant_lowest_lcoe, CAPEX_OPEX_for_comparison