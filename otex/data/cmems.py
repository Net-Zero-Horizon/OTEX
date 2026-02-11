# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:08:37 2023

@author: jkalanger
"""

import netCDF4
import pandas as pd
import numpy as np

# NumPy 2.0 compatibility fix for pickled data
# Older HDF5 files may have been pickled with numpy.core (pre-2.0)
# but NumPy 2.0+ uses numpy._core
import sys
if not hasattr(np, 'core'):
    np.core = np._core

import datetime
import os
import time as _time_module  # Avoid conflicts with numpy
import copernicusmarine
import threading

# For file locking: use fcntl on Unix/Linux, msvcrt on Windows
if sys.platform == 'win32':
    import msvcrt
    FCNTL_AVAILABLE = False
else:
    import fcntl
    FCNTL_AVAILABLE = True

# Keep a reference to time.sleep to avoid numpy conflicts
_sleep = _time_module.sleep
_time = _time_module.time

# Cross-platform file locking functions
def lock_file(file_handle, blocking=True):
    """Lock a file in a cross-platform way"""
    if FCNTL_AVAILABLE:
        # Unix/Linux using fcntl
        if blocking:
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX)
        else:
            fcntl.flock(file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    else:
        # Windows using msvcrt
        retry = 0
        max_retries = 10 if blocking else 1
        while retry < max_retries:
            try:
                msvcrt.locking(file_handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except IOError:
                if retry < max_retries - 1:
                    _sleep(0.1)
                    retry += 1
                else:
                    raise

def unlock_file(file_handle):
    """Unlock a file in a cross-platform way"""
    if FCNTL_AVAILABLE:
        # Unix/Linux using fcntl
        fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
    else:
        # Windows using msvcrt
        try:
            msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
        except IOError:
            pass  # Already unlocked

## We use seawater temperature data from CMEMS for our OTEC analysis. If the data does not exist in the work folder yet, then it is downloaded with the function
## below. Essentially, we contact CMEMS's servers via an url created from input data like desired year, water depth, coordinates, etc, and download the data
## after the connection to the server has been established successfully.




def download_data(cost_level,inputs,studied_region,new_path):

    ## The csv file below stores all countries and territories that have OTEC resources, their coordinates, and electricity demand in 2019.

    print(f"    [download_data] Reading regions CSV...", flush=True)
    from otex.data.resources import load_regions
    regions = load_regions()
    print(f"    [download_data] CSV loaded, checking region: {studied_region}", flush=True)

    if np.any(regions['region'] == studied_region):

        print(f"    [download_data] Region found, getting parts...", flush=True)
        parts = regions['region'].value_counts()[studied_region]
        print(f"    [download_data] Parts: {parts}", flush=True)
        
        ## OTEC uses warm surface seawater to evaporate a work fluid, while cold deep-sea water is used to condense said work fluid. We download
        ## seawater temperature data from depths representing warm water (WW) and cold water (CW). 
        
        depth_WW = inputs['length_WW_inlet']
        depth_CW = inputs['length_CW_inlet']
        
        ## Due to download limitations, we only download one year of data. We chose the year 2011, but any year between 1994 and 2012 could also work.
        
        date_start = inputs['date_start']
        date_end = inputs['date_end']
        
        ## We store the filenames and their paths, so that the seawater temperature data can be accessed by OTEX later.

        files = []
        # print(depth_WW,depth_CW)
        print(f"    [download_data] Starting download loop for depths: {depth_WW}, {depth_CW}", flush=True)
        for depth in [depth_WW,depth_CW]:
            print(f"    [download_data] Processing depth: {depth}m", flush=True)
            for part in range(0,parts):
                print(f"    [download_data] Processing part {part+1}/{parts}", flush=True)

                ## The coordinates for the download are pulled from the csv file. Alternatively, the user could define the coordinates themselves.

                print(f"    [download_data] Getting coordinates...", flush=True)
                north = float(regions[regions['region']==studied_region]['north'].iloc[part])
                south = float(regions[regions['region']==studied_region]['south'].iloc[part])
                west = float(regions[regions['region']==studied_region]['west'].iloc[part])
                east = float(regions[regions['region']==studied_region]['east'].iloc[part])
                print(f"    [download_data] Coordinates: N={north}, S={south}, W={west}, E={east}", flush=True)

                start_time = _time()
                print(f"    [download_data] Creating filename...", flush=True)
                filename = f'T_{round(depth,0)}m_{date_start[0:4]}_{studied_region}_{part+1}.nc'.replace(" ","_")
                filepath = os.path.join(new_path, filename)
                files.append(filepath)
                directory_data_results='Data_Results/'
                print(filepath, flush=True)
                print(f"    [download_data] Checking if file exists...", flush=True)

                # Check if file exists and is valid
                file_is_valid = False
                try:
                    if os.path.exists(filepath):
                        # Try to open the file to verify it's not corrupted
                        try:
                            test_nc = netCDF4.Dataset(filepath, 'r')
                            test_nc.close()
                            file_is_valid = True
                            print(f"    [download_data] File exists and is valid.", flush=True)
                        except:
                            # File exists but is corrupted, delete it
                            print(f"    [download_data] File exists but is corrupted, will re-download...", flush=True)
                            try:
                                os.remove(filepath)
                            except:
                                pass
                            file_is_valid = False
                    else:
                        print(f"    [download_data] File does not exist.", flush=True)
                except Exception as e:
                    print(f"    [download_data] ERROR checking file: {e}", flush=True)
                    file_is_valid = False

                if file_is_valid:
                    print('File already exists and is valid. No download necessary.', flush=True)
                    continue
                else:
                    # Download the subset of data
                    # Use netcdf3_compatible=True to avoid h5py/h5netcdf compatibility issues
                    try:
                        copernicusmarine.subset(
                            dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m",
                            dataset_version="202311",
                            variables = ['thetao'],
                            minimum_longitude = west,
                            maximum_longitude = east,
                            minimum_latitude = south,
                            maximum_latitude = north,
                            minimum_depth = depth,
                            maximum_depth = depth,
                            start_datetime = date_start,
                            end_datetime = date_end,
                            force_download = True,
                            output_directory = directory_data_results+studied_region.replace(" ","_"),
                            output_filename = filename,
                            netcdf3_compatible = True  # Avoid h5netcdf dimension scale issues
                        )
                    except RuntimeError as e:
                        if "H5DSis_scale" in str(e):
                            # Fallback: try with compression disabled
                            print(f"    Warning: h5netcdf error, retrying with compression disabled...")
                            copernicusmarine.subset(
                                dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m",
                                dataset_version="202311",
                                variables = ['thetao'],
                                minimum_longitude = west,
                                maximum_longitude = east,
                                minimum_latitude = south,
                                maximum_latitude = north,
                                minimum_depth = depth,
                                maximum_depth = depth,
                                start_datetime = date_start,
                                end_datetime = date_end,
                                force_download = True,
                                output_directory = directory_data_results+studied_region.replace(" ","_"),
                                output_filename = filename,
                                netcdf_compression_enabled = False
                            )
                        else:
                            raise


                    end_time = _time()
                    print(f'{filename} saved. Time for download: ' + str(round((end_time-start_time)/60,2)) + ' minutes.')



        print(f"    [download_data] Returning {len(files)} files", flush=True)
        return files    
        
    else:
        raise ValueError('Entered region not valid. Please check for typos and whether the region is included in "download_ranges_per_region.csv"')


def data_processing(files,sites_df,inputs,studied_region,new_path,water,nan_columns = None):
    ## Here we convert the pandas Dataframe storing site-specific data into a numpy array
    
    sites = np.vstack((sites_df['longitude'],sites_df['latitude'],sites_df['dist_shore'],sites_df['id'])).T
    ## The "for file in files" was made for countries and territories that stretch across the East/West border, like Fiji and New Zealand.
    ## These regions are split into two parts that cover the regions' Eastern and Western side, respectively.

    for file in files:
        ## It can happen that a corruped nc file is downloaded with 1 kB size. In that case, reading the file would raise an error. So, we try to read the file,
        ## and if it does not work, it means that the file is corrupted and needs to be downloaded again.
        try:
            T_water_nc = netCDF4.Dataset(file,'r')       
        except:
            raise Warning(f'{file} was not downloaded successfully. Please try downloading the file later.')
    
    ## Here, we convert the timestamp to year-month-day hour:minute:second    
    
    time = T_water_nc.variables['time'][:]
    time_origin = datetime.datetime.strptime(inputs['time_origin'], '%Y-%m-%d %H:%M:%S') 
    # print(time)
    # print(datetime.timedelta(hours=time[0]))
    
    
    timestamp = [time_origin + datetime.timedelta(hours=int(step)) for idx,step in enumerate(time)]  
    
    ## Earlier, we downloaded the data across a rectangular field defined by the input coordinates. However, not every data point is suitable
    ## for OTEC (e.g. points on land, too shallow/ deep water, inside marine protection areas, etc). In this loop, we check which downloaded data points
    ## could be occupied by OTEC plants, and store their coordinates and temperature profiles in a numpy array
    
    T_water_profiles = np.zeros((time.shape[0],0),dtype=np.float64)
    coordinates = np.zeros((0,2),dtype=np.float64)
    dist_shore = np.zeros((1,0),dtype=np.float64)
    id_sites = np.zeros((1,0),dtype=np.float64)
    
    # Create a dictionary for fast lookup of sites by coordinates
    # This significantly improves performance from O(n*m) to O(n+m)
    sites_dict = {}
    for i in range(sites.shape[0]):
        key = (np.round(sites[i, 0], 3), np.round(sites[i, 1], 3))
        sites_dict[key] = (sites[i, 2], sites[i, 3])  # dist_shore, id

    for file in files:
        T_water_nc = netCDF4.Dataset(file,'r')
        latitude = T_water_nc.variables['latitude'][:]
        longitude = T_water_nc.variables['longitude'][:]
        depth = int(T_water_nc.variables['depth'][:])
        T_water = T_water_nc.variables['thetao'][:]

        # Optimized: use meshgrid to create coordinate pairs efficiently
        lon_grid, lat_grid = np.meshgrid(longitude, latitude)
        lon_rounded = np.round(lon_grid, 3)
        lat_rounded = np.round(lat_grid, 3)

        # Iterate over all grid points
        for idx_lat in range(len(latitude)):
            for idx_lon in range(len(longitude)):
                lon_val = np.round(longitude[idx_lon], 3)
                lat_val = np.round(latitude[idx_lat], 3)
                key = (lon_val, lat_val)

                # Fast lookup in dictionary
                if key in sites_dict:
                    dist_shore_val, id_site_val = sites_dict[key]

                    if T_water_profiles.shape[1] == 0:
                        # Initialize as 2D array to ensure consistent shape even for single-site regions
                        coordinates = np.array([[lon_val, lat_val]])
                        dist_shore = np.array([[dist_shore_val]])
                        id_sites = np.array([[id_site_val]])
                        T_water_profiles = (np.array(T_water[:,:,idx_lat,idx_lon],dtype=np.float64))
                    else:
                        coordinates = np.vstack((coordinates, [lon_val, lat_val]))
                        dist_shore = np.hstack((dist_shore, [[dist_shore_val]]))
                        id_sites = np.hstack((id_sites, [[id_site_val]]))
                        T_water_profiles = np.hstack((T_water_profiles,(np.array(T_water[:,:,idx_lat,idx_lon],dtype=np.float64))))
    
    ## After obtaining the relevant CMEMS points, we calculate power transmission losses from OTEC plant offshore to the public grid onshore in kilometres.
    
    eff_trans = np.empty(np.shape(dist_shore),dtype=np.float64)
    # AC cables for distances below or equal to 50 km, source: Fragoso Rodrigues (2016) 
    eff_trans[dist_shore <= inputs['threshold_AC_DC']] = 0.979-1*10**-6*dist_shore[dist_shore <= 50]**2-9*10**-5*dist_shore[dist_shore <= 50]  
    # DC cables for distances beyond 50 km, source: Fragoso Rodrigues (2016) 
    eff_trans[dist_shore > inputs['threshold_AC_DC']] = 0.964-8*10**-5*dist_shore[dist_shore > 50]  

    ## Some data might either be missing (no timestamp) or faulty (e.g. T = -30000)
    ## First, we remove the faulty values

    T_water_profiles[T_water_profiles <= 0] = np.nan
    
    ## Here, we resample the dataset to the temporal resolution given in the parameters_and_constants file
    ## and to fill previously missing steps with NaN, which are then filled via linear interpolation
    T_water_profiles_df = pd.DataFrame(T_water_profiles)
    # coordinates is always 2D with shape (n_sites, 2), so iterate over rows
    T_water_profiles_df.columns = [str(val[0]) + '_' + str(val[1]) for val in coordinates]

    # Validate timestamp length matches data
    if len(timestamp) != T_water_profiles_df.shape[0]:
        raise ValueError(f"Timestamp length ({len(timestamp)}) does not match data rows ({T_water_profiles_df.shape[0]})")

    T_water_profiles_df['time'] = timestamp

    # Remove duplicate timestamps by keeping the first occurrence
    # This can happen when processing multiple files with overlapping time ranges
    if T_water_profiles_df['time'].duplicated().any():
        T_water_profiles_df = T_water_profiles_df.drop_duplicates(subset='time', keep='first')

    T_water_profiles_df = T_water_profiles_df.set_index('time').asfreq(f'{inputs["t_resolution"]}')
    T_water_profiles_df = T_water_profiles_df.interpolate(method='linear')
    
    # Calculating interquartiles. With a factor 3, we are less strict with outliers than the convention of 1.5
    # With this, we want to account for extreme seawater temperature conditions that would otherwise be removed from the dataset
    r = T_water_profiles_df.rolling(window=30)
    mps = (r.quantile(0.75) - r.quantile(0.25))*3 

    T_water_profiles_df[(T_water_profiles_df < T_water_profiles_df.quantile(0.25) - mps) |
                        (T_water_profiles_df > T_water_profiles_df.quantile(0.75) + mps)] = np.nan
    
    T_water_profiles_df = T_water_profiles_df.interpolate(method='linear')
    
    ## In some case, points don't have any data at all. If there are profiles solely consisting of NaN, they are removed from the dataset
    
    if nan_columns is None:
        nan_columns = np.where(T_water_profiles_df.isna())
    else:
        pass
    
    T_water_profiles_df = T_water_profiles_df.drop(T_water_profiles_df.iloc[:,nan_columns[1]],axis=1)
    T_water_profiles = np.array(T_water_profiles_df,dtype=np.float64)
    
    ## To assess OTEC's economic and technical performance under off-design conditions, we design the plants for different warm and cold seawater temperatures
    ## Using combinations of minimum, median, and maximum temperature, we assess a total of nine configurations. For example, the most conservative configuration is
    ## configuration 1 using minimum warm seawater temperature and maximum cold deep-seawater temperature.
    
    ## Here, we calculate the design temperatures from the cleaned datasets    
    
    if water == 'CW':       
        T_water_design = np.round(np.array([np.max(T_water_profiles_df,axis=0),
                                            np.median(T_water_profiles_df,axis=0),
                                            np.min(T_water_profiles_df,axis=0)]),1)
    elif water == 'WW':
        T_water_design = np.round(np.array([np.min(T_water_profiles_df,axis=0),
                                            np.median(T_water_profiles_df,axis=0),
                                            np.max(T_water_profiles_df,axis=0)]),1) 
    else:
        raise ValueError('Invalid input for seawater. Please select "CW" for cold deep seawater or "WW" for warm surface seawater.')
        
    coordinates = np.delete(coordinates,nan_columns[1],axis=0)
    dist_shore = np.delete(dist_shore,nan_columns[1],axis=1)
    inputs['dist_shore'] = dist_shore
    eff_trans = np.delete(eff_trans,nan_columns[1],axis=1)
    inputs['eff_trans'] = eff_trans
    id_sites = np.delete(id_sites,nan_columns[1],axis=1)
    
    ## Here we store the cleaned datasets as h5 files so that it does not have to recalculated later.

    year = inputs['date_start'][0:4]

    filename = f'T_{round(depth,0)}m_{year}_{studied_region}.h5'.replace(" ","_")
    h5_filepath = new_path + filename
    lockfile_path = h5_filepath + '.lock'

    # Use file locking to prevent multiple processes from writing simultaneously
    # This is critical for parallel execution
    max_retries = 30
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            # Create lock file and acquire exclusive lock
            with open(lockfile_path, 'w') as lockfile:
                try:
                    # Try to acquire exclusive lock (non-blocking first, then blocking)
                    lock_file(lockfile, blocking=False)
                except IOError:
                    # Lock is held by another process, wait and retry
                    if attempt < max_retries - 1:
                        print(f"    [data_processing] File {filename} is locked, waiting {retry_delay}s (attempt {attempt+1}/{max_retries})...", flush=True)
                        _sleep(retry_delay)
                        continue
                    else:
                        # Use blocking lock on last attempt
                        print(f"    [data_processing] Acquiring blocking lock for {filename}...", flush=True)
                        lock_file(lockfile, blocking=True)

                # Check if file already exists (another process may have created it while we waited)
                if os.path.exists(h5_filepath):
                    print(f'    [data_processing] File {filename} already exists (created by another process), skipping write.', flush=True)
                    # Release lock and exit
                    unlock_file(lockfile)
                    break

                # Write HDF5 file
                print(f'    [data_processing] Writing {filename} with exclusive lock...', flush=True)
                T_water_profiles_df.to_hdf(h5_filepath, key='T_water_profiles', mode='w')
                pd.DataFrame(T_water_design).to_hdf(h5_filepath, key='T_water_design')
                pd.DataFrame(dist_shore).to_hdf(h5_filepath, key='dist_shore')
                pd.DataFrame(eff_trans).to_hdf(h5_filepath, key='eff_trans')
                pd.DataFrame(coordinates).to_hdf(h5_filepath, key='coordinates')
                pd.DataFrame(nan_columns[1]).to_hdf(h5_filepath, key='nan_columns')
                pd.DataFrame(id_sites).to_hdf(h5_filepath, key='id_sites')

                print(f'Processing {filename} successful. h5 temperature profiles exported.\n', flush=True)

                # Release lock
                unlock_file(lockfile)
                break

        except Exception as e:
            print(f"    [data_processing] ERROR writing {filename}: {e}", flush=True)
            if attempt < max_retries - 1:
                _sleep(retry_delay)
                continue
            else:
                raise

    # Clean up lock file
    try:
        if os.path.exists(lockfile_path):
            os.remove(lockfile_path)
    except:
        pass
            
    return T_water_profiles, T_water_design, coordinates, id_sites, T_water_profiles_df.index, inputs, nan_columns
        
def load_temperatures(file,inputs):

    # If the h5 files for the cleaned seawater temperature data already exists, it is merely loaded with this function

    T_water_profiles_df = pd.read_hdf(file,key='T_water_profiles')
    timestamp = T_water_profiles_df.index
    T_water_profiles = np.array(T_water_profiles_df,dtype=np.float64)
    T_water_design = np.array(pd.read_hdf(file,key='T_water_design'),dtype=np.float64)

    inputs['dist_shore'] = np.array(pd.read_hdf(file,key='dist_shore'),dtype=np.float64)
    inputs['eff_trans'] = np.array(pd.read_hdf(file,key='eff_trans'),dtype=np.float64)

    coordinates = np.array(pd.read_hdf(file,key='coordinates'),dtype=np.float64)
    nan_columns = np.array(pd.read_hdf(file,key='nan_columns'),dtype=np.float64)

    id_sites = np.array(pd.read_hdf(file,key='id_sites'),dtype=np.float64)

    # Fix for single-site regions: ensure coordinates has shape (n_sites, 2)
    # Old buggy code saved single-site coordinates as [lon, lat] which became shape (2, 1) when loaded
    if coordinates.ndim == 1:
        # 1D array with 2 elements: [lon, lat] -> reshape to [[lon, lat]]
        coordinates = coordinates.reshape(1, -1)
    elif coordinates.ndim == 2 and coordinates.shape[1] == 1 and coordinates.shape[0] == 2:
        # Shape (2, 1) from buggy save: transpose to (1, 2)
        coordinates = coordinates.T
    # Ensure it's always 2D with at least 2 columns
    if coordinates.ndim == 2 and coordinates.shape[1] < 2:
        raise ValueError(f"Invalid coordinates shape {coordinates.shape} in file {file}. "
                        f"Expected shape (n_sites, 2). Please re-download data for this region.")

    return T_water_profiles, T_water_design, coordinates, id_sites, timestamp, inputs, nan_columns