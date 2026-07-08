# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Portions of this file derive from pyOTEC by Jannis K. A. Langer
# (TU Delft), originally distributed under the EUPL-1.2. See NOTICE.
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
import re
import time as _time_module  # Avoid conflicts with numpy
import threading
from collections import defaultdict

# copernicusmarine is only required for ``download_data`` (the actual
# CMEMS subset request). All the other helpers in this module — the
# multi-year loader, data_processing, load_temperatures — work with
# already-downloaded NetCDFs and do not need the SDK installed. Import
# it lazily so that environments without copernicusmarine (CI, users
# of the analysis-only path) can still import otex.data.cmems freely.
copernicusmarine = None  # populated on first call to download_data

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
    global copernicusmarine
    if copernicusmarine is None:
        import copernicusmarine as _cm
        copernicusmarine = _cm

    ## Region geometry is resolved on demand from Natural Earth admin-0
    ## boundaries (since 0.2.0). For OTEC we extend the country's tight
    ## land bbox by an offshore buffer so the download window captures
    ## relevant deep-water cells, mirroring how the legacy CSV bboxes
    ## were specified.

    print(f"    [download_data] Resolving region: {studied_region}", flush=True)
    from otex.data.regions import resolve_region
    from otex.data.sites import _DEFAULT_OFFSHORE_BUFFER_DEG, _expand_bbox
    region = resolve_region(studied_region)
    parts_list = [_expand_bbox(b, _DEFAULT_OFFSHORE_BUFFER_DEG) for b in region.bboxes]
    parts = len(parts_list)
    print(f"    [download_data] Resolved to {parts} bbox part(s)", flush=True)

    ## OTEC uses warm surface seawater to evaporate a work fluid, while cold deep-sea water is used to condense said work fluid. We download
    ## seawater temperature data from depths representing warm water (WW) and cold water (CW).

    depth_WW = inputs['length_WW_inlet']
    depth_CW = inputs['length_CW_inlet']

    ## Multi-year support: download one NetCDF per (depth, part, year). The
    ## per-year split keeps individual download payloads small enough for
    ## the CMEMS API and matches the per-year filename pattern that
    ## ``data_processing`` expects when it groups files by year.

    # Resolve year range: prefer explicit year_start/year_end, fall back to
    # legacy single-year inferred from date_start.
    year_start = inputs.get('year_start')
    year_end = inputs.get('year_end')
    if year_start is None or year_end is None:
        single_year = int(inputs['date_start'][0:4])
        year_start = year_start or single_year
        year_end = year_end or single_year
    years = list(range(int(year_start), int(year_end) + 1))

    ## We store the filenames and their paths, so that the seawater temperature data can be accessed by OTEX later.

    files = []
    print(f"    [download_data] Starting download loop for depths: {depth_WW}, {depth_CW}; years: {years}", flush=True)
    for depth in [depth_WW,depth_CW]:
        print(f"    [download_data] Processing depth: {depth}m", flush=True)
        for part in range(0,parts):
            print(f"    [download_data] Processing part {part+1}/{parts}", flush=True)

            ## The coordinates for the download are pulled from the csv file. Alternatively, the user could define the coordinates themselves.

            print(f"    [download_data] Getting coordinates...", flush=True)
            _bbox = parts_list[part]
            north = float(_bbox.north)
            south = float(_bbox.south)
            west = float(_bbox.west)
            east = float(_bbox.east)
            print(f"    [download_data] Coordinates: N={north}, S={south}, W={west}, E={east}", flush=True)

            for year in years:
                date_start = f'{year}-01-01 00:00:00'
                date_end = f'{year}-12-31 21:00:00'

                start_time = _time()
                print(f"    [download_data] Creating filename for year {year}...", flush=True)
                filename = f'T_{round(depth,0)}m_{year}_{studied_region}_{part+1}.nc'.replace(" ","_")
                filepath = os.path.join(new_path, filename)
                files.append(filepath)
                # Ensure the caller's destination exists; copernicusmarine
                # will write directly into it (no second hard-coded prefix).
                os.makedirs(new_path, exist_ok=True)
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
                            output_directory = new_path,
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
                                output_directory = new_path,
                                output_filename = filename,
                                netcdf_compression_enabled = False
                            )
                        else:
                            raise


                    end_time = _time()
                    print(f'{filename} saved. Time for download: ' + str(round((end_time-start_time)/60,2)) + ' minutes.')



    print(f"    [download_data] Returning {len(files)} files", flush=True)
    return files


# Regex matches the filename pattern produced by ``download_data``:
#   T_{depth}m_{YYYY}_{region}_{part}.nc   (with optional .0 in depth)
# The region segment may itself contain underscores, so the year is
# captured as the *first* 4-digit token after the depth segment.
_YEAR_FROM_FILENAME_RE = re.compile(r'T_[\d.]+m_(\d{4})_')


def _year_from_filename(path: str) -> int:
    """Extract the calendar year from a NetCDF filename produced by download_data.

    Raises ValueError if the filename does not match the expected pattern.
    """
    name = os.path.basename(path)
    m = _YEAR_FROM_FILENAME_RE.search(name)
    if not m:
        raise ValueError(
            f"Cannot extract year from filename '{name}'. Expected pattern "
            f"'T_<depth>m_<YYYY>_<region>_<part>.nc'."
        )
    return int(m.group(1))


def _group_files_by_year(files):
    """Group a flat file list into ``{year: [paths]}`` preserving input order."""
    grouped = defaultdict(list)
    for f in files:
        grouped[_year_from_filename(f)].append(f)
    # Return as a sorted regular dict for deterministic iteration.
    return {y: grouped[y] for y in sorted(grouped)}


def _extract_year_data(year_files, sites_dict, time_origin):
    """Read all NetCDF parts for one calendar year and return a DataFrame.

    Returns
    -------
    df : pd.DataFrame
        Index = timestamps for that year, columns = "lon_lat" labels.
    coordinates, dist_shore, id_sites : np.ndarray
        Aligned site metadata (one entry per column of df).
    depth : int
        Depth in metres read from the first file.
    """
    # Validate every file opens cleanly.
    for f in year_files:
        try:
            netCDF4.Dataset(f, 'r').close()
        except Exception:
            raise Warning(
                f'{f} was not downloaded successfully. Please try '
                f'downloading the file later.'
            )

    # Time axis is shared across parts within a single year.
    first = netCDF4.Dataset(year_files[0], 'r')
    time = first.variables['time'][:]
    timestamp = [time_origin + datetime.timedelta(hours=int(step)) for step in time]
    first.close()

    # Accumulate matches as Python lists, then concatenate ONCE at the
    # end. The original code did np.hstack on every match, which is
    # O(M^2) in memory ops because each hstack reallocates the whole
    # growing buffer. With this approach the per-file pass is O(N + M)
    # — about 50x faster on a 200x200 grid (Indonesia / Philippines).
    matched_T_columns: list = []
    matched_lon: list = []
    matched_lat: list = []
    matched_dist: list = []
    matched_ids: list = []
    depth = None

    for f in year_files:
        ds = netCDF4.Dataset(f, 'r')
        latitude = np.round(ds.variables['latitude'][:], 3)
        longitude = np.round(ds.variables['longitude'][:], 3)
        depth = int(ds.variables['depth'][:])
        T_water = np.asarray(ds.variables['thetao'][:], dtype=np.float64)
        # Collapse the depth axis (typically length 1) and the spatial
        # grid into a single (time, lat*lon) view so we can later slice
        # all matched cells with one fancy-indexing operation.
        T_flat = T_water.reshape(T_water.shape[0], -1)   # (time, n_lat*n_lon)
        n_lon = len(longitude)

        # Membership check: still a Python pass, but with constant-time
        # set lookups and *no* numpy reallocations along the way.
        flat_indices: list = []
        for idx_lat, lat_val in enumerate(latitude):
            for idx_lon, lon_val in enumerate(longitude):
                key = (float(lon_val), float(lat_val))
                hit = sites_dict.get(key)
                if hit is None:
                    continue
                dist_shore_val, id_site_val = hit
                flat_indices.append(idx_lat * n_lon + idx_lon)
                matched_lon.append(float(lon_val))
                matched_lat.append(float(lat_val))
                matched_dist.append(float(dist_shore_val))
                matched_ids.append(float(id_site_val))

        if flat_indices:
            # Single fancy-index slice instead of one slice per cell.
            matched_T_columns.append(T_flat[:, flat_indices])
        ds.close()

    if matched_T_columns:
        T_water_profiles = np.concatenate(matched_T_columns, axis=1)
        coordinates = np.column_stack([matched_lon, matched_lat])
        dist_shore = np.array([matched_dist], dtype=np.float64)
        id_sites = np.array([matched_ids], dtype=np.float64)
    else:
        T_water_profiles = np.zeros((time.shape[0], 0), dtype=np.float64)
        coordinates = np.zeros((0, 2), dtype=np.float64)
        dist_shore = np.zeros((1, 0), dtype=np.float64)
        id_sites = np.zeros((1, 0), dtype=np.float64)

    df = pd.DataFrame(T_water_profiles,
                      columns=[f'{c[0]}_{c[1]}' for c in coordinates])
    if len(timestamp) != df.shape[0]:
        raise ValueError(
            f"Timestamp length ({len(timestamp)}) does not match data rows "
            f"({df.shape[0]}) in year files {year_files}"
        )
    df['time'] = timestamp
    df = df.set_index('time')
    return df, coordinates, dist_shore, id_sites, depth


def data_processing(files,sites_df,inputs,studied_region,new_path,water,nan_columns = None):
    ## Here we convert the pandas Dataframe storing site-specific data into a numpy array

    sites = np.vstack((sites_df['longitude'],sites_df['latitude'],sites_df['dist_shore'],sites_df['id'])).T

    # Fast site lookup by (lon, lat). O(1) per grid point vs O(n) linear scan.
    sites_dict = {}
    for i in range(sites.shape[0]):
        key = (np.round(sites[i, 0], 3), np.round(sites[i, 1], 3))
        sites_dict[key] = (sites[i, 2], sites[i, 3])  # dist_shore, id

    time_origin = datetime.datetime.strptime(inputs['time_origin'], '%Y-%m-%d %H:%M:%S')

    # Group input files by calendar year. Single-year runs produce one group;
    # multi-year runs stack each year's DataFrame along the time axis.
    files_by_year = _group_files_by_year(files)

    per_year_dfs = []
    coordinates = dist_shore = id_sites = None
    depth = None

    for year, year_files in files_by_year.items():
        df_y, coords_y, dist_y, ids_y, depth_y = _extract_year_data(
            year_files, sites_dict, time_origin
        )
        if coordinates is None:
            coordinates, dist_shore, id_sites = coords_y, dist_y, ids_y
            depth = depth_y
        else:
            # Sanity check: same region across years must yield the same sites.
            if not np.array_equal(coords_y, coordinates):
                raise ValueError(
                    f"Site coordinates for year {year} do not match the "
                    f"reference year {next(iter(files_by_year))}. The region "
                    f"download window must be identical across all years."
                )
        per_year_dfs.append(df_y)

    T_water_profiles_df = pd.concat(per_year_dfs).sort_index()
    T_water_profiles = np.array(T_water_profiles_df, dtype=np.float64)
    timestamp = T_water_profiles_df.index
    
    ## After obtaining the relevant CMEMS points, we calculate power transmission losses from OTEC plant offshore to the public grid onshore in kilometres.
    
    eff_trans = np.empty(np.shape(dist_shore),dtype=np.float64)
    # AC cables for distances below or equal to 50 km, source: Fragoso Rodrigues (2016) 
    eff_trans[dist_shore <= inputs['threshold_AC_DC']] = 0.979-1*10**-6*dist_shore[dist_shore <= 50]**2-9*10**-5*dist_shore[dist_shore <= 50]  
    # DC cables for distances beyond 50 km, source: Fragoso Rodrigues (2016) 
    eff_trans[dist_shore > inputs['threshold_AC_DC']] = 0.964-8*10**-5*dist_shore[dist_shore > 50]  

    ## Some data might either be missing (no timestamp) or faulty (e.g. T = -30000)
    ## First, we remove the faulty values; the DataFrame was assembled above
    ## (single- or multi-year), so we operate directly on it.

    T_water_profiles_df = T_water_profiles_df.mask(T_water_profiles_df <= 0)

    ## Here, we resample the dataset to the temporal resolution given in the
    ## parameters_and_constants file and to fill previously missing steps with
    ## NaN, which are then filled via linear interpolation.

    # Drop duplicate timestamps that might appear if input files overlap.
    if T_water_profiles_df.index.duplicated().any():
        T_water_profiles_df = T_water_profiles_df[~T_water_profiles_df.index.duplicated(keep='first')]

    T_water_profiles_df = T_water_profiles_df.asfreq(f'{inputs["t_resolution"]}')
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

    # Use the multi-year label (e.g. '2020-2022') when available; fall back to
    # the single-year string for backward compatibility.
    year_label = inputs.get('year_label') or inputs['date_start'][0:4]

    filename = f'T_{round(depth,0)}m_{year_label}_{studied_region}.h5'.replace(" ","_")
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
