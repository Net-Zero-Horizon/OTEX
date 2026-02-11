# -*- coding: utf-8 -*-
"""
CMEMS Multi-Depth Data Download and Processing
Extends CMEMS download to support multiple depth levels for depth optimization

This module downloads ocean temperature data at multiple depths to enable
location-specific cold water intake depth optimization.

@author: Extended OTEX
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta


def get_depth_levels_for_optimization(depth_range=(400, 1500), step=100):
    """
    Generate list of depth levels to download for optimization

    Args:
        depth_range: Tuple (min_depth, max_depth) in meters
        step: Depth step in meters

    Returns:
        depth_levels: List of depths to download [m]
    """

    min_depth, max_depth = depth_range
    depth_levels = list(range(min_depth, max_depth + step, step))

    return depth_levels


def get_nearest_cmems_depths(target_depths, available_depths):
    """
    Match target depths to nearest available CMEMS depth levels

    CMEMS provides data at specific depth levels (not continuous)
    Standard depths: 0, 5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150,
                    200, 250, 300, 400, 500, 600, 700, 800, 900, 1000,
                    1100, 1200, 1300, 1400, 1500, ...

    Args:
        target_depths: List of desired depths [m]
        available_depths: List of available CMEMS depths [m]

    Returns:
        nearest_depths: List of nearest available depths
        depth_mapping: Dictionary mapping target -> actual depth
    """

    nearest_depths = []
    depth_mapping = {}

    for target in target_depths:
        # Find nearest available depth
        differences = [abs(target - available) for available in available_depths]
        nearest_idx = np.argmin(differences)
        nearest = available_depths[nearest_idx]

        nearest_depths.append(nearest)
        depth_mapping[target] = nearest

    # Remove duplicates while preserving order
    seen = set()
    nearest_depths = [x for x in nearest_depths if not (x in seen or seen.add(x))]

    return nearest_depths, depth_mapping


def download_cmems_multi_depth(region, depth_levels, inputs, dl_path):
    """
    Download CMEMS temperature data at multiple depths

    Args:
        region: Region name (e.g., 'Caribbean')
        depth_levels: List of depth levels to download [m]
        inputs: Dictionary with download parameters
        dl_path: Download path

    Returns:
        files: List of downloaded NetCDF files (grouped by depth)
    """

    print(f"\n++ Downloading multi-depth CMEMS data ++")
    print(f"Region: {region}")
    print(f"Depths: {depth_levels}")

    # Standard CMEMS depth levels (from global_analysis_phy dataset)
    cmems_standard_depths = [
        0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50,
        60, 70, 80, 90, 100, 125, 150, 200, 250, 300, 350, 400,
        500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500,
        2000, 2500, 3000, 3500, 4000, 4500, 5000
    ]

    # Map requested depths to available CMEMS depths
    cmems_depths, depth_mapping = get_nearest_cmems_depths(
        depth_levels,
        cmems_standard_depths
    )

    print(f"\nDepth mapping (target -> CMEMS):")
    for target, actual in depth_mapping.items():
        print(f"  {target}m -> {actual}m")

    # Get date range from inputs
    date_start = inputs.get('date_start', '2020-01-01 00:00:00')
    date_end = inputs.get('date_end', '2020-12-31 21:00:00')

    # Get geographic bounds for region
    # This should be loaded from download_ranges_per_region.csv
    # For now, use example coordinates
    region_bounds = _get_region_bounds(region)

    downloaded_files = {}

    # Download data for each depth level
    for depth in cmems_depths:
        print(f"\nDownloading data at {depth}m depth...")

        # Construct file name
        year = date_start[:4]
        filename = os.path.join(
            dl_path,
            f'CMEMS_T_{depth}m_{year}_{region}.nc'.replace(" ", "_")
        )

        # Check if already downloaded
        if os.path.exists(filename):
            print(f"  File already exists: {filename}")
            downloaded_files[depth] = filename
            continue

        # Download using copernicusmarine
        try:
            import copernicusmarine

            # Use netcdf3_compatible=True to avoid h5py/h5netcdf compatibility issues
            try:
                copernicusmarine.subset(
                    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
                    variables=["thetao"],  # Potential temperature
                    minimum_longitude=region_bounds['lon_min'],
                    maximum_longitude=region_bounds['lon_max'],
                    minimum_latitude=region_bounds['lat_min'],
                    maximum_latitude=region_bounds['lat_max'],
                    start_datetime=date_start,
                    end_datetime=date_end,
                    minimum_depth=depth,
                    maximum_depth=depth,
                    output_filename=filename,
                    force_download=False,
                    netcdf3_compatible=True  # Avoid h5netcdf dimension scale issues
                )
            except RuntimeError as e:
                if "H5DSis_scale" in str(e):
                    # Fallback: try with compression disabled
                    print(f"  Warning: h5netcdf error, retrying with compression disabled...")
                    copernicusmarine.subset(
                        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
                        variables=["thetao"],
                        minimum_longitude=region_bounds['lon_min'],
                        maximum_longitude=region_bounds['lon_max'],
                        minimum_latitude=region_bounds['lat_min'],
                        maximum_latitude=region_bounds['lat_max'],
                        start_datetime=date_start,
                        end_datetime=date_end,
                        minimum_depth=depth,
                        maximum_depth=depth,
                        output_filename=filename,
                        force_download=False,
                        netcdf_compression_enabled=False
                    )
                else:
                    raise

            downloaded_files[depth] = filename
            print(f"  Downloaded: {filename}")

        except Exception as e:
            print(f"  Error downloading {depth}m: {e}")
            print(f"  Skipping this depth level")

    print(f"\n++ Multi-depth download complete ++")
    print(f"Successfully downloaded {len(downloaded_files)} depth levels")

    return downloaded_files


def process_multi_depth_data(downloaded_files, sites_df, inputs):
    """
    Process multi-depth temperature data

    Args:
        downloaded_files: Dictionary {depth: filename}
        sites_df: DataFrame with site locations
        inputs: Processing parameters

    Returns:
        temperature_profiles: Dictionary {depth: temperature_array}
                             Shape: (time, locations)
        coordinates: Array of [lon, lat] for each location
        timestamps: Time series timestamps
    """

    import xarray as xr

    print(f"\n++ Processing multi-depth temperature data ++")

    temperature_profiles = {}
    coordinates = None
    timestamps = None

    # Process each depth level
    for depth, filename in sorted(downloaded_files.items()):
        print(f"\nProcessing {depth}m depth...")

        try:
            # Open NetCDF file
            ds = xr.open_dataset(filename)

            # Extract temperature data
            temp_data = ds['thetao'].values  # Shape: (time, lat, lon)

            # Extract coordinates if not already done
            if coordinates is None:
                lons = ds['longitude'].values
                lats = ds['latitude'].values
                times = pd.to_datetime(ds['time'].values)

                # Create coordinate grid
                lon_grid, lat_grid = np.meshgrid(lons, lats)
                coordinates = np.column_stack([
                    lon_grid.ravel(),
                    lat_grid.ravel()
                ])

                timestamps = times

            # Reshape to (time, locations)
            n_times = temp_data.shape[0]
            n_locations = temp_data.shape[1] * temp_data.shape[2]
            temp_reshaped = temp_data.reshape(n_times, n_locations)

            temperature_profiles[depth] = temp_reshaped

            ds.close()

            print(f"  Shape: {temp_reshaped.shape}")
            print(f"  Temperature range: {np.nanmin(temp_reshaped):.2f} - {np.nanmax(temp_reshaped):.2f}°C")

        except Exception as e:
            print(f"  Error processing {depth}m: {e}")

    print(f"\n++ Multi-depth processing complete ++")
    print(f"Processed {len(temperature_profiles)} depth levels")
    print(f"Locations: {coordinates.shape[0]}")
    print(f"Time points: {len(timestamps)}")

    return temperature_profiles, coordinates, timestamps


def optimize_depth_with_real_data(temperature_profiles, coordinates, inputs, cost_level='low_cost'):
    """
    Optimize cold water intake depth using real CMEMS temperature profiles

    Args:
        temperature_profiles: Dictionary {depth: temperature_array}
        coordinates: Array of [lon, lat] for each location
        inputs: Optimization parameters
        cost_level: 'low_cost' or 'high_cost'

    Returns:
        optimal_depths: Array of optimal depths for each location [m]
        optimal_temperatures: Array of temperatures at optimal depths [°C]
        lcoe_vs_depth: DataFrame with LCOE for each depth and location
    """

    from depth_optimization import calculate_pipe_cost_increment, calculate_thermal_benefit

    print(f"\n++ Optimizing depths with real CMEMS data ++")

    n_locations = coordinates.shape[0]
    depths = sorted(temperature_profiles.keys())

    # Reference depth (original implementation)
    depth_reference = 1062

    # Cost parameters
    if cost_level == 'low_cost':
        cost_per_meter = 9.0 * 1000
    else:
        cost_per_meter = 30.1 * 1000

    # Storage
    optimal_depths = np.zeros(n_locations)
    optimal_temperatures = np.zeros(n_locations)
    lcoe_matrix = np.zeros((len(depths), n_locations))

    # Assume warm water temperature (could also be from CMEMS surface data)
    T_WW = inputs.get('T_WW_design', 26.0)

    print(f"Analyzing {n_locations} locations...")

    for loc_idx in range(n_locations):
        if (loc_idx + 1) % 100 == 0:
            print(f"  Progress: {loc_idx + 1}/{n_locations}")

        lcoe_values = []

        for depth_idx, depth in enumerate(depths):
            # Average temperature at this depth and location
            T_CW = np.nanmean(temperature_profiles[depth][:, loc_idx])

            # Skip if insufficient data
            if np.isnan(T_CW):
                lcoe_values.append(np.inf)
                continue

            # Check minimum ΔT
            delta_T = T_WW - T_CW
            if delta_T < 15:  # Minimum for viable OTEC
                lcoe_values.append(np.inf)
                continue

            # Calculate cost increment
            pipe_cost = calculate_pipe_cost_increment(
                depth, depth_reference, cost_per_meter, inputs
            )

            # Calculate thermal benefit
            T_CW_ref = np.nanmean(temperature_profiles.get(depth_reference, temperature_profiles[depths[-1]])[:, loc_idx])
            power_benefit = calculate_thermal_benefit(
                T_CW, T_CW_ref, T_WW, inputs
            )

            # Simplified LCOE calculation
            lcoe_baseline = inputs.get('lcoe_reference', 15.0)
            capex_baseline = inputs.get('capex_reference', 500e6)

            capex_ratio = (capex_baseline + pipe_cost) / capex_baseline
            energy_baseline = abs(inputs.get('p_gross', -136000)) * 8760
            energy_increment = power_benefit * 8760
            energy_ratio = (energy_baseline + energy_increment) / energy_baseline

            lcoe = lcoe_baseline * capex_ratio / energy_ratio
            lcoe_values.append(lcoe)

            lcoe_matrix[depth_idx, loc_idx] = lcoe

        # Find optimal depth for this location
        lcoe_values = np.array(lcoe_values)
        if np.all(np.isinf(lcoe_values)):
            optimal_depths[loc_idx] = np.nan
            optimal_temperatures[loc_idx] = np.nan
        else:
            optimal_idx = np.argmin(lcoe_values)
            optimal_depths[loc_idx] = depths[optimal_idx]
            optimal_temperatures[loc_idx] = np.nanmean(
                temperature_profiles[depths[optimal_idx]][:, loc_idx]
            )

    # Create results DataFrame
    lcoe_vs_depth = pd.DataFrame(
        lcoe_matrix,
        index=[f'{d}m' for d in depths],
        columns=[f'Loc_{i}' for i in range(n_locations)]
    )

    print(f"\n++ Depth optimization complete ++")
    print(f"Mean optimal depth: {np.nanmean(optimal_depths):.1f}m")
    print(f"Depth range: {np.nanmin(optimal_depths):.0f} - {np.nanmax(optimal_depths):.0f}m")
    print(f"Std deviation: {np.nanstd(optimal_depths):.1f}m")

    return optimal_depths, optimal_temperatures, lcoe_vs_depth


def _get_region_bounds(region):
    """
    Get geographic bounds for a region

    This should load from download_ranges_per_region.csv
    For now, provides example bounds

    Args:
        region: Region name

    Returns:
        bounds: Dictionary with lon_min, lon_max, lat_min, lat_max
    """

    # Example bounds (should be loaded from CSV)
    region_bounds_dict = {
        'Caribbean': {
            'lon_min': -85.0,
            'lon_max': -60.0,
            'lat_min': 10.0,
            'lat_max': 25.0,
        },
        'Hawaii': {
            'lon_min': -162.0,
            'lon_max': -154.0,
            'lat_min': 18.0,
            'lat_max': 23.0,
        },
        'Indian_Ocean': {
            'lon_min': 50.0,
            'lon_max': 90.0,
            'lat_min': -20.0,
            'lat_max': 10.0,
        },
        # Add more regions as needed
    }

    if region in region_bounds_dict:
        return region_bounds_dict[region]
    else:
        # Default bounds (global)
        print(f"Warning: Unknown region '{region}', using default bounds")
        return {
            'lon_min': -180.0,
            'lon_max': 180.0,
            'lat_min': -60.0,
            'lat_max': 60.0,
        }


if __name__ == "__main__":
    # Test multi-depth download module
    print("Testing CMEMS Multi-Depth Download Module\n")
    print("="*70)

    # Test 1: Generate depth levels
    print("\n1. Generating depth levels for optimization:")
    depth_range = (400, 1500)
    step = 100
    depths = get_depth_levels_for_optimization(depth_range, step)
    print(f"   Depth range: {depth_range[0]}-{depth_range[1]}m, step: {step}m")
    print(f"   Generated depths: {depths}")

    # Test 2: Map to CMEMS depths
    print("\n2. Mapping to CMEMS standard depths:")
    cmems_standard = [0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500,
                      600, 700, 800, 900, 1000, 1200, 1500]
    nearest, mapping = get_nearest_cmems_depths(depths, cmems_standard)
    print(f"   CMEMS depths to download: {nearest}")
    print(f"\n   Mapping:")
    for target, actual in mapping.items():
        print(f"     {target}m -> {actual}m")

    # Test 3: Region bounds
    print("\n3. Getting region bounds:")
    regions = ['Caribbean', 'Hawaii', 'Unknown_Region']
    for region in regions:
        bounds = _get_region_bounds(region)
        print(f"   {region}: lon [{bounds['lon_min']}, {bounds['lon_max']}], "
              f"lat [{bounds['lat_min']}, {bounds['lat_max']}]")

    print("\n" + "="*70)
    print("Testing complete!")
    print("\nTo use this module:")
    print("  1. Ensure Copernicus Marine account is configured")
    print("  2. Call download_cmems_multi_depth() with your region")
    print("  3. Process with process_multi_depth_data()")
    print("  4. Optimize with optimize_depth_with_real_data()")
