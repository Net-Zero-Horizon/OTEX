
"""
OTEC Onshore vs Offshore Comparison Analysis

This script compares OTEC installations in two configurations:
1. OFFSHORE: Floating platform at optimal offshore location
2. ONSHORE: Land-based plant with long intake/outfall pipelines

Key differences:
- OFFSHORE: Higher structure/mooring/deployment costs, lower pipeline costs
- ONSHORE: Lower structure costs, no mooring, much higher pipeline costs

The analysis helps identify which configuration is more economical
for different regions based on distance to optimal thermal resource.

@author: OTEX Development Team
"""

import numpy as np

# NumPy 2.0 compatibility fix for pickled data
# Older HDF5 files may have been pickled with numpy.core (pre-2.0)
# but NumPy 2.0+ uses numpy._core
import sys
if not hasattr(np, 'core'):
    np.core = np._core

import pandas as pd
import xarray as xr
from datetime import datetime
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


# Load regional cold water depths
REGIONAL_CW_DEPTHS_FILE = 'regional_cw_depths.csv'
REGIONAL_CW_DEPTHS = {}


def calculate_depth_for_latitude(latitude):
    """
    Calculate recommended cold water intake depth based on latitude.

    The thermocline depth varies with latitude:
    - Near equator (0-10°): thermocline is shallower, ~800-900m needed
    - Mid-latitudes (10-25°): thermocline deeper, ~900-1100m needed
    - Higher latitudes (>25°): thermocline variable, ~1000-1200m needed

    This approximation is based on global ocean thermocline patterns
    and typical OTEC cold water temperature requirements (~4-6°C).

    Args:
        latitude: Latitude in degrees (absolute value used)

    Returns:
        depth: Recommended CW intake depth in meters
    """
    abs_lat = abs(latitude)

    # Piecewise linear approximation based on OTEC literature
    if abs_lat <= 10:
        # Tropical region: shallower thermocline
        depth = 800 + abs_lat * 10  # 800-900m
    elif abs_lat <= 25:
        # Subtropical region: deeper thermocline
        depth = 900 + (abs_lat - 10) * 13.3  # 900-1100m
    else:
        # Higher latitudes: deepest thermocline
        depth = 1100 + (abs_lat - 25) * 4  # 1100-1200m (capped)

    # Cap at reasonable OTEC limits
    depth = min(max(depth, 800), 1200)

    return int(depth)


def load_regional_depths(filepath='regional_cw_depths.csv'):
    """
    Load regional cold water intake depths from CSV file

    Returns:
        Dictionary {region_name: depth_m}
    """
    global REGIONAL_CW_DEPTHS

    if os.path.exists(filepath):
        depths_df = pd.read_csv(filepath)
        REGIONAL_CW_DEPTHS = dict(zip(depths_df['region'], depths_df['recommended_depth_m']))
        print(f"Loaded regional depths for {len(REGIONAL_CW_DEPTHS)} regions from {filepath}")
    else:
        print(f"Warning: Regional depths file not found: {filepath}")
        print("Will calculate depths from latitude on-the-fly for each region")
        print("(For better performance, run 'python regional_depth_calculator.py' to pre-generate depths)")

    return REGIONAL_CW_DEPTHS

def get_cw_depth_for_region(region_name, default=None):
    """
    Get recommended cold water intake depth for a region

    If regional depths CSV exists, uses precomputed value.
    If not found in CSV, calculates depth from region's mean latitude.
    If region not in sites database, falls back to default (1062m).

    Args:
        region_name: Name of the region
        default: Default depth if region not found anywhere [m]
                If None, uses 1062m

    Returns:
        depth: Recommended CW intake depth [m]
    """
    if default is None:
        default = 1062

    # Try to load from CSV first
    if not REGIONAL_CW_DEPTHS:
        load_regional_depths()

    # If found in CSV, use it
    if region_name in REGIONAL_CW_DEPTHS:
        return REGIONAL_CW_DEPTHS[region_name]

    # Not in CSV - calculate from latitude as fallback
    try:
        # Load sites to get mean latitude for this region
        sites_df = pd.read_csv('CMEMS_points_with_properties.csv', delimiter=';')
        region_sites = sites_df[sites_df['region'] == region_name]

        if len(region_sites) > 0:
            # Calculate mean latitude
            mean_lat = region_sites['latitude'].mean()

            # Calculate optimal depth for this latitude
            depth = calculate_depth_for_latitude(mean_lat)

            print(f"  Note: Calculated depth for {region_name} from latitude ({mean_lat:.2f}°): {depth}m")

            return int(depth)
        else:
            # Region not found in sites database
            print(f"  Warning: {region_name} not found in sites database, using default {default}m")
            return default

    except Exception as e:
        # Error loading sites or calculating
        print(f"  Warning: Could not calculate depth for {region_name}: {e}")
        print(f"  Using default depth: {default}m")
        return default


def save_netcdf_for_config(results_list, config_name, installation_type, cost_level, year, output_dir):
    """
    Save NetCDF file with spatial data for a specific configuration and installation type

    Args:
        results_list: List of result dictionaries from run_single_analysis
        config_name: Name of configuration (e.g., 'Rankine_Closed_Ammonia')
        installation_type: 'onshore' or 'offshore'
        cost_level: Cost level string
        year: Analysis year
        output_dir: Directory to save NetCDF files
    """
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    # Collect all site-level data from all regions
    all_p_net = []
    all_lcoe = []
    all_lats = []
    all_lons = []

    for result in results_list:
        if 'p_net_sites' in result and 'lcoe_sites' in result:
            # Convert to arrays and flatten to ensure 1D
            p_net_arr = np.atleast_1d(result['p_net_sites']).flatten()
            lcoe_arr = np.atleast_1d(result['lcoe_sites']).flatten()
            lats_arr = np.atleast_1d(result['lats']).flatten()
            lons_arr = np.atleast_1d(result['lons']).flatten()

            # Extend lists with individual values (not arrays)
            all_p_net.extend(p_net_arr.tolist())
            all_lcoe.extend(lcoe_arr.tolist())
            all_lats.extend(lats_arr.tolist())
            all_lons.extend(lons_arr.tolist())

    if len(all_p_net) == 0:
        print(f"Warning: No site-level data available for {config_name}_{installation_type}")
        return

    # Convert to numpy arrays
    all_p_net = np.array(all_p_net, dtype=np.float64)
    all_lcoe = np.array(all_lcoe, dtype=np.float64)
    all_lats = np.array(all_lats, dtype=np.float64)
    all_lons = np.array(all_lons, dtype=np.float64)

    # Create regular grid for raster (QGIS-compatible)
    # Use 0.05 degree resolution (~5.5 km at equator)
    lat_min, lat_max = all_lats.min(), all_lats.max()
    lon_min, lon_max = all_lons.min(), all_lons.max()

    # Add small margin
    lat_margin = (lat_max - lat_min) * 0.1
    lon_margin = (lon_max - lon_min) * 0.1

    lat_grid = np.arange(lat_min - lat_margin, lat_max + lat_margin, 0.05)
    lon_grid = np.arange(lon_min - lon_margin, lon_max + lon_margin, 0.05)

    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Interpolate scattered data to regular grid using nearest neighbor
    points = np.column_stack((all_lons, all_lats))

    p_net_grid = griddata(points, all_p_net, (lon_mesh, lat_mesh), method='nearest')
    lcoe_grid = griddata(points, all_lcoe, (lon_mesh, lat_mesh), method='nearest')

    # Set values far from any data point to NaN
    tree = cKDTree(points)
    distances, _ = tree.query(np.column_stack((lon_mesh.ravel(), lat_mesh.ravel())))
    distances = distances.reshape(lon_mesh.shape)

    # Mask out cells more than 0.15 degrees (~16 km) from any data point
    mask = distances > 0.15
    p_net_grid[mask] = np.nan
    lcoe_grid[mask] = np.nan

    # Create xarray Dataset with 2D spatial dimensions (QGIS raster format)
    ds = xr.Dataset(
        {
            'p_net_MW': (['lat', 'lon'], p_net_grid, {
                'long_name': 'Net power output',
                'units': 'MW',
                'description': 'Average net power production',
                'grid_mapping': 'crs'
            }),
            'lcoe': (['lat', 'lon'], lcoe_grid, {
                'long_name': 'Levelized Cost of Energy',
                'units': 'ct/kWh',
                'description': 'LCOE for each location',
                'grid_mapping': 'crs'
            }),
        },
        coords={
            'lat': (['lat'], lat_grid, {
                'long_name': 'Latitude',
                'units': 'degrees_north',
                'standard_name': 'latitude',
                'axis': 'Y'
            }),
            'lon': (['lon'], lon_grid, {
                'long_name': 'Longitude',
                'units': 'degrees_east',
                'standard_name': 'longitude',
                'axis': 'X'
            }),
        },
        attrs={
            'title': f'OTEC Analysis - {config_name} ({installation_type})',
            'configuration': config_name,
            'installation_type': installation_type,
            'cost_level': cost_level,
            'year': year,
            'creation_date': datetime.now().isoformat(),
            'description': f'OTEC {installation_type} - Gridded results for GIS',
            'data_source': 'CMEMS Global Ocean Physics Analysis',
            'Conventions': 'CF-1.8',
            'n_sites': len(all_p_net),
            'grid_resolution': '0.05 degrees',
        }
    )

    # Add CRS information for QGIS
    ds['crs'] = xr.DataArray(0, attrs={
        'grid_mapping_name': 'latitude_longitude',
        'longitude_of_prime_meridian': 0.0,
        'semi_major_axis': 6378137.0,
        'inverse_flattening': 298.257223563,
    })

    # Save NetCDF file with compression
    encoding = {
        'p_net_MW': {'zlib': True, 'complevel': 4, '_FillValue': -9999.0},
        'lcoe': {'zlib': True, 'complevel': 4, '_FillValue': -9999.0}
    }

    nc_filename = os.path.join(output_dir, f'{config_name}_{installation_type}_{cost_level}.nc')
    ds.to_netcdf(nc_filename, encoding=encoding, format='NETCDF4')
    print(f" Saved NetCDF raster: {nc_filename} ({len(lat_grid)}x{len(lon_grid)} grid, {len(all_p_net)} data points)")


def download_region_data_comparison(region, cost_level, year):
    """
    Download and process data for a single region (for comparison analysis)
    This is run ONCE per region before any calculations
    """
    from otex.data import cmems as Cdp
    import pandas as pd
    import os
    from otex.config import parameters_and_constants

    try:
        # Initialize inputs
        p_gross = -136000  # kW (136 MW)
        inputs = parameters_and_constants(
            p_gross=p_gross,
            cost_level=cost_level,
            fluid_type='ammonia',
            cycle_type='rankine_closed',
            use_coolprop=False,
            year=year,
        )

        # Get regional cold water depth (latitude-based)
        cw_depth = get_cw_depth_for_region(region)
        inputs['length_CW_inlet'] = cw_depth

        # Create data directory
        new_path = f'Data_Results/{region.replace(" ", "_")}/'
        os.makedirs(new_path, exist_ok=True)

        print(f"  [Download] {region}: Starting (CW depth: {cw_depth}m)...", flush=True)

        # Download CMEMS data
        files = Cdp.download_data(cost_level, inputs, region, new_path)

        # Load sites and process data
        sites_df = pd.read_csv('CMEMS_points_with_properties.csv', delimiter=';')
        sites_df = sites_df[(sites_df['region'] == region) &
                           (sites_df['water_depth'] <= inputs['min_depth']) &
                           (sites_df['water_depth'] >= inputs['max_depth'])]
        sites_df = sites_df.sort_values(by=['longitude', 'latitude'], ascending=True)

        if len(sites_df) == 0:
            print(f"  [Download] {region}: No valid sites", flush=True)
            return {'region': region, 'success': False}

        # Process cold water
        print(f"  [Download] {region}: Processing CW...", flush=True)
        T_CW_profiles, T_CW_design, coordinates_CW, id_sites, timestamp, inputs, nan_columns_CW = \
            Cdp.data_processing(files[int(len(files)/2):int(len(files))], sites_df,
                               inputs, region, new_path, 'CW')

        # Process warm water
        print(f"  [Download] {region}: Processing WW...", flush=True)
        T_WW_profiles, T_WW_design, coordinates_WW, id_sites, timestamp, inputs, nan_columns_WW = \
            Cdp.data_processing(files[0:int(len(files)/2)], sites_df,
                               inputs, region, new_path, 'WW', nan_columns_CW)

        print(f"   [Download] {region}: Complete", flush=True)
        return {'region': region, 'success': True}

    except Exception as e:
        print(f"  ✗ [Download] {region}: Failed - {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {'region': region, 'success': False}


def run_single_analysis(region, config, installation_type, cost_level, year):
    """
    Run OTEC analysis for a single region, configuration, and installation type
    ASSUMES DATA HAS ALREADY BEEN DOWNLOADED

    Args:
        region: Region name
        config: Configuration dictionary (cycle + fluid)
        installation_type: 'onshore' or 'offshore'
        cost_level: 'low_cost' or 'high_cost'
        year: Year for analysis

    Returns:
        Dictionary with results or None if failed
    """
    # Import modules (needed for each worker process)
    from otex.data import cmems as Cdp
    from otex.plant import off_design_analysis as oda
    from otex.config import parameters_and_constants
    from otex.core import get_working_fluid, get_thermodynamic_cycle
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    config_name = f"{config['cycle']}_{config['fluid']}_{installation_type}"

    try:
        # Initialize inputs
        p_gross = -136000  # kW (136 MW)

        # Get fluid and cycle type strings for unique file naming
        fluid_type_str = config['fluid_config']['type'] if config['fluid_config'] else 'seawater'
        cycle_type_str = config['cycle_config']['type']

        inputs = parameters_and_constants(
            p_gross=p_gross,
            cost_level=cost_level,
            fluid_type=fluid_type_str,
            cycle_type=cycle_type_str,
            use_coolprop=config['fluid_config']['use_coolprop'] if config['fluid_config'] else False,
            year=year,
        )

        # Set installation type
        inputs['installation_type'] = installation_type

        # Store config strings for unique HDF5 filename generation
        inputs['config_cycle_type'] = cycle_type_str
        inputs['config_fluid_type'] = fluid_type_str

        # Get regional cold water depth (latitude-based)
        cw_depth = get_cw_depth_for_region(region)
        inputs['length_CW_inlet'] = cw_depth

        # Data directory
        new_path = f"Data_Results/{region.replace(' ', '_')}/"

        # Load sites from CSV
        sites_df = pd.read_csv('CMEMS_points_with_properties.csv', delimiter=';')
        sites_df = sites_df[(sites_df['region'] == region) &
                           (sites_df['water_depth'] <= inputs['min_depth']) &
                           (sites_df['water_depth'] >= inputs['max_depth'])]
        sites_df = sites_df.sort_values(by=['longitude', 'latitude'], ascending=True)

        if len(sites_df) == 0:
            tqdm.write(f"  ✗ [{config_name}] {region}: No valid sites")
            return None

        # Load pre-processed data from .h5 files
        # Look for existing .h5 files in the directory (created during Phase 1)
        # These files are named based on the depth used during download
        import glob

        h5_files = glob.glob(f'{new_path}T_*m_{year}_{region}.h5'.replace(" ","_"))

        if len(h5_files) < 2:
            tqdm.write(f"  ✗ [{config_name}] {region}: Pre-processed data files not found (found {len(h5_files)} files)")
            return None

        # Sort files by depth (extract depth from filename)
        def extract_depth(filepath):
            import re
            match = re.search(r'T_(\d+)m_', os.path.basename(filepath))
            return int(match.group(1)) if match else 0

        h5_files_sorted = sorted(h5_files, key=extract_depth)
        h5_file_WW = h5_files_sorted[0]  # Shallowest (warm water)
        h5_file_CW = h5_files_sorted[-1]  # Deepest (cold water)

        # Load cold water data
        T_CW_profiles, T_CW_design, coordinates_CW, id_sites_CW, timestamp, inputs, nan_columns_CW = \
            Cdp.load_temperatures(h5_file_CW, inputs)

        # Load warm water data (need to save CW's dist_shore/eff_trans before they get overwritten)
        dist_shore_CW = inputs.get('dist_shore', None)
        eff_trans_CW = inputs.get('eff_trans', None)

        T_WW_profiles, T_WW_design, coordinates_WW, id_sites_WW, timestamp, inputs, nan_columns_WW = \
            Cdp.load_temperatures(h5_file_WW, inputs)

        dist_shore_WW = inputs.get('dist_shore', None)
        eff_trans_WW = inputs.get('eff_trans', None)

        # Check if data is valid
        if T_WW_profiles is None or T_CW_profiles is None:
            tqdm.write(f"  ✗ [{config_name}] {region}: No valid data")
            return None

        # =====================================================================
        # SYNCHRONIZE SITES: Ensure WW and CW have the same sites
        # This is critical because WW and CW may have different data availability
        # =====================================================================

        # Create site keys for both datasets (lon_lat format)
        def create_site_keys(coordinates):
            """Create unique site identifiers from coordinates"""
            coordinates = np.atleast_1d(coordinates)

            if coordinates.ndim == 1 and coordinates.size >= 2:
                # Single coordinate pair [lon, lat]
                return [f"{coordinates[0]:.3f}_{coordinates[1]:.3f}"]
            elif coordinates.ndim == 1:
                # Invalid: 1D array with less than 2 elements
                raise ValueError(f"Coordinate array has insufficient elements: {coordinates.size}")

            # 2D array of coordinate pairs - validate shape
            if coordinates.shape[1] < 2:
                raise ValueError(f"Coordinate array must have at least 2 columns (lon, lat), "
                               f"got shape {coordinates.shape}")

            return [f"{c[0]:.3f}_{c[1]:.3f}" for c in coordinates]

        keys_CW = create_site_keys(coordinates_CW)
        keys_WW = create_site_keys(coordinates_WW)

        # Find common sites (intersection)
        common_keys = set(keys_CW) & set(keys_WW)

        if len(common_keys) == 0:
            tqdm.write(f"  ✗ [{config_name}] {region}: No common sites between WW and CW data")
            return None

        # Log if there's a mismatch
        if len(keys_CW) != len(keys_WW) or len(common_keys) != len(keys_CW):
            tqdm.write(f"  ⚠ [{config_name}] {region}: Site mismatch - WW:{len(keys_WW)}, CW:{len(keys_CW)}, common:{len(common_keys)}")

        # Get indices of common sites in each dataset
        idx_CW = [i for i, k in enumerate(keys_CW) if k in common_keys]
        idx_WW = [i for i, k in enumerate(keys_WW) if k in common_keys]

        # Sort indices to maintain consistent ordering (by site key)
        # Create mapping from key to index for proper alignment
        key_to_idx_CW = {k: i for i, k in enumerate(keys_CW) if k in common_keys}
        key_to_idx_WW = {k: i for i, k in enumerate(keys_WW) if k in common_keys}

        # Sort by common keys to ensure WW and CW have matching order
        sorted_common_keys = sorted(common_keys)
        idx_CW_sorted = [key_to_idx_CW[k] for k in sorted_common_keys]
        idx_WW_sorted = [key_to_idx_WW[k] for k in sorted_common_keys]

        # Filter CW data
        T_CW_profiles = T_CW_profiles[:, idx_CW_sorted]
        T_CW_design = T_CW_design[:, idx_CW_sorted] if T_CW_design.ndim > 1 else T_CW_design
        coordinates_CW = coordinates_CW[idx_CW_sorted] if coordinates_CW.ndim > 1 else coordinates_CW

        # Filter WW data
        T_WW_profiles = T_WW_profiles[:, idx_WW_sorted]
        T_WW_design = T_WW_design[:, idx_WW_sorted] if T_WW_design.ndim > 1 else T_WW_design
        coordinates_WW = coordinates_WW[idx_WW_sorted] if coordinates_WW.ndim > 1 else coordinates_WW

        # Update inputs with synchronized dist_shore and eff_trans (use CW's values for common sites)
        if dist_shore_CW is not None and len(idx_CW_sorted) > 0:
            if dist_shore_CW.ndim > 1:
                inputs['dist_shore'] = dist_shore_CW[:, idx_CW_sorted]
            else:
                inputs['dist_shore'] = dist_shore_CW[idx_CW_sorted]

        if eff_trans_CW is not None and len(idx_CW_sorted) > 0:
            if eff_trans_CW.ndim > 1:
                inputs['eff_trans'] = eff_trans_CW[:, idx_CW_sorted]
            else:
                inputs['eff_trans'] = eff_trans_CW[idx_CW_sorted]

        # Use CW coordinates as the reference (they should now be identical)
        coordinates = coordinates_CW

        # =====================================================================
        # END SYNCHRONIZATION
        # =====================================================================

        # Check if design temperature arrays are not empty
        T_WW_design_arr = np.atleast_1d(T_WW_design)
        T_CW_design_arr = np.atleast_1d(T_CW_design)

        if T_WW_design_arr.size == 0 or T_CW_design_arr.size == 0:
            tqdm.write(f"  ✗ [{config_name}] {region}: Empty design temperature arrays after synchronization")
            return None

        # Set up working fluid and cycle
        if config['fluid_config']:
            working_fluid = get_working_fluid(
                config['fluid_config']['type'],
                use_coolprop=config['fluid_config']['use_coolprop']
            )
            inputs['working_fluid'] = working_fluid

            cycle = get_thermodynamic_cycle(
                config['cycle_config']['type'],
                working_fluid=working_fluid
            )
            inputs['thermodynamic_cycle'] = cycle
        else:
            cycle = get_thermodynamic_cycle(config['cycle_config']['type'])
            inputs['thermodynamic_cycle'] = cycle

        # Run OTEC analysis
        otec_plants, capex_opex_comparison = oda.off_design_analysis(
            T_WW_design, T_CW_design, T_WW_profiles, T_CW_profiles,
            inputs, coordinates_CW, timestamp, region, new_path, cost_level
        )

        if otec_plants is None or 'p_net' not in otec_plants:
            return None

        # Extract results
        p_net_sites = -np.nanmean(otec_plants['p_net'], axis=0) / 1000  # MW per site
        lcoe_sites = otec_plants['LCOE']  # LCOE per site
        lats = coordinates_CW[:, 1]
        lons = coordinates_CW[:, 0]

        # Get distance to shore - ensure scalar value
        dist_shore_array = inputs['dist_shore']
        if hasattr(dist_shore_array, '__len__'):
            # It's an array, take mean (they should all be the same for a region)
            dist_shore = float(np.mean(dist_shore_array))
        else:
            dist_shore = float(dist_shore_array)

        result = {
            'region': region,
            'configuration': config_name,
            'installation_type': installation_type,
            'p_net_avg_MW': float(p_net_sites.mean()),
            'p_net_max_MW': float(p_net_sites.max()),
            'lcoe_avg': float(lcoe_sites.mean()),
            'lcoe_min': float(lcoe_sites.min()),
            'dist_shore_km': dist_shore,
            'n_sites': int(len(coordinates_CW)),
            'T_WW_avg': float(T_WW_profiles.mean()),
            'T_CW_avg': float(T_CW_profiles.mean()),
            'deltaT_avg': float((T_WW_profiles - T_CW_profiles).mean()),
            # Site-level data (arrays)
            'p_net_sites': p_net_sites,
            'lcoe_sites': lcoe_sites,
            'lats': lats,
            'lons': lons,
            # Cost breakdown (from first site)
            'costs': capex_opex_comparison[0][0] if capex_opex_comparison else None
        }

        tqdm.write(f"   [{config_name}] {region}: P_net={result['p_net_avg_MW']:.1f} MW, LCOE={result['lcoe_avg']:.2f} ct/kWh")
        return result

    except Exception as e:
        tqdm.write(f"  ✗ [{config_name}] {region} failed: {str(e)}")
        import traceback
        tqdm.write(traceback.format_exc())
        return None


def compare_onshore_offshore(regions, configurations, cost_level='low_cost', year=2020, parallel=True):
    """
    Compare onshore vs offshore installations for given regions and configurations
    TWO-PHASE EXECUTION:
      PHASE 1: Download all data for all regions (sequential)
      PHASE 2: Parallel calculation for all (region, config, installation_type) combinations

    Args:
        regions: List of region names to analyze
        configurations: List of configuration dictionaries
        cost_level: 'low_cost' or 'high_cost'
        year: Year for analysis
        parallel: Whether to run in parallel

    Returns:
        DataFrame with comparison results
    """
    print("="*80)
    print("OTEC ONSHORE VS OFFSHORE COMPARISON (TWO-PHASE EXECUTION)")
    print("="*80)
    print(f"\nRegions: {len(regions)}")
    print(f"Configurations: {len(configurations)}")
    print(f"Installation types: 2 (onshore, offshore)")
    print(f"Total analyses: {len(regions) * len(configurations) * 2}")
    print(f"Cost level: {cost_level}")
    print(f"Year: {year}")
    print(f"Parallel: {parallel}")
    print("="*80 + "\n")

    # Load regional cold water depths
    print("Loading regional cold water intake depths...")
    regional_depths = load_regional_depths()
    if regional_depths:
        print(f"Using latitude-based depths for {len(regional_depths)} regions")
        print(f"Depth range: {min(regional_depths.values())}m - {max(regional_depths.values())}m")
    else:
        print("Using fixed depth (1062m) for all regions")
    print()

    # ========================================================================
    # PHASE 1: DOWNLOAD AND PRE-PROCESS DATA FOR ALL REGIONS
    # ========================================================================
    print("="*80)
    print(f"PHASE 1: DOWNLOADING DATA FOR {len(regions)} REGIONS")
    print("="*80)
    print("Each region is downloaded ONCE and shared across all configurations\n")

    downloaded_regions = []
    failed_regions = []

    for i, region in enumerate(regions):
        print(f"[{i+1}/{len(regions)}] Downloading {region}...")
        result = download_region_data_comparison(region, cost_level, year)
        if result['success']:
            downloaded_regions.append(region)
        else:
            failed_regions.append(region)

    print("\n" + "-"*80)
    print(f"PHASE 1 COMPLETE")
    print(f"   Successfully downloaded: {len(downloaded_regions)} regions")
    print(f"  ✗ Failed: {len(failed_regions)} regions")
    if failed_regions:
        print(f"  Failed regions: {', '.join(failed_regions)}")
    print("-"*80 + "\n")

    # ========================================================================
    # PHASE 2: PARALLEL CALCULATION
    # ========================================================================
    print("="*80)
    print(f"PHASE 2: RUNNING CALCULATIONS")
    print("="*80)
    print(f"Regions: {len(downloaded_regions)}")
    print(f"Configurations: {len(configurations)}")
    print(f"Installation types: 2")
    print(f"Total jobs: {len(downloaded_regions)} × {len(configurations)} × 2 = {len(downloaded_regions) * len(configurations) * 2}\n")

    # Create list of all calculation tasks (only for downloaded regions)
    tasks = []
    for region in downloaded_regions:
        for config in configurations:
            for installation_type in ['offshore', 'onshore']:
                tasks.append((region, config, installation_type, cost_level, year))

    results = []

    if parallel:
        # Parallel execution
        max_workers = min(multiprocessing.cpu_count() - 1, len(tasks))
        print(f"Using {max_workers} parallel workers for calculations\n")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single_analysis, *task): task for task in tasks}

            # Progress tracking
            completed = 0
            successful = 0
            failed = 0
            total_tasks = len(futures)

            with tqdm(total=total_tasks, desc="Progress", unit="task",
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                for future in as_completed(futures):
                    region, config, inst_type, _, _ = futures[future]
                    config_name = f"{config['cycle']}_{config['fluid']}_{inst_type}"

                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            successful += 1
                        else:
                            failed += 1
                        completed += 1
                    except Exception as e:
                        failed += 1
                        completed += 1
                        tqdm.write(f"  ✗ ERROR: {region} - {config_name}: {str(e)[:50]}")

                    # Update progress bar description with absolute counts
                    pbar.set_description(f"Progress [ {successful} | ✗ {failed}]")
                    pbar.update(1)

        # Print summary
        print("\n" + "-"*80)
        print(f"PHASE 2 COMPLETE")
        print(f"  Total tasks: {total_tasks}")
        print(f"   Successful: {successful}")
        print(f"  ✗ Failed: {failed}")
        print(f"  Success rate: {successful/total_tasks*100:.1f}%")
        print("-"*80 + "\n")
    else:
        # Sequential execution
        for task in tqdm(tasks, desc="Calculating", unit="task"):
            result = run_single_analysis(*task)
            if result:
                results.append(result)

    # Save intermediate results to pickle for later NetCDF regeneration
    import pickle
    intermediate_file = f'./OTEC_Comparison/intermediate_results_{cost_level}_{year}.pkl'

    # Load existing results if available
    existing_results = []
    if os.path.exists(intermediate_file):
        try:
            with open(intermediate_file, 'rb') as f:
                existing_results = pickle.load(f)
            print(f"\n Loaded {len(existing_results)} existing results from {intermediate_file}")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
            existing_results = []

    # Combine with current results, avoiding duplicates
    combined_results = existing_results.copy()

    # Create a set of existing (region, config, type) tuples
    existing_keys = {(r['region'], r['configuration'], r['installation_type'])
                     for r in existing_results}

    # Add new results that don't exist yet
    new_count = 0
    for r in results:
        key = (r['region'], r['configuration'], r['installation_type'])
        if key not in existing_keys:
            combined_results.append(r)
            new_count += 1

    print(f" Added {new_count} new results (total: {len(combined_results)})")

    # Save combined results
    os.makedirs('./OTEC_Comparison/', exist_ok=True)
    try:
        with open(intermediate_file, 'wb') as f:
            pickle.dump(combined_results, f)
        print(f" Saved intermediate results to {intermediate_file}")
    except Exception as e:
        print(f"Warning: Could not save intermediate results: {e}")

    # Create comparison DataFrame and NetCDF files
    if results:
        # Create DataFrame for CURRENT session results
        df = pd.DataFrame(results)

        # Reorder columns for better readability
        cols = ['region', 'configuration', 'installation_type', 'dist_shore_km',
                'p_net_avg_MW', 'lcoe_avg', 'lcoe_min',
                'T_WW_avg', 'T_CW_avg', 'deltaT_avg', 'n_sites']
        df = df[cols + [c for c in df.columns if c not in cols]]

        # Also create DataFrame with ALL accumulated results (for comprehensive CSV)
        # Remove site-level data arrays before creating full DataFrame (too large for CSV)
        combined_results_for_csv = []
        for r in combined_results:
            r_copy = r.copy()
            # Remove arrays that shouldn't be in CSV
            r_copy.pop('p_net_sites', None)
            r_copy.pop('lcoe_sites', None)
            r_copy.pop('lats', None)
            r_copy.pop('lons', None)
            combined_results_for_csv.append(r_copy)

        df_all = pd.DataFrame(combined_results_for_csv)
        if len(df_all) > 0:
            df_all = df_all[cols + [c for c in df_all.columns if c not in cols]]

        # Create NetCDF files for each configuration and installation type
        # IMPORTANT: Use combined_results (all regions) instead of results (current session only)
        print("\n" + "="*80)
        print("CREATING NETCDF FILES FOR SPATIAL VISUALIZATION")
        print("="*80)
        print(f"Using ALL accumulated results: {len(combined_results)} analyses")
        print(f"  Current session: {len(results)}")
        print(f"  Previous sessions: {len(existing_results)}")

        # Count unique regions
        unique_regions_combined = len(set(r['region'] for r in combined_results))
        print(f"  Total unique regions: {unique_regions_combined}")

        # Get output directory (same as CSV output)
        output_dir = './OTEC_Comparison/'
        os.makedirs(output_dir, exist_ok=True)

        # Group results by configuration base name and installation type
        for config in configurations:
            config_base = f"{config['cycle']}_{config['fluid']}"

            for inst_type in ['offshore', 'onshore']:
                # Filter results for this config and installation type
                # USE COMBINED_RESULTS instead of results to include ALL regions
                config_results = [r for r in combined_results
                                if r['configuration'] == f"{config_base}_{inst_type}"]

                if config_results:
                    print(f"  {config_base}_{inst_type}: {len(config_results)} regions")
                    save_netcdf_for_config(
                        results_list=config_results,
                        config_name=config_base,
                        installation_type=inst_type,
                        cost_level=cost_level,
                        year=year,
                        output_dir=output_dir
                    )

        print("="*80 + "\n")

        # Return both current session results and complete accumulated results
        return_dict = {
            'current_session': df,
            'all_accumulated': df_all if len(df_all) > 0 else df
        }
        return return_dict
    else:
        print("\nNo results generated!")
        return None


def regenerate_netcdf_from_pickle(cost_level='low_cost', year=2020, output_dir='./OTEC_Comparison/'):
    """
    Regenerate NetCDF files from saved pickle file without re-running analysis

    This is useful if:
    - You want to regenerate NetCDF with different settings
    - NetCDF files were corrupted or deleted
    - You want to update NetCDF after accumulating more results

    Args:
        cost_level: Cost level used in analysis
        year: Year used in analysis
        output_dir: Output directory for NetCDF files

    Returns:
        True if successful, False otherwise
    """
    import pickle

    intermediate_file = f'./OTEC_Comparison/intermediate_results_{cost_level}_{year}.pkl'

    if not os.path.exists(intermediate_file):
        print(f"Error: Intermediate results file not found: {intermediate_file}")
        print("You need to run the analysis first to generate this file.")
        return False

    print("="*80)
    print("REGENERATING NETCDF FILES FROM PICKLE")
    print("="*80)

    # Load results
    try:
        with open(intermediate_file, 'rb') as f:
            combined_results = pickle.load(f)
        print(f" Loaded {len(combined_results)} results from {intermediate_file}")
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return False

    unique_regions = len(set(r['region'] for r in combined_results))
    print(f"  Total unique regions: {unique_regions}")

    # Get all unique configurations
    unique_configs = set()
    for r in combined_results:
        config_name = r['configuration'].rsplit('_', 1)[0]  # Remove _onshore/_offshore
        unique_configs.add(config_name)

    print(f"  Total unique configurations: {len(unique_configs)}")

    # Define configurations (same as in main)
    configurations = [
        {'cycle': 'Rankine_Closed', 'fluid': 'Ammonia'},
        {'cycle': 'Rankine_Closed', 'fluid': 'R134a'},
        {'cycle': 'Rankine_Closed', 'fluid': 'R245fa'},
        {'cycle': 'Rankine_Closed', 'fluid': 'Propane'},
        {'cycle': 'Rankine_Closed', 'fluid': 'Isobutane'},
        {'cycle': 'Rankine_Open', 'fluid': 'Seawater'},
        {'cycle': 'Rankine_Hybrid', 'fluid': 'Ammonia'},
        {'cycle': 'Rankine_Hybrid', 'fluid': 'R134a'},
        {'cycle': 'Rankine_Hybrid', 'fluid': 'R245fa'},
        {'cycle': 'Rankine_Hybrid', 'fluid': 'Propane'},
        {'cycle': 'Rankine_Hybrid', 'fluid': 'Isobutane'},
        {'cycle': 'Kalina', 'fluid': 'NH3-H2O'},
        {'cycle': 'Uehara', 'fluid': 'NH3-H2O'},
    ]

    os.makedirs(output_dir, exist_ok=True)

    print("\nRegenerating NetCDF files...")
    print("-"*80)

    # Generate NetCDF for each configuration and installation type
    for config in configurations:
        config_base = f"{config['cycle']}_{config['fluid']}"

        for inst_type in ['offshore', 'onshore']:
            config_results = [r for r in combined_results
                            if r['configuration'] == f"{config_base}_{inst_type}"]

            if config_results:
                print(f"  {config_base}_{inst_type}: {len(config_results)} regions")
                save_netcdf_for_config(
                    results_list=config_results,
                    config_name=config_base,
                    installation_type=inst_type,
                    cost_level=cost_level,
                    year=year,
                    output_dir=output_dir
                )
            else:
                print(f"  {config_base}_{inst_type}: No data")

    print("="*80)
    print(" NetCDF regeneration complete!")
    print("="*80)
    return True


def create_comparison_plots(df, output_dir='./OTEC_Comparison/'):
    """
    Create comparison plots for onshore vs offshore

    Args:
        df: DataFrame with comparison results
        output_dir: Directory to save plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. LCOE comparison by distance
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: LCOE vs distance for each configuration
    for config in df['configuration'].unique():
        config_base = config.rsplit('_', 1)[0]  # Remove _onshore/_offshore
        df_config = df[df['configuration'].str.contains(config_base)]

        ax = axes[0, 0]
        for inst_type in ['offshore', 'onshore']:
            data = df_config[df_config['installation_type'] == inst_type]
            # Ensure data is numeric and not arrays
            x_data = pd.to_numeric(data['dist_shore_km'], errors='coerce').values
            y_data = pd.to_numeric(data['lcoe_avg'], errors='coerce').values
            ax.scatter(x_data, y_data,
                      label=f'{config_base} ({inst_type})', alpha=0.7, s=100)

    axes[0, 0].set_xlabel('Distance to Shore (km)', fontsize=12)
    axes[0, 0].set_ylabel('Average LCOE (ct/kWh)', fontsize=12)
    axes[0, 0].set_title('LCOE vs Distance to Shore', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: LCOE difference (onshore - offshore)
    configs_base = df['configuration'].str.rsplit('_', n=1).str[0].unique()
    for config_base in configs_base:
        df_offshore = df[(df['configuration'] == f'{config_base}_offshore')].copy()
        df_onshore = df[(df['configuration'] == f'{config_base}_onshore')].copy()

        if len(df_offshore) > 0 and len(df_onshore) > 0:
            # Merge on region
            merged = df_offshore.merge(df_onshore, on='region', suffixes=('_off', '_on'))
            merged['lcoe_diff'] = pd.to_numeric(merged['lcoe_avg_on'], errors='coerce') - pd.to_numeric(merged['lcoe_avg_off'], errors='coerce')

            # Ensure data is numeric
            x_data = pd.to_numeric(merged['dist_shore_km_off'], errors='coerce').values
            y_data = merged['lcoe_diff'].values
            axes[0, 1].scatter(x_data, y_data,
                              label=config_base, alpha=0.7, s=100)

    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Distance to Shore (km)', fontsize=12)
    axes[0, 1].set_ylabel('LCOE Difference: Onshore - Offshore (ct/kWh)', fontsize=12)
    axes[0, 1].set_title('Cost Advantage: Negative = Onshore Better', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Average power by installation type
    power_data = df.groupby(['configuration', 'installation_type'])['p_net_avg_MW'].mean().unstack()
    power_data.plot(kind='bar', ax=axes[1, 0], alpha=0.7)
    axes[1, 0].set_xlabel('Configuration', fontsize=12)
    axes[1, 0].set_ylabel('Average Power (MW)', fontsize=12)
    axes[1, 0].set_title('Average Power Output by Configuration', fontsize=14, fontweight='bold')
    axes[1, 0].legend(title='Installation Type')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 4: LCOE by region
    lcoe_pivot = df.pivot_table(values='lcoe_avg', index='region',
                                 columns='installation_type', aggfunc='mean')
    lcoe_pivot.plot(kind='bar', ax=axes[1, 1], alpha=0.7)
    axes[1, 1].set_xlabel('Region', fontsize=12)
    axes[1, 1].set_ylabel('Average LCOE (ct/kWh)', fontsize=12)
    axes[1, 1].set_title('LCOE by Region and Installation Type', fontsize=14, fontweight='bold')
    axes[1, 1].legend(title='Installation Type')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'onshore_offshore_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\nPlot saved: {os.path.join(output_dir, 'onshore_offshore_comparison.png')}")


def main(region_subset=None, node_id=None):
    """
    Main execution function

    Args:
        region_subset: Optional tuple (start_idx, end_idx) to process only a subset of regions
        node_id: Optional node identifier for logging
    """
    print("#"*80)
    print("# OTEC ONSHORE VS OFFSHORE COMPARISON ANALYSIS")
    if node_id:
        print(f"# NODE: {node_id}")
    if region_subset:
        print(f"# REGION SUBSET: {region_subset[0]} to {region_subset[1]}")
    print("#"*80)

    unique_regions = pd.read_csv('download_ranges_per_region.csv',delimiter=';',encoding='latin-1').drop_duplicates(subset=['region'])
    all_regions = list(unique_regions['region'])

    # Apply region subset if specified
    if region_subset:
        start_idx, end_idx = region_subset
        regions = all_regions[start_idx:end_idx]
        print(f"\nProcessing {len(regions)} regions (indices {start_idx}-{end_idx} of {len(all_regions)} total)")
    else:
        regions = all_regions
        print(f"\nProcessing all {len(regions)} regions")

    # # Define regions to test
    # regions = [
    #     'Jamaica',
    #     'Philippines',
    #     'Maldives',
    #     'Cuba',
    # ]


    # OPTION 2: Complete comparison (13 configurations - ALL CYCLES)
    # - 5 Rankine Closed configurations (5 working fluids)
    # - 1 Rankine Open configuration (seawater)
    # - 5 Rankine Hybrid configurations (5 working fluids)
    # - 1 Kalina configuration (NH3-H2O mixture)
    # - 1 Uehara configuration (NH3-H2O mixture)
    configurations = [
        # Rankine Closed Cycle - All working fluids
        {
            'cycle': 'Rankine_Closed',
            'fluid': 'Ammonia',
            'cycle_config': {'type': 'rankine_closed'},
            'fluid_config': {'type': 'ammonia', 'use_coolprop': True}
        },
        {
            'cycle': 'Rankine_Closed',
            'fluid': 'R134a',
            'cycle_config': {'type': 'rankine_closed'},
            'fluid_config': {'type': 'R134a', 'use_coolprop': True}
        },
        {
            'cycle': 'Rankine_Closed',
            'fluid': 'R245fa',
            'cycle_config': {'type': 'rankine_closed'},
            'fluid_config': {'type': 'R245fa', 'use_coolprop': True}
        },
        {
            'cycle': 'Rankine_Closed',
            'fluid': 'Propane',
            'cycle_config': {'type': 'rankine_closed'},
            'fluid_config': {'type': 'propane', 'use_coolprop': True}
        },
        {
            'cycle': 'Rankine_Closed',
            'fluid': 'Isobutane',
            'cycle_config': {'type': 'rankine_closed'},
            'fluid_config': {'type': 'isobutane', 'use_coolprop': True}
        },
        # Rankine Open Cycle
        {
            'cycle': 'Rankine_Open',
            'fluid': 'Seawater',
            'cycle_config': {'type': 'rankine_open'},
            'fluid_config': None  # Open cycle uses seawater directly
        },
        # Rankine Hybrid Cycle (Open-Closed) - All working fluids
        {
            'cycle': 'Rankine_Hybrid',
            'fluid': 'Ammonia',
            'cycle_config': {'type': 'rankine_hybrid'},
            'fluid_config': {'type': 'ammonia', 'use_coolprop': True}
        },
        {
            'cycle': 'Rankine_Hybrid',
            'fluid': 'R134a',
            'cycle_config': {'type': 'rankine_hybrid'},
            'fluid_config': {'type': 'R134a', 'use_coolprop': True}
        },
        {
            'cycle': 'Rankine_Hybrid',
            'fluid': 'R245fa',
            'cycle_config': {'type': 'rankine_hybrid'},
            'fluid_config': {'type': 'R245fa', 'use_coolprop': True}
        },
        {
            'cycle': 'Rankine_Hybrid',
            'fluid': 'Propane',
            'cycle_config': {'type': 'rankine_hybrid'},
            'fluid_config': {'type': 'propane', 'use_coolprop': True}
        },
        {
            'cycle': 'Rankine_Hybrid',
            'fluid': 'Isobutane',
            'cycle_config': {'type': 'rankine_hybrid'},
            'fluid_config': {'type': 'isobutane', 'use_coolprop': True}
        },
        # Kalina Cycle - uses NH3-H2O mixture (creates its own fluid internally)
        {
            'cycle': 'Kalina',
            'fluid': 'NH3-H2O',
            'cycle_config': {'type': 'kalina', 'ammonia_concentration': 0.7},
            'fluid_config': None  # Kalina cycle creates its own NH3-H2O mixture
        },
        # Uehara Cycle (Two-stage Rankine) - uses NH3-H2O mixture
        {
            'cycle': 'Uehara',
            'fluid': 'NH3-H2O',
            'cycle_config': {'type': 'uehara', 'ammonia_concentration': 0.7},
            'fluid_config': None  # Uehara cycle creates its own NH3-H2O mixture
        },
    ]
    
    print(f"Configurations: {len(configurations)}")
    print(f"Total analyses: {len(regions)} regions × {len(configurations)} configs × 2 installations = {len(regions) * len(configurations) * 2}")
    print()

    # Run comparison
    results = compare_onshore_offshore(
        regions=regions,
        configurations=configurations,
        cost_level='low_cost',
        year=2020,
        parallel=True
    )

    if results is not None:
        # Extract DataFrames
        df_current = results['current_session']
        df_all = results['all_accumulated']

        # Save results
        output_dir = './OTEC_Comparison/'
        os.makedirs(output_dir, exist_ok=True)

        # Save current session results
        if node_id:
            csv_filename = f'onshore_offshore_comparison_{node_id}.csv'
        else:
            csv_filename = 'onshore_offshore_comparison_session.csv'

        csv_path = os.path.join(output_dir, csv_filename)
        df_current.to_csv(csv_path, index=False)
        print(f"\n Current session results saved: {csv_path}")

        # Save ALL accumulated results (complete dataset)
        csv_all_path = os.path.join(output_dir, 'onshore_offshore_comparison_ALL.csv')
        df_all.to_csv(csv_all_path, index=False)
        print(f" Complete accumulated results saved: {csv_all_path}")
        print(f"  - Current session: {len(df_current)} rows ({df_current['region'].nunique()} regions)")
        print(f"  - All accumulated: {len(df_all)} rows ({df_all['region'].nunique()} regions)")

        # Display summary for ALL results
        print("\n" + "="*80)
        print("SUMMARY - ALL ACCUMULATED RESULTS")
        print("="*80)
        print(df_all.groupby(['configuration', 'installation_type'])[['lcoe_avg', 'p_net_avg_MW']].mean())

        # Create plots using ALL accumulated data
        print("\nCreating comparison plots (using ALL accumulated data)...")
        create_comparison_plots(df_all, output_dir)

        print("\n" + "#"*80)
        print("# COMPARISON COMPLETE!")
        print("#"*80)
        print(f"\nOutput directory: {output_dir}")
        print(f"Results file: onshore_offshore_comparison.csv")
        print(f"Plots: onshore_offshore_comparison.png")


if __name__ == "__main__":
    import sys

    # Check for special commands
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'regenerate_netcdf':
            # Regenerate NetCDF from pickle without re-running analysis
            cost_level = sys.argv[2] if len(sys.argv) > 2 else 'low_cost'
            year = int(sys.argv[3]) if len(sys.argv) > 3 else 2020

            print(f"Regenerating NetCDF files from pickle...")
            print(f"  Cost level: {cost_level}")
            print(f"  Year: {year}\n")

            success = regenerate_netcdf_from_pickle(cost_level, year)
            sys.exit(0 if success else 1)

        elif command == 'help':
            print("="*80)
            print("OTEC ONSHORE VS OFFSHORE COMPARISON - USAGE")
            print("="*80)
            print("\nUsage:")
            print("  python compare_onshore_offshore.py [command] [options]")
            print("\nCommands:")
            print("  (no arguments)         Run full analysis for all regions")
            print("  regenerate_netcdf      Regenerate NetCDF from saved pickle file")
            print("                         Usage: python compare_onshore_offshore.py regenerate_netcdf [cost_level] [year]")
            print("                         Example: python compare_onshore_offshore.py regenerate_netcdf low_cost 2020")
            print("  help                   Show this help message")
            print("\nExamples:")
            print("  # Run full analysis")
            print("  python compare_onshore_offshore.py")
            print("")
            print("  # Regenerate NetCDF from existing results")
            print("  python compare_onshore_offshore.py regenerate_netcdf")
            print("="*80)
            sys.exit(0)

        else:
            print(f"Unknown command: {command}")
            print("Run 'python compare_onshore_offshore.py help' for usage information")
            sys.exit(1)

    # Default: run main analysis
    main()
