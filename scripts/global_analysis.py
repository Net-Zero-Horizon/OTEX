# -*- coding: utf-8 -*-
"""
OTEX Global Analysis using CMEMS Data
Analyzes all countries/regions with actual oceanographic data from CMEMS
Generates NetCDF outputs for each cycle-fluid combination

Workflow:
1. Download CMEMS data for each region
2. Process temperature data
3. Run OTEC analysis
4. Generate NetCDF outputs

@author: OTEX Development Team
"""

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import os
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing

# Import OTEX modules
from otex.config import parameters_and_constants
from otex.data.cmems import download_data, data_processing, load_temperatures
from otex.plant.off_design_analysis import off_design_analysis
from otex.core import get_working_fluid, get_thermodynamic_cycle

# Check CoolProp availability
try:
    import CoolProp
    COOLPROP_AVAILABLE = True
    print("CoolProp available - using high-accuracy fluid properties")
except ImportError:
    COOLPROP_AVAILABLE = False
    print("CoolProp not available - using polynomial correlations (NH3 only)")


def download_region_data(region, cost_level, year):
    """
    Download and process data for a single region (all depths)
    This is run ONCE per region before any calculations

    Args:
        region: Region name
        cost_level: 'low_cost' or 'high_cost'
        year: Year for analysis

    Returns:
        dict: {'region': region, 'files': files, 'new_path': new_path, 'success': True/False}
    """
    from otex.data.cmems import download_data
    import pandas as pd
    import os
    from otex.config import parameters_and_constants

    try:
        # Initialize inputs (we just need basic params for download)
        p_gross = -136000  # kW (136 MW)

        inputs = parameters_and_constants(
            p_gross=p_gross,
            cost_level=cost_level,
            fluid_type='ammonia',  # Doesn't matter for download
            cycle_type='rankine_closed',  # Doesn't matter for download
            use_coolprop=False,
            optimize_depth=False,
            data='CMEMS',
            year=year,
        )

        # Create data directory for this region
        new_path = f'Data_Results/{region.replace(" ", "_")}/'
        os.makedirs(new_path, exist_ok=True)

        # Download CMEMS data (NetCDF files for both depths)
        print(f"  [Download] {region}: Starting download...", flush=True)
        files = download_data(cost_level, inputs, region, new_path)

        # Load sites and process data to create .h5 files
        sites_df = pd.read_csv('CMEMS_points_with_properties.csv', delimiter=';')
        sites_df = sites_df[(sites_df['region'] == region) &
                           (sites_df['water_depth'] <= inputs['min_depth']) &
                           (sites_df['water_depth'] >= inputs['max_depth'])]
        sites_df = sites_df.sort_values(by=['longitude', 'latitude'], ascending=True)

        if len(sites_df) == 0:
            print(f"  [Download] {region}: No valid sites", flush=True)
            return {'region': region, 'files': None, 'new_path': new_path, 'success': False}

        # Process cold water temperatures (creates .h5 file)
        print(f"  [Download] {region}: Processing cold water data...", flush=True)
        T_CW_profiles, T_CW_design, coordinates_CW, id_sites, timestamp, inputs, nan_columns_CW = \
            data_processing(files[int(len(files)/2):int(len(files))], sites_df,
                               inputs, region, new_path, 'CW')

        # Process warm water temperatures (creates .h5 file)
        print(f"  [Download] {region}: Processing warm water data...", flush=True)
        T_WW_profiles, T_WW_design, coordinates_WW, id_sites, timestamp, inputs, nan_columns_WW = \
            data_processing(files[0:int(len(files)/2)], sites_df,
                               inputs, region, new_path, 'WW', nan_columns_CW)

        print(f"  ✓ [Download] {region}: Complete ({len(sites_df)} sites)", flush=True)
        return {'region': region, 'files': files, 'new_path': new_path, 'success': True}

    except Exception as e:
        print(f"  ✗ [Download] {region}: Failed - {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {'region': region, 'files': None, 'new_path': None, 'success': False}


def process_single_region_config(region, config, cost_level, year):
    """
    Worker function to process a single (region, configuration) pair
    ASSUMES DATA HAS ALREADY BEEN DOWNLOADED
    This function is called in parallel by ProcessPoolExecutor

    Args:
        region: Region name
        config: Configuration dictionary (cycle + fluid)
        cost_level: 'low_cost' or 'high_cost'
        year: Year for analysis

    Returns:
        result: Dictionary with results (or None if failed)
    """
    # Import modules needed in worker process (each worker is a separate process)
    from otex.data.cmems import download_data, data_processing, load_temperatures
    from otex.plant.off_design_analysis import off_design_analysis
    import pandas as pd
    import os
    import numpy as np
    from otex.config import parameters_and_constants
    from otex.core.fluids import get_working_fluid
    from otex.core.cycles import get_thermodynamic_cycle

    config_name = f"{config['cycle']}_{config['fluid']}"

    try:
        # Initialize inputs for this region/config
        p_gross = -136000  # kW (136 MW)

        inputs = parameters_and_constants(
            p_gross=p_gross,
            cost_level=cost_level,
            fluid_type=config['fluid_config']['type'] if config['fluid_config'] else 'ammonia',
            cycle_type=config['cycle_config']['type'],
            use_coolprop=config['fluid_config']['use_coolprop'] if config['fluid_config'] else False,
            optimize_depth=False,
            data='CMEMS',
            year=year,
        )

        # Data directory for this region
        new_path = f'Data_Results/{region.replace(" ", "_")}/'

        # Load sites from CSV
        sites_df = pd.read_csv('CMEMS_points_with_properties.csv', delimiter=';')
        sites_df = sites_df[(sites_df['region'] == region) &
                           (sites_df['water_depth'] <= inputs['min_depth']) &
                           (sites_df['water_depth'] >= inputs['max_depth'])]
        sites_df = sites_df.sort_values(by=['longitude', 'latitude'], ascending=True)

        if len(sites_df) == 0:
            return None

        # Load pre-processed temperature data from .h5 files
        # Look for existing .h5 files in the directory (created during Phase 1)
        # These files are named based on the depth used during download
        import glob

        h5_files = glob.glob(f'{new_path}T_*m_{year}_{region}.h5'.replace(" ","_"))

        if len(h5_files) < 2:
            print(f"✗ [{config_name}] {region}: Pre-processed data files not found (found {len(h5_files)} files)", flush=True)
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
        T_CW_profiles, T_CW_design, coordinates_CW, id_sites, timestamp, inputs, nan_columns_CW = \
            load_temperatures(h5_file_CW, inputs)

        # Load warm water data
        T_WW_profiles, T_WW_design, coordinates_WW, id_sites, timestamp, inputs, nan_columns_WW = \
            load_temperatures(h5_file_WW, inputs)

        # Create working fluid and cycle
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
        otec_plants, capex_opex_comparison = off_design_analysis(
            T_WW_design, T_CW_design, T_WW_profiles, T_CW_profiles,
            inputs, coordinates_CW, timestamp, region, new_path, cost_level
        )

        if otec_plants is None or 'p_net' not in otec_plants:
            return None

        # Extract results
        # NOTE: p_net is negative in OTEC convention (negative = power production)
        # so we negate it to get positive values for power output

        # Calculate site-level results
        # otec_plants['p_net'] has shape (timesteps, sites)
        # We want mean power over time for each site
        p_net_sites = -np.nanmean(otec_plants['p_net'], axis=0) / 1000  # MW per site
        lcoe_sites = otec_plants['LCOE']  # LCOE per site

        # Extract lat/lon for each site
        lats = coordinates_CW[:, 1]  # latitude
        lons = coordinates_CW[:, 0]  # longitude

        # Summary statistics for the region
        result = {
            'region': region,
            'configuration': config_name,
            'p_net_avg_MW': p_net_sites.mean(),  # Regional average
            'p_net_max_MW': p_net_sites.max(),  # Regional max
            'p_net_min_MW': p_net_sites.min(),  # Regional min
            'lcoe_avg': lcoe_sites.mean(),
            'lcoe_min': lcoe_sites.min(),
            'n_sites': len(coordinates_CW),
            'T_WW_avg': T_WW_profiles.mean(),
            'T_CW_avg': T_CW_profiles.mean(),
            'deltaT_avg': (T_WW_profiles - T_CW_profiles).mean(),
            # Add site-level data for NetCDF generation
            'p_net_sites': p_net_sites,  # Array of P_net per site
            'lcoe_sites': lcoe_sites,    # Array of LCOE per site
            'lats': lats,                # Array of latitudes
            'lons': lons,                # Array of longitudes
        }

        print(f"✓ [{config_name}] {region}: P_net={result['p_net_avg_MW']:.1f} MW, LCOE={result['lcoe_avg']:.2f} ct/kWh", flush=True)
        return result

    except Exception as e:
        print(f"✗ [Worker] {config_name} - {region} failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None


def process_configuration_parallel(config, regions_to_analyze, cost_level, year, output_dir):
    """
    Worker function to process a single configuration across all regions
    This function is called in parallel by ProcessPoolExecutor

    Args:
        config: Configuration dictionary (cycle + fluid)
        regions_to_analyze: List of regions to analyze
        cost_level: 'low_cost' or 'high_cost'
        year: Year for analysis
        output_dir: Output directory

    Returns:
        config_results: List of result dictionaries
        df_config: DataFrame with results (or None if no results)
    """
    config_name = f"{config['cycle']}_{config['fluid']}"
    print(f"\n[Worker] Starting {config_name}...")

    config_results = []

    # Process each region for this configuration
    for region in regions_to_analyze:
        try:
            # Initialize inputs for this region/config
            p_gross = -136000  # kW (136 MW)

            inputs = parameters_and_constants(
                p_gross=p_gross,
                cost_level=cost_level,
                fluid_type=config['fluid_config']['type'] if config['fluid_config'] else 'ammonia',
                cycle_type=config['cycle_config']['type'],
                use_coolprop=config['fluid_config']['use_coolprop'] if config['fluid_config'] else False,
                optimize_depth=False,
                data='CMEMS',
                year=year,
            )

            # Create data directory for this region
            new_path = f'Data_Results/{region.replace(" ", "_")}/'
            os.makedirs(new_path, exist_ok=True)

            # Download CMEMS data
            files = download_data(cost_level, inputs, region, new_path)

            # Load sites from CSV
            sites_df = pd.read_csv('CMEMS_points_with_properties.csv', delimiter=';')
            sites_df = sites_df[(sites_df['region'] == region) &
                               (sites_df['water_depth'] <= inputs['min_depth']) &
                               (sites_df['water_depth'] >= inputs['max_depth'])]
            sites_df = sites_df.sort_values(by=['longitude', 'latitude'], ascending=True)

            if len(sites_df) == 0:
                continue

            # Process cold water temperatures
            T_CW_profiles, T_CW_design, coordinates_CW, id_sites, timestamp, inputs, nan_columns_CW = \
                data_processing(files[int(len(files)/2):int(len(files))], sites_df,
                                   inputs, region, new_path, 'CW')

            # Process warm water temperatures
            T_WW_profiles, T_WW_design, coordinates_WW, id_sites, timestamp, inputs, nan_columns_WW = \
                data_processing(files[0:int(len(files)/2)], sites_df,
                                   inputs, region, new_path, 'WW', nan_columns_CW)

            # Create working fluid and cycle
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
            otec_plants, capex_opex_comparison = off_design_analysis(
                T_WW_design, T_CW_design, T_WW_profiles, T_CW_profiles,
                inputs, coordinates_CW, timestamp, region, new_path, cost_level
            )

            # Extract key results
            # NOTE: p_net is negative in OTEC convention (negative = power production)
            if otec_plants is not None and 'p_net' in otec_plants:
                # Calculate site-level results
                p_net_sites = -np.nanmean(otec_plants['p_net'], axis=0) / 1000  # MW per site
                lcoe_sites = otec_plants['LCOE']  # LCOE per site

                # Extract lat/lon for each site
                lats = coordinates_CW[:, 1]  # latitude
                lons = coordinates_CW[:, 0]  # longitude

                result = {
                    'region': region,
                    'p_net_avg_MW': p_net_sites.mean(),  # Regional average
                    'p_net_max_MW': p_net_sites.max(),  # Regional max
                    'p_net_min_MW': p_net_sites.min(),  # Regional min
                    'lcoe_avg': lcoe_sites.mean(),
                    'lcoe_min': lcoe_sites.min(),
                    'n_sites': len(coordinates_CW),
                    'T_WW_avg': T_WW_profiles.mean(),
                    'T_CW_avg': T_CW_profiles.mean(),
                    'deltaT_avg': (T_WW_profiles - T_CW_profiles).mean(),
                    'configuration': config_name,
                    # Add site-level data for NetCDF generation
                    'p_net_sites': p_net_sites,
                    'lcoe_sites': lcoe_sites,
                    'lats': lats,
                    'lons': lons,
                    'cycle': config['cycle'],
                    'fluid': config['fluid']
                }
                config_results.append(result)

        except Exception as e:
            print(f"[Worker] {config_name} - {region} failed: {e}")
            continue

    # Save results if we have any
    if config_results:
        # Create DataFrame (exclude site-level arrays)
        df_data = []
        for result in config_results:
            # Copy only scalar values for DataFrame
            row = {k: v for k, v in result.items()
                   if k not in ['p_net_sites', 'lcoe_sites', 'lats', 'lons']}
            df_data.append(row)

        df_config = pd.DataFrame(df_data)

        # Save CSV (regional aggregates)
        csv_filename = os.path.join(output_dir, f'{config_name}_{cost_level}.csv')
        df_config.to_csv(csv_filename, index=False)

        # Save NetCDF (site-level spatial data)
        save_netcdf_standalone(config_results, config_name, cost_level, year, output_dir)

        print(f"[Worker] {config_name} completed: {len(config_results)} regions")
        return config_results, df_config
    else:
        print(f"[Worker] {config_name} - No valid results")
        return [], None


def save_netcdf_standalone(results_list, config_name, cost_level, year, output_dir):
    """Standalone function to save NetCDF with spatial data (for use in parallel workers)"""

    # Collect all site-level data from all regions
    all_p_net = []
    all_lcoe = []
    all_lats = []
    all_lons = []
    all_regions = []

    for result in results_list:
        if 'p_net_sites' in result and 'lcoe_sites' in result:
            n_sites = len(result['p_net_sites'])
            all_p_net.extend(result['p_net_sites'])
            all_lcoe.extend(result['lcoe_sites'])
            all_lats.extend(result['lats'])
            all_lons.extend(result['lons'])
            all_regions.extend([result['region']] * n_sites)

    if len(all_p_net) == 0:
        print(f"Warning: No site-level data available for {config_name}")
        return

    # Convert to numpy arrays
    all_p_net = np.array(all_p_net)
    all_lcoe = np.array(all_lcoe)
    all_lats = np.array(all_lats)
    all_lons = np.array(all_lons)

    # Create xarray Dataset with spatial coordinates
    ds = xr.Dataset(
        {
            'p_net_MW': (['site'], all_p_net, {
                'long_name': 'Net power output',
                'units': 'MW',
                'description': 'Average net power production over the year'
            }),
            'lcoe': (['site'], all_lcoe, {
                'long_name': 'Levelized Cost of Energy',
                'units': 'ct/kWh',
                'description': 'LCOE for each site'
            }),
            'region': (['site'], all_regions, {
                'long_name': 'Region name',
                'description': 'Name of the region containing this site'
            }),
        },
        coords={
            'lat': (['site'], all_lats, {
                'long_name': 'Latitude',
                'units': 'degrees_north',
                'standard_name': 'latitude'
            }),
            'lon': (['site'], all_lons, {
                'long_name': 'Longitude',
                'units': 'degrees_east',
                'standard_name': 'longitude'
            }),
        },
        attrs={
            'title': f'Global OTEC Analysis - {config_name}',
            'configuration': config_name,
            'cycle': results_list[0].get('cycle', 'unknown') if results_list else 'unknown',
            'fluid': results_list[0].get('fluid', 'unknown') if results_list else 'unknown',
            'cost_level': cost_level,
            'year': year,
            'creation_date': datetime.now().isoformat(),
            'description': 'OTEC feasibility analysis using CMEMS oceanographic data - Site-level results',
            'data_source': 'CMEMS Global Ocean Physics Analysis',
            'conventions': 'CF-1.8',
            'n_sites': len(all_p_net),
            'n_regions': len(set(all_regions)),
        }
    )

    nc_filename = os.path.join(output_dir, f'{config_name}_{cost_level}.nc')
    ds.to_netcdf(nc_filename)
    print(f"[Worker] Saved NetCDF: {nc_filename} ({len(all_p_net)} sites)")


class GlobalOTECCMEMSAnalysis:
    """
    Comprehensive global OTEC analysis using real CMEMS oceanographic data
    """

    def __init__(self, output_dir='./Global_OTEC_CMEMS/', year=2020):
        """
        Initialize analysis

        Args:
            output_dir: Directory to save results
            year: Year for CMEMS data (2020 recommended, 1993-2023 available)
        """
        self.output_dir = output_dir
        self.year = year
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load regions from CSV
        self.load_regions()

        # Define configurations
        self.define_configurations()

        # Results storage
        self.results_dict = {}

    def load_regions(self):
        """
        Load all regions/countries from CSV file
        """
        print("\nLoading regions from download_ranges_per_region.csv...")

        self.regions_df = pd.read_csv('download_ranges_per_region.csv', delimiter=';')

        # Filter out regions with invalid demand (NaN or #N/A)
        # Keep all regions for analysis regardless of demand
        valid_regions = self.regions_df[pd.notna(self.regions_df['region'])]

        print(f"  Total regions available: {len(valid_regions)}")
        print(f"  Regions with electricity demand data: {valid_regions['demand'].notna().sum()}")

        self.regions_list = valid_regions['region'].unique().tolist()

        print(f"\nFirst 20 regions:")
        for i, region in enumerate(self.regions_list[:20]):
            print(f"  {i+1}. {region}")
        print(f"  ... and {len(self.regions_list) - 20} more")

    def define_configurations(self):
        """
        Define cycle-fluid combinations to analyze
        """

        # Thermodynamic cycles
        self.cycles = {
            'Rankine_Closed': {
                'type': 'rankine_closed',
                'requires_fluid': True,
                'description': 'Standard closed Rankine cycle'
            },
            'Rankine_Open': {
                'type': 'rankine_open',
                'requires_fluid': False,
                'description': 'Open Rankine flash cycle (seawater)',
                'note': 'Now implemented with simplified flash model'
            },
        }

        # Working fluids (for Rankine Closed only)
        # NOTE: Ammonia uses polynomial correlations for better performance with large arrays
        # Other fluids require CoolProp (element-by-element iteration, slower but accurate)
        if COOLPROP_AVAILABLE:
            self.fluids = {
                'Ammonia': {'type': 'ammonia', 'use_coolprop': True},
                'R134a': {'type': 'r134a', 'use_coolprop': True},
                'R245fa': {'type': 'r245fa', 'use_coolprop': True},
                'Propane': {'type': 'propane', 'use_coolprop': True},
                'Isobutane': {'type': 'isobutane', 'use_coolprop': True},
            }
        else:
            self.fluids = {
                'Ammonia': {'type': 'ammonia', 'use_coolprop': False},
            }

        # Generate all configurations
        self.configurations = []

        for cycle_name, cycle_config in self.cycles.items():
            if cycle_config.get('requires_fluid', True):
                # Cycle needs working fluid
                for fluid_name, fluid_config in self.fluids.items():
                    self.configurations.append({
                        'cycle': cycle_name,
                        'fluid': fluid_name,
                        'cycle_config': cycle_config,
                        'fluid_config': fluid_config,
                    })
            else:
                # Cycle doesn't need working fluid (Rankine Open)
                self.configurations.append({
                    'cycle': cycle_name,
                    'fluid': 'Seawater',
                    'cycle_config': cycle_config,
                    'fluid_config': None,
                })

        print(f"\nTotal configurations: {len(self.configurations)}")
        for cfg in self.configurations:
            print(f"  - {cfg['cycle']} + {cfg['fluid']}")

    def analyze_region(self, region_name, config, cost_level='low_cost'):
        """
        Analyze OTEC potential for a specific region using CMEMS data

        Args:
            region_name: Name of region (from CSV)
            config: Configuration dictionary (cycle + fluid)
            cost_level: 'low_cost' or 'high_cost'

        Returns:
            results: Dictionary with regional results or None if failed
        """

        try:
            print(f"\n  Processing region: {region_name}", flush=True)
            print(f"    Initializing parameters...", flush=True)

            # Initialize inputs
            p_gross = -136000  # kW (136 MW)

            inputs = parameters_and_constants(
                p_gross=p_gross,
                cost_level=cost_level,
                fluid_type=config['fluid_config']['type'] if config['fluid_config'] else 'ammonia',
                cycle_type=config['cycle_config']['type'],
                use_coolprop=config['fluid_config']['use_coolprop'] if config['fluid_config'] else False,
                optimize_depth=False,
                data='CMEMS',
                year=self.year,
            )
            print(f"    Parameters initialized.", flush=True)

            # Create data directory for this region
            new_path = f'Data_Results/{region_name.replace(" ", "_")}/'
            os.makedirs(new_path, exist_ok=True)
            print(f"    Created directory: {new_path}", flush=True)

            # Download CMEMS data for this region
            print(f"    Downloading CMEMS data...", flush=True)
            sys.stderr.write(f"[DEBUG] About to call download_data\n")
            sys.stderr.flush()
            try:
                files = download_data(cost_level, inputs, region_name, new_path)
                sys.stderr.write(f"[DEBUG] Returned from download_data\n")
                sys.stderr.flush()
                print(f"    Download complete. Files: {files}", flush=True)
                sys.stderr.write(f"[DEBUG] Print completed\n")
                sys.stderr.flush()
            except Exception as e:
                print(f"    ERROR downloading data: {e}")
                return None

            # Process the downloaded data (following OTEX.py workflow)
            sys.stderr.write(f"[DEBUG] About to print 'Processing oceanographic data'\n")
            sys.stderr.flush()
            print(f"    Processing oceanographic data...", flush=True)
            sys.stderr.write(f"[DEBUG] Print completed\n")
            sys.stderr.flush()
            try:
                # Load sites from CSV
                print(f"    Loading sites CSV...", flush=True)
                sites_df = pd.read_csv('CMEMS_points_with_properties.csv', delimiter=';')
                print(f"    CSV loaded ({len(sites_df)} rows). Filtering...", flush=True)
                sites_df = sites_df[(sites_df['region'] == region_name) &
                                   (sites_df['water_depth'] <= inputs['min_depth']) &
                                   (sites_df['water_depth'] >= inputs['max_depth'])]
                print(f"    Filtered to {len(sites_df)} sites. Sorting...", flush=True)
                sites_df = sites_df.sort_values(by=['longitude', 'latitude'], ascending=True)
                print(f"    Sorted. Checking site count...", flush=True)

                # Check if we have sites for this region
                if len(sites_df) == 0:
                    print(f"    WARNING: No valid ocean sites found for {region_name}")
                    return None

                # Process cold water temperatures
                print(f"    Processing cold water data...", flush=True)
                T_CW_profiles, T_CW_design, coordinates_CW, id_sites, timestamp, inputs, nan_columns_CW = \
                    data_processing(files[int(len(files)/2):int(len(files))], sites_df,
                                       inputs, region_name, new_path, 'CW')

                # Process warm water temperatures
                print(f"    Processing warm water data...", flush=True)
                T_WW_profiles, T_WW_design, coordinates_WW, id_sites, timestamp, inputs, nan_columns_WW = \
                    data_processing(files[0:int(len(files)/2)], sites_df,
                                       inputs, region_name, new_path, 'WW', nan_columns_CW)

            except Exception as e:
                print(f"    ERROR processing data: {e}")
                import traceback
                traceback.print_exc()
                return None

            # Create working fluid and cycle (if needed)
            if config['fluid_config']:
                working_fluid = get_working_fluid(
                    config['fluid_config']['type'],
                    use_coolprop=config['fluid_config']['use_coolprop']
                )
                inputs['working_fluid'] = working_fluid

                # Create cycle
                cycle = get_thermodynamic_cycle(
                    config['cycle_config']['type'],
                    working_fluid=working_fluid
                )
                inputs['thermodynamic_cycle'] = cycle
            else:
                # Rankine Open cycle
                cycle = get_thermodynamic_cycle(config['cycle_config']['type'])
                inputs['thermodynamic_cycle'] = cycle

            # Run OTEC off-design analysis
            print(f"    Running OTEC off-design analysis...")
            try:
                otec_plants, capex_opex_comparison = off_design_analysis(
                    T_WW_design, T_CW_design, T_WW_profiles, T_CW_profiles,
                    inputs, coordinates_CW, timestamp, region_name, new_path, cost_level
                )
            except Exception as e:
                print(f"    ERROR in OTEC analysis: {e}")
                import traceback
                traceback.print_exc()
                return None

            # Extract key results
            if otec_plants is not None and 'p_net' in otec_plants:
                # Calculate average metrics
                p_net_avg = -np.nanmean(otec_plants['p_net'], axis=0) / 1000  # MW
                lcoe_avg = otec_plants['LCOE'].T

                region_results = {
                    'region': region_name,
                    'p_net_avg_MW': np.nanmean(p_net_avg),
                    'p_net_max_MW': np.nanmax(p_net_avg),
                    'p_net_min_MW': np.nanmin(p_net_avg),
                    'lcoe_avg': np.nanmean(lcoe_avg),
                    'lcoe_min': np.nanmin(lcoe_avg),
                    'n_sites': otec_plants['p_net'].shape[1] if len(otec_plants['p_net'].shape) > 1 else 1,
                    'T_WW_avg': np.nanmean(T_WW_design[1, :]),  # Median temperature
                    'T_CW_avg': np.nanmean(T_CW_design[1, :]),  # Median temperature
                    'deltaT_avg': np.nanmean(T_WW_design[1, :] - T_CW_design[1, :]),
                }

                print(f"    SUCCESS: P_net_avg = {region_results['p_net_avg_MW']:.2f} MW, LCOE = {region_results['lcoe_avg']:.2f} ct/kWh, Sites = {region_results['n_sites']}")
                return region_results
            else:
                print(f"    WARNING: No valid results for this region")
                return None

        except Exception as e:
            print(f"    EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_analysis(self, regions_to_analyze=None, cost_level='low_cost', parallel=False,
                    max_workers=None, parallel_level='region'):
        """
        Run analysis for specified regions and all configurations

        Args:
            regions_to_analyze: List of region names (None = all regions)
            cost_level: 'low_cost' or 'high_cost'
            parallel: If True, run in parallel (default: False)
            max_workers: Maximum number of parallel workers (default: CPU count - 1)
            parallel_level: 'config' for config-level parallelization (6 workers max)
                           'region' for region-level parallelization (region × config workers)

        Returns:
            results_dict: Dictionary of DataFrames, one per configuration
        """

        print("\n" + "="*80)
        print("GLOBAL OTEC ANALYSIS USING CMEMS DATA")
        print("="*80)

        # Determine which regions to analyze
        if regions_to_analyze is None:
            regions_to_analyze = self.regions_list
        else:
            # Validate regions
            invalid = [r for r in regions_to_analyze if r not in self.regions_list]
            if invalid:
                print(f"WARNING: Invalid regions will be skipped: {invalid}")
            regions_to_analyze = [r for r in regions_to_analyze if r in self.regions_list]

        print(f"\nRegions to analyze: {len(regions_to_analyze)}")
        print(f"Configurations: {len(self.configurations)}")
        print(f"Cost level: {cost_level}")
        print(f"Year: {self.year}")
        print(f"Total analyses: {len(regions_to_analyze) * len(self.configurations)}")

        if parallel:
            if max_workers is None:
                max_workers = max(1, multiprocessing.cpu_count() - 1)

            if parallel_level == 'region':
                print(f"PARALLEL MODE (REGION-LEVEL): Using up to {max_workers} workers")
                print(f"  → Max concurrent jobs: {len(regions_to_analyze) * len(self.configurations)}")
                print(f"  → Actual workers: min({max_workers}, {len(regions_to_analyze) * len(self.configurations)})")
                return self._run_analysis_parallel_region_level(regions_to_analyze, cost_level, max_workers)
            else:  # config level
                print(f"PARALLEL MODE (CONFIG-LEVEL): Using {max_workers} workers")
                print(f"  → Max concurrent jobs: {len(self.configurations)}")
                return self._run_analysis_parallel(regions_to_analyze, cost_level, max_workers)
        else:
            print("SEQUENTIAL MODE")
            return self._run_analysis_sequential(regions_to_analyze, cost_level)

    def _run_analysis_sequential(self, regions_to_analyze, cost_level):
        """Sequential analysis (original behavior)"""
        # Process each configuration
        for i, config in enumerate(self.configurations):
            config_name = f"{config['cycle']}_{config['fluid']}"
            print(f"\n{'='*80}")
            print(f"Configuration {i+1}/{len(self.configurations)}: {config_name}")
            print(f"{'='*80}")

            # Results for this configuration
            config_results = []

            # Process each region
            print(f"Starting to process {len(regions_to_analyze)} regions...", flush=True)
            for region in tqdm(regions_to_analyze, desc=f"Analyzing {config_name}", file=sys.stdout):
                print(f"\n  Processing region: {region}", flush=True)
                result = self.analyze_region(region, config, cost_level)

                if result is not None:
                    result['configuration'] = config_name
                    result['cycle'] = config['cycle']
                    result['fluid'] = config['fluid']
                    config_results.append(result)

            # Convert to DataFrame (exclude site-level arrays)
            if config_results:
                df_data = []
                for result in config_results:
                    # Copy only scalar values for DataFrame
                    row = {k: v for k, v in result.items()
                           if k not in ['p_net_sites', 'lcoe_sites', 'lats', 'lons']}
                    df_data.append(row)

                df_config = pd.DataFrame(df_data)
                self.results_dict[config_name] = df_config

                # Save CSV (regional aggregates)
                csv_filename = os.path.join(self.output_dir, f'{config_name}_{cost_level}.csv')
                df_config.to_csv(csv_filename, index=False)
                print(f"\nSaved CSV: {csv_filename}")

                # Save NetCDF (site-level spatial data)
                self.save_netcdf(config_results, config_name, cost_level)

            else:
                print(f"\nWARNING: No valid results for configuration {config_name}")

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)

        return self.results_dict

    def _run_analysis_parallel(self, regions_to_analyze, cost_level, max_workers):
        """
        Parallel analysis - runs multiple configurations simultaneously
        Each configuration processes all its regions independently
        """
        print(f"\nSubmitting {len(self.configurations)} configuration jobs to parallel executor...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all configuration jobs
            future_to_config = {}
            for i, config in enumerate(self.configurations):
                config_name = f"{config['cycle']}_{config['fluid']}"
                future = executor.submit(
                    process_configuration_parallel,
                    config, regions_to_analyze, cost_level, self.year, self.output_dir
                )
                future_to_config[future] = (i, config_name)

            # Collect results as they complete
            for future in as_completed(future_to_config):
                i, config_name = future_to_config[future]
                try:
                    config_results, df_config = future.result()
                    if df_config is not None:
                        self.results_dict[config_name] = df_config
                        print(f"\n✓ Completed {i+1}/{len(self.configurations)}: {config_name} "
                              f"({len(config_results)} regions)")
                    else:
                        print(f"\n✗ No valid results for {config_name}")
                except Exception as e:
                    print(f"\n✗ Configuration {config_name} failed with error: {e}")
                    import traceback
                    traceback.print_exc()

        print("\n" + "="*80)
        print("PARALLEL ANALYSIS COMPLETE")
        print("="*80)

        return self.results_dict

    def _run_analysis_parallel_region_level(self, regions_to_analyze, cost_level, max_workers):
        """
        TWO-PHASE PARALLEL ANALYSIS:
        PHASE 1: Download all data for all regions (sequential or parallel)
        PHASE 2: Parallel calculation for all (region, config) combinations

        This eliminates redundant downloads and file locking issues.
        """
        print("\n" + "="*80)
        print("TWO-PHASE PARALLEL EXECUTION")
        print("="*80)

        # ========================================================================
        # PHASE 1: DOWNLOAD AND PRE-PROCESS DATA FOR ALL REGIONS
        # ========================================================================
        print("\n" + "="*80)
        print(f"PHASE 1: DOWNLOADING DATA FOR {len(regions_to_analyze)} REGIONS")
        print("="*80)
        print("This will download NetCDF files and create .h5 processed files")
        print("Each region is downloaded ONCE and shared across all configurations\n")

        downloaded_regions = []
        failed_regions = []

        # Option 1: Sequential download (safer, prevents API throttling)
        print("Download mode: SEQUENTIAL (prevents API throttling)\n")
        for i, region in enumerate(regions_to_analyze):
            print(f"[{i+1}/{len(regions_to_analyze)}] Downloading {region}...")
            result = download_region_data(region, cost_level, self.year)
            if result['success']:
                downloaded_regions.append(region)
            else:
                failed_regions.append(region)

        # Option 2: Parallel download (faster but may hit API limits)
        # Uncomment to use parallel downloads:
        # print("Download mode: PARALLEL (faster but may hit API limits)\n")
        # with ProcessPoolExecutor(max_workers=min(4, max_workers)) as executor:
        #     futures = {executor.submit(download_region_data, region, cost_level, self.year): region
        #                for region in regions_to_analyze}
        #
        #     for i, future in enumerate(as_completed(futures)):
        #         region = futures[future]
        #         result = future.result()
        #         if result['success']:
        #             downloaded_regions.append(region)
        #             print(f"[{i+1}/{len(regions_to_analyze)}] ✓ {region}")
        #         else:
        #             failed_regions.append(region)
        #             print(f"[{i+1}/{len(regions_to_analyze)}] ✗ {region}")

        print("\n" + "-"*80)
        print(f"PHASE 1 COMPLETE")
        print(f"  ✓ Successfully downloaded: {len(downloaded_regions)} regions")
        print(f"  ✗ Failed: {len(failed_regions)} regions")
        if failed_regions:
            print(f"  Failed regions: {', '.join(failed_regions)}")
        print("-"*80)

        # ========================================================================
        # PHASE 2: PARALLEL CALCULATION FOR ALL CONFIGURATIONS
        # ========================================================================
        print("\n" + "="*80)
        print(f"PHASE 2: RUNNING CALCULATIONS")
        print("="*80)
        print(f"Regions: {len(downloaded_regions)}")
        print(f"Configurations: {len(self.configurations)}")
        print(f"Total jobs: {len(downloaded_regions)} × {len(self.configurations)} = {len(downloaded_regions) * len(self.configurations)}")
        print(f"Workers: {max_workers}\n")

        # Create all (region, config) pairs (only for successfully downloaded regions)
        all_jobs = []
        for region in downloaded_regions:
            for config in self.configurations:
                all_jobs.append((region, config))

        print(f"Submitting {len(all_jobs)} calculation jobs...\n")

        # Dictionary to collect results by configuration
        results_by_config = {f"{c['cycle']}_{c['fluid']}": [] for c in self.configurations}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all calculation jobs
            future_to_job = {}
            for region, config in all_jobs:
                config_name = f"{config['cycle']}_{config['fluid']}"
                future = executor.submit(
                    process_single_region_config,
                    region, config, cost_level, self.year
                )
                future_to_job[future] = (region, config_name)

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_job):
                region, config_name = future_to_job[future]
                completed += 1

                try:
                    result = future.result()
                    if result is not None:
                        results_by_config[config_name].append(result)
                        print(f"[{completed}/{len(all_jobs)}] ✓ {config_name} - {region}", flush=True)
                    else:
                        print(f"[{completed}/{len(all_jobs)}] ✗ {config_name} - {region} (no results)", flush=True)
                except Exception as e:
                    print(f"[{completed}/{len(all_jobs)}] ✗ {config_name} - {region} failed: {e}", flush=True)

        # ========================================================================
        # SAVE RESULTS
        # ========================================================================
        print("\n" + "="*80)
        print("SAVING RESULTS...")
        print("="*80)

        for config in self.configurations:
            config_name = f"{config['cycle']}_{config['fluid']}"
            config_results = results_by_config[config_name]

            if len(config_results) > 0:
                # Add config info to each result
                for result in config_results:
                    result['configuration'] = config_name
                    result['cycle'] = config['cycle']
                    result['fluid'] = config['fluid']

                # Create DataFrame (for CSV and summary) - exclude site-level arrays
                df_data = []
                for result in config_results:
                    # Copy only scalar values for DataFrame
                    row = {k: v for k, v in result.items()
                           if k not in ['p_net_sites', 'lcoe_sites', 'lats', 'lons']}
                    df_data.append(row)

                df = pd.DataFrame(df_data)
                self.results_dict[config_name] = df

                # Save CSV (regional aggregates)
                csv_filename = os.path.join(self.output_dir, f"{config_name}_{cost_level}.csv")
                df.to_csv(csv_filename, index=False)
                print(f"✓ Saved {config_name}: {len(df)} regions → {csv_filename}")

                # Save NetCDF (site-level spatial data)
                try:
                    self.save_netcdf(config_results, config_name, cost_level)
                except Exception as e:
                    print(f"✗ Failed to save NetCDF for {config_name}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"✗ No valid results for {config_name}")

        print("\n" + "="*80)
        print("TWO-PHASE PARALLEL ANALYSIS COMPLETE")
        print("="*80)

        return self.results_dict

    def save_netcdf(self, results_list, config_name, cost_level):
        """
        Save results to NetCDF format with spatial data (lat/lon grid)

        Args:
            results_list: List of result dictionaries (one per region)
            config_name: Configuration name
            cost_level: Cost level
        """

        # Collect all site-level data from all regions
        all_p_net = []
        all_lcoe = []
        all_lats = []
        all_lons = []
        all_regions = []

        for result in results_list:
            if 'p_net_sites' in result and 'lcoe_sites' in result:
                # Convert to arrays and flatten to ensure 1D
                p_net_arr = np.atleast_1d(result['p_net_sites']).flatten()
                lcoe_arr = np.atleast_1d(result['lcoe_sites']).flatten()
                lats_arr = np.atleast_1d(result['lats']).flatten()
                lons_arr = np.atleast_1d(result['lons']).flatten()

                n_sites = len(p_net_arr)

                # Extend lists with individual values (not arrays)
                all_p_net.extend(p_net_arr.tolist())
                all_lcoe.extend(lcoe_arr.tolist())
                all_lats.extend(lats_arr.tolist())
                all_lons.extend(lons_arr.tolist())
                all_regions.extend([result['region']] * n_sites)

        if len(all_p_net) == 0:
            print(f"Warning: No site-level data available for {config_name}")
            return

        # Convert to numpy arrays - now all elements are scalars
        all_p_net = np.array(all_p_net, dtype=np.float64)
        all_lcoe = np.array(all_lcoe, dtype=np.float64)
        all_lats = np.array(all_lats, dtype=np.float64)
        all_lons = np.array(all_lons, dtype=np.float64)

        # Create xarray Dataset with spatial coordinates
        ds = xr.Dataset(
            {
                'p_net_MW': (['site'], all_p_net, {
                    'long_name': 'Net power output',
                    'units': 'MW',
                    'description': 'Average net power production over the year'
                }),
                'lcoe': (['site'], all_lcoe, {
                    'long_name': 'Levelized Cost of Energy',
                    'units': 'ct/kWh',
                    'description': 'LCOE for each site'
                }),
                'region': (['site'], all_regions, {
                    'long_name': 'Region name',
                    'description': 'Name of the region containing this site'
                }),
            },
            coords={
                'lat': (['site'], all_lats, {
                    'long_name': 'Latitude',
                    'units': 'degrees_north',
                    'standard_name': 'latitude'
                }),
                'lon': (['site'], all_lons, {
                    'long_name': 'Longitude',
                    'units': 'degrees_east',
                    'standard_name': 'longitude'
                }),
            },
            attrs={
                'title': f'Global OTEC Analysis - {config_name}',
                'configuration': config_name,
                'cycle': results_list[0].get('cycle', 'unknown') if results_list else 'unknown',
                'fluid': results_list[0].get('fluid', 'unknown') if results_list else 'unknown',
                'cost_level': cost_level,
                'year': self.year,
                'creation_date': datetime.now().isoformat(),
                'description': 'OTEC feasibility analysis using CMEMS oceanographic data - Site-level results',
                'data_source': 'CMEMS Global Ocean Physics Analysis',
                'conventions': 'CF-1.8',
                'n_sites': len(all_p_net),
                'n_regions': len(set(all_regions)),
            }
        )

        # Save NetCDF
        nc_filename = os.path.join(self.output_dir, f'{config_name}_{cost_level}.nc')
        ds.to_netcdf(nc_filename)
        print(f"✓ Saved NetCDF: {nc_filename} ({len(all_p_net)} sites across {len(set(all_regions))} regions)")

    def create_summary(self):
        """
        Create summary statistics across all configurations
        """

        print("\n" + "="*80)
        print("CREATING SUMMARY")
        print("="*80)

        summary_data = []

        for config_name, df in self.results_dict.items():
            summary_entry = {
                'Configuration': config_name,
                'Regions_Analyzed': len(df),
                'Avg_Power_MW': df['p_net_avg_MW'].mean(),
                'Max_Power_MW': df['p_net_max_MW'].max(),
                'Min_Power_MW': df['p_net_min_MW'].min(),
                'Avg_LCOE_ct/kWh': df['lcoe_avg'].mean(),
                'Min_LCOE_ct/kWh': df['lcoe_min'].min(),
                'Best_Region': df.loc[df['lcoe_min'].idxmin(), 'region'],
            }
            # Add efficiency if available
            if 'efficiency_avg' in df.columns:
                summary_entry['Avg_Efficiency_%'] = df['efficiency_avg'].mean()
            summary_data.append(summary_entry)

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Avg_LCOE_ct/kWh')

        # Save summary
        csv_filename = os.path.join(self.output_dir, 'summary_all_configurations.csv')
        summary_df.to_csv(csv_filename, index=False)
        print(f"\nSaved: {csv_filename}")

        # Print summary
        print("\nSummary Statistics:")
        print(summary_df.to_string(index=False))

        return summary_df


def main():
    """
    Main execution function
    """

    print("#"*80)
    print("# GLOBAL OTEC FEASIBILITY ANALYSIS")
    print("# Using Real CMEMS Oceanographic Data")
    print("# All Regions, Multiple Cycle-Fluid Combinations")
    print("#"*80)

    # Create analysis object
    analysis = GlobalOTECCMEMSAnalysis(
        output_dir='./Global_OTEC_CMEMS/',
        year=2020  # Use 2020 data
    )

    # For testing: analyze a subset of regions
    # For full analysis: set test_regions = None

    # TESTING MODE: Analyze only a few regions
    test_regions = [
        'Hawaii',
        'Jamaica',
        'Philippines',
        'Maldives',
        'Cuba',
    ]

    # FULL MODE: Analyze all regions (comment out test_regions above)
    # test_regions = None

    print(f"\n{'='*80}")
    print(f"ANALYSIS MODE")
    print(f"{'='*80}")
    if test_regions:
        print(f"TEST MODE: Analyzing {len(test_regions)} regions")
        print(f"Regions: {', '.join(test_regions)}")
    else:
        print(f"FULL MODE: Analyzing ALL {len(analysis.regions_list)} regions")
    print(f"{'='*80}\n")

    # Run analysis
    # Set parallel=True to run in parallel (much faster!)
    # Set parallel=False for sequential execution (easier debugging)
    USE_PARALLEL = True  # Change this to False for sequential mode

    # Parallelization level:
    # 'region' = region-level parallelization (RECOMMENDED - uses all cores)
    # 'config' = config-level parallelization (only uses 6 cores max)
    PARALLEL_LEVEL = 'region'  # Change to 'config' for old behavior

    results_dict = analysis.run_analysis(
        regions_to_analyze=test_regions,
        cost_level='low_cost',
        parallel=USE_PARALLEL,
        max_workers=None,  # None = auto-detect (CPU count - 1)
        parallel_level=PARALLEL_LEVEL
    )

    # Create summary
    if results_dict:
        summary_df = analysis.create_summary()

    print("\n" + "#"*80)
    print("# ANALYSIS COMPLETE!")
    print("#"*80)
    print(f"\nOutput directory: {analysis.output_dir}")
    print(f"Configurations analyzed: {len(results_dict)}")
    print(f"Summary: summary_all_configurations.csv")

    # List generated files
    print("\nGenerated files:")
    for config_name in results_dict.keys():
        print(f"  - {config_name}_low_cost.csv")
        print(f"  - {config_name}_low_cost.nc")
    print(f"  - summary_all_configurations.csv")


if __name__ == "__main__":
    main()
