"""
Merge multiple CSV files and regenerate NetCDF from aggregated results

If you have multiple CSV files from distributed execution (e.g., onshore_offshore_comparison_node1.csv,
onshore_offshore_comparison_node2.csv, etc.), this script will:
1. Merge them into a single comprehensive CSV
2. Extract site-level data from .h5 files
3. Regenerate NetCDF files with ALL regions

Usage:
    # If you have multiple CSV files with node_id:
    python merge_csv_and_regenerate_netcdf.py --merge

    # If you just want to regenerate NetCDF from current CSV:
    python merge_csv_and_regenerate_netcdf.py --regenerate
"""

import numpy as np
import pandas as pd
import os
import glob
import argparse
from datetime import datetime


def merge_csv_files(input_pattern='OTEC_Comparison/onshore_offshore_comparison*.csv',
                    output_file='OTEC_Comparison/onshore_offshore_comparison_merged.csv'):
    """
    Merge multiple CSV result files into one

    Args:
        input_pattern: Glob pattern for input CSV files
        output_file: Output merged CSV file
    """

    print("="*80)
    print("MERGING CSV FILES")
    print("="*80)

    csv_files = glob.glob(input_pattern)

    if len(csv_files) == 0:
        print(f"No CSV files found matching pattern: {input_pattern}")
        return None

    print(f"\nFound {len(csv_files)} CSV files:")
    for f in csv_files:
        size_mb = os.path.getsize(f) / (1024*1024)
        print(f"  - {f} ({size_mb:.2f} MB)")

    # Read and merge all CSVs
    dfs = []
    total_rows = 0

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
        total_rows += len(df)
        print(f"  Loaded {csv_file}: {len(df)} rows")

    # Concatenate
    merged_df = pd.concat(dfs, ignore_index=True)

    # Remove duplicates (in case some regions were processed multiple times)
    print(f"\nTotal rows before deduplication: {len(merged_df)}")

    # Deduplicate based on region, configuration, installation_type
    merged_df = merged_df.drop_duplicates(subset=['region', 'configuration', 'installation_type'],
                                          keep='last')

    print(f"Total rows after deduplication: {len(merged_df)}")
    print(f"Unique regions: {merged_df['region'].nunique()}")
    print(f"Unique configurations: {merged_df['configuration'].nunique()}")

    # Save merged file
    merged_df.to_csv(output_file, index=False)
    print(f"\n✓ Merged CSV saved to: {output_file}")
    print(f"  Size: {os.path.getsize(output_file)/(1024*1024):.2f} MB")

    return merged_df


def check_data_availability():
    """
    Check what data is available for NetCDF regeneration
    """

    print("\n" + "="*80)
    print("CHECKING DATA AVAILABILITY")
    print("="*80)

    # Check CSV files
    csv_files = glob.glob('OTEC_Comparison/onshore_offshore_comparison*.csv')
    print(f"\nCSV files: {len(csv_files)}")

    if len(csv_files) > 0:
        # Read the main or most recent CSV
        main_csv = 'OTEC_Comparison/onshore_offshore_comparison.csv'
        if not os.path.exists(main_csv) and len(csv_files) > 0:
            main_csv = csv_files[0]

        df = pd.read_csv(main_csv)
        print(f"  Main CSV: {main_csv}")
        print(f"  Regions: {df['region'].nunique()}")
        print(f"  Total rows: {len(df)}")

    # Check .h5 files
    data_dirs = [d for d in os.listdir('Data_Results')
                 if os.path.isdir(os.path.join('Data_Results', d))]

    regions_with_h5 = []
    for region_dir in data_dirs:
        h5_files = glob.glob(f'Data_Results/{region_dir}/T_*m_2020_*.h5')
        if len(h5_files) >= 2:
            regions_with_h5.append(region_dir.replace('_', ' '))

    print(f"\nRegions with .h5 data: {len(regions_with_h5)}")

    # Check NetCDF files
    nc_files = glob.glob('OTEC_Comparison/*.nc')
    print(f"\nExisting NetCDF files: {len(nc_files)}")

    if len(nc_files) > 0:
        # Check one NetCDF to see how many regions it has
        import xarray as xr
        sample_nc = nc_files[0]
        ds = xr.open_dataset(sample_nc)
        n_sites = ds.attrs.get('n_sites', 'unknown')
        print(f"  Sample: {os.path.basename(sample_nc)}")
        print(f"  Data points: {n_sites}")
        ds.close()

    return {
        'csv_files': csv_files,
        'regions_with_h5': regions_with_h5,
        'nc_files': nc_files
    }


def show_region_summary():
    """
    Show summary of which regions are in CSV vs have .h5 data
    """

    print("\n" + "="*80)
    print("REGION SUMMARY")
    print("="*80)

    # Get regions from CSV
    csv_file = 'OTEC_Comparison/onshore_offshore_comparison.csv'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        csv_regions = set(df['region'].unique())
    else:
        csv_regions = set()

    # Get regions with .h5 data
    data_dirs = [d for d in os.listdir('Data_Results')
                 if os.path.isdir(os.path.join('Data_Results', d))]

    h5_regions = set()
    for region_dir in data_dirs:
        h5_files = glob.glob(f'Data_Results/{region_dir}/T_*m_2020_*.h5')
        if len(h5_files) >= 2:
            h5_regions.add(region_dir.replace('_', ' '))

    print(f"\nRegions in CSV: {len(csv_regions)}")
    print(f"Regions with .h5 data: {len(h5_regions)}")

    # Regions in CSV but not in .h5
    csv_only = csv_regions - h5_regions
    if len(csv_only) > 0:
        print(f"\nRegions in CSV but missing .h5 data: {len(csv_only)}")
        for r in sorted(csv_only)[:10]:
            print(f"  - {r}")
        if len(csv_only) > 10:
            print(f"  ... and {len(csv_only)-10} more")

    # Regions with .h5 but not in CSV
    h5_only = h5_regions - csv_regions
    if len(h5_only) > 0:
        print(f"\nRegions with .h5 data but not in CSV: {len(h5_only)}")
        print("These regions can be added by regenerating NetCDF:")
        for r in sorted(h5_only)[:20]:
            print(f"  - {r}")
        if len(h5_only) > 20:
            print(f"  ... and {len(h5_only)-20} more")

    # Regions in both
    both = csv_regions & h5_regions
    print(f"\nRegions in both CSV and .h5: {len(both)}")

    return {
        'csv_regions': csv_regions,
        'h5_regions': h5_regions,
        'csv_only': csv_only,
        'h5_only': h5_only,
        'both': both
    }


def main():
    parser = argparse.ArgumentParser(description='Merge CSV files and regenerate NetCDF')
    parser.add_argument('--merge', action='store_true',
                       help='Merge multiple CSV files')
    parser.add_argument('--regenerate', action='store_true',
                       help='Regenerate NetCDF from .h5 files')
    parser.add_argument('--check', action='store_true',
                       help='Check data availability')
    parser.add_argument('--summary', action='store_true',
                       help='Show region summary')

    args = parser.parse_args()

    if not any([args.merge, args.regenerate, args.check, args.summary]):
        # Default: show summary
        print("\nNo action specified. Showing data summary...\n")
        check_data_availability()
        show_region_summary()

        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        print("\nTo regenerate NetCDF files with ALL regions that have .h5 data:")
        print("  python regenerate_netcdf_from_h5.py")
        print("\nTo merge multiple CSV files (if you have node-specific CSVs):")
        print("  python merge_csv_and_regenerate_netcdf.py --merge")
        print("\nTo just check what data you have:")
        print("  python merge_csv_and_regenerate_netcdf.py --check")
        print("="*80)

    else:
        if args.check:
            check_data_availability()

        if args.summary:
            show_region_summary()

        if args.merge:
            merged_df = merge_csv_files()
            if merged_df is not None:
                print("\nMerge successful! Now you can regenerate NetCDF files.")
                print("Run: python regenerate_netcdf_from_h5.py")

        if args.regenerate:
            print("\nTo regenerate NetCDF files, please run:")
            print("  python regenerate_netcdf_from_h5.py")


if __name__ == "__main__":
    main()
