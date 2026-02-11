"""
Regional Cold Water Depth Calculator

Calculates optimal CW intake depth for each region based on its
mean latitude. This provides a practical middle-ground between:
- Fixed global depth (current approach)
- Site-specific depths (requires multiple downloads per region)

Usage:
    python regional_depth_calculator.py

Outputs:
    - CSV with recommended depths for each region
    - Comparison statistics

@author: OTEX extension
"""

import pandas as pd
import numpy as np
from scripts.latitude_based_depth import calculate_depth_for_latitude
import os


def calculate_regional_depths(regions_file='download_ranges_per_region.csv',
                              sites_file='CMEMS_points_with_properties.csv',
                              output_file='regional_cw_depths.csv'):
    """
    Calculate optimal CW intake depth for each region

    Args:
        regions_file: Path to regions CSV
        sites_file: Path to sites CSV (to get actual site coordinates)
        output_file: Path to save output CSV

    Returns:
        DataFrame with regional depth recommendations
    """

    print("="*80)
    print("CALCULATING REGIONAL COLD WATER DEPTHS")
    print("="*80)

    # Load regions
    regions_df = pd.read_csv(regions_file, delimiter=';', encoding='latin-1')
    print(f"\nLoaded {len(regions_df)} region definitions")

    # Load sites to get actual coordinates
    sites_df = pd.read_csv(sites_file, delimiter=';')
    print(f"Loaded {len(sites_df)} CMEMS sites")

    # Calculate depth for each region
    results = []

    for idx, region_row in regions_df.iterrows():
        region_name = region_row['region']

        # Get all sites in this region
        region_sites = sites_df[sites_df['region'] == region_name]

        if len(region_sites) == 0:
            print(f"WARNING: {region_name}: No sites found")
            continue

        # Calculate mean latitude for the region
        mean_lat = region_sites['latitude'].mean()
        lat_std = region_sites['latitude'].std()
        lat_min = region_sites['latitude'].min()
        lat_max = region_sites['latitude'].max()

        # Calculate optimal depth for mean latitude
        optimal_depth = calculate_depth_for_latitude(mean_lat)

        # Also calculate depths at extremes to assess variability
        depth_at_min_lat = calculate_depth_for_latitude(lat_min)
        depth_at_max_lat = calculate_depth_for_latitude(lat_max)
        depth_range = abs(depth_at_max_lat - depth_at_min_lat)

        # Current fixed depth
        current_depth = 1062

        results.append({
            'region': region_name,
            'n_sites': len(region_sites),
            'mean_latitude': mean_lat,
            'lat_std': lat_std,
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lat_range': lat_max - lat_min,
            'recommended_depth_m': int(optimal_depth),
            'depth_at_min_lat': int(depth_at_min_lat),
            'depth_at_max_lat': int(depth_at_max_lat),
            'depth_variability_m': int(depth_range),
            'current_fixed_depth_m': current_depth,
            'depth_change_m': int(optimal_depth - current_depth),
            'depth_change_pct': round((optimal_depth - current_depth) / current_depth * 100, 1),
        })

    # Create DataFrame
    df_results = pd.DataFrame(results)

    # Sort by latitude
    df_results = df_results.sort_values('mean_latitude')

    # Save to CSV
    df_results.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")

    return df_results


def summarize_regional_depths(df):
    """
    Print summary statistics of regional depth recommendations
    """

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\nTotal regions: {len(df)}")
    print(f"Total sites: {df['n_sites'].sum():,}")

    print(f"\nDepth Statistics:")
    print(f"  Mean recommended depth: {df['recommended_depth_m'].mean():.1f} m")
    print(f"  Depth range: {df['recommended_depth_m'].min()} - {df['recommended_depth_m'].max()} m")
    print(f"  Std deviation: {df['recommended_depth_m'].std():.1f} m")

    print(f"\nComparison to current fixed depth (1062m):")
    shallower = (df['depth_change_m'] < 0).sum()
    deeper = (df['depth_change_m'] > 0).sum()
    same = (df['depth_change_m'] == 0).sum()

    print(f"  Regions needing shallower depth: {shallower} ({shallower/len(df)*100:.1f}%)")
    print(f"  Regions needing deeper depth: {deeper} ({deeper/len(df)*100:.1f}%)")
    print(f"  Regions at current depth: {same} ({same/len(df)*100:.1f}%)")
    print(f"  Mean depth change: {df['depth_change_m'].mean():.1f} m ({df['depth_change_pct'].mean():.1f}%)")

    print(f"\nRegions with largest depth reduction:")
    top_reductions = df.nsmallest(5, 'depth_change_m')[['region', 'mean_latitude', 'recommended_depth_m', 'depth_change_m']]
    print(top_reductions.to_string(index=False))

    print(f"\nRegions needing deepest intakes:")
    top_depths = df.nlargest(5, 'recommended_depth_m')[['region', 'mean_latitude', 'recommended_depth_m', 'depth_change_m']]
    print(top_depths.to_string(index=False))

    print(f"\nRegions with highest intra-regional depth variability:")
    top_variability = df.nlargest(5, 'depth_variability_m')[['region', 'lat_range', 'depth_variability_m', 'recommended_depth_m']]
    print(top_variability.to_string(index=False))


def create_depth_modification_dict(csv_file='regional_cw_depths.csv'):
    """
    Create a dictionary mapping region names to recommended depths

    This can be used directly in the code to override default depths

    Args:
        csv_file: Path to regional depths CSV

    Returns:
        Dictionary {region_name: depth_m}
    """

    df = pd.read_csv(csv_file)

    depth_dict = dict(zip(df['region'], df['recommended_depth_m']))

    return depth_dict


def generate_code_snippet(csv_file='regional_cw_depths.csv'):
    """
    Generate Python code snippet to use regional depths

    This can be copy-pasted into other scripts
    """

    depth_dict = create_depth_modification_dict(csv_file)

    print("\n" + "="*80)
    print("PYTHON CODE SNIPPET FOR REGIONAL DEPTHS")
    print("="*80)
    print("\n# Regional Cold Water Intake Depths (based on latitude)")
    print("# Generated by regional_depth_calculator.py")
    print("REGIONAL_CW_DEPTHS = {")

    for region, depth in sorted(depth_dict.items()):
        print(f"    '{region}': {depth},")

    print("}")

    print("\n# Usage example:")
    print("region_name = 'Philippines'")
    print("cw_depth = REGIONAL_CW_DEPTHS.get(region_name, 1062)  # Default to 1062m if not found")
    print("inputs['length_CW_inlet'] = cw_depth")

    print("\n" + "="*80)


def compare_with_current():
    """
    Compare proposed regional depths with current fixed depth approach
    """

    df = pd.read_csv('regional_cw_depths.csv')

    print("\n" + "="*80)
    print("COMPARISON: REGIONAL vs FIXED DEPTH")
    print("="*80)

    # Group regions by latitude zones
    def get_zone(lat):
        abs_lat = abs(lat)
        if abs_lat < 10:
            return 'Equatorial/Near-Eq (0-10°)'
        elif abs_lat < 20:
            return 'Tropical (10-20°)'
        elif abs_lat < 30:
            return 'Subtropical (20-30°)'
        else:
            return 'Mid-Latitude (30-40°)'

    df['zone'] = df['mean_latitude'].apply(get_zone)

    # Statistics by zone
    zone_stats = df.groupby('zone').agg({
        'region': 'count',
        'n_sites': 'sum',
        'recommended_depth_m': ['mean', 'min', 'max'],
        'depth_change_m': ['mean', 'min', 'max'],
    }).round(1)

    print("\nStatistics by Latitude Zone:")
    print(zone_stats)

    # Estimated cost impact
    print("\n" + "-"*80)
    print("ESTIMATED COST IMPACT")
    print("-"*80)

    # Rough estimate: $30-50 per meter of pipe length per kW
    cost_per_meter_per_kw = 40  # $/m/kW
    plant_size_kw = 136000  # 136 MW

    total_depth_reduction = df['depth_change_m'].sum()
    avg_depth_reduction_per_region = df['depth_change_m'].mean()

    cost_savings_per_region = abs(avg_depth_reduction_per_region) * cost_per_meter_per_kw * plant_size_kw / 1e6

    print(f"\nAssuming:")
    print(f"  - Plant size: {plant_size_kw/1000:.0f} MW")
    print(f"  - Pipe cost: ${cost_per_meter_per_kw}/m/kW")
    print(f"\nEstimated impact per region:")
    print(f"  - Average depth change: {avg_depth_reduction_per_region:.1f} m")

    if avg_depth_reduction_per_region < 0:
        print(f"  - Estimated CAPEX savings: ${abs(cost_savings_per_region):.2f} million")
        print(f"    (Shorter pipes for {(df['depth_change_m'] < 0).sum()} regions)")
    else:
        print(f"  - Estimated CAPEX increase: ${cost_savings_per_region:.2f} million")

    print(f"\nNote: Actual cost impact depends on:")
    print(f"  - Thermal benefit from colder water (can offset pipe costs)")
    print(f"  - Installation depth factors")
    print(f"  - Regional bathymetry constraints")


def main():
    """
    Main execution
    """

    # Calculate regional depths
    df = calculate_regional_depths()

    # Summarize results
    summarize_regional_depths(df)

    # Compare with current approach
    compare_with_current()

    # Generate code snippet
    generate_code_snippet()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review regional_cw_depths.csv")
    print("2. Integrate depths into compare_onshore_offshore.py")
    print("3. Re-run analyses with variable depths")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
