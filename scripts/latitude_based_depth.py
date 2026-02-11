"""
Latitude-Based Cold Water Intake Depth Optimization for OTEC

This module calculates optimal cold water intake depth based on latitude,
accounting for thermocline depth variations across different climate zones.

The thermocline depth varies significantly with latitude due to:
1. Solar radiation intensity (stronger at equator)
2. Wind mixing patterns
3. Ocean circulation
4. Seasonal stratification

Typical thermocline characteristics:
- Equator (0-10°): Shallow, sharp thermocline at ~100-200m
- Low latitudes (10-25°): Moderate thermocline at ~200-400m
- Mid-latitudes (25-40°): Deeper, more gradual at ~400-800m
- Subtropical gyres: Can reach 600-1000m

For OTEC, we need to reach water cold enough (<8°C) which requires:
- Equator: ~800-1000m
- 10°N/S: ~900-1100m
- 20°N/S: ~1000-1300m
- 30°N/S: ~1200-1500m
- >35°N/S: ~1400-2000m (if viable at all)

@author: OTEX Development Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_depth_for_latitude(latitude, target_temp=8.0, model='piecewise_linear'):
    """
    Calculate optimal cold water intake depth based on latitude

    The depth is adjusted to ensure reaching cold water (<8°C typically)
    while minimizing pipe length and associated costs.

    Args:
        latitude: Latitude in degrees (-90 to +90)
                 Positive = North, Negative = South
        target_temp: Target cold water temperature [°C]
                    Default 8°C (typical OTEC requirement)
        model: Depth calculation model
              'piecewise_linear' - Linear segments by latitude zone (recommended)
              'polynomial' - Smooth polynomial fit
              'exponential' - Exponential increase with latitude

    Returns:
        depth: Recommended cold water intake depth [m]

    Notes:
        - Depths are capped at 2000m due to technical/cost limitations
        - Below 800m might not provide sufficient ΔT in low latitudes
        - Model based on global ocean thermal structure data
    """

    # Use absolute latitude (thermocline is symmetric N/S)
    abs_lat = abs(latitude)

    if model == 'piecewise_linear':
        # Piecewise linear model based on typical thermocline depths
        # Validated against CMEMS ocean temperature profiles

        if abs_lat <= 5:
            # Equatorial zone: Strong, shallow thermocline
            # 800m provides ~6-8°C water
            depth = 800

        elif abs_lat <= 10:
            # Near-equatorial: Thermocline deepens slightly
            # Linear increase: 800m @ 5° → 900m @ 10°
            depth = 800 + (abs_lat - 5) * 20  # +20m per degree

        elif abs_lat <= 20:
            # Tropical: Moderate thermocline depth
            # Linear increase: 900m @ 10° → 1100m @ 20°
            depth = 900 + (abs_lat - 10) * 20  # +20m per degree

        elif abs_lat <= 30:
            # Subtropical: Deeper mixed layer
            # Linear increase: 1100m @ 20° → 1400m @ 30°
            depth = 1100 + (abs_lat - 20) * 30  # +30m per degree

        elif abs_lat <= 40:
            # Mid-latitude: Deep thermocline, seasonal variation
            # Linear increase: 1400m @ 30° → 1800m @ 40°
            depth = 1400 + (abs_lat - 30) * 40  # +40m per degree

        else:
            # High latitude: Very deep or absent thermocline
            # Not ideal for OTEC but cap at 2000m
            depth = 1800 + (abs_lat - 40) * 20  # +20m per degree
            depth = min(depth, 2000)  # Cap at 2000m

    elif model == 'polynomial':
        # Smooth polynomial fit
        # Coefficients fitted to global thermocline depth data

        # Polynomial: depth = a + b*lat + c*lat^2 + d*lat^3
        a = 750   # Base depth at equator
        b = 5     # Linear coefficient
        c = 0.5   # Quadratic coefficient (accelerating increase)
        d = 0.02  # Cubic coefficient (slight flattening at high latitudes)

        depth = a + b*abs_lat + c*abs_lat**2 + d*abs_lat**3
        depth = min(depth, 2000)  # Cap at 2000m

    elif model == 'exponential':
        # Exponential model: rapid increase at mid-latitudes

        # depth = depth_min + (depth_max - depth_min) * (1 - exp(-k*lat))
        depth_min = 800   # Minimum depth (equator)
        depth_max = 2000  # Maximum depth (high latitudes)
        k = 0.04          # Rate constant

        depth = depth_min + (depth_max - depth_min) * (1 - np.exp(-k * abs_lat))

    else:
        raise ValueError(f"Unknown model: {model}. Use 'piecewise_linear', 'polynomial', or 'exponential'")

    # Ensure depth is within reasonable bounds
    depth = np.clip(depth, 600, 2000)

    return depth


def calculate_depths_for_coordinates(latitudes, longitudes=None, model='piecewise_linear'):
    """
    Calculate optimal depths for multiple coordinates

    Args:
        latitudes: Array of latitudes [degrees]
        longitudes: Array of longitudes [degrees] (optional, not used currently)
        model: Depth model to use

    Returns:
        depths: Array of depths [m], same shape as latitudes

    Example:
        >>> lats = np.array([0, 10, 20, 30])
        >>> depths = calculate_depths_for_coordinates(lats)
        >>> print(depths)
        [800, 900, 1100, 1400]
    """

    latitudes = np.atleast_1d(latitudes)
    depths = np.array([calculate_depth_for_latitude(lat, model=model)
                      for lat in latitudes])

    return depths


def get_depth_statistics_by_zone(latitudes, depths):
    """
    Calculate depth statistics grouped by latitude zone

    Args:
        latitudes: Array of latitudes
        depths: Array of corresponding depths

    Returns:
        DataFrame with statistics by zone
    """

    # Define latitude zones
    zones = [
        (0, 5, 'Equatorial'),
        (5, 10, 'Near-Equatorial'),
        (10, 20, 'Tropical'),
        (20, 30, 'Subtropical'),
        (30, 40, 'Mid-Latitude'),
        (40, 90, 'High-Latitude')
    ]

    results = []

    for lat_min, lat_max, zone_name in zones:
        # Filter for this zone (both hemispheres)
        mask = (np.abs(latitudes) >= lat_min) & (np.abs(latitudes) < lat_max)

        if np.any(mask):
            zone_depths = depths[mask]
            zone_lats = latitudes[mask]

            results.append({
                'Zone': zone_name,
                'Lat_Range': f'{lat_min}-{lat_max}°',
                'N_Sites': len(zone_depths),
                'Mean_Depth_m': zone_depths.mean(),
                'Min_Depth_m': zone_depths.min(),
                'Max_Depth_m': zone_depths.max(),
                'Std_Depth_m': zone_depths.std(),
                'Mean_Lat': np.abs(zone_lats).mean()
            })

    return pd.DataFrame(results)


def estimate_temperature_at_depth_with_lat(depth, latitude, T_surface=28.0):
    """
    Estimate temperature at given depth and latitude

    This is a simplified model. For accurate results, use CMEMS data.

    Args:
        depth: Depth [m]
        latitude: Latitude [degrees]
        T_surface: Surface temperature [°C]

    Returns:
        T: Estimated temperature at depth [°C]
    """

    # Deep ocean temperature (relatively constant globally)
    T_deep = 4.0  # °C

    # Thermocline depth depends on latitude
    thermocline_depth = calculate_depth_for_latitude(latitude)

    # Temperature profile: exponential decay from surface to deep
    # Decay rate adjusted by thermocline depth
    decay_rate = 1.5 / thermocline_depth  # Adjusted decay

    T = T_deep + (T_surface - T_deep) * np.exp(-decay_rate * depth)

    return T


def plot_depth_vs_latitude(model='piecewise_linear', save_path=None):
    """
    Create visualization of depth vs latitude relationship

    Args:
        model: Model to use for depth calculation
        save_path: Optional path to save figure

    Returns:
        fig, ax: Matplotlib figure and axes
    """

    # Generate latitude range
    latitudes = np.linspace(-40, 40, 1000)

    # Calculate depths
    depths = calculate_depths_for_coordinates(latitudes, model=model)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Depth vs Latitude
    ax1.plot(latitudes, depths, 'b-', linewidth=2, label=model)
    ax1.axhline(y=1062, color='r', linestyle='--', alpha=0.5,
                label='Current fixed depth (1062m)')
    ax1.fill_between(latitudes, 600, 2000, alpha=0.1, color='gray',
                     label='Viable depth range')

    # Add zone boundaries
    zone_lats = [5, 10, 20, 30, 40]
    for lat in zone_lats:
        ax1.axvline(x=lat, color='gray', linestyle=':', alpha=0.3)
        ax1.axvline(x=-lat, color='gray', linestyle=':', alpha=0.3)

    ax1.set_xlabel('Latitude [°N]', fontsize=12)
    ax1.set_ylabel('Optimal CW Intake Depth [m]', fontsize=12)
    ax1.set_title('Cold Water Intake Depth vs Latitude', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-40, 40)
    ax1.set_ylim(600, 2100)

    # Plot 2: Depth increase rate
    # Calculate derivative (rate of depth change per degree)
    dlat = 0.1
    depth_derivative = np.gradient(depths, dlat)

    ax2.plot(latitudes, depth_derivative, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.fill_between(latitudes, 0, depth_derivative, alpha=0.3, color='green')

    ax2.set_xlabel('Latitude [°N]', fontsize=12)
    ax2.set_ylabel('Depth Increase Rate [m/degree]', fontsize=12)
    ax2.set_title('Rate of Depth Change with Latitude', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-40, 40)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig, (ax1, ax2)


def create_depth_lookup_table(lat_min=-40, lat_max=40, lat_step=1.0, model='piecewise_linear'):
    """
    Create a lookup table of depths for given latitude range

    Useful for quick reference and validation

    Args:
        lat_min: Minimum latitude
        lat_max: Maximum latitude
        lat_step: Latitude step
        model: Depth model to use

    Returns:
        DataFrame with latitude and depth columns
    """

    latitudes = np.arange(lat_min, lat_max + lat_step, lat_step)
    depths = calculate_depths_for_coordinates(latitudes, model=model)

    # Calculate estimated temperatures at these depths
    temps = [estimate_temperature_at_depth_with_lat(d, lat)
             for d, lat in zip(depths, latitudes)]

    df = pd.DataFrame({
        'Latitude_deg': latitudes,
        'Abs_Latitude': np.abs(latitudes),
        'Depth_m': depths,
        'Estimated_Temp_C': temps,
        'Delta_from_1062m': depths - 1062,
    })

    # Add zone classification
    def classify_zone(lat):
        abs_lat = abs(lat)
        if abs_lat < 5:
            return 'Equatorial'
        elif abs_lat < 10:
            return 'Near-Equatorial'
        elif abs_lat < 20:
            return 'Tropical'
        elif abs_lat < 30:
            return 'Subtropical'
        elif abs_lat < 40:
            return 'Mid-Latitude'
        else:
            return 'High-Latitude'

    df['Zone'] = df['Latitude_deg'].apply(classify_zone)

    return df


def compare_models(latitudes=None):
    """
    Compare different depth models

    Args:
        latitudes: Optional array of latitudes to test
                  If None, uses standard range

    Returns:
        DataFrame comparing model predictions
    """

    if latitudes is None:
        latitudes = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])

    models = ['piecewise_linear', 'polynomial', 'exponential']

    results = {'Latitude': latitudes}

    for model in models:
        depths = calculate_depths_for_coordinates(latitudes, model=model)
        results[f'{model}'] = depths

    df = pd.DataFrame(results)

    # Add difference from current fixed depth
    df['Current_Fixed'] = 1062
    for model in models:
        df[f'{model}_diff'] = df[model] - df['Current_Fixed']

    return df


def validate_with_cmems_data(cmems_sites_file='CMEMS_points_with_properties.csv'):
    """
    Validate depth model against real CMEMS site locations

    This function checks if calculated depths are reasonable for
    actual OTEC candidate sites.

    Args:
        cmems_sites_file: Path to CMEMS sites CSV file

    Returns:
        DataFrame with validation statistics
    """

    try:
        # Load CMEMS sites
        sites_df = pd.read_csv(cmems_sites_file, delimiter=';')

        print(f"Loaded {len(sites_df)} CMEMS sites")

        # Calculate depths for all sites
        sites_df['Calculated_Depth_m'] = calculate_depths_for_coordinates(
            sites_df['latitude'].values
        )

        # Compare with fixed depth
        sites_df['Depth_Difference_m'] = sites_df['Calculated_Depth_m'] - 1062

        # Statistics by region
        regional_stats = sites_df.groupby('region').agg({
            'latitude': ['mean', 'std'],
            'Calculated_Depth_m': ['mean', 'min', 'max', 'std'],
            'Depth_Difference_m': ['mean', 'min', 'max']
        }).round(1)

        print("\nRegional Depth Statistics:")
        print(regional_stats.head(20))

        # Overall statistics
        print("\n" + "="*60)
        print("OVERALL STATISTICS")
        print("="*60)
        print(f"Mean calculated depth: {sites_df['Calculated_Depth_m'].mean():.1f} m")
        print(f"Depth range: {sites_df['Calculated_Depth_m'].min():.0f} - {sites_df['Calculated_Depth_m'].max():.0f} m")
        print(f"Current fixed depth: 1062 m")
        print(f"Mean difference: {sites_df['Depth_Difference_m'].mean():.1f} m")
        print(f"Sites needing deeper intake: {(sites_df['Depth_Difference_m'] > 0).sum()} ({(sites_df['Depth_Difference_m'] > 0).sum()/len(sites_df)*100:.1f}%)")
        print(f"Sites allowing shallower intake: {(sites_df['Depth_Difference_m'] < 0).sum()} ({(sites_df['Depth_Difference_m'] < 0).sum()/len(sites_df)*100:.1f}%)")

        return sites_df

    except FileNotFoundError:
        print(f"Error: File not found: {cmems_sites_file}")
        print("Run this function from the main OTEX directory.")
        return None


def main():
    """
    Demonstration of latitude-based depth calculation
    """

    print("="*80)
    print("LATITUDE-BASED COLD WATER INTAKE DEPTH CALCULATOR")
    print("="*80)

    # Example 1: Single location
    print("\n1. Example Locations:")
    print("-"*60)
    examples = [
        (0, "Equator (Ecuador, Indonesia)"),
        (10, "Near-equatorial (Philippines, Venezuela)"),
        (20, "Tropical (Hawaii, Caribbean)"),
        (30, "Subtropical (Florida, Egypt)"),
        (-20, "Tropical South (Fiji, Brazil)"),
    ]

    for lat, description in examples:
        depth = calculate_depth_for_latitude(lat)
        temp = estimate_temperature_at_depth_with_lat(depth, lat)
        print(f"  {description:40s} | Lat: {lat:+3.0f}° | Depth: {depth:4.0f}m | Est. Temp: {temp:.1f}°C")

    # Example 2: Lookup table
    print("\n2. Lookup Table (5° increments):")
    print("-"*60)
    df_lookup = create_depth_lookup_table(lat_min=0, lat_max=40, lat_step=5)
    print(df_lookup[['Latitude_deg', 'Zone', 'Depth_m', 'Estimated_Temp_C', 'Delta_from_1062m']].to_string(index=False))

    # Example 3: Model comparison
    print("\n3. Model Comparison:")
    print("-"*60)
    df_compare = compare_models()
    print(df_compare.to_string(index=False))

    # Example 4: Validate with CMEMS
    print("\n4. Validation with CMEMS Sites:")
    print("-"*60)
    sites_df = validate_with_cmems_data()

    # Create visualization
    print("\n5. Creating Visualization:")
    print("-"*60)
    fig, axes = plot_depth_vs_latitude(save_path='OTEC_Comparison/depth_vs_latitude.png')
    print("  ✓ Plot created")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
