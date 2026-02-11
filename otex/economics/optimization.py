# -*- coding: utf-8 -*-
"""
Depth Optimization Module for OTEX
Optimizes cold water intake depth for each location based on LCOE

Current implementation assumes fixed depth globally (e.g., 1062m)
This module enables location-specific depth optimization considering:
- Available temperature difference
- Pipe cost vs thermal benefit tradeoff
- Local bathymetry constraints

@author: Extended OTEX
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def estimate_temperature_at_depth(depth, T_surface, depth_profile_model='exponential'):
    """
    Estimate seawater temperature at given depth

    This is a simplified model. In practice, should use actual CMEMS data.

    Args:
        depth: Depth in meters (positive down)
        T_surface: Surface temperature in °C
        depth_profile_model: 'exponential', 'linear', or 'polynomial'

    Returns:
        T: Temperature at depth in °C
    """

    if depth_profile_model == 'exponential':
        # Exponential decay model: T(z) = T_deep + (T_surf - T_deep) * exp(-z/z0)
        T_deep = 4.0  # Deep ocean temperature ~4°C
        z0 = 500  # Characteristic depth scale
        T = T_deep + (T_surface - T_deep) * np.exp(-depth / z0)

    elif depth_profile_model == 'linear':
        # Simple linear interpolation
        T_deep = 4.0
        depth_thermocline = 1000
        if depth < depth_thermocline:
            T = T_surface + (T_deep - T_surface) * depth / depth_thermocline
        else:
            T = T_deep

    elif depth_profile_model == 'polynomial':
        # Polynomial fit (typical tropical ocean profile)
        # Based on empirical data
        z_norm = depth / 1000  # Normalize to km
        T = T_surface - 10*z_norm - 5*z_norm**2 + 2*z_norm**3
        T = max(T, 4.0)  # Floor at deep ocean temp

    else:
        raise ValueError(f"Unknown depth profile model: {depth_profile_model}")

    return T


def calculate_pipe_cost_increment(depth, depth_reference, cost_per_meter, inputs):
    """
    Calculate incremental pipe cost for different depths

    Args:
        depth: Target depth [m]
        depth_reference: Reference depth [m] (e.g., 1062m)
        cost_per_meter: Pipe cost per meter [$/m]
        inputs: Dictionary with pipe parameters

    Returns:
        delta_cost: Incremental cost [$]
    """

    # Get pipe diameter (assumed constant for first-order analysis)
    # In reality, would need to recalculate based on flow rate

    delta_length = abs(depth - depth_reference)

    # Cost includes material + installation
    # Installation cost is higher for deeper depths
    depth_factor = 1.0 + 0.0001 * depth  # 10% increase per 1000m

    delta_cost = delta_length * cost_per_meter * depth_factor

    return delta_cost


def calculate_thermal_benefit(T_CW, T_CW_reference, T_WW, inputs):
    """
    Calculate thermal benefit from colder water

    Colder water increases Carnot efficiency and reduces required seawater flow

    Args:
        T_CW: Cold water temperature [°C]
        T_CW_reference: Reference cold water temperature [°C]
        T_WW: Warm water temperature [°C]
        inputs: Dictionary with cycle parameters

    Returns:
        power_benefit: Estimated power increase [kW] (or cost reduction)
    """

    # Simplified analysis: Carnot efficiency proportional to ΔT
    # Real analysis would run full OTEC sizing

    delta_T = T_WW - T_CW
    delta_T_ref = T_WW - T_CW_reference

    # Thermal efficiency approximately proportional to ΔT (first order)
    efficiency_ratio = delta_T / delta_T_ref

    # Power benefit (assuming same gross power target)
    # Better efficiency means less pumping power needed
    p_gross = inputs.get('p_gross', -136000)

    # Rough estimate: 1°C colder water = 2-3% efficiency improvement
    # This is a simplified model
    benefit_per_degree = 0.025 * abs(p_gross)  # 2.5% per °C

    power_benefit = (T_CW_reference - T_CW) * benefit_per_degree

    return power_benefit


def optimize_cw_depth_for_location(T_WW, T_surface, depth_range, inputs, cost_level='low_cost'):
    """
    Optimize cold water intake depth for a specific location

    Args:
        T_WW: Warm water temperature [°C]
        T_surface: Surface water temperature [°C] (for profile estimation)
        depth_range: Tuple (min_depth, max_depth) to search [m]
        inputs: Dictionary with parameters
        cost_level: 'low_cost' or 'high_cost'

    Returns:
        optimal_depth: Optimized depth [m]
        T_CW_optimal: Temperature at optimal depth [°C]
        lcoe_vs_depth: Dictionary with depth sweep results
    """

    # Reference depth (original implementation)
    depth_reference = 1062.4
    T_CW_reference = estimate_temperature_at_depth(depth_reference, T_surface)

    # Cost parameters
    if cost_level == 'low_cost':
        cost_per_meter = 9.0 * 1000  # $/m (scaled by mass assumption)
    else:
        cost_per_meter = 30.1 * 1000

    # Depth candidates to evaluate
    min_depth, max_depth = depth_range
    depth_candidates = np.arange(min_depth, max_depth, 50)  # Every 50m

    lcoe_values = []
    T_CW_values = []

    for depth in depth_candidates:
        # Estimate temperature at this depth
        T_CW = estimate_temperature_at_depth(depth, T_surface)
        T_CW_values.append(T_CW)

        # Check if sufficient ΔT available
        delta_T_available = T_WW - T_CW
        if delta_T_available < 15:  # Minimum ΔT for viable OTEC
            lcoe_values.append(np.inf)
            continue

        # Calculate incremental cost
        pipe_cost_increment = calculate_pipe_cost_increment(
            depth, depth_reference, cost_per_meter, inputs
        )

        # Calculate thermal benefit
        power_benefit = calculate_thermal_benefit(
            T_CW, T_CW_reference, T_WW, inputs
        )

        # Simplified LCOE calculation
        # Full implementation would call otec_sizing and capex_opex_lcoe

        # Baseline LCOE (assumed from reference depth)
        lcoe_baseline = inputs.get('lcoe_reference', 15.0)  # ct/kWh

        # CAPEX increment
        capex_baseline = inputs.get('capex_reference', 500e6)  # $
        capex_increment = pipe_cost_increment
        capex_ratio = (capex_baseline + capex_increment) / capex_baseline

        # Energy benefit
        energy_baseline = abs(inputs.get('p_gross', -136000)) * 8760  # kWh/yr
        energy_increment = power_benefit * 8760
        energy_ratio = (energy_baseline + energy_increment) / energy_baseline

        # Adjusted LCOE (simplified)
        lcoe_adjusted = lcoe_baseline * capex_ratio / energy_ratio

        lcoe_values.append(lcoe_adjusted)

    # Find optimal depth
    lcoe_values = np.array(lcoe_values)
    optimal_idx = np.argmin(lcoe_values)
    optimal_depth = depth_candidates[optimal_idx]
    T_CW_optimal = T_CW_values[optimal_idx]
    lcoe_optimal = lcoe_values[optimal_idx]

    # Package results
    lcoe_vs_depth = {
        'depth': depth_candidates,
        'T_CW': T_CW_values,
        'LCOE': lcoe_values,
        'optimal_depth': optimal_depth,
        'T_CW_optimal': T_CW_optimal,
        'LCOE_optimal': lcoe_optimal,
    }

    return optimal_depth, T_CW_optimal, lcoe_vs_depth


def optimize_depths_for_all_locations(coordinates, T_WW_profiles, inputs, cost_level='low_cost'):
    """
    Optimize cold water depths for all locations in study area

    Args:
        coordinates: Array of [lon, lat] for each location
        T_WW_profiles: Warm water temperature profiles (for average calculation)
        inputs: Dictionary with parameters
        cost_level: 'low_cost' or 'high_cost'

    Returns:
        optimal_depths: Array of optimal depths for each location [m]
        T_CW_optimal: Array of optimal CW temperatures [°C]
    """

    n_locations = coordinates.shape[0]
    optimal_depths = np.zeros(n_locations)
    T_CW_optimal = np.zeros(n_locations)

    # Depth search range
    min_depth = inputs.get('min_depth', -600)
    max_depth = inputs.get('max_depth', -3000)
    depth_range = (abs(max_depth), abs(min_depth))  # Convert to positive

    print(f"\nOptimizing cold water intake depths for {n_locations} locations...")
    print(f"Depth range: {depth_range[0]}-{depth_range[1]} m")

    for i in range(n_locations):
        # Average warm water temperature for this location
        T_WW = np.nanmean(T_WW_profiles[:, i])

        # Estimate surface temperature (assume ~2°C higher than WW intake)
        T_surface = T_WW + 2.0

        # Optimize depth for this location
        depth_opt, T_CW_opt, _ = optimize_cw_depth_for_location(
            T_WW, T_surface, depth_range, inputs, cost_level
        )

        optimal_depths[i] = depth_opt
        T_CW_optimal[i] = T_CW_opt

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_locations} locations")

    print(f"\nDepth optimization complete!")
    print(f"  Mean optimal depth: {np.mean(optimal_depths):.1f} m")
    print(f"  Depth range: {np.min(optimal_depths):.1f} - {np.max(optimal_depths):.1f} m")
    print(f"  Std deviation: {np.std(optimal_depths):.1f} m")

    return optimal_depths, T_CW_optimal


def analyze_depth_sensitivity(T_WW, T_surface, inputs, cost_level='low_cost'):
    """
    Perform sensitivity analysis on depth optimization

    Shows how LCOE varies with depth for a given location
    Useful for understanding tradeoffs

    Args:
        T_WW: Warm water temperature [°C]
        T_surface: Surface temperature [°C]
        inputs: Dictionary with parameters
        cost_level: 'low_cost' or 'high_cost'

    Returns:
        sensitivity_df: DataFrame with depth sweep results
    """

    depth_range = (400, 1500)

    optimal_depth, T_CW_optimal, results = optimize_cw_depth_for_location(
        T_WW, T_surface, depth_range, inputs, cost_level
    )

    sensitivity_df = pd.DataFrame({
        'Depth_m': results['depth'],
        'Temperature_C': results['T_CW'],
        'LCOE_ct_per_kWh': results['LCOE'],
    })

    # Add markers for optimal and reference
    sensitivity_df['Is_Optimal'] = sensitivity_df['Depth_m'] == optimal_depth
    sensitivity_df['Is_Reference'] = np.abs(sensitivity_df['Depth_m'] - 1062) < 25

    return sensitivity_df


if __name__ == "__main__":
    # Test the depth optimization module
    print("Testing Depth Optimization Module\n")
    print("="*60)

    # Mock inputs
    inputs = {
        'p_gross': -136000,  # kW
        'min_depth': -600,
        'max_depth': -3000,
        'lcoe_reference': 15.0,  # ct/kWh at reference depth
        'capex_reference': 500e6,  # $
    }

    # Test case: Tropical location
    T_WW = 26.0  # °C
    T_surface = 28.0  # °C
    depth_range = (400, 1500)  # m

    print(f"\nTest Case:")
    print(f"Warm water temperature: {T_WW}°C")
    print(f"Surface temperature: {T_surface}°C")
    print(f"Depth search range: {depth_range[0]}-{depth_range[1]} m")

    # Run optimization
    optimal_depth, T_CW_optimal, results = optimize_cw_depth_for_location(
        T_WW, T_surface, depth_range, inputs, 'low_cost'
    )

    print(f"\nOptimization Results:")
    print(f"Optimal depth: {optimal_depth:.1f} m")
    print(f"Temperature at optimal depth: {T_CW_optimal:.2f}°C")
    print(f"LCOE at optimal depth: {results['LCOE_optimal']:.2f} ct/kWh")
    print(f"Reference depth (1062m): T = {estimate_temperature_at_depth(1062, T_surface):.2f}°C")

    # Sensitivity analysis
    print(f"\n\nSensitivity Analysis:")
    sensitivity_df = analyze_depth_sensitivity(T_WW, T_surface, inputs, 'low_cost')

    print(f"\nDepth vs LCOE (sample):")
    print(sensitivity_df.iloc[::5].to_string(index=False))  # Every 5th row

    print("\n" + "="*60)
    print("Testing complete!")
    print("\nNote: This uses simplified thermal and cost models.")
    print("For production use, integrate with full OTEC sizing calculations.")
