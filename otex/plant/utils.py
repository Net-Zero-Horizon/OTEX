# -*- coding: utf-8 -*-
"""
OTEX Plant Utilities
Helper functions for pressure drop, saturation properties, and enthalpies.
Extended to support multiple working fluids and thermodynamic cycles.

@author: OTEX Development Team
"""

import numpy as np

def pressure_drop(T_water_ts,u_water_ts,d_pipes,rho_water,roughness_pipe,length,K_L,u_HX):
    
    dyn_visc = 0.000000344285714*T_water_ts**2-0.000047107142857*T_water_ts+0.001766642857143        
    Re = u_water_ts*rho_water*d_pipes/dyn_visc                     
    f = 0.25/(np.log10((roughness_pipe/d_pipes)/3.7+5.74/(Re**0.9))**2)        
   
    p_drop = ((f*rho_water*length/d_pipes*0.5*u_water_ts**2)+(K_L*rho_water*0.5*u_HX**2))/1000
    
    return p_drop

def saturation_pressures_and_temperatures(T_WW_in,T_CW_in,del_T_WW,del_T_CW,inputs):
    """
    Calculate evaporator and condenser temperatures and pressures

    This function can work with:
    1. Legacy mode: uses hardcoded NH3 correlations (original implementation)
    2. Modern mode: uses working_fluid object if provided in inputs

    Args:
        T_WW_in: Warm water inlet temperature [°C]
        T_CW_in: Cold water inlet temperature [°C]
        del_T_WW: Warm water temperature drop [°C]
        del_T_CW: Cold water temperature rise [°C]
        inputs: Dictionary with parameters (may include 'working_fluid')

    Returns:
        T_evap, T_cond, p_evap, p_cond
    """

    # Ensure inputs are arrays for broadcasting
    T_WW_in_arr = np.atleast_1d(T_WW_in)
    T_CW_in_arr = np.atleast_1d(T_CW_in)
    del_T_WW_arr = np.atleast_1d(del_T_WW)
    del_T_CW_arr = np.atleast_1d(del_T_CW)

    # Check if any input array is empty
    if T_WW_in_arr.size == 0 or T_CW_in_arr.size == 0:
        raise ValueError(f"Empty temperature array detected: T_WW_in has {T_WW_in_arr.size} elements, "
                        f"T_CW_in has {T_CW_in_arr.size} elements. Cannot perform OTEC analysis with empty data.")

    T_evap = np.round(T_WW_in - del_T_WW - inputs['T_pinch_WW'],1)
    T_cond = np.round(T_CW_in + del_T_CW + inputs['T_pinch_CW'],1)

    # Convert to arrays for consistent handling
    T_evap_arr = np.atleast_1d(T_evap)
    T_cond_arr = np.atleast_1d(T_cond)

    # Check for shape mismatch after calculation
    if T_evap_arr.shape != T_cond_arr.shape:
        raise ValueError(f"Shape mismatch after calculation: T_evap has shape {T_evap_arr.shape}, "
                        f"T_cond has shape {T_cond_arr.shape}. "
                        f"Input shapes: T_WW_in={T_WW_in_arr.shape}, T_CW_in={T_CW_in_arr.shape}, "
                        f"del_T_WW={del_T_WW_arr.shape}, del_T_CW={del_T_CW_arr.shape}")

    # Handle both scalar and array inputs
    is_scalar = np.isscalar(T_evap) or (isinstance(T_evap, np.ndarray) and T_evap.ndim == 0)

    if is_scalar:
        # Scalar case
        if T_evap <= T_cond:
            T_cond = np.nan
            T_evap = np.nan
    else:
        # Array case - check for empty arrays
        if T_evap_arr.size == 0:
            # Return empty arrays with consistent shapes
            pass  # T_evap and T_cond are already empty and same shape
        else:
            infeasible_T = np.where(T_evap <= T_cond,1,0)
            T_cond[infeasible_T==1] = np.nan
            T_evap[infeasible_T==1] = np.nan

    # Check if modern working fluid object is available
    if 'working_fluid' in inputs and inputs['working_fluid'] is not None:
        working_fluid = inputs['working_fluid']
        p_evap = working_fluid.saturation_pressure(T_evap)
        p_cond = working_fluid.saturation_pressure(T_cond)
    else:
        # Legacy NH3 polynomial correlation (original implementation)
        p_evap = 0.00002196*T_evap**3+0.00193103*T_evap**2+0.1695763*T_evap+4.25739601
        p_cond = 0.00002196*T_cond**3+0.00193103*T_cond**2+0.1695763*T_cond+4.25739601

    return T_evap,T_cond,p_evap,p_cond

def enthalpies_entropies(p_evap,p_cond,inputs):
    """
    Calculate cycle enthalpies and entropies

    This function can work with:
    1. Legacy mode: uses hardcoded NH3 correlations (original implementation)
    2. Modern mode: uses thermodynamic_cycle object if provided in inputs

    Args:
        p_evap: Evaporator pressure [bar]
        p_cond: Condenser pressure [bar]
        inputs: Dictionary with parameters (may include 'thermodynamic_cycle')

    Returns:
        Dictionary with h_1, h_2, h_3, h_4
    """

    # Check if modern thermodynamic cycle object is available
    if 'thermodynamic_cycle' in inputs and inputs['thermodynamic_cycle'] is not None:
        cycle = inputs['thermodynamic_cycle']
        cycle_name = cycle.name.lower() if hasattr(cycle, 'name') else ''

        # Get temperatures from pressures using working fluid
        if 'working_fluid' in inputs and inputs['working_fluid'] is not None:
            working_fluid = inputs['working_fluid']
            T_evap = working_fluid.saturation_temperature(p_evap)
            T_cond = working_fluid.saturation_temperature(p_cond)
        elif ('kalina' in cycle_name or 'uehara' in cycle_name) and hasattr(cycle, 'mixture'):
            # Kalina/Uehara cycle: use internal NH3-H2O mixture for temperature calculation
            # Use basic solution concentration for saturation temperature
            x_basic = getattr(cycle, 'x_basic', 0.7)

            # Handle arrays of any shape by flattening, processing, and reshaping
            p_evap_arr = np.asarray(p_evap)
            p_cond_arr = np.asarray(p_cond)
            original_shape_evap = p_evap_arr.shape
            original_shape_cond = p_cond_arr.shape

            # Flatten for element-wise processing
            p_evap_flat = p_evap_arr.ravel()
            p_cond_flat = p_cond_arr.ravel()

            # Calculate saturation temperatures for each pressure
            T_evap_flat = np.array([cycle.mixture.saturation_temperature(p, x_basic) for p in p_evap_flat])
            T_cond_flat = np.array([cycle.mixture.saturation_temperature(p, x_basic) for p in p_cond_flat])

            # Reshape to original shape
            T_evap = T_evap_flat.reshape(original_shape_evap)
            T_cond = T_cond_flat.reshape(original_shape_cond)

            # Return to scalar if input was scalar
            if np.isscalar(p_evap):
                T_evap = float(T_evap)
                T_cond = float(T_cond)
        else:
            # Fallback: estimate temperature (inverse of pressure correlation for NH3)
            # This is approximate for legacy support
            T_evap = (p_evap - 4.25739601) / 0.1695763  # Linear approximation
            T_cond = (p_cond - 4.25739601) / 0.1695763

        # Calculate cycle states using modern approach
        states = cycle.calculate_cycle_states(T_evap, T_cond, p_evap, p_cond, inputs)

        # Map states based on cycle type (cycle_name already defined above)
        if 'uehara' in cycle_name or 'two-stage' in cycle_name:
            # Uehara cycle: two-stage Rankine with HP and LP loops
            # Map HP loop states to standard format (primary power generation)
            # The LP loop contribution will be handled via mass flow calculations
            enthalpies = {
                'h_1': states['h_1_HP'],
                'h_2': states['h_2_HP'],
                'h_3': states['h_3_HP'],
                'h_4': states['h_4_HP'],
                # Store additional states for LP loop calculations
                'h_5_LP': states.get('h_5_LP', states['h_1_HP']),
                'h_6_LP': states.get('h_6_LP', states['h_2_HP']),
                'h_7_LP': states.get('h_7_LP', states['h_3_HP']),
                'h_8_LP': states.get('h_8_LP', states['h_4_HP']),
                'p_int': states.get('p_int', np.sqrt(p_evap * p_cond)),
                'cycle_type': 'uehara',
            }

        elif 'kalina' in cycle_name:
            # Kalina cycle: uses different state numbering
            # State 1: Condenser exit (rich stream)
            # State 7: Turbine inlet (separator vapor)
            # State 10: Turbine exit
            enthalpies = {
                'h_1': states['h_1'],
                'h_2': states['h_2'],
                'h_3': states['h_7'],  # Turbine inlet (separator vapor)
                'h_4': states['h_10'],  # Turbine exit
                # Store additional Kalina-specific states
                'h_evap_in': states.get('h_5', states['h_2']),
                'h_evap_out': states.get('h_6', states['h_7']),
                'split_ratio': states.get('split_ratio', 0.7),
                'cycle_type': 'kalina',
            }

        else:
            # Standard Rankine cycles (closed, open, hybrid)
            enthalpies = {
                'h_1': states['h_1'],
                'h_2': states['h_2'],
                'h_3': states['h_3'],
                'h_4': states['h_4'],
            }

        return enthalpies

    else:
        # Legacy NH3 polynomial correlation (original implementation)
        eff_isen_turb = inputs['eff_isen_turb']

        # Enthalpy and Entropy at Inlet (Evaporator Outlet, 100% Steam Quality, using approximation functions from Excel)
        h_3 = 28.276*np.log(p_evap)+1418.1
        s_3 = -0.352*np.log(p_evap)+6.1284

        # Enthalpy and Entropy at Outlet, using approximation functions from Excel

        s_4_liq = 0.3947*np.log(p_cond)+0.4644
        s_4_vap = -0.352*np.log(p_cond)+6.1284

        # Enthalpies of Liquid and Vapour Phase (Enthalpy at Liquid Phase equals Enthalpy and NH3 Pump Inlet)

        h_4_liq = -0.0235*p_cond**4+0.9083*p_cond**3-12.93*p_cond**2+97.316*p_cond-39.559
        h_4_vap = 28.276*np.log(p_cond)+1418.1

        x_4_isen = (s_3-s_4_liq)/(s_4_vap-s_4_liq)
        h_4_isen = h_4_vap*x_4_isen+h_4_liq*(1-x_4_isen)
        h_4 = (h_4_isen-h_3)*eff_isen_turb+h_3

        x_4 = (h_4-h_4_liq)/(h_4_vap-h_4_liq)
        s_4 = s_4_vap*x_4+s_4_liq*(1-x_4)

        h_1 = h_4_liq # inlet enthalpy
        h_2 = 1/inputs['rho_NH3']*(p_evap-p_cond)*100000/1000/inputs['eff_isen_pump']+h_1 # outlet enthalpy

        enthalpies = {
            'h_1': h_1,
            'h_2': h_2,
            'h_3': h_3,
            'h_4': h_4,
            }

        return enthalpies