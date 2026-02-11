# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:56:12 2023

@author: jkalanger
"""

import numpy as np

def capex_opex_lcoe(otec_plant_nom,inputs,cost_level='low_cost'):
    """
    Calculate CAPEX, OPEX, and LCOE for OTEC plant

    Extended to support:
    - Uncertainty analysis through cost multipliers in inputs
    - Onshore vs offshore installations

    Cost multipliers:
    - capex_turbine_factor, capex_HX_factor, etc.: Multiply base costs
    - opex_factor: Multiply OPEX rate

    Installation type (inputs['installation_type']):
    - 'offshore': Floating platform with mooring (default)
    - 'onshore': Land-based installation with intake/outfall pipelines

    Args:
        otec_plant_nom: Nominal plant design dictionary
        inputs: Parameters including cost factors and installation_type
        cost_level: 'low_cost' or 'high_cost'

    Returns:
        CAPEX_OPEX_dict: Dictionary with component costs
        CAPEX_total: Total CAPEX
        OPEX: Annual OPEX
        LCOE_nom: Levelized cost of energy
    """

    ## Unpack results from otec_steady_state

    p_gross = otec_plant_nom['p_gross_nom']
    p_pump_total = otec_plant_nom['p_pump_total_nom']
    p_net = otec_plant_nom['p_net_nom']

    A_evap = otec_plant_nom['A_evap']
    A_cond = otec_plant_nom['A_cond']

    m_pipes_WW = otec_plant_nom['m_pipes_WW']
    m_pipes_CW = otec_plant_nom['m_pipes_CW']

    dist_shore = inputs['dist_shore']

    # Get installation type (default to offshore for backward compatibility)
    installation_type = inputs.get('installation_type', 'offshore')

    # Get uncertainty multipliers (default = 1.0 if not provided)
    turbine_factor = inputs.get('capex_turbine_factor', 1.0)
    hx_factor = inputs.get('capex_HX_factor', 1.0)
    pump_factor = inputs.get('capex_pump_factor', 1.0)
    pipes_factor = inputs.get('capex_pipes_factor', 1.0)
    structure_factor = inputs.get('capex_structure_factor', 1.0)
    cable_factor = inputs.get('capex_cable_factor', 1.0)
    deploy_factor = inputs.get('capex_deploy_factor', 1.0)
    opex_factor = inputs.get('opex_factor', 1.0)

    if cost_level == 'low_cost':
        capex_turbine = 328*(136000/-p_gross)**0.16 * turbine_factor
        capex_HX = 226*(80000/-p_gross)**0.16 * hx_factor
        capex_pump = 1674*(5600/p_pump_total)**0.38 * pump_factor
        capex_pipes = 9 * pipes_factor  # to divide : material, fabrication, transport, installation cost
        capex_structure = 4465*(28100/-p_gross)**0.35 * structure_factor
        capex_deploy = 650 * deploy_factor
        capex_controls = 3113*(3960/-p_gross)**0.70
        capex_extra = 0.05
        opex = 0.03 * opex_factor
    elif cost_level == 'high_cost':
        capex_turbine = 512*(136000/-p_gross)**0.16 * turbine_factor
        capex_HX = 916*(4400/-p_gross)**0.093 * hx_factor
        capex_pump = 2480*(5600/p_pump_total)**0.38 * pump_factor
        capex_pipes = 30.1 * pipes_factor
        capex_structure = 7442*(28100/-p_gross)**0.35 * structure_factor
        capex_deploy = 667 * deploy_factor
        capex_controls = 6085*(4400/-p_gross)**0.70
        capex_extra = 0.2
        opex = 0.05 * opex_factor
    else:
        raise ValueError('Invalid cost level. Valid inputs are "low_cost" and "high_cost"')
    
    # ===========================================================================
    # INSTALLATION TYPE-SPECIFIC COST ADJUSTMENTS
    # ===========================================================================

    if installation_type == 'onshore':
        # ONSHORE INSTALLATION
        # - No mooring needed (plant is on land)
        # - Foundation/civil works instead of floating platform
        # - Additional intake/outfall pipeline costs (must reach optimal offshore location)
        # - Reduced deployment costs (no marine operations)
        # - Minimal transmission cable costs (connected directly to grid)

        # Structure costs become foundation/civil works (typically cheaper than offshore platform)
        capex_structure_onshore = capex_structure * 0.4  # 40% of offshore structure cost
        capex_deploy_onshore = capex_deploy * 0.3  # 30% of offshore deployment (simpler installation)

        # Additional pipeline costs to reach optimal offshore point
        # Pipes must extend from shore to the offshore location
        # Cost includes: intake pipeline, outfall pipeline, trenching, installation
        #
        # Reference costs for submarine HDPE pipelines:
        # - Large diameter (1.5-2m) HDPE pipe: ~$2-4M/km installed
        # - Trenching and burial: ~$1-2M/km
        # - Total per pipeline: ~$4-5M/km
        # - For intake + outfall (2 pipelines): ~$8-10M/km
        #
        # For a 100 MW plant, this is a FIXED cost, not scaled by capacity
        # Larger plants need larger pipes but cost doesn't scale linearly with power
        #
        # Cost model: Base cost + scaling factor for plant size
        # Base: $8M/km for both pipelines
        # Scaling: mild increase for larger plants (sqrt scaling)
        plant_size_MW = -p_gross / 1000  # Convert kW to MW
        size_scaling = np.sqrt(plant_size_MW / 100)  # Normalized to 100 MW plant
        size_scaling = np.clip(size_scaling, 0.7, 1.5)  # Bound the scaling factor

        # Pipeline cost per km (both intake and outfall) in $M/km
        pipeline_cost_per_km = 8.0 * size_scaling * pipes_factor  # $8M/km base for 100MW plant

        # Total pipeline cost in $ (not $/kW!)
        CAPEX_intake_outfall = pipeline_cost_per_km * dist_shore * 1e6  # Convert $M to $

        # Cable costs minimal for onshore (short connection to grid)
        # Fixed cost ~$50-100/kW for short grid connection
        capex_cable_onshore = 75 * cable_factor  # $/kW

        # Additional pipe mass for intake/outfall pipelines
        # Large diameter HDPE: ~200-400 kg/m per pipeline (varies with diameter)
        # For 100 MW plant: ~300 kg/m per pipeline
        pipe_mass_per_m = 300 * size_scaling  # kg/m per pipeline
        additional_pipe_mass = dist_shore * 1000 * pipe_mass_per_m * 2  # 2 pipelines, dist in km

        # Additional pumping head for longer pipelines
        # Each km adds ~1-2m of friction head loss
        pump_head_factor = 1 + 0.01 * dist_shore  # 1% increase per km

        CAPEX_turbine = capex_turbine*-p_gross
        CAPEX_evap = capex_HX*A_evap
        CAPEX_cond = capex_HX*A_cond
        CAPEX_pump = capex_pump*p_pump_total * pump_head_factor  # Scale with distance
        CAPEX_pipes = capex_pipes*(m_pipes_WW+m_pipes_CW+additional_pipe_mass)
        CAPEX_cable = capex_cable_onshore*-p_gross
        CAPEX_structure = capex_structure_onshore*(-p_gross)
        CAPEX_deploy = capex_deploy_onshore*(-p_gross)
        CAPEX_man = capex_controls*(-p_gross)

        # For onshore, structure breakdown is different
        CAPEX_mooring = np.zeros(np.shape(dist_shore))  # No mooring needed
        CAPEX_platform = CAPEX_structure  # All structure cost is foundation/civil works

    else:
        # OFFSHORE INSTALLATION (default)
        # - Floating platform with mooring system
        # - Marine deployment operations
        # - Submarine transmission cables
        # - Plant located at optimal offshore point

        capex_cable = np.empty(np.shape(dist_shore),dtype=np.float64)
        # AC cables for distances below or equal to 50 km, source: Bosch et al. (2019), costs converted from US$(2016) to US$(2021) with conversion factor 1.10411)
        capex_cable[dist_shore <= 50] = (8.5*dist_shore[dist_shore <= 50]+56.8)*1.10411 * cable_factor
        # DC cables for distances beyond 50 km, source: Bosch et al. (2019), costs converted from US$(2016) to US$(2021) with conversion factor 1.10411)
        capex_cable[dist_shore > 50] = (2.2*dist_shore[dist_shore > 50]+387.8)*1.10411 * cable_factor

        CAPEX_turbine = capex_turbine*-p_gross
        CAPEX_evap = capex_HX*A_evap
        CAPEX_cond = capex_HX*A_cond
        CAPEX_pump = capex_pump*p_pump_total
        CAPEX_pipes = capex_pipes*(m_pipes_WW+m_pipes_CW)
        CAPEX_cable = capex_cable*-p_gross
        CAPEX_structure = capex_structure*(-p_gross)
        CAPEX_deploy = capex_deploy*(-p_gross)
        CAPEX_man = capex_controls*(-p_gross)
        CAPEX_intake_outfall = np.zeros(np.shape(dist_shore))  # No additional intake/outfall needed

        CAPEX_mooring = CAPEX_structure / 4
        CAPEX_platform = 3* CAPEX_structure / 4

    # ===========================================================================
    # COMMON CALCULATIONS (both onshore and offshore)
    # ===========================================================================

    CAPEX_wo_extra = CAPEX_turbine + CAPEX_evap + CAPEX_cond + CAPEX_pump + CAPEX_pipes + CAPEX_cable + CAPEX_structure + CAPEX_deploy + CAPEX_man + CAPEX_intake_outfall
    CAPEX_extra = CAPEX_wo_extra*capex_extra

    CAPEX_total = CAPEX_wo_extra+CAPEX_extra

    OPEX = CAPEX_total*opex

    LCOE_nom = (CAPEX_total*inputs['crf']+OPEX)*100/(-p_net*inputs['availability_factor']*8760) # LCOE in ct/kWh 
    
    CAPEX_OPEX_dict = {
    'turbine_CAPEX': CAPEX_turbine[0],
    'evap_CAPEX': CAPEX_evap[0],
    'cond_CAPEX': CAPEX_cond[0],
    'pump_CAPEX': CAPEX_pump,
    'pipes_CAPEX': CAPEX_pipes,
    # 'structure_CAPEX': CAPEX_structure[0],
    'mooring_CAPEX': CAPEX_mooring[0],
    'platform_CAPEX': CAPEX_platform[0],
    'deploy_CAPEX': CAPEX_deploy[0],
    'man_CAPEX': CAPEX_man[0],
    'cable_CAPEX': CAPEX_cable[0],
    'intake_outfall_CAPEX': CAPEX_intake_outfall[0],  # New: for onshore installations
    'extra_CAPEX': CAPEX_extra[0],
    'OPEX': OPEX[0],
    'LCOE':LCOE_nom[0],
    'installation_type': installation_type  # Record which type was used
}
    # np.where(LCOE_nom < 0)
    # CAPEX_total[0,1061]
    # OPEX[0,1061]
    # p_net[0,1061]
    # LCOE_nom[0,1061]
    # a = T_WW_profiles[:,1061]
    
    if np.any(LCOE_nom <= 0):
        raise ValueError('Invalid LCOE found, please check inputs.')
    else:
        pass
    
    # print(LCOE_nom)
    # print(len(LCOE_nom[0]))
    return CAPEX_OPEX_dict,CAPEX_total,OPEX,LCOE_nom

def lcoe_time_series(otec_plant_nom,inputs,p_net_ts):
    
    p_net_mean = np.nanmean(p_net_ts,axis=0)
    e_mean_annual = -p_net_mean*8760
    
    lcoe_ts = (otec_plant_nom['CAPEX']*inputs['crf']+otec_plant_nom['OPEX'])*100/(e_mean_annual*inputs['availability_factor'])
    
    return lcoe_ts