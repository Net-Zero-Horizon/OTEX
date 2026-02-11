628839# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:04:43 2023

@author: jkalanger
"""

import numpy as np
import pandas as pd
from .utils import saturation_pressures_and_temperatures, enthalpies_entropies, pressure_drop
from ..economics.costs import lcoe_time_series
from .components import initial_NTU_and_epsilon, pressure_regulation, evaporator_regulation, condenser_regulation, seawater_pipes_operation

def otec_operation(otec_plant_nom,T_WW_profiles,T_CW_profiles,inputs):
                
    # inputs = parameters_and_constants()
    
    # to avoid confusion with the variables used for plant sizing, we use the suffix ts for time series
    
    T_evap_ts,p_evap_ts = pressure_regulation(T_WW_profiles,otec_plant_nom,inputs,'evap')
    T_cond_ts,p_cond_ts = pressure_regulation(T_CW_profiles,otec_plant_nom,inputs,'cond')
    
    NTU_evap_initial,epsilon_evap_initial = initial_NTU_and_epsilon(inputs,otec_plant_nom,'evap')
    NTU_cond_initial,epsilon_cond_initial = initial_NTU_and_epsilon(inputs,otec_plant_nom,'cond')
       
    enthalpies_ts = enthalpies_entropies(p_evap_ts,p_cond_ts,inputs)
           
    m_NH3_ts,m_WW_ts,T_WW_out_ts,Q_evap_ts = evaporator_regulation(enthalpies_ts,T_WW_profiles,T_evap_ts,epsilon_evap_initial,inputs,otec_plant_nom)
    t_CW_out_ts,m_CW_ts,Q_cond_ts = condenser_regulation(T_CW_profiles,inputs,otec_plant_nom,m_NH3_ts,T_cond_ts,enthalpies_ts,epsilon_cond_initial)
           
    p_pump_NH3_ts = m_NH3_ts*(enthalpies_ts['h_2']-enthalpies_ts['h_1'])/inputs['eff_pump_NH3_mech']      
    p_gross_ts = m_NH3_ts*(enthalpies_ts['h_4']-enthalpies_ts['h_3'])
    
    p_drop_WW_ts,p_pump_WW_ts = seawater_pipes_operation(otec_plant_nom,
                                                         inputs,
                                                         m_WW_ts,
                                                         T_WW_profiles,
                                                         inputs['rho_WW'],
                                                         inputs['length_WW'],
                                                         otec_plant_nom['A_pipes_WW'],
                                                         otec_plant_nom['d_pipes_WW'],
                                                         inputs['roughness_pipe'],
                                                         inputs['K_L'],
                                                         inputs['u_HX'])
    
    p_drop_CW_ts,p_pump_CW_ts = seawater_pipes_operation(otec_plant_nom,
                                                          inputs,
                                                          m_CW_ts,
                                                          T_CW_profiles,
                                                          inputs['rho_CW'],
                                                          inputs['length_CW'],
                                                          otec_plant_nom['A_pipes_CW'],
                                                          otec_plant_nom['d_pipes_CW'],
                                                          inputs['roughness_pipe'],
                                                          inputs['K_L'],
                                                          inputs['u_HX'])
    
    p_pump_total_ts = p_pump_NH3_ts + p_pump_WW_ts + p_pump_CW_ts
    
    p_net_ts = (p_gross_ts*inputs['eff_turb_el']*inputs['eff_turb_mech'] + p_pump_total_ts)*inputs['eff_trans']
    
    if np.any(p_net_ts > 0):
        # print('Infeasible systems detected and replaced by NaN')
        p_net_ts = np.where(p_net_ts > 0, np.nan, p_net_ts)
    
    eff_net_ts = -p_net_ts/Q_evap_ts
    
    LCOE_ts = lcoe_time_series(otec_plant_nom,inputs,p_net_ts)
    
    otec_plant_ts = {
        'm_NH3': m_NH3_ts,
        
        'T_evap' : T_evap_ts,
        'T_WW_out' : T_WW_out_ts,
        'm_WW' : m_WW_ts,
        'Q_evap' : Q_evap_ts,


        'T_cond' : T_cond_ts,
        'T_CW_out' : t_CW_out_ts,
        'm_CW' : m_CW_ts,
        'Q_cond' : Q_cond_ts,

        'p_gross' : p_gross_ts,
        'p_net': p_net_ts,
        'p_pump_total': p_pump_total_ts,
        'p_pump_NH3': p_pump_NH3_ts,
        'p_pump_WW' : p_pump_WW_ts,
        'p_pump_CW' : p_pump_CW_ts,
        'eff_net' : eff_net_ts,
        
        'LCOE' : LCOE_ts
        }

    return otec_plant_ts
        
    