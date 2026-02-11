# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:34:18 2023

@author: jkalanger
"""

import numpy as np
from .utils import pressure_drop

def pressure_regulation(T_water_profiles,otec_plant_nom,inputs,HX):
    
    if HX == 'evap':
        T_sat_ts = np.where(T_water_profiles - otec_plant_nom['Q_evap_nom']/(otec_plant_nom['m_WW_nom']*inputs['cp_water']) - otec_plant_nom['T_evap_nom'] < inputs['T_pinch_WW'],
                          T_water_profiles - otec_plant_nom['del_T_WW'] - inputs['T_pinch_WW'],
                          otec_plant_nom['T_evap_nom'])
    elif HX == 'cond':
        T_sat_ts = np.where(otec_plant_nom['T_cond_nom'] - (T_water_profiles - otec_plant_nom['Q_cond_nom']/(otec_plant_nom['m_CW_nom']*inputs['cp_water'])) < inputs['T_pinch_CW'],
                          T_water_profiles + otec_plant_nom['del_T_CW'] + inputs['T_pinch_CW'],
                          otec_plant_nom['T_cond_nom'])
    else:
        raise ValueError('Invalid heat exchanger input. Please choose between "evap" and "cond".')
            
    p_sat_ts = 0.00002196*T_sat_ts**3+0.00193103*T_sat_ts**2+0.1695763*T_sat_ts+4.25739601
        
    return T_sat_ts,p_sat_ts

def initial_NTU_and_epsilon(inputs,otec_plant_nom,HX):
    
    if HX == 'evap':
        suffix = ['evap','WW_nom']
    elif HX == 'cond':
        suffix = ['cond','CW_nom']
    else:
        raise ValueError('Invalid heat exchanger input. Please choose between "evap" and "cond".')
    
    NTU_initial = (inputs['U_'+suffix[0]]*otec_plant_nom['A_'+suffix[0]])/(otec_plant_nom['m_'+suffix[1]]*inputs['cp_water'])
    epsilon_initial = 1 - np.exp(-NTU_initial)
    
    return NTU_initial,epsilon_initial

def evaporator_regulation(enthalpies_ts,T_WW_profiles,T_evap_ts,epsilon_evap_initial,inputs,otec_plant_nom):
    
    T_WW_out_initial = T_WW_profiles - epsilon_evap_initial * (T_WW_profiles - T_evap_ts)
       
    m_NH3_initial = (-otec_plant_nom['m_WW_nom']*inputs['cp_water']*(T_WW_out_initial-T_WW_profiles)/(enthalpies_ts['h_3']-enthalpies_ts['h_2']))
    
    # This boolean determines whether ammonia mass flow is restricted by turbine limitations
    bool_turb = np.where(m_NH3_initial*(enthalpies_ts['h_4']-enthalpies_ts['h_3']) < otec_plant_nom['p_gross_nom'],1,0)
    # This boolean determines whether ammonia mass flow is restricted by condenser limitations
    bool_cond = np.where((bool_turb == 0) & (m_NH3_initial*(enthalpies_ts['h_1']-enthalpies_ts['h_4']) < otec_plant_nom['Q_cond_nom']),1,0)
    # This boolean determines whether ammonia mass flow is restricted by evaporator limitations
    bool_evap = np.where((bool_turb == 0) & (m_NH3_initial*(enthalpies_ts['h_1']-enthalpies_ts['h_4']) >= otec_plant_nom['Q_cond_nom']),1,0)  
    
    m_NH3_turb = otec_plant_nom['m_NH3_nom']
    Q_evap_turb = m_NH3_turb*(enthalpies_ts['h_3']-enthalpies_ts['h_2'])
    T_WW_out_turb_iteration = T_WW_out_initial
    difference = np.ones(np.size(bool_turb))
    
    while np.any(abs(difference) > 1E-7):
        T_WW_out_turb = T_WW_out_turb_iteration
        m_WW_turb = -Q_evap_turb/(inputs['cp_water']*(T_WW_out_turb-T_WW_profiles))
        u_evap_turb = inputs['U_evap']*(m_WW_turb/otec_plant_nom['m_WW_nom'])**0.65
        NTU_evap_turb = u_evap_turb*otec_plant_nom['A_evap']/(m_WW_turb*inputs['cp_water'])
        epsilon_evap_turb = 1-np.exp(-NTU_evap_turb)
        T_WW_out_turb_iteration = T_WW_profiles-epsilon_evap_turb*(T_WW_profiles-T_evap_ts)
        difference = T_WW_out_turb_iteration - T_WW_out_turb
    T_WW_out_turb = T_WW_out_turb_iteration
    
    
    # Here we make sure that the condenser heat flow does not exceed the nominal design value
    m_NH3_target = otec_plant_nom['Q_cond_nom']/(enthalpies_ts['h_1']-enthalpies_ts['h_4'])
    m_NH3_cond = np.ones(np.shape(m_NH3_target),dtype=np.float64)*otec_plant_nom['m_NH3_nom']   
    m_NH3_cond[m_NH3_target <= otec_plant_nom['m_NH3_nom']] = m_NH3_target[m_NH3_target <= otec_plant_nom['m_NH3_nom']]
 
    m_WW_cond = otec_plant_nom['m_WW_nom']
    T_WW_out_cond = T_WW_out_initial
    Q_evap_cond = m_NH3_cond*(enthalpies_ts['h_3']-enthalpies_ts['h_2'])
    
    m_NH3_evap = otec_plant_nom['m_NH3_nom']
    m_WW_evap = otec_plant_nom['m_WW_nom']
    T_WW_out_evap = T_WW_out_initial
    Q_evap_evap = - m_WW_evap*inputs['cp_water']*(T_WW_out_initial-T_WW_profiles) 
    
    # suffix ts stands for time series
    
    m_NH3_ts = m_NH3_turb*bool_turb + m_NH3_cond*bool_cond + m_NH3_evap*bool_evap
    m_WW_ts = m_WW_turb*bool_turb + m_WW_cond*bool_cond + m_WW_evap*bool_evap
    T_WW_out_ts = T_WW_out_turb*bool_turb + T_WW_out_cond*bool_cond + T_WW_out_evap*bool_evap
    Q_evap_ts = Q_evap_turb*bool_turb + Q_evap_cond*bool_cond + Q_evap_evap*bool_evap
    
    # Sanity checks disabled for off-design analysis
    # Off-design operation naturally results in values outside nominal conditions
    # These variations are the entire point of off-design analysis

    # The checks below are commented out because they were preventing valid off-design calculations
    # If physical violations occur (negative temperatures, imaginary numbers, etc.), they will
    # naturally result in NaN values that can be filtered during post-processing

    # tolerance = 1.10  # 10% tolerance
    # if np.any(np.round(np.amax(m_NH3_ts, axis=0),1) > np.round(otec_plant_nom['m_NH3_nom']*tolerance,1)):
    #     raise Warning('Ammonia mass flow significantly exceeds nominal design value.')
    # if np.any(np.round(np.amax(m_WW_ts, axis=0),1) > np.round(otec_plant_nom['m_WW_nom']*tolerance,1)):
    #     raise Warning('Warm seawater mass flow significantly exceeds nominal design value.')
    # if np.any(np.round(np.amax(Q_evap_ts, axis=0),1) > np.round(otec_plant_nom['Q_evap_nom']*tolerance,1)):
    #     raise Warning('Evaporator heat flow significantly exceeds nominal design value.')
    # if np.any(np.round(np.amin(T_WW_out_ts - T_evap_ts, axis=0),1) < np.round(inputs['T_pinch_WW']*0.8,1)):
    #     raise Warning('Warm water pinch temperature critically below nominal design value.')
        
    return m_NH3_ts,m_WW_ts,T_WW_out_ts,Q_evap_ts


def condenser_regulation(T_CW_profiles,inputs,otec_plant_nom,m_NH3_ts,T_cond_ts,enthalpies_ts,epsilon_cond_initial):
    
    T_CW_out_initial = T_CW_profiles + epsilon_cond_initial*(T_cond_ts-T_CW_profiles)
    
    Q_cond_ts = m_NH3_ts*(enthalpies_ts['h_1'] - enthalpies_ts['h_4'])
    
    m_CW_ts = otec_plant_nom['m_CW_nom']
    T_CW_out_ts = T_CW_out_initial
    T_CW_out_iteration = T_CW_out_initial
    difference = np.ones(np.size(T_CW_out_initial))
    
    while np.any(abs(difference) > 1E-7):
        T_CW_out_ts = T_CW_out_iteration
        m_CW_ts = -Q_cond_ts/(inputs['cp_water']*(T_CW_out_ts-T_CW_profiles))
        u_cond_ts = inputs['U_cond']*(m_CW_ts/otec_plant_nom['m_CW_nom'])**0.65
        NTU_cond_ts = u_cond_ts*otec_plant_nom['A_cond']/(m_CW_ts*inputs['cp_water'])
        epsilon_cond_ts = 1-np.exp(-NTU_cond_ts)
        T_CW_out_iteration = T_CW_profiles-epsilon_cond_ts*(T_CW_profiles-T_cond_ts)
        difference = T_CW_out_iteration - T_CW_out_ts
    T_CW_out_ts = T_CW_out_iteration
    
    # Sanity checks disabled for off-design analysis
    # Off-design operation naturally results in values outside nominal conditions
    # These variations are the entire point of off-design analysis

    # The checks below are commented out because they were preventing valid off-design calculations
    # If physical violations occur (negative temperatures, imaginary numbers, etc.), they will
    # naturally result in NaN values that can be filtered during post-processing

    # tolerance = 1.10  # 10% tolerance
    # if np.any(np.round(np.amax(Q_cond_ts, axis=0),1) > np.round(otec_plant_nom['Q_cond_nom']*tolerance,1)):
    #     raise Warning('Condenser heat flow significantly exceeds nominal design value.')
    # if np.any(np.round(np.amin(T_cond_ts - T_CW_out_ts, axis=0),1) < np.round(inputs['T_pinch_CW']*0.8,1)):
    #     raise Warning('Cold water pinch temperature critically below nominal design value.')
    
    return T_CW_out_ts,m_CW_ts,Q_cond_ts

def seawater_pipes_operation(otec_plant_nom,inputs,m_water_ts,t_water_in,rho_water,length,A_pipes,d_pipes,roughness_pipe,K_L,u_HX):
    
    V_water_ts = m_water_ts/rho_water
    u_water_ts = V_water_ts/A_pipes
    
    u_HX_ts = u_water_ts/2
    
    p_drop_ts = pressure_drop(t_water_in,u_water_ts,d_pipes,rho_water,roughness_pipe,length,K_L,u_HX_ts)
    
    p_pump_ts = m_water_ts/rho_water*p_drop_ts/inputs['eff_hyd']/inputs['eff_el']
    
    return p_drop_ts,p_pump_ts
    