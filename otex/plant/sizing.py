# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:16:02 2021

@author: jkalanger
"""

import numpy as np
from .utils import saturation_pressures_and_temperatures, enthalpies_entropies, pressure_drop
from ..config import parameters_and_constants


def ammonia_pump_sizing(p_evap,p_cond,enthalpies,inputs):
    m_NH3 = inputs['p_gross']/(enthalpies['h_4']-enthalpies['h_3']) # mass flow           
    p_pump_NH3 = m_NH3*(enthalpies['h_2']-enthalpies['h_1']) # power consumption
    
    return m_NH3,p_pump_NH3

def evaporator_sizing(T_WW_in,del_T_WW,T_evap,m_NH3,enthalpies,inputs):
    
    del_T_log_evap = ((T_WW_in-T_evap) - ((T_WW_in-del_T_WW)-T_evap))/np.log((T_WW_in-T_evap)/((T_WW_in-del_T_WW)-T_evap))
    
    Q_evap = m_NH3*(enthalpies['h_3']-enthalpies['h_2'])
    m_WW = Q_evap/(inputs['cp_water']*del_T_WW)
    A_evap = Q_evap/(inputs['U_evap']*del_T_log_evap)
    
    evaporator = {
        'm_WW': m_WW,
        'A_evap': A_evap,
        'Q_evap': Q_evap
        }
   
    return evaporator

def condenser_sizing(T_CW_in,del_T_CW,T_cond,m_NH3,enthalpies,inputs):
  
    del_T_log_cond = ((T_cond-T_CW_in) - (T_cond-(T_CW_in+del_T_CW)))/np.log((T_cond-T_CW_in)/(T_cond-(T_CW_in+del_T_CW)))  
    
    Q_cond = m_NH3*(enthalpies['h_1']-enthalpies['h_4'])
    m_CW = -Q_cond/(inputs['cp_water']*del_T_CW)
    A_cond = -Q_cond/(inputs['U_cond']*del_T_log_cond)
    
    condenser = {
        'm_CW': m_CW,
        'A_cond': A_cond,
        'Q_cond': Q_cond
        }
    
    return condenser

def seawater_pipe_sizing(T_in,m_water,rho,length,inputs):
    
    ## Load and unpack inputs to improve readibility of code below   
    
    rho_pipe,roughness_pipe = inputs['pipe_material']
    length_WW, \
    length_CW, \
    SDR_ratio, \
    K_L, \
    u_pipes, \
    u_HX, \
    pressure_drop_nom, \
    max_d, \
    max_p = inputs['pipe_properties']
    
    u_pipes = np.ones(np.size(T_in),dtype=np.float64)*u_pipes
    p_drop = np.ones(np.size(T_in),dtype=np.float64)*pressure_drop_nom

    # Add safety measures to prevent infinite loops
    max_iterations_outer = 200  # Maximum iterations for pressure drop loop
    min_velocity = 0.1  # Minimum velocity in m/s to prevent infeasible designs
    iteration_count_outer = 0

    while np.any(p_drop >= max_p) and iteration_count_outer < max_iterations_outer:
        iteration_count_outer += 1

        u_pipes[p_drop >= max_p] = u_pipes[p_drop >= max_p] - 0.1

        # Check if velocity becomes too low (infeasible design)
        if np.any(u_pipes <= min_velocity):
            # Mark these designs as infeasible by setting results to NaN
            print(f"    WARNING: Pipe velocity too low (<{min_velocity} m/s). Setting infeasible designs to NaN.")
            u_pipes[u_pipes <= min_velocity] = np.nan
            # Ensure variables are calculated before break
            num_pipes = pipe_pairs * 2
            m_pipes = np.pi/4*((d_pipes+2*thickness)**2-d_pipes**2)*length*rho_pipe*pipe_pairs
            break

        u_HX = u_pipes/2

        A_pipes = m_water/rho/u_pipes

        # To avoid making the same calculation twice for inlet and outlet pipe, we assume that they form one single pipe
        # with the same properties (diameter, thickness, etc.) and loading (water flow, pressure drop, etc.)
        # Therefore, we don't count the individual inlet and outlet pipes, but the pipe pairs. Once we have the required pairs,
        # we multiply the pairs with 2 to obtain the number of actual pipes needed.

        pipe_pairs = np.zeros(np.size(T_in),dtype=np.float64)
        num_pipes = np.zeros(np.size(T_in),dtype=np.float64)  # Initialize num_pipes
        m_pipes = np.zeros(np.size(T_in),dtype=np.float64)  # Initialize m_pipes

        d_pipes = np.ones(np.size(T_in),dtype=np.float64)*(max_d+0.1)
        thickness = d_pipes/SDR_ratio

        # Add safety measure for inner loop
        max_iterations_inner = 1000  # Maximum iterations for diameter loop
        iteration_count_inner = 0

        while np.any(d_pipes > max_d) and iteration_count_inner < max_iterations_inner:
            iteration_count_inner += 1
            pipe_pairs[d_pipes > max_d] = pipe_pairs[d_pipes > max_d] + 1
            d_pipes = np.sqrt(A_pipes*4/(np.pi*pipe_pairs))
            thickness = d_pipes/SDR_ratio # adjust the thickness of the pipe based on its diameter

        if iteration_count_inner >= max_iterations_inner:
            print(f"    WARNING: Inner loop max iterations reached. Setting infeasible designs to NaN.")
            d_pipes[d_pipes > max_d] = np.nan
            # Ensure variables are calculated before break
            num_pipes = pipe_pairs * 2
            m_pipes = np.pi/4*((d_pipes+2*thickness)**2-d_pipes**2)*length*rho_pipe*pipe_pairs
            break

        m_pipes = np.pi/4*((d_pipes+2*thickness)**2-d_pipes**2)*length*rho_pipe*pipe_pairs

        num_pipes = pipe_pairs*2

        # print(d_pipes,thickness)

        p_drop = pressure_drop(T_in,u_pipes,d_pipes,rho,roughness_pipe,length,K_L,u_HX)

    if iteration_count_outer >= max_iterations_outer:
        print(f"    WARNING: Outer loop max iterations reached. Setting infeasible designs to NaN.")
        p_drop[p_drop >= max_p] = np.nan
       
    p_pump = m_water/rho*p_drop/inputs['eff_hyd']/inputs['eff_el']
    
    pipes = {
        'd_pipes': d_pipes,
        'num_pipes': num_pipes,
        'm_pipes': m_pipes,
        'A_pipes': A_pipes,
        'p_pump': p_pump
        }
    
    return pipes


def otec_sizing(T_WW_in,T_CW_in,del_T_WW,del_T_CW,inputs,cost_level):
        
    # inputs = parameters_and_constants(cost_level)
    
    T_evap,T_cond,p_evap,p_cond = saturation_pressures_and_temperatures(T_WW_in,T_CW_in,del_T_WW,del_T_CW,inputs)
            
    enthalpies = enthalpies_entropies(p_evap,p_cond,inputs)
    
    m_NH3,p_pump_NH3 = ammonia_pump_sizing(p_evap,p_cond,enthalpies,inputs)
    
    evaporator = evaporator_sizing(T_WW_in,del_T_WW,T_evap,m_NH3,enthalpies,inputs)
    
    condenser = condenser_sizing(T_CW_in,del_T_CW,T_cond,m_NH3,enthalpies,inputs)
    
    pipes_WW = seawater_pipe_sizing(T_WW_in,
                                    evaporator['m_WW'],
                                    inputs['rho_WW'],
                                    inputs['length_WW'],
                                    inputs)
    pipes_CW = seawater_pipe_sizing(T_CW_in,
                                    condenser['m_CW'],
                                    inputs['rho_CW'],
                                    inputs['length_CW'],
                                    inputs)
    
    p_pump_total = p_pump_NH3/inputs['eff_pump_NH3_mech'] + pipes_WW['p_pump'] + pipes_CW['p_pump']
    
    p_net = (inputs['p_gross']*inputs['eff_turb_el']*inputs['eff_turb_mech'] + p_pump_total)*inputs['eff_trans']  
    eff_net = -p_net/evaporator['Q_evap']
    
    if np.any(p_net > 0):
        # print('Infeasible systems detected and replaced by NaN')
        p_net = np.where(p_net > 0, np.nan, p_net)
   
    
    # Pack results
    
    otec_plant_nominal = {
                'm_NH3_nom': m_NH3,
                
                'del_T_WW' : del_T_WW,
                'T_evap_nom' : T_evap,
                'p_evap_nom' : p_evap,
                'm_WW_nom' : evaporator['m_WW'],
                'A_evap' : evaporator['A_evap'],
                'Q_evap_nom' : evaporator['Q_evap'],
                
                'p_gross_nom': np.ones(np.shape(p_net),dtype=np.float64)*inputs['p_gross'],
                'p_net_nom': p_net,
                'p_pump_total_nom': p_pump_total,
                'p_pump_NH3_nom': p_pump_NH3,
                'p_pump_WW_nom' : pipes_WW['p_pump'],
                'p_pump_CW_nom' : pipes_CW['p_pump'],
                'eff_net_nom' : eff_net,
                
                'del_T_CW' : del_T_CW,
                'T_cond_nom' : T_cond,
                'p_cond_nom' : p_cond,
                'm_CW_nom' : condenser['m_CW'],
                'A_cond' : condenser['A_cond'],
                'Q_cond_nom' : condenser['Q_cond'],

                'd_pipes_WW': pipes_WW['d_pipes'],
                'num_pipes_WW': pipes_WW['num_pipes'],
                'm_pipes_WW': pipes_WW['m_pipes'],
                'A_pipes_WW': pipes_WW['A_pipes'],
                'd_pipes_CW': pipes_CW['d_pipes'],
                'num_pipes_CW': pipes_CW['num_pipes'],
                'm_pipes_CW': pipes_CW['m_pipes'],  
                'A_pipes_CW': pipes_CW['A_pipes'],
                
                'h_1_nom': enthalpies['h_1'],
                'h_2_nom': enthalpies['h_2'],
                'h_3_nom': enthalpies['h_3'],
                'h_4_nom': enthalpies['h_4']
                }
     
    return otec_plant_nominal