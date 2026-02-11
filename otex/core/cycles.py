# -*- coding: utf-8 -*-
"""
OTEX Thermodynamic Cycles Module
Implements different OTEC power cycles

Supported cycles:
- Rankine Closed Cycle (original)
- Rankine Open Cycle (Flash)
- Rankine Hybrid Cycle (Open-Closed)
- Kalina Cycle (NH3-H2O mixture)
- Uehara Cycle (Two-stage with NH3-H2O)

@author: OTEX Development Team
"""

import numpy as np
from abc import ABC, abstractmethod
from .fluids import WorkingFluid


class ThermodynamicCycle(ABC):
    """
    Abstract base class for thermodynamic power cycles
    """

    def __init__(self, name, working_fluid):
        """
        Args:
            name: Name of the cycle
            working_fluid: WorkingFluid instance
        """
        self.name = name
        self.fluid = working_fluid

    @abstractmethod
    def calculate_cycle_states(self, T_evap, T_cond, p_evap, p_cond, inputs):
        """
        Calculate thermodynamic states at all points in the cycle

        Args:
            T_evap: Evaporator temperature [°C]
            T_cond: Condenser temperature [°C]
            p_evap: Evaporator pressure [bar]
            p_cond: Condenser pressure [bar]
            inputs: Dictionary with efficiencies and other parameters

        Returns:
            states: Dictionary with thermodynamic states
                   e.g., {'h_1': ..., 'h_2': ..., 's_1': ..., etc.}
        """
        pass

    @abstractmethod
    def calculate_mass_flow(self, p_gross, states):
        """
        Calculate required mass flow rate for target gross power

        Args:
            p_gross: Target gross power output [kW] (negative value)
            states: Dictionary with thermodynamic states

        Returns:
            m_fluid: Mass flow rate [kg/s]
        """
        pass

    @abstractmethod
    def calculate_pump_power(self, m_fluid, states, inputs):
        """
        Calculate pump power consumption

        Args:
            m_fluid: Mass flow rate [kg/s]
            states: Dictionary with thermodynamic states
            inputs: Dictionary with efficiencies

        Returns:
            p_pump: Pump power [kW] (negative value)
        """
        pass

    def calculate_heat_transfer(self, m_fluid, states):
        """
        Calculate heat transfer in evaporator and condenser

        Args:
            m_fluid: Mass flow rate [kg/s]
            states: Dictionary with thermodynamic states

        Returns:
            Q_evap: Evaporator heat transfer [kW]
            Q_cond: Condenser heat transfer [kW]
        """
        Q_evap = m_fluid * (states['h_3'] - states['h_2'])
        Q_cond = m_fluid * (states['h_1'] - states['h_4'])
        return Q_evap, Q_cond


class RankineClosedCycle(ThermodynamicCycle):
    """
    Closed Rankine Cycle (original OTEX implementation)

    Cycle diagram:
    1 → 2: Pump (liquid compression)
    2 → 3: Evaporator (heating to saturated vapor)
    3 → 4: Turbine (expansion to two-phase)
    4 → 1: Condenser (cooling to saturated liquid)
    """

    def __init__(self, working_fluid):
        super().__init__('Rankine Closed', working_fluid)

    def calculate_cycle_states(self, T_evap, T_cond, p_evap, p_cond, inputs):
        """
        Calculate Rankine cycle states

        Returns states dictionary with:
        - h_1, s_1: Condenser outlet (saturated liquid)
        - h_2, s_2: Pump outlet (compressed liquid)
        - h_3, s_3: Evaporator outlet (saturated vapor)
        - h_4, s_4: Turbine outlet (two-phase mixture)
        - x_4: Vapor quality at turbine outlet
        """

        eff_isen_turb = inputs['eff_isen_turb']
        eff_isen_pump = inputs['eff_isen_pump']

        # Use scalar density value - liquid density is nearly constant with pressure
        if 'rho_NH3' in inputs:
            rho_fluid = inputs['rho_NH3']
        else:
            # Use a scalar pressure for density calculation (take mean if array)
            p_cond_scalar = np.mean(p_cond) if isinstance(p_cond, np.ndarray) else p_cond
            rho_fluid = self.fluid.density_liquid(p=p_cond_scalar)

        # State 1: Condenser outlet (saturated liquid at p_cond)
        h_1 = self.fluid.enthalpy_liquid(p=p_cond)
        s_1 = self.fluid.entropy_liquid(p=p_cond)

        # State 2: Pump outlet (compressed liquid at p_evap)
        # Pump work: W_pump = v * (p2 - p1) / eff
        # For liquid, v ≈ 1/rho
        h_2 = 1/rho_fluid * (p_evap - p_cond) * 100000/1000 / eff_isen_pump + h_1
        # Entropy approximately constant for liquid compression
        s_2 = s_1

        # State 3: Evaporator outlet (saturated vapor at p_evap)
        h_3 = self.fluid.enthalpy_vapor(p=p_evap)
        s_3 = self.fluid.entropy_vapor(p=p_evap)

        # State 4: Turbine outlet (two-phase at p_cond)
        # First calculate isentropic expansion (s_4s = s_3)
        h_4_liq = self.fluid.enthalpy_liquid(p=p_cond)
        h_4_vap = self.fluid.enthalpy_vapor(p=p_cond)
        s_4_liq = self.fluid.entropy_liquid(p=p_cond)
        s_4_vap = self.fluid.entropy_vapor(p=p_cond)

        # Quality at isentropic state
        x_4_isen = self.fluid.quality_from_entropy(s_3, s_4_liq, s_4_vap)
        h_4_isen = self.fluid.enthalpy_two_phase(h_4_liq, h_4_vap, x_4_isen)

        # Actual state accounting for turbine efficiency
        h_4 = (h_4_isen - h_3) * eff_isen_turb + h_3

        # Actual quality
        x_4 = self.fluid.quality_from_enthalpy(h_4, h_4_liq, h_4_vap)
        s_4 = self.fluid.entropy_two_phase(s_4_liq, s_4_vap, x_4)

        states = {
            'h_1': h_1,
            's_1': s_1,
            'h_2': h_2,
            's_2': s_2,
            'h_3': h_3,
            's_3': s_3,
            'h_4': h_4,
            's_4': s_4,
            'x_4': x_4,
            'h_4_isen': h_4_isen,
            'x_4_isen': x_4_isen,
        }

        return states

    def calculate_mass_flow(self, p_gross, states):
        """
        Calculate mass flow from gross power output
        p_gross is negative (convention: power out is negative)

        W_gross = m * (h_4 - h_3)
        """
        m_fluid = p_gross / (states['h_4'] - states['h_3'])
        return m_fluid

    def calculate_pump_power(self, m_fluid, states, inputs):
        """
        Calculate pump power consumption

        W_pump = m * (h_2 - h_1) / eff_mech
        """
        eff_mech = inputs['eff_pump_NH3_mech']
        p_pump = m_fluid * (states['h_2'] - states['h_1']) / eff_mech
        return p_pump


class RankineOpenCycle(ThermodynamicCycle):
    """
    Open Rankine Cycle (Flash Steam)
    Uses seawater directly as working fluid

    This is a simplified implementation suitable for OTEC analysis.
    Assumes:
    - Flash evaporation of warm seawater under vacuum
    - Steam expansion through turbine
    - Surface condensation with cold seawater
    - Simplified water/steam properties
    """

    def __init__(self):
        # Open cycle doesn't need a separate working fluid
        # It uses seawater directly
        super().__init__('Rankine Open (Flash)', None)

    def calculate_cycle_states(self, T_evap, T_cond, p_evap, p_cond, inputs):
        """
        Flash steam cycle calculation

        States (simplified):
        1: Warm seawater inlet (liquid)
        2: Flash chamber exit (two-phase)
        3: Turbine inlet (saturated vapor from flash chamber)
        4: Turbine exit (two-phase after expansion)

        Simplified water/steam properties used
        """

        # For open cycle, T_evap is flash temperature, T_cond is condenser temp
        # These are approximately equal to warm/cold seawater temperatures

        # State 1: Warm seawater inlet (liquid at T_evap)
        # Simplified: h = cp * T (liquid water)
        cp_water = 4.18  # kJ/kg·K (specific heat of water)
        h_1 = cp_water * T_evap
        s_1 = cp_water * np.log((T_evap + 273.15) / 273.15)  # Simplified entropy

        # State 2: After flashing (mixture of liquid and vapor)
        # Flash occurs at saturation pressure corresponding to flash chamber vacuum
        # The flash temperature is LOWER than inlet temperature (vacuum chamber)

        # Flash chamber operates at a lower temperature than warm water inlet
        # Typical flash temperature drop: by ~2-4°C for OTEC applications
        T_flash = T_evap - 3.0  # Flash at 3°C below warm water inlet

        # Latent heat of vaporization at flash temperature (temperature dependent)
        h_fg_flash = 2500 - 2.4 * T_flash  # Approximate kJ/kg at flash temperature

        # Saturation enthalpy of liquid at flash temperature
        h_f_flash = cp_water * T_flash
        h_g_flash = h_f_flash + h_fg_flash

        # Flash quality (fraction of vapor produced)
        # Energy balance: h_inlet = h_f_flash + x * h_fg_flash
        # x = (h_inlet - h_f_flash) / h_fg_flash
        # Since h_inlet = cp * T_evap and h_f_flash = cp * T_flash:
        # x = cp * (T_evap - T_flash) / h_fg_flash
        quality_flash = np.where(h_fg_flash > 0,
                                 cp_water * (T_evap - T_flash) / h_fg_flash,
                                 0)
        # Typical flash quality for OTEC: 0.5-1.5% (0.005-0.015)
        quality_flash = np.clip(quality_flash, 0.001, 0.02)

        h_2 = h_1  # Isenthalpic flash (energy conserved)
        s_2 = s_1  # Approximate

        # State 3: Saturated vapor to turbine (separator output)
        h_3 = h_g_flash
        # Simplified entropy for saturated vapor at flash temperature
        s_3 = cp_water * np.log((T_flash + 273.15) / 273.15) + h_fg_flash / (T_flash + 273.15)

        # State 4: Turbine exit (isentropic expansion, then actual with efficiency)
        # Expansion to condenser pressure

        # Condenser saturation properties
        h_f_cond = cp_water * T_cond
        h_fg_cond = 2500 - 2.4 * T_cond
        h_g_cond = h_f_cond + h_fg_cond

        # Isentropic entropy at condenser pressure (simplified)
        s_f_cond = cp_water * np.log((T_cond + 273.15) / 273.15)
        s_fg_cond = h_fg_cond / (T_cond + 273.15)
        s_g_cond = s_f_cond + s_fg_cond

        # Quality at isentropic state
        # Handle both scalar and array inputs
        x_4_isen = np.where(s_fg_cond > 0,
                            np.clip((s_3 - s_f_cond) / s_fg_cond, 0, 1),
                            0)

        h_4_isen = h_f_cond + x_4_isen * h_fg_cond

        # Actual state accounting for turbine efficiency
        eff_isen_turb = inputs.get('eff_isen_turb', 0.82)
        h_4 = (h_4_isen - h_3) * eff_isen_turb + h_3

        # Actual quality
        x_4 = np.where(h_fg_cond > 0, (h_4 - h_f_cond) / h_fg_cond, 0)
        x_4 = np.clip(x_4, 0, 1)
        s_4 = s_f_cond + x_4 * s_fg_cond

        states = {
            'h_1': h_1,
            's_1': s_1,
            'h_2': h_2,
            's_2': s_2,
            'h_3': h_3,
            's_3': s_3,
            'h_4': h_4,
            's_4': s_4,
            'x_4': x_4,
            'h_4_isen': h_4_isen,
            'x_4_isen': x_4_isen,
            'quality_flash': quality_flash,  # Flash vapor fraction
        }

        return states

    def calculate_mass_flow(self, p_gross, states):
        """
        Calculate mass flow from gross power output

        For open cycle, p_gross is the steam turbine power
        Note: This is the STEAM mass flow, not total seawater flow
        Total seawater flow = m_steam / quality_flash

        W_gross = m_steam * (h_4 - h_3)
        """
        # Mass flow of steam through turbine
        m_steam = p_gross / (states['h_4'] - states['h_3'])

        # Store for later use
        states['m_steam'] = m_steam

        # Total warm seawater flow needed
        if states['quality_flash'] > 0:
            m_seawater_total = m_steam / states['quality_flash']
        else:
            m_seawater_total = m_steam * 100  # Assume 1% flash if not calculated

        return m_seawater_total

    def calculate_pump_power(self, m_fluid, states, inputs):
        """
        Calculate pump power consumption for open cycle

        For open cycle, pump power includes:
        1. Warm water pumps (large flow rate for flash evaporation)
        2. Cold water pumps (for condensation)
        3. Vacuum pumps for non-condensable gases removal
        4. Condensate extraction pumps

        Note: Open cycle requires MUCH larger seawater flow rates than closed cycle
        because only a small fraction (~0.5-1.5%) flashes to steam.

        Args:
            m_fluid: Total warm seawater mass flow [kg/s]
            states: Dictionary with thermodynamic states
            inputs: Dictionary with pump parameters

        Returns:
            p_pump_total: Total pump power [kW] (positive value)
        """
        # Get pump efficiencies
        eff_pump_mech = inputs.get('eff_pump_SW_mech', 0.90)
        eff_pump_hydr = inputs.get('eff_pump_SW_hyd', 0.80)
        eff_pump = eff_pump_mech * eff_pump_hydr

        # Seawater density
        rho_sw = inputs.get('rho_SW', 1025)  # kg/m³

        # --- Warm water pump ---
        # Head required: intake depth + friction losses + flash chamber vacuum head
        # Typical: 5-10 m for intake + 3-5 m friction + 5-8 m vacuum head
        head_ww = inputs.get('head_WW_pump', 15)  # meters
        g = 9.81  # m/s²

        # Warm water flow is the total seawater flow (m_fluid)
        m_ww = m_fluid
        p_pump_ww = m_ww * g * head_ww / (eff_pump * 1000)  # kW

        # --- Cold water pump ---
        # Cold water flow determined by condenser heat balance
        # Typically 1.5-2x warm water flow for open cycle
        # Cold water comes from ~1000m depth, so higher head
        cw_ww_ratio = inputs.get('CW_WW_ratio', 1.8)
        m_cw = m_ww * cw_ww_ratio
        head_cw = inputs.get('head_CW_pump', 25)  # meters (deeper intake)
        p_pump_cw = m_cw * g * head_cw / (eff_pump * 1000)  # kW

        # --- Vacuum pump ---
        # Non-condensable gas removal (dissolved gases released during flash)
        # Typically 2-4% of gross power for open cycle
        vacuum_pump_factor = inputs.get('vacuum_pump_factor', 0.035)  # 3.5%
        p_gross = inputs.get('p_gross', -136000)
        p_vacuum = abs(p_gross) * vacuum_pump_factor

        # --- Condensate pump ---
        # Extract fresh water from condenser
        m_steam = states.get('m_steam', m_fluid * 0.01)  # Steam flow
        if hasattr(m_steam, '__len__'):
            m_condensate = m_steam
        else:
            m_condensate = np.atleast_1d(m_steam)
        head_condensate = 10  # meters
        p_pump_condensate = m_condensate * g * head_condensate / (eff_pump * 1000)

        # Total pump power
        p_pump_total = p_pump_ww + p_pump_cw + p_vacuum + p_pump_condensate

        return p_pump_total


class KalinaCycle(ThermodynamicCycle):
    """
    Kalina Cycle - uses ammonia-water mixture
    More efficient at low temperature differentials

    Key components:
    1. Turbine - expands NH3-rich vapor
    2. Separator - splits mixture into NH3-rich and NH3-poor streams
    3. Evaporator - heats basic solution
    4. Condenser - condenses NH3-rich vapor
    5. Recuperator - heat exchange between streams
    6. Mixer - recombines streams
    7. Pumps - two pumps for different streams

    Advantages over Rankine:
    - Variable boiling point matches heat source better
    - Higher efficiency at low temperature differences
    - 5-10% better performance for OTEC

    Note: Uses simplified Kalina Cycle System 11 (KCS-11)
    """

    def __init__(self, ammonia_concentration=0.7):
        """
        Args:
            ammonia_concentration: Basic solution NH3 mass fraction (0.6-0.9 typical)
        """
        from .mixtures import AmmoniaWaterMixture

        # Kalina uses NH3-H2O mixture, not single-component fluid
        mixture_fluid = AmmoniaWaterMixture()

        super().__init__(f'Kalina (NH3 {ammonia_concentration:.0%})', mixture_fluid)
        self.x_basic = ammonia_concentration  # Basic solution concentration
        self.mixture = mixture_fluid

    def calculate_cycle_states(self, T_evap, T_cond, p_evap, p_cond, inputs):
        """
        Calculate Kalina cycle states (KCS-11 variant)

        States:
        1: Condenser exit (saturated liquid, NH3-rich)
        2: Pump 1 exit (compressed liquid, NH3-rich)
        3: Mixer exit (basic solution)
        4: Pump 2 exit (compressed basic solution)
        5: Recuperator hot side exit
        6: Evaporator exit (saturated vapor-liquid, basic solution)
        7: Separator vapor exit (NH3-rich vapor)
        8: Separator liquid exit (NH3-poor liquid)
        9: Recuperator cold side exit
        10: Turbine exit (NH3-rich two-phase)

        Returns:
            states: Dictionary with thermodynamic states
        """

        # Determine separator conditions
        # Typically T_separator = T_evap, P_separator = p_evap
        T_separator = T_evap
        P_separator = p_evap

        # Vapor-liquid equilibrium at separator
        y_rich = self.mixture.vapor_liquid_equilibrium(T_separator, P_separator, self.x_basic)
        x_poor = self._calculate_poor_stream_concentration(self.x_basic, y_rich)

        # Mass split ratio (fraction going to rich stream)
        split_ratio = 0.7  # Typically 60-80%, can be optimized

        # State 1: Condenser exit (saturated liquid, rich stream)
        T_1 = T_cond
        P_1 = p_cond
        h_1 = self.mixture.enthalpy_liquid(T_1, P_1, y_rich)
        s_1 = self.mixture.entropy_liquid(T_1, P_1, y_rich)

        # State 2: Pump 1 exit (rich stream pump)
        P_2 = P_separator
        h_2 = h_1 + (P_2 - P_1) * 100 / (inputs.get('rho_NH3', 640) * inputs.get('eff_isen_pump', 0.8))
        T_2 = T_1 + 1.0  # Small temperature rise
        s_2 = s_1

        # State 3: Mixer exit (basic solution)
        # Energy and mass balances at mixer
        # Simplified: assume poor stream enters at T_9
        T_3 = T_2 + 5.0  # Approximate mixing temperature
        P_3 = P_separator
        h_3 = self.mixture.enthalpy_liquid(T_3, P_3, self.x_basic)
        s_3 = self.mixture.entropy_liquid(T_3, P_3, self.x_basic)

        # State 4: Pump 2 exit (basic solution pump)
        P_4 = P_separator  # Already at high pressure (simplified)
        h_4 = h_3
        T_4 = T_3
        s_4 = s_3

        # State 5: Recuperator hot side exit
        T_5 = T_separator - 10.0  # Approach temperature in recuperator
        P_5 = P_separator
        h_5 = self.mixture.enthalpy_liquid(T_5, P_5, self.x_basic)
        s_5 = self.mixture.entropy_liquid(T_5, P_5, self.x_basic)

        # State 6: Evaporator exit (saturated, basic solution)
        T_6 = T_separator
        P_6 = P_separator
        # Two-phase mixture at separator inlet
        h_6_liq = self.mixture.enthalpy_liquid(T_6, P_6, self.x_basic)
        h_6_vap = self.mixture.enthalpy_vapor(T_6, P_6, y_rich)
        quality_6 = split_ratio  # Vapor quality equals split ratio
        h_6 = h_6_liq * (1 - quality_6) + h_6_vap * quality_6

        # State 7: Separator vapor exit (NH3-rich vapor to turbine)
        T_7 = T_separator
        P_7 = P_separator
        h_7 = self.mixture.enthalpy_vapor(T_7, P_7, y_rich)
        s_7 = self.mixture.entropy_vapor(T_7, P_7, y_rich)  # Proper entropy calculation

        # State 8: Separator liquid exit (NH3-poor liquid)
        T_8 = T_separator
        P_8 = P_separator
        h_8 = self.mixture.enthalpy_liquid(T_8, P_8, x_poor)
        s_8 = self.mixture.entropy_liquid(T_8, P_8, x_poor)

        # State 9: Recuperator cold side exit (poor stream heated)
        T_9 = T_3 - 5.0  # Approach temperature
        P_9 = P_8
        h_9 = self.mixture.enthalpy_liquid(T_9, P_9, x_poor)
        s_9 = self.mixture.entropy_liquid(T_9, P_9, x_poor)

        # State 10: Turbine exit (NH3-rich two-phase)
        P_10 = p_cond
        # Isentropic expansion (simplified)
        h_10_isen = h_7 - (h_7 - h_1) * 0.5  # Approximate
        h_10 = h_7 - (h_7 - h_10_isen) * inputs.get('eff_isen_turb', 0.82)
        T_10 = T_cond + 2.0  # Slightly superheated

        states = {
            'h_1': h_1, 's_1': s_1, 'T_1': T_1, 'P_1': P_1, 'x_1': y_rich,
            'h_2': h_2, 's_2': s_2, 'T_2': T_2, 'P_2': P_2,
            'h_3': h_3, 's_3': s_3, 'T_3': T_3, 'P_3': P_3, 'x_3': self.x_basic,
            'h_4': h_4, 's_4': s_4, 'T_4': T_4, 'P_4': P_4,
            'h_5': h_5, 's_5': s_5, 'T_5': T_5, 'P_5': P_5,
            'h_6': h_6, 'T_6': T_6, 'P_6': P_6,
            'h_7': h_7, 's_7': s_7, 'T_7': T_7, 'P_7': P_7, 'x_7': y_rich,
            'h_8': h_8, 's_8': s_8, 'T_8': T_8, 'P_8': P_8, 'x_8': x_poor,
            'h_9': h_9, 's_9': s_9, 'T_9': T_9, 'P_9': P_9,
            'h_10': h_10, 'T_10': T_10, 'P_10': P_10,
            'split_ratio': split_ratio,
            'y_rich': y_rich,
            'x_poor': x_poor,
        }

        return states

    def calculate_mass_flow(self, p_gross, states):
        """
        Calculate mass flow from gross power output

        W_gross = m_rich * (h_10 - h_7)
        """
        split_ratio = states['split_ratio']
        m_basic = p_gross / (split_ratio * (states['h_10'] - states['h_7']))
        m_rich = m_basic * split_ratio
        m_poor = m_basic * (1 - split_ratio)

        return {
            'm_basic': m_basic,
            'm_rich': m_rich,
            'm_poor': m_poor,
        }

    def calculate_pump_power(self, m_flows, states, inputs):
        """
        Calculate pump power consumption (two pumps)

        W_pump1 = m_rich * (h_2 - h_1) / eff_mech
        W_pump2 = m_basic * (h_4 - h_3) / eff_mech
        """
        eff_mech = inputs.get('eff_pump_NH3_mech', 0.95)

        m_rich = m_flows['m_rich']
        m_basic = m_flows['m_basic']

        p_pump1 = m_rich * (states['h_2'] - states['h_1']) / eff_mech
        p_pump2 = m_basic * (states['h_4'] - states['h_3']) / eff_mech

        p_pump_total = p_pump1 + p_pump2

        return p_pump_total

    def calculate_heat_transfer(self, m_flows, states):
        """
        Calculate heat transfer in evaporator, condenser, and recuperator

        Args:
            m_flows: Dictionary with mass flows
            states: Dictionary with thermodynamic states

        Returns:
            Q_evap: Evaporator heat transfer [kW]
            Q_cond: Condenser heat transfer [kW]
            Q_recup: Recuperator heat transfer [kW]
        """
        m_basic = m_flows['m_basic']
        m_rich = m_flows['m_rich']
        m_poor = m_flows['m_poor']

        Q_evap = m_basic * (states['h_6'] - states['h_5'])
        Q_cond = m_rich * (states['h_1'] - states['h_10'])
        Q_recup = m_poor * (states['h_9'] - states['h_8'])

        return Q_evap, Q_cond, Q_recup

    def _calculate_poor_stream_concentration(self, x_basic, y_rich):
        """
        Calculate poor stream concentration from mass balance (vectorized)

        Simplified approximation: x_poor ≈ x_basic - Δx
        More accurate: solve mass balance with split ratio
        """
        # For typical Kalina cycle:
        # y_rich > x_basic > x_poor
        # Approximate relationship
        x_poor = x_basic - (y_rich - x_basic) * 0.5

        # Bounds check (vectorized)
        x_poor = np.maximum(0.3, np.minimum(x_poor, x_basic - 0.05))

        return x_poor


class RankineHybridCycle(ThermodynamicCycle):
    """
    Hybrid Rankine Cycle (Open-Closed Combined)
    Combines closed Rankine cycle with open flash cycle for maximum power generation

    Configuration:
    1. Primary closed cycle (NH3) - main power generation
    2. Secondary open cycle (flash steam) - extracts additional power from residual heat

    Flow sequence:
    - Warm seawater first heats NH3 in closed cycle evaporator
    - Partially cooled warm water then flashes in vacuum chamber
    - Flash steam expands through secondary turbine
    - Both cycles share cold seawater for condensation

    Advantages:
    - 8-15% higher power output compared to closed cycle alone
    - Better utilization of available thermal energy
    - Modest increase in complexity
    - Proven concept for OTEC applications

    States:
    Closed cycle: 1-4 (same as RankineClosedCycle)
    Open cycle: 5-8 (flash and steam expansion)
    """

    def __init__(self, working_fluid):
        super().__init__('Rankine Hybrid (Open-Closed)', working_fluid)

    def calculate_cycle_states(self, T_evap, T_cond, p_evap, p_cond, inputs):
        """
        Calculate hybrid cycle states

        Closed cycle (primary):
        1: Condenser outlet (saturated liquid NH3)
        2: Pump outlet (compressed liquid NH3)
        3: Evaporator outlet (saturated vapor NH3)
        4: Turbine outlet (two-phase NH3)

        Open cycle (secondary):
        5: Warm seawater after closed evaporator (liquid)
        6: Flash chamber (two-phase water)
        7: Flash steam to turbine (saturated vapor)
        8: Flash turbine outlet (two-phase water)

        Returns:
            states: Dictionary with all thermodynamic states
        """

        eff_isen_turb = inputs['eff_isen_turb']
        eff_isen_pump = inputs['eff_isen_pump']

        # === CLOSED CYCLE (NH3) - Primary power generation ===

        # Use scalar density value
        if 'rho_NH3' in inputs:
            rho_fluid = inputs['rho_NH3']
        else:
            p_cond_scalar = np.mean(p_cond) if isinstance(p_cond, np.ndarray) else p_cond
            rho_fluid = self.fluid.density_liquid(p=p_cond_scalar)

        # State 1: Condenser outlet (saturated liquid at p_cond)
        h_1 = self.fluid.enthalpy_liquid(p=p_cond)
        s_1 = self.fluid.entropy_liquid(p=p_cond)

        # State 2: Pump outlet (compressed liquid at p_evap)
        h_2 = 1/rho_fluid * (p_evap - p_cond) * 100000/1000 / eff_isen_pump + h_1
        s_2 = s_1

        # State 3: Evaporator outlet (saturated vapor at p_evap)
        h_3 = self.fluid.enthalpy_vapor(p=p_evap)
        s_3 = self.fluid.entropy_vapor(p=p_evap)

        # State 4: Turbine outlet (two-phase at p_cond)
        h_4_liq = self.fluid.enthalpy_liquid(p=p_cond)
        h_4_vap = self.fluid.enthalpy_vapor(p=p_cond)
        s_4_liq = self.fluid.entropy_liquid(p=p_cond)
        s_4_vap = self.fluid.entropy_vapor(p=p_cond)

        x_4_isen = self.fluid.quality_from_entropy(s_3, s_4_liq, s_4_vap)
        h_4_isen = self.fluid.enthalpy_two_phase(h_4_liq, h_4_vap, x_4_isen)

        h_4 = (h_4_isen - h_3) * eff_isen_turb + h_3
        x_4 = self.fluid.quality_from_enthalpy(h_4, h_4_liq, h_4_vap)
        s_4 = self.fluid.entropy_two_phase(s_4_liq, s_4_vap, x_4)

        # === OPEN CYCLE (Flash Steam) - Secondary power generation ===

        # Temperature drop across closed cycle evaporator
        # Warm water exits evaporator at reduced temperature
        dT_evap = inputs.get('dT_WW_evap', 3.0)  # Typical 2-4°C drop
        T_ww_post_evap = T_evap - dT_evap

        # Flash chamber temperature (lower than warm water exit)
        # Optimized to be between warm and cold water temperatures
        T_flash = T_ww_post_evap - 2.0  # Additional 2°C for flash chamber

        # Simplified water properties for flash cycle
        cp_water = 4.18  # kJ/kg·K

        # State 5: Warm seawater after closed evaporator (liquid)
        h_5 = cp_water * T_ww_post_evap
        s_5 = cp_water * np.log((T_ww_post_evap + 273.15) / 273.15)

        # State 6: Flash chamber (isenthalpic flash)
        h_6 = h_5  # Isenthalpic flash process

        # Latent heat at flash temperature
        h_fg_flash = 2500 - 2.4 * T_flash  # Approximate kJ/kg
        h_f_flash = cp_water * T_flash
        h_g_flash = h_f_flash + h_fg_flash

        # Flash quality (fraction of vapor produced)
        quality_flash = np.where(h_fg_flash > 0, (h_6 - h_f_flash) / h_fg_flash, 0)
        quality_flash = np.clip(quality_flash, 0, 0.02)  # Typically 0.5-2% for hybrid OTEC

        s_6 = s_5

        # State 7: Saturated vapor to flash turbine
        h_7 = h_g_flash
        s_7 = cp_water * np.log((T_flash + 273.15) / 273.15) + h_fg_flash / (T_flash + 273.15)

        # State 8: Flash turbine outlet (expansion to condenser pressure)
        # Condenser properties
        h_f_cond = cp_water * T_cond
        h_fg_cond = 2500 - 2.4 * T_cond
        h_g_cond = h_f_cond + h_fg_cond

        s_f_cond = cp_water * np.log((T_cond + 273.15) / 273.15)
        s_fg_cond = h_fg_cond / (T_cond + 273.15)
        s_g_cond = s_f_cond + s_fg_cond

        # Isentropic expansion quality
        x_8_isen = np.where(s_fg_cond > 0,
                            np.clip((s_7 - s_f_cond) / s_fg_cond, 0, 1),
                            0)
        h_8_isen = h_f_cond + x_8_isen * h_fg_cond

        # Actual state with turbine efficiency
        # Use same turbine efficiency as closed cycle
        h_8 = (h_8_isen - h_7) * eff_isen_turb + h_7

        x_8 = np.where(h_fg_cond > 0, (h_8 - h_f_cond) / h_fg_cond, 0)
        x_8 = np.clip(x_8, 0, 1)
        s_8 = s_f_cond + x_8 * s_fg_cond

        states = {
            # Closed cycle states
            'h_1': h_1,
            's_1': s_1,
            'h_2': h_2,
            's_2': s_2,
            'h_3': h_3,
            's_3': s_3,
            'h_4': h_4,
            's_4': s_4,
            'x_4': x_4,
            'h_4_isen': h_4_isen,
            'x_4_isen': x_4_isen,

            # Open cycle states
            'h_5': h_5,
            's_5': s_5,
            'h_6': h_6,
            's_6': s_6,
            'h_7': h_7,
            's_7': s_7,
            'h_8': h_8,
            's_8': s_8,
            'x_8': x_8,
            'h_8_isen': h_8_isen,
            'x_8_isen': x_8_isen,

            # Flash parameters
            'quality_flash': quality_flash,
            'T_flash': T_flash,
            'T_ww_post_evap': T_ww_post_evap,
        }

        return states

    def calculate_mass_flow(self, p_gross, states):
        """
        Calculate mass flows for both cycles

        Total gross power is split between:
        - Closed cycle turbine (primary, ~85-90%)
        - Flash turbine (secondary, ~10-15%)

        Args:
            p_gross: Total target gross power output [kW] (negative value)
            states: Dictionary with thermodynamic states

        Returns:
            Dictionary with mass flows:
            - m_NH3: NH3 mass flow in closed cycle [kg/s]
            - m_steam: Steam mass flow in flash turbine [kg/s]
            - m_seawater: Total warm seawater flow [kg/s]
        """

        # Power split: optimize based on available enthalpy drops
        # Typical: 85-90% from closed cycle, 10-15% from flash
        power_split_closed = 0.88  # 88% from closed cycle

        # Closed cycle mass flow
        W_closed = states['h_4'] - states['h_3']  # Negative (power out)
        m_NH3 = (p_gross * power_split_closed) / W_closed

        # Flash cycle mass flow
        W_flash = states['h_8'] - states['h_7']  # Negative (power out)
        m_steam = (p_gross * (1 - power_split_closed)) / W_flash

        # Total warm seawater flow
        # Steam is fraction of total seawater (quality_flash)
        quality_flash = states['quality_flash']
        if np.any(quality_flash > 0):
            m_seawater = m_steam / quality_flash
        else:
            m_seawater = m_steam * 100  # Assume 1% if not calculated

        return {
            'm_NH3': m_NH3,
            'm_steam': m_steam,
            'm_seawater': m_seawater,
            'power_split_closed': power_split_closed,
        }

    def calculate_pump_power(self, m_flows, states, inputs):
        """
        Calculate pump power consumption for both cycles

        Closed cycle:
        - NH3 working fluid pump

        Open cycle:
        - Vacuum pumps (non-condensable gas removal)
        - Already included in seawater pumps

        Args:
            m_flows: Dictionary with mass flows
            states: Dictionary with thermodynamic states
            inputs: Dictionary with efficiencies

        Returns:
            p_pump_total: Total pump power [kW] (positive value)
        """

        eff_mech = inputs['eff_pump_NH3_mech']
        m_NH3 = m_flows['m_NH3']

        # Closed cycle NH3 pump
        p_pump_NH3 = m_NH3 * (states['h_2'] - states['h_1']) / eff_mech

        # Flash cycle vacuum pump (for non-condensable gases)
        # Simplified: 2-3% of flash turbine power
        power_split_closed = m_flows.get('power_split_closed', 0.88)
        p_gross = inputs.get('p_gross', -136000)
        p_flash_turbine = abs(p_gross) * (1 - power_split_closed)
        p_vacuum = p_flash_turbine * 0.025  # 2.5% of flash turbine power

        p_pump_total = p_pump_NH3 + p_vacuum

        return p_pump_total

    def calculate_heat_transfer(self, m_flows, states):
        """
        Calculate heat transfer in evaporators and condensers

        Args:
            m_flows: Dictionary with mass flows
            states: Dictionary with thermodynamic states

        Returns:
            Q_evap_closed: Closed cycle evaporator heat transfer [kW]
            Q_evap_flash: Flash evaporator (warm water cooling) [kW]
            Q_cond_total: Total condenser heat transfer [kW]
        """

        m_NH3 = m_flows['m_NH3']
        m_steam = m_flows['m_steam']
        m_seawater = m_flows['m_seawater']

        # Closed cycle evaporator
        Q_evap_closed = m_NH3 * (states['h_3'] - states['h_2'])

        # Flash cycle heat (from warm water latent heat of vaporization)
        # The steam produced takes energy from the warm water
        # Q = m_steam * h_fg (latent heat)
        cp_water = 4.18  # kJ/kg-K
        T_flash = states.get('T_flash', 18.0)
        T_flash_val = T_flash[0] if hasattr(T_flash, '__getitem__') else T_flash
        h_fg_flash = 2500 - 2.4 * T_flash_val  # Approximate latent heat
        Q_evap_flash = m_steam * h_fg_flash

        # Total condenser load (both cycles)
        Q_cond_NH3 = m_NH3 * (states['h_1'] - states['h_4'])

        # Flash steam condensation
        # Steam condenses from state 8 to saturated liquid
        # The heat released is approximately the enthalpy difference
        # Since state 8 is already two-phase, we calculate the heat to condense it to liquid
        Q_cond_steam = m_steam * states['h_8']  # Heat released when condensing

        Q_cond_total = Q_cond_NH3 + Q_cond_steam

        return Q_evap_closed, Q_evap_flash, Q_cond_total


class UeharaCycle(ThermodynamicCycle):
    """
    Uehara Cycle (Two-stage Rankine with NH3-H2O mixture)
    Uses two-stage evaporation for better thermal matching

    Key components:
    1. LP Evaporator - uses partially cooled warm water
    2. HP Evaporator - uses hottest warm water
    3. LP Turbine - expands from p_int to p_cond
    4. HP Turbine - expands from p_evap to p_int
    5. Condenser - shared by both streams
    6. Pumps - two pumps (LP and HP)
    7. Separator - splits NH3-rich vapor from NH3-poor liquid

    Advantages:
    - Better thermal matching → higher efficiency (3-5% improvement)
    - More power extraction from available ΔT
    - Proven design for OTEC (Uehara & Ikegami, 1990)

    Uses NH3-H2O mixture like Kalina cycle.

    States:
    HP loop: 1-4
    LP loop: 5-8
    """

    def __init__(self, ammonia_concentration=0.7):
        """
        Args:
            ammonia_concentration: Basic solution NH3 mass fraction (0.6-0.9 typical)
        """
        from .mixtures import AmmoniaWaterMixture

        # Uehara uses NH3-H2O mixture
        mixture_fluid = AmmoniaWaterMixture()

        super().__init__(f'Uehara (NH3 {ammonia_concentration:.0%})', mixture_fluid)
        self.x_basic = ammonia_concentration  # Basic solution concentration
        self.mixture = mixture_fluid

    def calculate_cycle_states(self, T_evap, T_cond, p_evap, p_cond, inputs):
        """
        Two-stage Rankine cycle calculation with NH3-H2O mixture

        States numbering:
        HP Loop:
        1: HP condenser exit (saturated liquid)
        2: HP pump exit (compressed liquid)
        3: HP evaporator exit (saturated vapor)
        4: HP turbine exit (two-phase, at p_int)

        LP Loop:
        5: LP condenser exit (saturated liquid)
        6: LP pump exit (compressed liquid, to p_int)
        7: LP evaporator exit (saturated vapor, at p_int)
        8: LP turbine exit (two-phase, at p_cond)

        Returns:
            states: Dictionary with thermodynamic states for both loops
        """

        # Optimize intermediate pressure
        # Typically p_int = sqrt(p_evap * p_cond) for equal pressure ratios
        p_int = np.sqrt(p_evap * p_cond)

        # Get saturation temperatures using mixture properties
        # Handle arrays by element-wise calculation
        p_int_arr = np.atleast_1d(p_int)
        original_shape = p_int_arr.shape
        p_int_flat = p_int_arr.ravel()
        T_int_flat = np.array([self.mixture.saturation_temperature(p, self.x_basic) for p in p_int_flat])
        T_int = T_int_flat.reshape(original_shape)
        if np.isscalar(p_int):
            T_int = float(T_int)

        # Vapor-liquid equilibrium to get vapor composition
        y_rich = self.mixture.vapor_liquid_equilibrium(T_evap, p_evap, self.x_basic)
        y_rich_int = self.mixture.vapor_liquid_equilibrium(T_int, p_int, self.x_basic)

        # Temperature levels for evaporators
        T_evap_HP = T_evap  # Highest temperature
        T_evap_LP = T_int   # Intermediate temperature

        eff_pump = inputs.get('eff_isen_pump', 0.80)
        eff_turb = inputs.get('eff_isen_turb', 0.82)
        rho = inputs.get('rho_NH3', 640)  # Approximate liquid density

        #=== HP Loop States ===

        # State 1: HP condenser exit (saturated liquid at p_cond)
        h_1_HP = self.mixture.enthalpy_liquid(T_cond, p_cond, y_rich)
        s_1_HP = self.mixture.entropy_liquid(T_cond, p_cond, y_rich)

        # State 2: HP pump exit (compressed to p_evap)
        h_2_HP = h_1_HP + (p_evap - p_cond) * 100000 / 1000 / eff_pump / rho
        s_2_HP = s_1_HP  # Approximately constant for liquid
        T_2_HP = T_cond + 1.0  # Small temperature rise

        # State 3: HP evaporator exit (saturated vapor at p_evap)
        h_3_HP = self.mixture.enthalpy_vapor(T_evap, p_evap, y_rich)
        s_3_HP = self.mixture.entropy_vapor(T_evap, p_evap, y_rich)

        # State 4: HP turbine exit (at p_int)
        # Simplified isentropic expansion
        h_4_liq = self.mixture.enthalpy_liquid(T_int, p_int, self.x_basic)
        h_4_vap = self.mixture.enthalpy_vapor(T_int, p_int, y_rich_int)

        # Isentropic expansion approximation
        h_4_isen = h_3_HP - (h_3_HP - h_4_liq) * 0.5
        h_4_HP = h_3_HP - (h_3_HP - h_4_isen) * eff_turb

        # Estimate quality
        dh_4 = h_4_vap - h_4_liq
        dh_4 = np.where(np.abs(dh_4) < 1e-6, 1e-6, dh_4)  # Avoid division by zero
        x_4_HP = (h_4_HP - h_4_liq) / dh_4
        x_4_HP = np.clip(x_4_HP, 0, 1)

        s_4_HP = self.mixture.entropy_liquid(T_int, p_int, self.x_basic) * (1 - x_4_HP) + \
                 self.mixture.entropy_vapor(T_int, p_int, y_rich_int) * x_4_HP

        #=== LP Loop States ===

        # State 5: LP condenser exit (saturated liquid at p_cond)
        h_5_LP = h_1_HP  # Same condenser, same composition
        s_5_LP = s_1_HP

        # State 6: LP pump exit (compressed to p_int)
        h_6_LP = h_5_LP + (p_int - p_cond) * 100000 / 1000 / eff_pump / rho
        s_6_LP = s_5_LP
        T_6_LP = T_cond + 0.5

        # State 7: LP evaporator exit (saturated vapor at p_int)
        h_7_LP = self.mixture.enthalpy_vapor(T_int, p_int, y_rich_int)
        s_7_LP = self.mixture.entropy_vapor(T_int, p_int, y_rich_int)

        # State 8: LP turbine exit (at p_cond)
        h_8_liq = self.mixture.enthalpy_liquid(T_cond, p_cond, y_rich)
        h_8_vap = self.mixture.enthalpy_vapor(T_cond, p_cond, y_rich)

        # Isentropic expansion approximation
        h_8_isen = h_7_LP - (h_7_LP - h_8_liq) * 0.5
        h_8_LP = h_7_LP - (h_7_LP - h_8_isen) * eff_turb

        # Estimate quality
        dh_8 = h_8_vap - h_8_liq
        dh_8 = np.where(np.abs(dh_8) < 1e-6, 1e-6, dh_8)  # Avoid division by zero
        x_8_LP = (h_8_LP - h_8_liq) / dh_8
        x_8_LP = np.clip(x_8_LP, 0, 1)

        s_8_LP = self.mixture.entropy_liquid(T_cond, p_cond, y_rich) * (1 - x_8_LP) + \
                 self.mixture.entropy_vapor(T_cond, p_cond, y_rich) * x_8_LP

        # Pack states
        states = {
            # HP loop
            'h_1_HP': h_1_HP, 's_1_HP': s_1_HP,
            'h_2_HP': h_2_HP, 's_2_HP': s_2_HP,
            'h_3_HP': h_3_HP, 's_3_HP': s_3_HP,
            'h_4_HP': h_4_HP, 's_4_HP': s_4_HP, 'x_4_HP': x_4_HP,

            # LP loop
            'h_5_LP': h_5_LP, 's_5_LP': s_5_LP,
            'h_6_LP': h_6_LP, 's_6_LP': s_6_LP,
            'h_7_LP': h_7_LP, 's_7_LP': s_7_LP,
            'h_8_LP': h_8_LP, 's_8_LP': s_8_LP, 'x_8_LP': x_8_LP,

            # Pressures and temperatures
            'p_evap': p_evap,
            'p_int': p_int,
            'p_cond': p_cond,
            'T_evap_HP': T_evap_HP,
            'T_evap_LP': T_evap_LP,
            'T_int': T_int,

            # Mixture concentrations
            'x_basic': self.x_basic,
            'y_rich': y_rich,
            'y_rich_int': y_rich_int,
        }

        return states

    def calculate_mass_flow(self, p_gross, states):
        """
        Calculate mass flows for both loops

        Total power split between HP and LP turbines
        Optimally, split to maximize efficiency

        For simplicity, assume equal power split (can be optimized)
        """

        # Power from each turbine
        W_HP = states['h_4_HP'] - states['h_3_HP']  # Negative (power out)
        W_LP = states['h_8_LP'] - states['h_7_LP']  # Negative (power out)

        # Assume 60% HP, 40% LP (can be optimized)
        power_split_HP = 0.6
        power_split_LP = 0.4

        m_HP = (p_gross * power_split_HP) / W_HP
        m_LP = (p_gross * power_split_LP) / W_LP

        return {
            'm_HP': m_HP,
            'm_LP': m_LP,
            'm_total': m_HP + m_LP,
        }

    def calculate_pump_power(self, m_flows, states, inputs):
        """
        Calculate pump power for both loops

        W_pump_HP = m_HP * (h_2 - h_1) / eff_mech
        W_pump_LP = m_LP * (h_6 - h_5) / eff_mech
        """
        eff_mech = inputs.get('eff_pump_NH3_mech', 0.95)

        m_HP = m_flows['m_HP']
        m_LP = m_flows['m_LP']

        p_pump_HP = m_HP * (states['h_2_HP'] - states['h_1_HP']) / eff_mech
        p_pump_LP = m_LP * (states['h_6_LP'] - states['h_5_LP']) / eff_mech

        p_pump_total = p_pump_HP + p_pump_LP

        return p_pump_total

    def calculate_heat_transfer(self, m_flows, states):
        """
        Calculate heat transfer in both evaporators and condenser

        Args:
            m_flows: Dictionary with mass flows
            states: Dictionary with thermodynamic states

        Returns:
            Q_evap_HP: HP evaporator heat transfer [kW]
            Q_evap_LP: LP evaporator heat transfer [kW]
            Q_cond: Condenser heat transfer [kW]
        """
        m_HP = m_flows['m_HP']
        m_LP = m_flows['m_LP']

        Q_evap_HP = m_HP * (states['h_3_HP'] - states['h_2_HP'])
        Q_evap_LP = m_LP * (states['h_7_LP'] - states['h_6_LP'])

        # Condenser handles both streams
        Q_cond = m_HP * (states['h_1_HP'] - states['h_4_HP']) + \
                 m_LP * (states['h_5_LP'] - states['h_8_LP'])

        return Q_evap_HP, Q_evap_LP, Q_cond


def get_thermodynamic_cycle(cycle_type='rankine_closed', working_fluid=None, **kwargs):
    """
    Factory function to get a thermodynamic cycle instance

    Args:
        cycle_type: Type of cycle ('rankine_closed', 'rankine_open', 'rankine_hybrid', 'kalina', 'uehara')
        working_fluid: WorkingFluid instance (required for closed cycles)
        **kwargs: Additional cycle-specific parameters

    Returns:
        ThermodynamicCycle instance
    """

    cycle_type = cycle_type.lower()

    if cycle_type == 'rankine_closed':
        if working_fluid is None:
            raise ValueError("Rankine closed cycle requires a working fluid")
        return RankineClosedCycle(working_fluid)

    elif cycle_type == 'rankine_open':
        return RankineOpenCycle()

    elif cycle_type == 'rankine_hybrid':
        if working_fluid is None:
            raise ValueError("Rankine hybrid cycle requires a working fluid")
        return RankineHybridCycle(working_fluid)

    elif cycle_type == 'kalina':
        # Kalina cycle creates its own NH3-H2O mixture
        concentration = kwargs.get('ammonia_concentration', 0.7)
        return KalinaCycle(concentration)

    elif cycle_type == 'uehara':
        # Uehara cycle creates its own NH3-H2O mixture (like Kalina)
        concentration = kwargs.get('ammonia_concentration', 0.7)
        return UeharaCycle(concentration)

    else:
        raise ValueError(f"Unknown cycle type: {cycle_type}")


if __name__ == "__main__":
    # Test the thermodynamic cycles module
    from working_fluids import get_working_fluid

    print("Testing Thermodynamic Cycles Module\n")
    print("="*60)

    # Get ammonia working fluid
    nh3 = get_working_fluid('ammonia', use_coolprop=True)

    # Create Rankine closed cycle
    cycle = RankineClosedCycle(nh3)

    # Test conditions (similar to original OTEX)
    T_WW_in = 26.0  # °C
    T_CW_in = 5.0   # °C
    T_evap = 23.0   # °C (T_WW - dT_WW - T_pinch)
    T_cond = 8.0    # °C (T_CW + dT_CW + T_pinch)

    p_evap = nh3.saturation_pressure(T_evap)
    p_cond = nh3.saturation_pressure(T_cond)

    print(f"\nTest Conditions:")
    print(f"Evaporator: T = {T_evap}°C, P = {p_evap[0]:.3f} bar")
    print(f"Condenser:  T = {T_cond}°C, P = {p_cond[0]:.3f} bar")

    # Mock inputs similar to original code
    inputs = {
        'eff_isen_turb': 0.82,
        'eff_isen_pump': 0.80,
        'eff_pump_NH3_mech': 0.95,
        'rho_NH3': 625,
    }

    # Calculate cycle states
    states = cycle.calculate_cycle_states(T_evap, T_cond, p_evap, p_cond, inputs)

    print(f"\nCycle States:")
    print(f"State 1 (Condenser out): h = {states['h_1'][0]:.2f} kJ/kg, s = {states['s_1'][0]:.4f} kJ/kgK")
    print(f"State 2 (Pump out):      h = {states['h_2'][0]:.2f} kJ/kg, s = {states['s_2'][0]:.4f} kJ/kgK")
    print(f"State 3 (Evaporator out): h = {states['h_3'][0]:.2f} kJ/kg, s = {states['s_3'][0]:.4f} kJ/kgK")
    print(f"State 4 (Turbine out):    h = {states['h_4'][0]:.2f} kJ/kg, s = {states['s_4'][0]:.4f} kJ/kgK")
    print(f"Vapor quality at turbine exit: x = {states['x_4'][0]:.4f}")

    # Calculate mass flow for 136 MW gross
    p_gross = -136000  # kW
    m_fluid = cycle.calculate_mass_flow(p_gross, states)
    print(f"\nMass Flow Rate:")
    print(f"m = {m_fluid[0]:.2f} kg/s for {-p_gross/1000} MW gross power")

    # Calculate pump power
    p_pump = cycle.calculate_pump_power(m_fluid, states, inputs)
    print(f"\nPump Power:")
    print(f"P_pump = {p_pump[0]:.2f} kW")

    # Calculate heat transfer
    Q_evap, Q_cond = cycle.calculate_heat_transfer(m_fluid, states)
    print(f"\nHeat Transfer:")
    print(f"Q_evap = {Q_evap[0]:.2f} kW")
    print(f"Q_cond = {Q_cond[0]:.2f} kW")

    # Calculate efficiency
    W_net = p_gross + p_pump  # Both negative
    eff_thermal = -W_net / Q_evap
    print(f"\nCycle Efficiency:")
    print(f"Thermal efficiency = {eff_thermal[0]*100:.2f}%")

    print("\n" + "="*60)
    print("Testing complete!")
    print("\nNote: Kalina and Uehara cycles require additional implementation.")
    print("They are provided as templates for future development.")
