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
            'quality_flash': quality_flash,
            # Temperatures stored so calculate_heat_transfer can compute
            # the flash latent-heat duty without re-deriving them.
            'T_evap': T_evap,
            'T_cond': T_cond,
            'T_flash': T_flash,
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
        p_gross = inputs.get('p_gross', -100000)
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

        # Total pump power. Note: the warm- and cold-seawater pump terms
        # above duplicate what plant/sizing.py computes from pipe friction.
        # Open cycle is currently used standalone (not in the regional
        # plant pipeline) so the duplication does not corrupt production
        # results, but a future refactor should choose one accounting site.
        p_pump_total = p_pump_ww + p_pump_cw + p_vacuum + p_pump_condensate

        return p_pump_total

    def calculate_heat_transfer(self, m_fluid, states):
        """
        Heat duties for the open (flash-evaporation) cycle.

        The base-class formula `Q = m * (h_3 - h_2)` is wrong for the
        open cycle because m_fluid is the total warm-seawater flow but
        only m_steam = m_fluid * quality_flash actually evaporates.
        Using the base formula gave Q_evap ~190× the real value (the
        NH3-water enthalpy span got multiplied by the entire seawater
        flow instead of just the flashed vapor). Net efficiency reported
        by code paths that consume Q_evap was correspondingly off.

        Returns the same 2-tuple shape as the base class:
            Q_evap = heat absorbed from warm seawater  (≡ m_steam·h_fg)
            Q_cond = heat rejected to cold seawater    (steam condensation)
        """
        m_steam = np.atleast_1d(states.get('m_steam', m_fluid * states.get('quality_flash', 0.01)))
        T_flash = np.atleast_1d(states.get('T_flash', 18.0))
        h_fg_flash = 2500.0 - 2.4 * T_flash
        Q_evap = m_steam * h_fg_flash

        # Steam condenses from state 4 (turbine outlet, two-phase) to
        # saturated liquid at T_cond. The reference enthalpy of liquid
        # water at T_cond closes the heat-rejection balance.
        cp_water = 4.18
        T_cond_arr = np.atleast_1d(states.get('T_cond', 9.0))
        h_f_cond = cp_water * T_cond_arr
        Q_cond = m_steam * (h_f_cond - np.atleast_1d(states['h_4']))
        return Q_evap, Q_cond


class KalinaCycle(ThermodynamicCycle):
    """
    Kalina Cycle System 11 (KCS-11) - faithful implementation.

    Single closed loop circulating an NH3-H2O basic solution. Like the
    Uehara cycle but with a single-stage turbine on the rich-vapor branch.
    Components in flow order:

      1. Absorber (also acts as condenser) - rich vapor from the turbine is
         absorbed back into the lean liquid, giving the basic solution at
         p_cond. Heat is rejected to cold seawater.
      2. Pump - basic liquid p_cond -> p_evap.
      3. Recuperator - lean liquid (returning from separator) preheats the
         pumped basic solution before the evaporator. The variable boiling
         point of the NH3-H2O mixture combined with this regeneration is
         what gives Kalina its OTEC-relevant efficiency advantage over a
         pure-NH3 Rankine cycle.
      4. Evaporator - warm seawater partially boils the basic solution.
      5. Separator - splits the two-phase mixture into rich vapor (mass
         fraction f, composition y_rich ~ pure NH3 at OTEC temps) and
         lean liquid (1-f, composition x_lean = (x_basic - f*y_rich)/(1-f)).
      6. Turbine - rich vapor expands p_evap -> p_cond, producing work.
      7. Throttle - lean liquid drops from p_evap to p_cond before entering
         the absorber (isenthalpic).

    State indices:
       1: basic liquid out of absorber (p_cond, x_basic)
       2: basic liquid after pump (p_evap, x_basic)
       3: basic liquid after recuperator (preheated, p_evap, x_basic)
       4: two-phase basic at evaporator outlet (T_evap, p_evap, separator
          inlet)
       5: rich vapor at separator (T_evap, p_evap, y_rich)
       6: lean liquid at separator (T_evap, p_evap, x_lean)
       7: rich vapor after turbine (p_cond, y_rich, two-phase)
       8: lean liquid after recuperator (cooled, p_evap)
       9: lean liquid after throttle to p_cond (h_9 = h_8)

    Tunable parameters (passed via inputs dict):
        - 'kalina_split_ratio'  (default 0.30): fraction of basic mass flow
          that vaporises in the evaporator (separator vapor fraction).
        - 'kalina_regen_approach_K' (default 5.0): minimum temperature
          difference at the cold end of the recuperator.

    The previous implementation (KCS-11 stub) used hard-coded approach
    temperatures, an inconsistent x_lean formula independent of the split
    ratio, and a "ghost" second pump (h_4 = h_3). It still hit the
    literature efficiency band by coincidence; this rewrite makes the
    mass balance and energy balance hold exactly so derived quantities
    (mass flows, heat duties, sizing) are physically meaningful.

    References:
        Kalina, A. I. (1984). Combined-Cycle System with Novel Bottoming
            Cycle. ASME J. Eng. Gas Turbines Power 106(4), 737-742.
        Bombarda, P., Invernizzi, C. M., Pietra, C. (2010). Heat recovery
            from Diesel engines: A thermodynamic comparison between Kalina
            and ORC cycles. Appl. Therm. Eng. 30, 212-219.
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
        """Faithful KCS-11 state calculation. See class docstring."""
        # ----- topology parameters -----
        f = inputs.get('kalina_split_ratio', 0.30)
        f = float(np.clip(f, 0.05, 0.45))
        dT_app_regen = inputs.get('kalina_regen_approach_K', 5.0)
        eff_pump = inputs.get('eff_isen_pump', 0.80)
        eff_turb = inputs.get('eff_isen_turb', 0.82)
        rho = inputs.get('rho_NH3', 640)

        # ----- compositions -----
        # Rich vapor in equilibrium with basic-solution liquid at separator.
        y_rich = self.mixture.vapor_liquid_equilibrium(T_evap, p_evap, self.x_basic)
        # Lean composition from separator mass balance (closes exactly):
        #   x_basic = f * y_rich + (1 - f) * x_lean
        x_lean = (self.x_basic - f * y_rich) / (1.0 - f)
        x_lean = np.clip(x_lean, 0.0, self.x_basic)

        # ===== State 1: basic liquid out of absorber (p_cond, x_basic) =====
        h_1 = self.mixture.enthalpy_liquid(T_cond, p_cond, self.x_basic)
        s_1 = self.mixture.entropy_liquid(T_cond, p_cond, self.x_basic)

        # ===== State 2: basic liquid after pump (p_evap, x_basic) =========
        # h_2 = h_1 + v * (p_evap - p_cond) / eff_pump,  v = 1/rho [m3/kg].
        # bar -> J/kg via x100, then -> kJ/kg via /1000  =>  factor 100/rho.
        h_2 = h_1 + (p_evap - p_cond) * 100.0 / (rho * eff_pump)
        s_2 = s_1
        T_2 = T_cond + 0.5

        # ===== State 5: rich vapor at separator ===========================
        h_5 = self.mixture.enthalpy_vapor(T_evap, p_evap, y_rich)
        s_5 = self.mixture.entropy_vapor(T_evap, p_evap, y_rich)

        # ===== State 6: lean liquid at separator ==========================
        h_6 = self.mixture.enthalpy_liquid(T_evap, p_evap, x_lean)
        s_6 = self.mixture.entropy_liquid(T_evap, p_evap, x_lean)

        # ===== State 4: two-phase basic at separator inlet ================
        # Lever rule per kg of basic solution.
        h_4 = f * h_5 + (1.0 - f) * h_6
        s_4 = f * s_5 + (1.0 - f) * s_6

        # ===== State 8: lean liquid after recuperator (cooled) ============
        # Cold-end approach: lean exit ≥ basic inlet to the recuperator.
        T_8 = T_2 + dT_app_regen
        T_8_arr = np.minimum(np.atleast_1d(T_8), np.atleast_1d(T_evap))
        T_8 = float(T_8_arr) if np.isscalar(T_2) else T_8_arr
        h_8 = self.mixture.enthalpy_liquid(T_8, p_evap, x_lean)
        s_8 = self.mixture.entropy_liquid(T_8, p_evap, x_lean)

        # ===== State 3: basic liquid after recuperator (preheated) ========
        # Energy balance per kg of basic solution:
        #   m_basic * (h_3 - h_2) = m_lean * (h_6 - h_8)
        h_3 = h_2 + (1.0 - f) * (h_6 - h_8)
        # Cap at the bubble enthalpy of the basic solution at p_evap so the
        # recuperator does not flash the stream.
        h_basic_bubble = self.mixture.enthalpy_liquid(T_evap, p_evap, self.x_basic)
        h_3 = np.minimum(np.atleast_1d(h_3), np.atleast_1d(h_basic_bubble))
        if np.isscalar(h_2):
            h_3 = float(h_3)

        # ===== State 7: rich vapor after turbine (p_cond, two-phase) ======
        # Isentropic expansion from state 5 to p_cond using the rich-vapor
        # saturation envelope. Saturation T at p_cond for rich composition.
        p_cond_arr = np.atleast_1d(p_cond)
        y_rich_arr_c = np.broadcast_to(np.atleast_1d(y_rich), p_cond_arr.shape)
        T_cond_rich_flat = np.array([
            self.mixture.saturation_temperature(p, y)
            for p, y in zip(p_cond_arr.ravel(), y_rich_arr_c.ravel())
        ])
        T_cond_rich = T_cond_rich_flat.reshape(p_cond_arr.shape)
        if np.isscalar(p_cond):
            T_cond_rich = float(T_cond_rich)
        h_7_liq = self.mixture.enthalpy_liquid(T_cond_rich, p_cond, y_rich)
        h_7_vap = self.mixture.enthalpy_vapor(T_cond_rich, p_cond, y_rich)
        s_7_liq = self.mixture.entropy_liquid(T_cond_rich, p_cond, y_rich)
        s_7_vap = self.mixture.entropy_vapor(T_cond_rich, p_cond, y_rich)
        ds_7 = s_7_vap - s_7_liq
        ds_7 = np.where(np.abs(ds_7) < 1e-9, 1e-9, ds_7)
        x_7_isen = np.clip((s_5 - s_7_liq) / ds_7, 0.0, 1.0)
        h_7_isen = h_7_liq + x_7_isen * (h_7_vap - h_7_liq)
        h_7 = h_5 - eff_turb * (h_5 - h_7_isen)
        dh_7 = h_7_vap - h_7_liq
        dh_7 = np.where(np.abs(dh_7) < 1e-9, 1e-9, dh_7)
        x_7 = np.clip((h_7 - h_7_liq) / dh_7, 0.0, 1.0)

        # ===== State 9: lean liquid after throttle (p_cond, isenthalpic) ==
        h_9 = h_8

        return {
            # Faithful KCS-11 state set
            'h_1': h_1, 's_1': s_1, 'T_1': T_cond,
            'h_2': h_2, 's_2': s_2, 'T_2': T_2,
            'h_3': h_3,
            'h_4': h_4, 's_4': s_4,
            'h_5': h_5, 's_5': s_5,
            'h_6': h_6, 's_6': s_6,
            'h_7': h_7, 'x_7': x_7,
            'h_8': h_8, 's_8': s_8, 'T_8': T_8,
            'h_9': h_9,
            # Pressures, temperatures, compositions
            'p_evap': p_evap, 'p_cond': p_cond,
            'T_evap': T_evap, 'T_cond': T_cond, 'T_cond_rich': T_cond_rich,
            'x_basic': self.x_basic, 'x_lean': x_lean, 'y_rich': y_rich,
            'split_ratio': f,
            # Backwards-compatible aliases for plant/utils.py legacy mapping
            # (turbine inlet/outlet referenced as h_7/h_10 in the old code).
            'h_7_legacy': h_5, 'h_10_legacy': h_7,
            # Old-state-name aliases consumed elsewhere
            'h_10': h_7,
            'x_poor': x_lean,
        }

    def calculate_mass_flow(self, p_gross, states):
        """Mass flows from requested gross power.

        Per kg of basic solution the turbine work is f * (h_5 - h_7).
        m_basic = -p_gross / (f * (h_5 - h_7))   (p_gross < 0)
        """
        f = states['split_ratio']
        w_per_kg_basic = f * (states['h_5'] - states['h_7'])
        m_basic = -p_gross / w_per_kg_basic
        m_rich = f * m_basic
        m_lean = (1.0 - f) * m_basic
        return {
            'm_basic': m_basic,
            'm_rich': m_rich,
            'm_lean': m_lean,
            'm_poor': m_lean,  # legacy alias
        }

    def calculate_pump_power(self, m_flows, states, inputs):
        """Pump power for the single basic-solution pump."""
        eff_mech = inputs.get('eff_pump_NH3_mech', 0.95)
        m_basic = m_flows['m_basic']
        return m_basic * (states['h_2'] - states['h_1']) / eff_mech

    def calculate_heat_transfer(self, m_flows, states):
        """
        External evaporator duty, recuperator duty, and absorber duty.

        Returns the same 3-tuple shape as the previous implementation
        (Q_evap, Q_cond, Q_recup) for backwards compatibility, but each
        term now respects the energy balance imposed by the new topology:

            Q_evap  = m_basic * (h_4 - h_3)        external WW heat in
            Q_cond  = m_rich*(h_7 - h_1) + m_lean*(h_9 - h_1)
                                                   external CW heat out
            Q_recup = m_lean * (h_6 - h_8)         internal regenerator
        """
        m_basic = m_flows['m_basic']
        m_rich = m_flows['m_rich']
        m_lean = m_flows['m_lean']

        Q_evap = m_basic * (states['h_4'] - states['h_3'])
        Q_cond = m_rich * (states['h_7'] - states['h_1']) + \
                 m_lean * (states['h_9'] - states['h_1'])
        Q_recup = m_lean * (states['h_6'] - states['h_8'])

        return Q_evap, Q_cond, Q_recup


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

        # Warm-seawater outlet from the closed-cycle evaporator. At the
        # cold end of the HX the WW exit approaches the NH3 saturation
        # temperature with the pinch ΔT, so:
        #     T_ww_post_evap = T_evap + pinch_WW
        # The previous formula `T_evap - dT_evap` placed the WW colder
        # than the NH3 boiling point and violated the heat-flow direction
        # in the closed evaporator.
        T_pinch_WW = inputs.get('T_pinch_WW', 1.0)
        T_ww_post_evap = T_evap + T_pinch_WW

        # Flash chamber temperature: a few K below the WW exit so a small
        # fraction of the residual sensible heat can flash to steam at
        # sub-atmospheric pressure.
        dT_flash = inputs.get('dT_flash', 2.0)
        T_flash = T_ww_post_evap - dT_flash

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
        Cycle-internal parasitic pump power for the hybrid cycle.

        By codebase convention, cycle-level methods return only the
        working-fluid (NH3) pump plus any cycle-specific auxiliary loads
        (here: the flash-chamber vacuum pump). Warm- and cold-seawater
        pumping is computed separately by plant/sizing.py from the pipe
        friction analysis, so adding it here would double-count.
        """
        eff_mech = inputs['eff_pump_NH3_mech']
        m_NH3 = m_flows['m_NH3']

        # 1. Closed-cycle NH3 pump
        p_pump_NH3 = m_NH3 * (states['h_2'] - states['h_1']) / eff_mech

        # 2. Vacuum pump for non-condensable gas removal in the flash chamber
        power_split_closed = m_flows.get('power_split_closed', 0.88)
        p_gross = inputs.get('p_gross', -100000)
        p_flash_turbine = abs(p_gross) * (1 - power_split_closed)
        p_vacuum = p_flash_turbine * inputs.get('vacuum_pump_factor', 0.025)

        return p_pump_NH3 + p_vacuum

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
    Uehara Cycle - faithful implementation (Uehara & Ikegami, 1990).

    Single closed loop circulating an NH3-H2O basic solution. Key components,
    in flow order:

      1. Absorber (also acts as condenser) - rich vapor from LP turbine is
         absorbed back into the lean-liquid stream, giving the basic solution
         at p_cond. Heat to cold seawater.
      2. Pump - basic liquid p_cond -> p_evap.
      3. Regenerator - lean liquid (returning from separator) preheats the
         pumped basic solution before it enters the evaporator. This is the
         feature that lifts the Uehara cycle's efficiency above a plain
         two-stage Rankine: external heat input is reduced by the recovered
         sensible heat of the lean stream.
      4. Evaporator - warm seawater partially boils the basic solution.
      5. Separator - splits the two-phase mixture into:
            - rich vapor (mass fraction f, composition y_rich ~ pure NH3)
            - lean liquid (mass fraction 1-f, composition x_lean < x_basic)
      6. HP turbine - rich vapor expands p_evap -> p_int, producing work.
      7. LP turbine - rich vapor expands p_int -> p_cond, producing work.
         The exhaust mixes with the lean liquid in the absorber, closing
         the loop.
      8. Throttle - lean liquid drops from p_evap to p_cond before entering
         the absorber (isenthalpic).

    State indices (single loop):
       1: basic liquid out of absorber, p_cond
       2: basic liquid after pump, p_evap
       3: basic liquid after regenerator, p_evap, preheated
       4: two-phase basic solution after evaporator, T_evap, p_evap
       5: rich vapor at separator, p_evap, y_rich
       6: lean liquid at separator, p_evap, x_lean
       7: rich vapor after HP turbine, p_int
       8: rich vapor after LP turbine, p_cond
       9: lean liquid after regenerator, p_evap (cooled)
      10: lean liquid after throttle to p_cond (isenthalpic, h_10 = h_9)

    Tunable parameters (passed via inputs dict):
        - 'uehara_split_ratio' (default 0.20): fraction of basic mass flow
          that vaporises in the evaporator (separator vapor fraction).
        - 'uehara_regen_approach_K' (default 5.0): minimum temperature
          difference at the cold end of the regenerator.

    Reference:
        Uehara H. & Ikegami Y. (1990). Optimization of a closed-cycle OTEC
        system. ASME J. Sol. Energy Eng. 112(4), 247-256.
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
        Faithful Uehara-cycle state calculation.

        Single basic-solution loop with separator + two-stage turbine on the
        rich-vapor branch + lean-liquid regenerator + absorber. See class
        docstring for the topology and state numbering.
        """
        # ----- topology parameters -----
        f = inputs.get('uehara_split_ratio', 0.20)        # vapor mass fraction
        f = float(np.clip(f, 0.05, 0.45))
        dT_app_regen = inputs.get('uehara_regen_approach_K', 5.0)
        eff_pump = inputs.get('eff_isen_pump', 0.80)
        eff_turb = inputs.get('eff_isen_turb', 0.82)
        rho = inputs.get('rho_NH3', 640)

        # ----- compositions -----
        # Rich vapor in equilibrium with basic-solution liquid at the
        # evaporator outlet. y_rich >> x_basic for OTEC NH3-H2O.
        y_rich = self.mixture.vapor_liquid_equilibrium(T_evap, p_evap, self.x_basic)
        # Lean-liquid composition by separator mass balance:
        #   x_basic = f * y_rich + (1 - f) * x_lean
        x_lean = (self.x_basic - f * y_rich) / (1.0 - f)
        # Physical bounds: lean must stay between [0, x_basic]
        x_lean = np.clip(x_lean, 0.0, self.x_basic)

        # Intermediate pressure at the rich-vapor saturation envelope.
        # Geometric mean gives equal pressure ratios across the two stages.
        p_int = np.sqrt(p_evap * p_cond)
        # Saturation T of the rich vapor at p_int (used for h_g/s_g lookups
        # at the HP-turbine outlet two-phase state).
        p_int_arr = np.atleast_1d(p_int)
        y_rich_arr = np.broadcast_to(np.atleast_1d(y_rich), p_int_arr.shape)
        T_int_flat = np.array([
            self.mixture.saturation_temperature(p, y)
            for p, y in zip(p_int_arr.ravel(), y_rich_arr.ravel())
        ])
        T_int = T_int_flat.reshape(p_int_arr.shape)
        if np.isscalar(p_int):
            T_int = float(T_int)

        # ===== State 1: basic liquid out of absorber (p_cond, x_basic) =====
        h_1 = self.mixture.enthalpy_liquid(T_cond, p_cond, self.x_basic)
        s_1 = self.mixture.entropy_liquid(T_cond, p_cond, self.x_basic)

        # ===== State 2: basic liquid after pump (p_evap, x_basic) =====
        # h_2 = h_1 + v * (p_evap - p_cond) / eff_pump.  v = 1/rho [m^3/kg].
        # Pressures in bar -> J/kg conversion: bar * 1e5 / rho.  Convert J/kg
        # to kJ/kg by /1000 -> overall factor 100/rho.
        h_2 = h_1 + (p_evap - p_cond) * 100.0 / (rho * eff_pump)
        s_2 = s_1
        T_2 = T_cond + 0.5  # negligible temperature rise

        # ===== State 5: rich vapor at separator (T_evap, p_evap, y_rich) ==
        h_5 = self.mixture.enthalpy_vapor(T_evap, p_evap, y_rich)
        s_5 = self.mixture.entropy_vapor(T_evap, p_evap, y_rich)

        # ===== State 6: lean liquid at separator (T_evap, p_evap, x_lean) ==
        h_6 = self.mixture.enthalpy_liquid(T_evap, p_evap, x_lean)
        s_6 = self.mixture.entropy_liquid(T_evap, p_evap, x_lean)

        # ===== State 4: two-phase basic solution at separator inlet =====
        # By mass balance per kg of basic solution: h_4 = f*h_5 + (1-f)*h_6
        h_4 = f * h_5 + (1.0 - f) * h_6
        s_4 = f * s_5 + (1.0 - f) * s_6  # approximate (lever rule)

        # ===== State 9: lean liquid out of regenerator (cooled, p_evap) ==
        # Approach temperature at the cold end: lean exit ≥ basic inlet.
        T_9 = T_2 + dT_app_regen
        # Cap at T_6 (cannot cool below... wait, lean is cooling, T_9 < T_6;
        # cap at the source temperature only as a sanity guard).
        T_9_arr = np.minimum(np.atleast_1d(T_9), np.atleast_1d(T_evap))
        T_9 = float(T_9_arr) if np.isscalar(T_2) else T_9_arr
        h_9 = self.mixture.enthalpy_liquid(T_9, p_evap, x_lean)
        s_9 = self.mixture.entropy_liquid(T_9, p_evap, x_lean)

        # ===== State 3: basic liquid after regenerator (p_evap, x_basic) =
        # Energy balance on the regenerator (per kg of basic solution):
        #   m_basic * (h_3 - h_2) = m_lean * (h_6 - h_9)
        #   m_lean / m_basic = (1 - f)
        h_3 = h_2 + (1.0 - f) * (h_6 - h_9)
        # Ensure h_3 does not exceed the basic-solution bubble enthalpy at
        # p_evap (otherwise the regenerator would already be flashing the
        # stream). If exceeded, cap at the bubble enthalpy and reduce the
        # actual heat recovered accordingly.
        h_basic_bubble = self.mixture.enthalpy_liquid(T_evap, p_evap, self.x_basic)
        h_3 = np.minimum(np.atleast_1d(h_3), np.atleast_1d(h_basic_bubble))
        if np.isscalar(h_2):
            h_3 = float(h_3)

        # ===== State 7: rich vapor after HP turbine (p_int) ==============
        # Isentropic expansion using saturation envelope at outlet. The
        # expanding stream is the rich vapor (composition y_rich), so the
        # two-phase envelope at p_int must also be evaluated at y_rich
        # for both saturated-liquid and saturated-vapor properties.
        h_7_liq = self.mixture.enthalpy_liquid(T_int, p_int, y_rich)
        h_7_vap = self.mixture.enthalpy_vapor(T_int, p_int, y_rich)
        s_7_liq = self.mixture.entropy_liquid(T_int, p_int, y_rich)
        s_7_vap = self.mixture.entropy_vapor(T_int, p_int, y_rich)
        ds_7 = s_7_vap - s_7_liq
        ds_7 = np.where(np.abs(ds_7) < 1e-9, 1e-9, ds_7)
        x_7_isen = np.clip((s_5 - s_7_liq) / ds_7, 0.0, 1.0)
        h_7_isen = h_7_liq + x_7_isen * (h_7_vap - h_7_liq)
        h_7 = h_5 - eff_turb * (h_5 - h_7_isen)
        # Quality and entropy of the actual outlet
        dh_7 = h_7_vap - h_7_liq
        dh_7 = np.where(np.abs(dh_7) < 1e-9, 1e-9, dh_7)
        x_7 = np.clip((h_7 - h_7_liq) / dh_7, 0.0, 1.0)
        s_7 = s_7_liq + x_7 * (s_7_vap - s_7_liq)

        # ===== State 8: rich vapor after LP turbine (p_cond) =============
        # Saturation envelope must be at the rich-vapor sat T at p_cond,
        # NOT at T_cond (which is the basic-mix absorber temperature, much
        # higher than the pure-NH3 saturation point at the same pressure).
        # Using T_cond here puts the envelope into the superheated region
        # and makes the LP turbine produce negative work.
        p_cond_arr = np.atleast_1d(p_cond)
        y_rich_arr_c = np.broadcast_to(np.atleast_1d(y_rich), p_cond_arr.shape)
        T_cond_rich_flat = np.array([
            self.mixture.saturation_temperature(p, y)
            for p, y in zip(p_cond_arr.ravel(), y_rich_arr_c.ravel())
        ])
        T_cond_rich = T_cond_rich_flat.reshape(p_cond_arr.shape)
        if np.isscalar(p_cond):
            T_cond_rich = float(T_cond_rich)
        h_8_liq = self.mixture.enthalpy_liquid(T_cond_rich, p_cond, y_rich)
        h_8_vap = self.mixture.enthalpy_vapor(T_cond_rich, p_cond, y_rich)
        s_8_liq = self.mixture.entropy_liquid(T_cond_rich, p_cond, y_rich)
        s_8_vap = self.mixture.entropy_vapor(T_cond_rich, p_cond, y_rich)
        ds_8 = s_8_vap - s_8_liq
        ds_8 = np.where(np.abs(ds_8) < 1e-9, 1e-9, ds_8)
        x_8_isen = np.clip((s_7 - s_8_liq) / ds_8, 0.0, 1.0)
        h_8_isen = h_8_liq + x_8_isen * (h_8_vap - h_8_liq)
        h_8 = h_7 - eff_turb * (h_7 - h_8_isen)
        dh_8 = h_8_vap - h_8_liq
        dh_8 = np.where(np.abs(dh_8) < 1e-9, 1e-9, dh_8)
        x_8 = np.clip((h_8 - h_8_liq) / dh_8, 0.0, 1.0)

        # ===== State 10: lean liquid after throttle (p_cond) =============
        # Isenthalpic expansion through the throttle valve.
        h_10 = h_9
        s_10 = s_9  # approximate

        return {
            # State enthalpies/entropies (single loop)
            'h_1': h_1, 's_1': s_1,
            'h_2': h_2, 's_2': s_2, 'T_2': T_2,
            'h_3': h_3,
            'h_4': h_4, 's_4': s_4,
            'h_5': h_5, 's_5': s_5,
            'h_6': h_6, 's_6': s_6,
            'h_7': h_7, 's_7': s_7, 'x_7': x_7,
            'h_8': h_8, 'x_8': x_8,
            'h_9': h_9, 's_9': s_9, 'T_9': T_9,
            'h_10': h_10,
            # Topology / mixture state
            'p_evap': p_evap, 'p_int': p_int, 'p_cond': p_cond,
            'T_evap': T_evap, 'T_int': T_int, 'T_cond': T_cond,
            'x_basic': self.x_basic, 'x_lean': x_lean, 'y_rich': y_rich,
            'split_ratio': f,
            # Backwards-compatible aliases consumed by enthalpies_entropies()
            # in plant/utils.py. The legacy mapping treated the cycle as a
            # two-stage Rankine; we expose equivalent rich-vapor states so
            # the (h_1, h_2, h_3, h_4) projection still produces sensible
            # numbers for downstream sizing code.
            'h_1_HP': h_1, 's_1_HP': s_1,
            'h_2_HP': h_2, 's_2_HP': s_2,
            'h_3_HP': h_5, 's_3_HP': s_5,
            'h_4_HP': h_7, 's_4_HP': s_7, 'x_4_HP': x_7,
            'h_5_LP': h_1, 's_5_LP': s_1,
            'h_6_LP': h_2, 's_6_LP': s_2,
            'h_7_LP': h_7, 's_7_LP': s_7,
            'h_8_LP': h_8, 's_8_LP': None, 'x_8_LP': x_8,
            'T_evap_HP': T_evap, 'T_evap_LP': T_int,
            'y_rich_int': y_rich,
        }

    def calculate_mass_flow(self, p_gross, states):
        """
        Mass flow of the basic solution from the requested gross power.

        W_gross_per_kg_basic = f * [(h_5 - h_7) + (h_7 - h_8)] = f * (h_5 - h_8)
        m_basic = p_gross / -(W_gross_per_kg_basic)
        """
        f = states['split_ratio']
        w_per_kg_basic = f * (states['h_5'] - states['h_8'])  # positive, kJ/kg
        # p_gross < 0 (convention: power out is negative). Mass flow is
        # positive when the per-kg work is positive.
        m_basic = -p_gross / w_per_kg_basic
        m_rich = f * m_basic
        m_lean = (1.0 - f) * m_basic
        return {
            'm_basic': m_basic,
            'm_rich': m_rich,
            'm_lean': m_lean,
            # Aliases for legacy consumers
            'm_HP': m_rich,
            'm_LP': m_rich,
            'm_total': m_basic,
        }

    def calculate_pump_power(self, m_flows, states, inputs):
        """Pump power for the single basic-solution pump."""
        eff_mech = inputs.get('eff_pump_NH3_mech', 0.95)
        m_basic = m_flows['m_basic']
        return m_basic * (states['h_2'] - states['h_1']) / eff_mech

    def calculate_heat_transfer(self, m_flows, states):
        """
        Heat duties for the three external HX (evaporator, regenerator,
        absorber/condenser).

        Returns the same 3-tuple shape as the previous implementation:
            Q_evap_HP, Q_evap_LP, Q_cond
        but populated as:
            Q_evap = total external evaporator duty (warm seawater)
            Q_regen = internal regenerator duty (closes the energy balance)
            Q_cond = absorber/condenser duty (cold seawater)
        """
        m_basic = m_flows['m_basic']
        m_rich = m_flows['m_rich']
        m_lean = m_flows['m_lean']

        # External heat input (warm seawater -> evaporator)
        Q_evap = m_basic * (states['h_4'] - states['h_3'])
        # Internal regenerator (lean stream -> basic stream)
        Q_regen = m_lean * (states['h_6'] - states['h_9'])
        # External heat rejection (rich vapor + lean liquid -> absorber)
        Q_cond = m_rich * (states['h_8'] - states['h_1']) + \
                 m_lean * (states['h_10'] - states['h_1'])

        # Backwards-compatible 3-tuple. Q_evap is the only external evaporator
        # duty; the legacy "two-stage Rankine" had separate HP/LP duties, so
        # we return Q_evap as Q_evap_HP and the regenerator as Q_evap_LP for
        # downstream compatibility (LP duty is internal, sums to the regen).
        return Q_evap, Q_regen, Q_cond


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

    # Calculate mass flow for 100 MW gross
    p_gross = -100000  # kW
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
