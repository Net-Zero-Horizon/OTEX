# -*- coding: utf-8 -*-
"""
OTEX Configuration Module
Centralized configuration management using dataclasses.

All configurable parameters for OTEC plant design, simulation, and analysis
are defined here with sensible defaults.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Tuple, Dict, Any
import numpy as np


@dataclass
class PhysicalProperties:
    """Physical properties of fluids and materials."""

    # Working fluid (NH3)
    rho_NH3: float = 625.0              # kg/m³ - Liquid ammonia density

    # Seawater
    rho_WW: float = 1024.0              # kg/m³ - Warm seawater density
    rho_CW: float = 1027.0              # kg/m³ - Cold seawater density
    cp_water: float = 4.0               # kJ/kg·K - Seawater specific heat

    # Pipe materials
    roughness_pipe: float = 0.03        # mm - Pipe roughness
    rho_pipe_hdpe: float = 995.0        # kg/m³ - HDPE pipe density
    rho_pipe_frp: float = 1016.0        # kg/m³ - FRP sandwich pipe density


@dataclass
class HeatTransfer:
    """Heat transfer coefficients and temperature differences."""

    # Overall heat transfer coefficients
    U_evap: float = 4.5                 # kW/m²K - Evaporator
    U_cond: float = 3.5                 # kW/m²K - Condenser

    # Pinch point temperature differences
    T_pinch_evap: float = 1.0           # °C - Evaporator pinch point
    T_pinch_cond: float = 1.0           # °C - Condenser pinch point

    # Heat exchanger pressure drop
    K_L: float = 100.0                  # Pressure drop coefficient (dimensionless)


@dataclass
class TemperatureDeltas:
    """Temperature difference ranges for optimization loops."""

    # Warm water temperature drop range
    dT_WW_min: float = 2.0              # °C - Minimum ΔT warm water
    dT_WW_max: float = 5.0              # °C - Maximum ΔT warm water

    # Cold water temperature rise range
    dT_CW_min: float = 2.0              # °C - Minimum ΔT cold water
    dT_CW_max: float = 5.0              # °C - Maximum ΔT cold water

    # Loop interval
    interval: float = 0.5               # °C - Step size for optimization loops


@dataclass
class Efficiencies:
    """Component efficiencies."""

    # Turbine
    turbine_isentropic: float = 0.82    # Isentropic efficiency
    turbine_mechanical: float = 0.95    # Mechanical efficiency
    turbine_electrical: float = 0.95    # Electrical/generator efficiency

    # NH3 pump
    pump_isentropic: float = 0.80       # Isentropic efficiency
    pump_mechanical: float = 0.95       # Mechanical efficiency

    # Seawater pumps
    sw_pump_hydraulic: float = 0.80     # Hydraulic efficiency
    sw_pump_electrical: float = 0.95    # Electrical efficiency

    # Transmission (set dynamically based on distance)
    transmission: float = 0.0           # Placeholder, updated during analysis


@dataclass
class SeawaterPipes:
    """Seawater pipe configuration."""

    # Warm water pipes
    ww_inlet_length: float = 21.6       # m - Inlet pipe length
    ww_outlet_length: float = 60.0      # m - Outlet pipe length
    ww_depth: float = 20.0              # m - Intake depth

    # Cold water pipes
    cw_inlet_length: float = 1062.4     # m - Inlet pipe length (default 1000m depth)
    cw_outlet_length: float = 60.0      # m - Outlet pipe length
    cw_depth: float = 1000.0            # m - Intake depth

    # Pipe design parameters
    SDR_ratio: float = 16.0             # Standard Dimension Ratio (D/t)
    max_diameter: float = 8.0           # m - Maximum pipe inner diameter
    max_pressure_drop: float = 100.0    # kPa - Maximum allowed pressure drop
    nominal_velocity: float = 2.1       # m/s - Design flow velocity
    hx_velocity: float = 1.05           # m/s - Heat exchanger flow velocity

    @property
    def ww_total_length(self) -> float:
        """Total warm water pipe length."""
        return self.ww_inlet_length + self.ww_outlet_length

    @property
    def cw_total_length(self) -> float:
        """Total cold water pipe length."""
        return self.cw_inlet_length + self.cw_outlet_length


@dataclass
class DepthLimits:
    """Depth constraints for cold water intake."""

    min_depth: float = 600.0            # m - Minimum CW intake depth
    max_depth: float = 3000.0           # m - Maximum CW intake depth (mooring limit)

    @property
    def optimization_range(self) -> Tuple[float, float]:
        """Range for depth optimization (min, max)."""
        return (self.min_depth, self.max_depth)


@dataclass
class Economics:
    """Economic parameters for LCOE calculation."""

    lifetime_years: int = 30            # Plant lifetime
    discount_rate: float = 0.10         # Discount rate (10%)
    availability: float = 0.914         # Capacity factor (8000/8760 hours)

    # Cost level affects pipe material and component costs
    cost_level: Literal['low_cost', 'high_cost'] = 'low_cost'

    # Transmission
    threshold_AC_DC: float = 50.0       # km - Distance threshold for DC vs AC

    @property
    def crf(self) -> float:
        """Capital Recovery Factor."""
        r = self.discount_rate
        n = self.lifetime_years
        return r * (1 + r)**n / ((1 + r)**n - 1)


@dataclass
class CycleConfig:
    """Thermodynamic cycle configuration."""

    cycle_type: Literal[
        'rankine_closed',
        'rankine_open',
        'rankine_hybrid',
        'kalina',
        'uehara'
    ] = 'rankine_closed'

    fluid_type: Literal[
        'ammonia',
        'r134a',
        'r245fa',
        'propane',
        'isobutane'
    ] = 'ammonia'

    use_coolprop: bool = True           # Use CoolProp for fluid properties
    ammonia_concentration: float = 0.7  # NH3 mass fraction for Kalina/Uehara


@dataclass
class PlantConfig:
    """OTEC plant configuration."""

    gross_power: float = -136000.0      # kW (negative = power output)
    installation_type: Literal['onshore', 'offshore'] = 'offshore'
    optimize_depth: bool = False        # Optimize CW intake depth


@dataclass
class DataConfig:
    """Data source configuration."""

    source: Literal['CMEMS', 'HYCOM'] = 'CMEMS'
    time_resolution: str = '24H'

    # Year and date range (auto-computed from year if not specified)
    year: int = 2020
    date_start: Optional[str] = None
    date_end: Optional[str] = None

    # CMEMS specific
    cmems_time_origin: str = '1950-01-01 00:00:00'

    # HYCOM specific
    hycom_glb: str = 'GLBu0.08'
    hycom_horizontal_stride: int = 3
    hycom_time_origin: str = '2000-01-01 00:00:00'

    def __post_init__(self):
        """Auto-compute date_start and date_end from year if not specified."""
        if self.date_start is None:
            self.date_start = f'{self.year}-01-01 00:00:00'
        if self.date_end is None:
            self.date_end = f'{self.year}-12-31 21:00:00'


@dataclass
class OTEXConfig:
    """
    Complete OTEX configuration.

    This is the main configuration class that aggregates all parameter groups.
    Use get_default_config() to create an instance with default values.

    Example:
        >>> config = OTEXConfig()
        >>> config.cycle.cycle_type = 'kalina'
        >>> config.economics.discount_rate = 0.08
        >>> inputs = config.to_legacy_dict()  # For compatibility with existing code
    """

    physical: PhysicalProperties = field(default_factory=PhysicalProperties)
    heat_transfer: HeatTransfer = field(default_factory=HeatTransfer)
    temperature_deltas: TemperatureDeltas = field(default_factory=TemperatureDeltas)
    efficiencies: Efficiencies = field(default_factory=Efficiencies)
    pipes: SeawaterPipes = field(default_factory=SeawaterPipes)
    depth_limits: DepthLimits = field(default_factory=DepthLimits)
    economics: Economics = field(default_factory=Economics)
    cycle: CycleConfig = field(default_factory=CycleConfig)
    plant: PlantConfig = field(default_factory=PlantConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to nested dictionary."""
        return asdict(self)

    def _create_working_fluid(self):
        """Create working fluid instance based on cycle configuration."""
        from otex.core.fluids import get_working_fluid
        # Open cycles and mixture-based cycles (kalina, uehara) don't use external working fluids
        if self.cycle.cycle_type in ('rankine_open', 'kalina', 'uehara'):
            return None
        return get_working_fluid(self.cycle.fluid_type, self.cycle.use_coolprop)

    def _create_thermodynamic_cycle(self):
        """Create thermodynamic cycle instance based on configuration."""
        from otex.core.cycles import get_thermodynamic_cycle
        wf = self._create_working_fluid()
        kwargs = {}
        if self.cycle.cycle_type in ('kalina', 'uehara'):
            kwargs['ammonia_concentration'] = self.cycle.ammonia_concentration
        return get_thermodynamic_cycle(self.cycle.cycle_type, working_fluid=wf, **kwargs)

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to legacy dictionary format for compatibility with existing code.

        Returns a dictionary matching the structure returned by the old
        parameters_and_constants() function.
        """
        # Pipe material based on cost level
        rho_pipe = (self.physical.rho_pipe_hdpe
                   if self.economics.cost_level == 'low_cost'
                   else self.physical.rho_pipe_frp)

        # Build legacy format
        legacy = {
            # Physical properties
            'rho_NH3': self.physical.rho_NH3,
            'rho_WW': self.physical.rho_WW,
            'rho_CW': self.physical.rho_CW,
            'cp_water': self.physical.cp_water,
            'fluid_properties': [
                self.physical.rho_NH3,
                self.physical.rho_WW,
                self.physical.rho_CW,
                self.physical.cp_water
            ],
            'roughness_pipe': self.physical.roughness_pipe,
            'rho_pipe': rho_pipe,
            'pipe_material': [rho_pipe, self.physical.roughness_pipe],

            # Depth limits
            'min_depth': -self.depth_limits.min_depth,
            'max_depth': -self.depth_limits.max_depth,

            # Temperatures
            'T_pinch_WW': self.heat_transfer.T_pinch_evap,
            'T_pinch_CW': self.heat_transfer.T_pinch_cond,
            'del_T_WW_min': int(self.temperature_deltas.dT_WW_min * 10),
            'del_T_CW_min': int(self.temperature_deltas.dT_CW_min * 10),
            'del_T_WW_max': int(self.temperature_deltas.dT_WW_max * 10),
            'del_T_CW_max': int(self.temperature_deltas.dT_CW_max * 10),
            'interval_WW': int(self.temperature_deltas.interval * 10),
            'interval_CW': int(self.temperature_deltas.interval * 10),
            'del_T_for_looping': [
                int(self.temperature_deltas.dT_WW_min * 10),
                int(self.temperature_deltas.dT_CW_min * 10),
                int(self.temperature_deltas.dT_WW_max * 10),
                int(self.temperature_deltas.dT_CW_max * 10),
                int(self.temperature_deltas.interval * 10),
                int(self.temperature_deltas.interval * 10),
            ],
            'temperatures': [
                self.heat_transfer.T_pinch_evap,
                self.heat_transfer.T_pinch_cond,
                [
                    int(self.temperature_deltas.dT_WW_min * 10),
                    int(self.temperature_deltas.dT_CW_min * 10),
                    int(self.temperature_deltas.dT_WW_max * 10),
                    int(self.temperature_deltas.dT_CW_max * 10),
                    int(self.temperature_deltas.interval * 10),
                    int(self.temperature_deltas.interval * 10),
                ]
            ],

            # Heat transfer
            'U_evap': self.heat_transfer.U_evap,
            'U_cond': self.heat_transfer.U_cond,
            'U': [self.heat_transfer.U_evap, self.heat_transfer.U_cond],
            'K_L': self.heat_transfer.K_L,

            # Pipes
            'length_WW': self.pipes.ww_total_length,
            'length_CW': self.pipes.cw_total_length,
            'length_WW_inlet': self.pipes.ww_inlet_length,
            'length_WW_outlet': self.pipes.ww_outlet_length,
            'length_CW_inlet': self.pipes.cw_inlet_length,
            'length_CW_outlet': self.pipes.cw_outlet_length,
            'SDR_ratio': self.pipes.SDR_ratio,
            'u_pipes': self.pipes.nominal_velocity,
            'u_HX': self.pipes.hx_velocity,
            'pressure_drop_nom': self.pipes.max_pressure_drop,
            'max_d': self.pipes.max_diameter,
            'max_p': self.pipes.max_pressure_drop,
            'pipe_properties': [
                self.pipes.ww_total_length,
                self.pipes.cw_total_length,
                self.pipes.SDR_ratio,
                self.heat_transfer.K_L,
                self.pipes.nominal_velocity,
                self.pipes.hx_velocity,
                self.pipes.max_pressure_drop,
                self.pipes.max_diameter,
                self.pipes.max_pressure_drop,
            ],

            # Efficiencies
            'eff_isen_turb': self.efficiencies.turbine_isentropic,
            'eff_isen_pump': self.efficiencies.pump_isentropic,
            'eff_pump_NH3_mech': self.efficiencies.pump_mechanical,
            'eff_turb_el': self.efficiencies.turbine_electrical,
            'eff_turb_mech': self.efficiencies.turbine_mechanical,
            'eff_trans': self.efficiencies.transmission,
            'eff_hyd': self.efficiencies.sw_pump_hydraulic,
            'eff_el': self.efficiencies.sw_pump_electrical,
            'efficiencies': [
                self.efficiencies.turbine_isentropic,
                self.efficiencies.pump_isentropic,
                self.efficiencies.pump_mechanical,
                self.efficiencies.turbine_electrical,
                self.efficiencies.turbine_mechanical,
                self.efficiencies.transmission,
                self.efficiencies.sw_pump_hydraulic,
                self.efficiencies.sw_pump_electrical,
            ],

            # Economics
            'lifetime': self.economics.lifetime_years,
            'discount_rate': self.economics.discount_rate,
            'crf': self.economics.crf,
            'availability_factor': self.economics.availability,
            'threshold_AC_DC': self.economics.threshold_AC_DC,
            'cost_level': self.economics.cost_level,
            'economic_inputs': [
                self.economics.lifetime_years,
                self.economics.crf,
                self.economics.discount_rate,
                self.economics.availability,
            ],

            # Plant
            'p_gross': self.plant.gross_power,
            'installation_type': self.plant.installation_type,

            # Cycle configuration
            'fluid_type': self.cycle.fluid_type,
            'cycle_type': self.cycle.cycle_type,
            'use_coolprop': self.cycle.use_coolprop,
            'ammonia_concentration': self.cycle.ammonia_concentration,
            'optimize_depth': self.plant.optimize_depth,
            'depth_optimization_range': self.depth_limits.optimization_range,

            # Working fluid and cycle instances (auto-created)
            'working_fluid': self._create_working_fluid(),
            'thermodynamic_cycle': self._create_thermodynamic_cycle(),

            # Config strings for unique file naming
            'config_cycle_type': self.cycle.cycle_type,
            'config_fluid_type': self.cycle.fluid_type,

            # Data configuration
            'data': self.data.source,
            't_resolution': self.data.time_resolution,
            'time_origin': (self.data.cmems_time_origin
                           if self.data.source == 'CMEMS'
                           else self.data.hycom_time_origin),

            # Date range
            'year': self.data.year,
            'date_start': self.data.date_start,
            'date_end': self.data.date_end,
        }

        return legacy


def get_default_config(**kwargs) -> OTEXConfig:
    """
    Create an OTEXConfig with default values, optionally overriding specific parameters.

    Args:
        **kwargs: Override specific top-level config sections or individual parameters.
                  Examples:
                  - cycle=CycleConfig(cycle_type='kalina')
                  - gross_power=-50000 (convenience alias)

    Returns:
        OTEXConfig instance

    Example:
        >>> config = get_default_config()
        >>> config = get_default_config(cycle=CycleConfig(cycle_type='kalina'))
    """
    config = OTEXConfig()

    # Handle convenience aliases
    if 'gross_power' in kwargs:
        config.plant.gross_power = kwargs.pop('gross_power')
    if 'cycle_type' in kwargs:
        config.cycle.cycle_type = kwargs.pop('cycle_type')
    if 'fluid_type' in kwargs:
        config.cycle.fluid_type = kwargs.pop('fluid_type')
    if 'cost_level' in kwargs:
        config.economics.cost_level = kwargs.pop('cost_level')
    if 'year' in kwargs:
        config.data.year = kwargs.pop('year')
        # Recompute dates after year change
        config.data.date_start = f'{config.data.year}-01-01 00:00:00'
        config.data.date_end = f'{config.data.year}-12-31 21:00:00'

    # Handle section overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


# Legacy compatibility function
def parameters_and_constants(
    p_gross: float = -136000,
    cost_level: str = 'low_cost',
    data: str = 'CMEMS',
    fluid_type: str = 'ammonia',
    cycle_type: str = 'rankine_closed',
    use_coolprop: bool = True,
    optimize_depth: bool = False,
    year: int = 2020
) -> Dict[str, Any]:
    """
    Legacy compatibility function.

    Creates an OTEXConfig and returns the legacy dictionary format.
    New code should use OTEXConfig directly.

    Args:
        p_gross: Gross power output in kW (negative = power output)
        cost_level: 'low_cost' or 'high_cost'
        data: Data source ('CMEMS' or 'HYCOM')
        fluid_type: Working fluid type
        cycle_type: Thermodynamic cycle type
        use_coolprop: Whether to use CoolProp for fluid properties
        optimize_depth: Whether to optimize cold water intake depth
        year: Year for analysis (date_start/date_end auto-computed)

    Returns:
        Dictionary with all configuration parameters
    """
    config = OTEXConfig(
        plant=PlantConfig(gross_power=p_gross, optimize_depth=optimize_depth),
        economics=Economics(cost_level=cost_level),
        cycle=CycleConfig(
            cycle_type=cycle_type,
            fluid_type=fluid_type,
            use_coolprop=use_coolprop
        ),
        data=DataConfig(source=data, year=year)
    )

    return config.to_legacy_dict()
