# -*- coding: utf-8 -*-
"""
OTEX Configuration Module
Centralized configuration management using dataclasses.

All configurable parameters for OTEC plant design, simulation, and analysis
are defined here with sensible defaults.
"""

import calendar
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Tuple, Dict, Any, Union, List
import numpy as np

from .economics.cost_schemes import CostScheme
from .economics.degradation import DegradationConfig, OpexEscalationConfig


def hours_in_year(year: int) -> int:
    """Return 8784 for leap years, 8760 otherwise."""
    return 8784 if calendar.isleap(year) else 8760


def hours_in_span(year_start: int, year_end: int) -> int:
    """Total hours across an inclusive year range, accounting for leap years."""
    return sum(hours_in_year(y) for y in range(year_start, year_end + 1))


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

    # Cost level affects pipe material and component costs.
    # Accepts built-in scheme names ('low_cost', 'high_cost') or a CostScheme object.
    cost_level: Union[Literal['low_cost', 'high_cost'], CostScheme] = 'low_cost'

    # Transmission
    threshold_AC_DC: float = 50.0       # km - Distance threshold for DC vs AC

    # Multi-year NPV controls (added in 0.2.0). Default ``constant`` /
    # ``flat`` reproduce the legacy single-rate behaviour.
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    opex_escalation: OpexEscalationConfig = field(default_factory=OpexEscalationConfig)

    @property
    def crf(self) -> float:
        """Capital Recovery Factor (legacy single-rate annualisation).

        Retained for backward compatibility. The 0.2.0 NPV LCOE bypasses
        ``crf`` and discounts annual cashflows directly.
        """
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

    gross_power: float = -100000.0      # kW (negative = power output)
    installation_type: Literal['onshore', 'offshore'] = 'offshore'
    optimize_depth: bool = False        # Optimize CW intake depth


@dataclass
class DataConfig:
    """Data source configuration.

    Multi-year simulations are configured via ``year_start`` and ``year_end``
    (inclusive). The legacy ``year`` parameter is still accepted and is
    equivalent to ``year_start = year_end = year``.
    """

    source: Literal['CMEMS', 'HYCOM'] = 'CMEMS'
    # pandas >= 2.2 deprecated 'H' in favour of 'h'; use the new spelling.
    time_resolution: str = '24h'

    # Year range. If only `year` is given, year_start = year_end = year.
    # If neither is given, defaults to 2020.
    year: Optional[int] = None
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None

    # CMEMS specific
    cmems_time_origin: str = '1950-01-01 00:00:00'

    # HYCOM specific
    hycom_glb: str = 'GLBy0.08'
    hycom_horizontal_stride: int = 3
    hycom_time_origin: str = '2000-01-01 00:00:00'

    def __post_init__(self):
        """Resolve year/year_start/year_end and auto-compute date range."""
        if self.year is not None:
            warnings.warn(
                "DataConfig.year is deprecated since 0.2.0 and will be removed "
                "in a future release; use year_start/year_end instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.year_start is None:
                self.year_start = self.year
            if self.year_end is None:
                self.year_end = self.year

        if self.year_start is None:
            self.year_start = 2020
        if self.year_end is None:
            self.year_end = self.year_start

        if self.year_end < self.year_start:
            raise ValueError(
                f"year_end ({self.year_end}) must be >= year_start ({self.year_start})"
            )

        # Keep `year` populated for legacy consumers; equals year_start.
        self.year = self.year_start

        if self.date_start is None:
            self.date_start = f'{self.year_start}-01-01 00:00:00'
        if self.date_end is None:
            self.date_end = f'{self.year_end}-12-31 21:00:00'

    @property
    def n_years(self) -> int:
        """Number of simulated years (inclusive)."""
        return self.year_end - self.year_start + 1

    @property
    def years(self) -> List[int]:
        """List of simulated calendar years."""
        return list(range(self.year_start, self.year_end + 1))

    @property
    def hours_total(self) -> int:
        """Total hours over the simulation span, accounting for leap years."""
        return hours_in_span(self.year_start, self.year_end)

    @property
    def year_label(self) -> str:
        """String label for filenames: '2020' for single year, '2020-2022' for range."""
        if self.year_start == self.year_end:
            return str(self.year_start)
        return f'{self.year_start}-{self.year_end}'


@dataclass
class SitingConfig:
    """Site-screening configuration: protected areas, shipping lanes, hazards."""

    # Master switches
    enable_mpa_filter: bool = False        # Exclude WDPA IUCN I-IV
    enable_ais_filter: bool = False        # Exclude high-traffic shipping lanes
    enable_hazard_costs: bool = False      # Apply seismic + cyclone cost multipliers

    # Buffers (km) applied around polygons/lines before point-in-buffer test
    mpa_buffer_km: float = 5.0
    ais_buffer_km: float = 5.0

    # AIS exclusion percentile: density above this percentile excludes the site
    ais_exclusion_pct: float = 95.0

    # Multiplier weights. Final factor: 1 + w * normalized_value, value in [0, 1]
    w_ais: float = 0.20            # Applied to BOTH CAPEX and OPEX
    w_seismic: float = 0.15        # Applied to CAPEX only
    w_cyclone: float = 0.25        # Applied to OPEX only

    # Normalization references (values >= ref clamp to 1.0)
    pga_ref_g: float = 0.4                 # PGA at 475-yr return period [g]
    cyclone_ref_per_yr: float = 0.5        # cyclone tracks/yr in 100km radius

    # Cache & refresh
    cache_dir: Optional[str] = None        # Defaults to ~/.otex/siting_cache
    refresh: bool = False                  # If True, re-download even if cached


@dataclass
class ClimateConfig:
    """CMIP6 climate-scenario configuration (added in 0.3.0).

    When ``scenario != 'historical'`` and ``target_year`` is set, OTEX
    pulls thetao deltas from the CMIP6 ensemble (see
    :mod:`otex.data.climate`) and adds them to the CMEMS time series
    before the off-design analysis runs. ``historical`` is the
    explicit no-op default — behaviour is identical to 0.2.0.
    """

    scenario: Literal['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585'] = 'historical'
    target_year: Optional[int] = None
    models: Tuple[str, ...] = ('MPI-ESM1-2-LR', 'EC-Earth3', 'CanESM5')

    # Reference baseline (IPCC AR6 standard).
    baseline_start: int = 1995
    baseline_end: int = 2014
    # Future window centred on `target_year`.
    future_window_years: int = 30

    @property
    def enabled(self) -> bool:
        """True iff a non-historical scenario delta will be applied."""
        return self.scenario != 'historical' and self.target_year is not None

    @property
    def label(self) -> str:
        """Filename-friendly label, e.g. ``'ssp245_2050'`` or ``'historical'``."""
        if self.enabled:
            return f'{self.scenario}_{int(self.target_year)}'
        return 'historical'


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
    siting: SitingConfig = field(default_factory=SitingConfig)
    climate: ClimateConfig = field(default_factory=ClimateConfig)

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
        if isinstance(self.economics.cost_level, CostScheme):
            rho_pipe = self.economics.cost_level.pipe_density
        elif self.economics.cost_level == 'low_cost':
            rho_pipe = self.physical.rho_pipe_hdpe
        else:
            rho_pipe = self.physical.rho_pipe_frp

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

            # Multi-year NPV controls (0.2.0+). The full configs are passed
            # through so that economics functions can call into degradation
            # and OPEX escalation models without re-importing config.
            'degradation_config': self.economics.degradation,
            'opex_escalation_config': self.economics.opex_escalation,

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
            'year': self.data.year_start,            # legacy alias
            'year_start': self.data.year_start,
            'year_end': self.data.year_end,
            'years': self.data.years,
            'n_years': self.data.n_years,
            'hours_total': self.data.hours_total,
            'year_label': self.data.year_label,
            'date_start': self.data.date_start,
            'date_end': self.data.date_end,

            # Siting
            'siting_enable_mpa_filter': self.siting.enable_mpa_filter,
            'siting_enable_ais_filter': self.siting.enable_ais_filter,
            'siting_enable_hazard_costs': self.siting.enable_hazard_costs,
            'siting_mpa_buffer_km': self.siting.mpa_buffer_km,
            'siting_ais_buffer_km': self.siting.ais_buffer_km,
            'siting_ais_exclusion_pct': self.siting.ais_exclusion_pct,
            'siting_w_ais': self.siting.w_ais,
            'siting_w_seismic': self.siting.w_seismic,
            'siting_w_cyclone': self.siting.w_cyclone,
            'siting_pga_ref_g': self.siting.pga_ref_g,
            'siting_cyclone_ref_per_yr': self.siting.cyclone_ref_per_yr,
            'siting_cache_dir': self.siting.cache_dir,
            'siting_refresh': self.siting.refresh,

            # Climate scenario (0.3.0+). When `climate_enabled=False` the
            # downstream pipeline behaves identically to 0.2.0 (no delta).
            'climate_config': self.climate,
            'climate_enabled': self.climate.enabled,
            'climate_label': self.climate.label,
            'climate_scenario': self.climate.scenario,
            'climate_target_year': self.climate.target_year,
            'climate_models': list(self.climate.models),
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
    # Year handling: support legacy `year` and new `year_start`/`year_end`.
    year_kw = kwargs.pop('year', None)
    year_start_kw = kwargs.pop('year_start', None)
    year_end_kw = kwargs.pop('year_end', None)
    if year_kw is not None or year_start_kw is not None or year_end_kw is not None:
        config.data = DataConfig(
            source=config.data.source,
            time_resolution=config.data.time_resolution,
            year=year_kw,
            year_start=year_start_kw,
            year_end=year_end_kw,
            cmems_time_origin=config.data.cmems_time_origin,
            hycom_glb=config.data.hycom_glb,
            hycom_horizontal_stride=config.data.hycom_horizontal_stride,
            hycom_time_origin=config.data.hycom_time_origin,
        )

    # Handle section overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


# Legacy compatibility function
def parameters_and_constants(
    p_gross: float = -100000,
    cost_level: Union[str, CostScheme] = 'low_cost',
    data: str = 'CMEMS',
    fluid_type: str = 'ammonia',
    cycle_type: str = 'rankine_closed',
    use_coolprop: bool = True,
    optimize_depth: bool = False,
    installation_type: str = 'offshore',
    year: Optional[int] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    climate_scenario: str = 'historical',
    climate_year: Optional[int] = None,
    climate_models: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    """
    Legacy compatibility function.

    Creates an OTEXConfig and returns the legacy dictionary format.
    New code should use OTEXConfig directly.

    Args:
        p_gross: Gross power output in kW (negative = power output)
        cost_level: 'low_cost', 'high_cost', or a CostScheme object
        data: Data source ('CMEMS' or 'HYCOM')
        fluid_type: Working fluid type
        cycle_type: Thermodynamic cycle type
        use_coolprop: Whether to use CoolProp for fluid properties
        optimize_depth: Whether to optimize cold water intake depth
        year: Single year for analysis (deprecated; use year_start/year_end).
        year_start: First simulated calendar year (inclusive).
        year_end: Last simulated calendar year (inclusive).
            If neither year nor year_start is provided, defaults to 2020.

    Returns:
        Dictionary with all configuration parameters
    """
    if year is None and year_start is None:
        year_start = 2020

    climate_kwargs = {'scenario': climate_scenario}
    if climate_year is not None:
        climate_kwargs['target_year'] = climate_year
    if climate_models is not None:
        climate_kwargs['models'] = tuple(climate_models)

    config = OTEXConfig(
        plant=PlantConfig(
            gross_power=p_gross,
            optimize_depth=optimize_depth,
            installation_type=installation_type,
        ),
        economics=Economics(cost_level=cost_level),
        cycle=CycleConfig(
            cycle_type=cycle_type,
            fluid_type=fluid_type,
            use_coolprop=use_coolprop
        ),
        data=DataConfig(
            source=data,
            year=year,
            year_start=year_start,
            year_end=year_end,
        ),
        climate=ClimateConfig(**climate_kwargs),
    )

    return config.to_legacy_dict()
