# -*- coding: utf-8 -*-
"""
Cost scheme definitions for OTEX.

Provides the CostScheme dataclass and built-in schemes (LOW_COST, HIGH_COST).
Users can define custom schemes or derive them from built-in ones:

    from otex.economics import CostScheme, LOW_COST
    from dataclasses import replace

    # Custom scheme from scratch
    my_scheme = CostScheme(turbine_coeff=400, opex_fraction=0.04, ...)

    # Derived from an existing scheme
    my_scheme = replace(LOW_COST, turbine_coeff=400, opex_fraction=0.04)

    # Use in any function that accepts cost_level
    results = capex_opex_lcoe(plant, inputs, cost_level=my_scheme)
"""

from dataclasses import dataclass
from typing import Union


@dataclass
class CostScheme:
    """
    Parametric cost scheme for OTEC plant economic analysis.

    Cost formulas follow the scaling pattern:
        component_cost [$/kW] = coeff * (ref_power / actual_power) ** exp

    All monetary values are in USD (2021).

    Attributes
    ----------
    turbine_coeff : float
        Turbine cost coefficient [$/kW].
    turbine_ref_power : float
        Reference gross power for turbine scaling [kW].
    turbine_exp : float
        Turbine cost scaling exponent [-].
    hx_coeff : float
        Heat exchanger cost coefficient [$/m²].
    hx_ref_power : float
        Reference gross power for heat exchanger scaling [kW].
    hx_exp : float
        Heat exchanger cost scaling exponent [-].
    pump_coeff : float
        Seawater pump cost coefficient [$/kW].
    pump_ref_power : float
        Reference pump power for pump scaling [kW].
    pump_exp : float
        Pump cost scaling exponent [-].
    pipes_coeff : float
        Pipe material cost per unit mass [$/kg].
    structure_coeff : float
        Structure/platform cost coefficient [$/kW].
    structure_ref_power : float
        Reference gross power for structure scaling [kW].
    structure_exp : float
        Structure cost scaling exponent [-].
    deploy_coeff : float
        Deployment/installation cost [$/kW].
    controls_coeff : float
        Controls and management cost coefficient [$/kW].
    controls_ref_power : float
        Reference gross power for controls scaling [kW].
    controls_exp : float
        Controls cost scaling exponent [-].
    capex_extra_fraction : float
        Contingency/extras as a fraction of the CAPEX subtotal [-].
    opex_fraction : float
        Annual OPEX as a fraction of total CAPEX [-].
    pipe_density : float
        Pipe material density [kg/m³]. Used to determine pipe mass.
        Default 995 kg/m³ corresponds to HDPE; 1016 kg/m³ to FRP sandwich.
    """

    # Turbine [$/kW]
    turbine_coeff: float = 328.0
    turbine_ref_power: float = 136000.0
    turbine_exp: float = 0.16

    # Heat exchangers [$/m²]
    hx_coeff: float = 226.0
    hx_ref_power: float = 80000.0
    hx_exp: float = 0.16

    # Seawater pumps [$/kW]
    pump_coeff: float = 1674.0
    pump_ref_power: float = 5600.0
    pump_exp: float = 0.38

    # Pipes [$/kg]
    pipes_coeff: float = 9.0

    # Structure/platform [$/kW]
    structure_coeff: float = 4465.0
    structure_ref_power: float = 28100.0
    structure_exp: float = 0.35

    # Deployment [$/kW]
    deploy_coeff: float = 650.0

    # Controls and management [$/kW]
    controls_coeff: float = 3113.0
    controls_ref_power: float = 3960.0
    controls_exp: float = 0.70

    # Contingency and OPEX fractions
    capex_extra_fraction: float = 0.05
    opex_fraction: float = 0.03

    # Pipe material density [kg/m³]
    pipe_density: float = 995.0  # HDPE by default


# ---------------------------------------------------------------------------
# Built-in schemes
# ---------------------------------------------------------------------------

LOW_COST = CostScheme(
    turbine_coeff=328.0,
    turbine_ref_power=136000.0,
    turbine_exp=0.16,
    hx_coeff=226.0,
    hx_ref_power=80000.0,
    hx_exp=0.16,
    pump_coeff=1674.0,
    pump_ref_power=5600.0,
    pump_exp=0.38,
    pipes_coeff=9.0,
    structure_coeff=4465.0,
    structure_ref_power=28100.0,
    structure_exp=0.35,
    deploy_coeff=650.0,
    controls_coeff=3113.0,
    controls_ref_power=3960.0,
    controls_exp=0.70,
    capex_extra_fraction=0.05,
    opex_fraction=0.03,
    pipe_density=995.0,   # HDPE
)

HIGH_COST = CostScheme(
    turbine_coeff=512.0,
    turbine_ref_power=136000.0,
    turbine_exp=0.16,
    hx_coeff=916.0,
    hx_ref_power=4400.0,
    hx_exp=0.093,
    pump_coeff=2480.0,
    pump_ref_power=5600.0,
    pump_exp=0.38,
    pipes_coeff=30.1,
    structure_coeff=7442.0,
    structure_ref_power=28100.0,
    structure_exp=0.35,
    deploy_coeff=667.0,
    controls_coeff=6085.0,
    controls_ref_power=4400.0,
    controls_exp=0.70,
    capex_extra_fraction=0.2,
    opex_fraction=0.05,
    pipe_density=1016.0,  # FRP sandwich
)

_BUILTIN_SCHEMES = {
    'low_cost': LOW_COST,
    'high_cost': HIGH_COST,
}


def get_cost_scheme(cost_level: Union[str, 'CostScheme']) -> 'CostScheme':
    """
    Resolve a cost level identifier to a CostScheme instance.

    Parameters
    ----------
    cost_level : str or CostScheme
        Either a built-in scheme name ('low_cost', 'high_cost') or a
        CostScheme instance.

    Returns
    -------
    CostScheme

    Raises
    ------
    ValueError
        If cost_level is a string that does not match a built-in scheme.
    TypeError
        If cost_level is neither a string nor a CostScheme.
    """
    if isinstance(cost_level, CostScheme):
        return cost_level
    if isinstance(cost_level, str):
        if cost_level not in _BUILTIN_SCHEMES:
            raise ValueError(
                f'cost_level "{cost_level}" is not a recognised built-in scheme. '
                f'Built-in options: {list(_BUILTIN_SCHEMES)}. '
                f'Alternatively, pass a CostScheme object directly.'
            )
        return _BUILTIN_SCHEMES[cost_level]
    raise TypeError(
        f'cost_level must be a str or CostScheme, got {type(cost_level).__name__}.'
    )
