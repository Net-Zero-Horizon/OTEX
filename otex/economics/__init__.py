# -*- coding: utf-8 -*-
"""
OTEX Economics Module
Cost analysis, LCOE calculations, and optimization.
"""

from .costs import (
    capex_opex_lcoe,
    lcoe_time_series,
    lcoe_npv,
)
from .cost_schemes import (
    CostScheme,
    LOW_COST,
    HIGH_COST,
    get_cost_scheme,
)
from .degradation import (
    DegradationConfig,
    OpexEscalationConfig,
    degradation_factor,
    opex_escalation_factor,
    extrapolate_cyclic,
)

__all__ = [
    "capex_opex_lcoe",
    "lcoe_time_series",
    "lcoe_npv",
    "CostScheme",
    "LOW_COST",
    "HIGH_COST",
    "get_cost_scheme",
    "DegradationConfig",
    "OpexEscalationConfig",
    "degradation_factor",
    "opex_escalation_factor",
    "extrapolate_cyclic",
]
