# -*- coding: utf-8 -*-
"""
OTEX - Ocean Thermal Energy eXchange
A Python library for OTEC plant design, simulation, and analysis.

Main modules:
- otex.core: Thermodynamic cycles, working fluids, and mixture properties
- otex.plant: Plant sizing, operation, and component modeling
- otex.economics: Cost analysis, LCOE calculations, and optimization
- otex.data: Oceanographic data processing (CMEMS, NetCDF)
- otex.config: Centralized configuration management
"""

__version__ = "0.1.0"
__author__ = "OTEX Development Team"

from .config import OTEXConfig, get_default_config

# Core exports
from .core import (
    ThermodynamicCycle,
    RankineClosedCycle,
    RankineOpenCycle,
    RankineHybridCycle,
    KalinaCycle,
    UeharaCycle,
    get_thermodynamic_cycle,
    WorkingFluid,
    CoolPropFluid,
    PolynomialAmmonia,
    get_working_fluid,
    AmmoniaWaterMixture,
)

# Plant exports
from .plant import (
    otec_sizing,
    otec_operation,
    enthalpies_entropies,
)

# Economics exports
from .economics import (
    capex_opex_lcoe,
    lcoe_time_series,
    CostScheme,
    LOW_COST,
    HIGH_COST,
    get_cost_scheme,
)

# Regional analysis - lazy import to avoid copernicusmarine dependency at package import
# from .regional import run_regional_analysis  # Moved to __getattr__

__all__ = [
    # Version
    "__version__",
    # Config
    "OTEXConfig",
    "get_default_config",
    # Core - Cycles
    "ThermodynamicCycle",
    "RankineClosedCycle",
    "RankineOpenCycle",
    "RankineHybridCycle",
    "KalinaCycle",
    "UeharaCycle",
    "get_thermodynamic_cycle",
    # Core - Fluids
    "WorkingFluid",
    "CoolPropFluid",
    "PolynomialAmmonia",
    "get_working_fluid",
    # Core - Mixtures
    "AmmoniaWaterMixture",
    # Plant
    "otec_sizing",
    "otec_operation",
    "enthalpies_entropies",
    # Economics
    "capex_opex_lcoe",
    "lcoe_time_series",
    "CostScheme",
    "LOW_COST",
    "HIGH_COST",
    "get_cost_scheme",
    # Regional
    "run_regional_analysis",
]


def __getattr__(name):
    """Lazy import for regional analysis to avoid copernicusmarine dependency."""
    if name == "run_regional_analysis":
        from .regional import run_regional_analysis
        return run_regional_analysis
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
