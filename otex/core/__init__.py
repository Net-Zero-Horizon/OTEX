# -*- coding: utf-8 -*-
"""
OTEX Core Module
Thermodynamic cycles, working fluids, and mixture properties.
"""

from .cycles import (
    ThermodynamicCycle,
    RankineClosedCycle,
    RankineOpenCycle,
    RankineHybridCycle,
    KalinaCycle,
    UeharaCycle,
    get_thermodynamic_cycle,
)

from .fluids import (
    WorkingFluid,
    CoolPropFluid,
    PolynomialAmmonia,
    get_working_fluid,
    COOLPROP_AVAILABLE,
)

from .mixtures import AmmoniaWaterMixture

__all__ = [
    # Cycles
    "ThermodynamicCycle",
    "RankineClosedCycle",
    "RankineOpenCycle",
    "RankineHybridCycle",
    "KalinaCycle",
    "UeharaCycle",
    "get_thermodynamic_cycle",
    # Fluids
    "WorkingFluid",
    "CoolPropFluid",
    "PolynomialAmmonia",
    "get_working_fluid",
    "COOLPROP_AVAILABLE",
    # Mixtures
    "AmmoniaWaterMixture",
]
