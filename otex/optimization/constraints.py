# -*- coding: utf-8 -*-
"""Physical and technical inequality constraints g_i(x) ≤ 0.

Each constraint maps a :class:`DesignResult` (the output of the
evaluator) to a single float. A non-positive value means the
constraint is satisfied; a positive value is the magnitude of the
violation, used directly by the penalty objective.

Constraint catalogue (MVP — 8 hard physical/technical limits):

* ``g_dT_total``      – ``dT_WW + dT_CW`` mustn't exceed the available
                        ocean ΔT minus pinch and pipe ΔT margins.
* ``g_pinch_evap``    – evaporator pinch point (T_evap below T_WW_out
                        by at least T_pinch_evap).
* ``g_pinch_cond``    – condenser pinch point.
* ``g_pipe_d_WW``     – warm-pipe inner diameter ≤ ``max_d``.
* ``g_pipe_d_CW``     – cold-pipe inner diameter ≤ ``max_d``.
* ``g_parasitic``     – ``p_pump_total / |p_gross|`` ≤ 0.40.
* ``g_depth_site``    – ``depth_CW`` ≤ site bathymetric water depth.
* ``g_p_net_sign``    – ``p_net < 0`` (i.e. positive net output).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .design_vector import DesignVector
from .evaluator import DesignResult, SiteContext


# Total parasitic fraction beyond which the plant is considered
# practically infeasible. Real OTEC plants sit around 25-35 %.
MAX_PARASITIC_FRACTION = 0.40

CONSTRAINT_NAMES: List[str] = [
    'g_dT_total',
    'g_pinch_evap',
    'g_pinch_cond',
    'g_pipe_d_WW',
    'g_pipe_d_CW',
    'g_parasitic',
    'g_depth_site',
    'g_p_net_sign',
]


@dataclass
class ConstraintResult:
    """All eight constraint values for a single ``(x, site)`` evaluation."""

    values: Dict[str, float] = field(default_factory=dict)

    @property
    def violations(self) -> Dict[str, float]:
        """Only the strictly positive (violated) entries."""
        return {k: v for k, v in self.values.items() if v > 0}

    @property
    def max_violation(self) -> float:
        if not self.values:
            return 0.0
        return max(0.0, max(self.values.values()))

    @property
    def feasible(self) -> bool:
        return self.max_violation == 0.0

    def as_array(self) -> np.ndarray:
        """Return values in :data:`CONSTRAINT_NAMES` order."""
        return np.array([self.values.get(k, np.nan) for k in CONSTRAINT_NAMES],
                        dtype=np.float64)


def evaluate_constraints(
    x: DesignVector,
    result: DesignResult,
    site: SiteContext,
) -> ConstraintResult:
    """Compute all eight g_i values for one design evaluation."""
    inputs = site.inputs_template

    # --- g_dT_total: leave at least T_pinch each side as margin. -----
    available_dT = site.T_WW_in - site.T_CW_in
    pinch_margin = inputs.get('T_pinch_WW', 1.0) + inputs.get('T_pinch_CW', 1.0)
    g_dT_total = (x.dT_WW + x.dT_CW + pinch_margin) - available_dT

    # --- g_pinch_evap: T_evap ≤ T_WW_in - dT_WW - T_pinch_evap. -----
    T_WW_out = site.T_WW_in - x.dT_WW
    g_pinch_evap = (result.T_evap + inputs.get('T_pinch_WW', 1.0)) - T_WW_out

    # --- g_pinch_cond: T_cond ≥ T_CW_in + dT_CW + T_pinch_cond. -----
    T_CW_out = site.T_CW_in + x.dT_CW
    g_pinch_cond = (T_CW_out + inputs.get('T_pinch_CW', 1.0)) - result.T_cond

    # --- g_pipe_d_*: per-pipe inner diameter ≤ max_d.  --------------
    max_d = inputs.get('max_d', 8.0)
    g_pipe_d_WW = result.D_pipe_WW - max_d
    g_pipe_d_CW = result.D_pipe_CW - max_d

    # --- g_parasitic: total pump power vs gross.  -------------------
    p_gross_abs = abs(x.p_gross)
    if p_gross_abs <= 0:
        g_parasitic = 1.0
    else:
        g_parasitic = (result.p_pump_total / p_gross_abs) - MAX_PARASITIC_FRACTION

    # --- g_depth_site: cold intake must fit the bathymetry.  --------
    # Site bathymetric depth is stored as a NEGATIVE elevation (OTEX
    # convention). Available water depth = abs(elevation).
    site_water_depth = abs(site.inputs_template.get('site_water_depth_m',
                                                     x.depth_CW))
    g_depth_site = x.depth_CW - site_water_depth

    # --- g_p_net_sign: net power must be negative (= output).  ------
    # If p_net is positive or NaN, the plant doesn't produce — heavy
    # violation so the optimiser walks away from this corner.
    if not np.isfinite(result.p_net):
        g_p_net_sign = 1.0
    else:
        g_p_net_sign = float(result.p_net)   # > 0 means consuming

    return ConstraintResult(values={
        'g_dT_total':   float(g_dT_total),
        'g_pinch_evap': float(g_pinch_evap),
        'g_pinch_cond': float(g_pinch_cond),
        'g_pipe_d_WW':  float(g_pipe_d_WW),
        'g_pipe_d_CW':  float(g_pipe_d_CW),
        'g_parasitic':  float(g_parasitic),
        'g_depth_site': float(g_depth_site),
        'g_p_net_sign': float(g_p_net_sign),
    })
