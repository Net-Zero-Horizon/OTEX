# -*- coding: utf-8 -*-
"""Penalty-method objective J(x) = LCOE(x) + Σ λ_i max(0, g_i(x))².

This is the function the solver receives. It collapses the
constrained NLP

    min   LCOE(x)
    s.t.  g_i(x) ≤ 0   ∀ i

into an unconstrained problem by adding quadratic penalty terms for
every violated constraint. Penalty weights default to large numbers
so the optimiser is strongly biased towards feasibility, but the
user can tune them via :data:`DEFAULT_PENALTY_WEIGHTS` or pass a
custom dict to :func:`build_objective`.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from .constraints import (
    CONSTRAINT_NAMES,
    ConstraintResult,
    evaluate_constraints,
)
from .design_vector import DesignVector
from .evaluator import DesignResult, SiteContext, evaluate
from .user_constraints import (
    USER_CONSTRAINT_DEFAULT_WEIGHT,
    UserConstraints,
    evaluate_user_constraints,
)


# Penalty weights. The units of g_i differ wildly (Kelvins for pinch
# constraints, dimensionless for parasitic ratio, kW for the
# p_net_sign one), so each gets its own λ. Values were chosen so a
# 1-unit violation costs ~10-100 ¢/kWh — large compared to the
# typical LCOE of 20-30 ¢/kWh.
DEFAULT_PENALTY_WEIGHTS: Dict[str, float] = {
    'g_dT_total':   50.0,    # K
    'g_pinch_evap': 50.0,    # K
    'g_pinch_cond': 50.0,    # K
    'g_pipe_d_WW':   5.0,    # m
    'g_pipe_d_CW':   5.0,    # m
    'g_parasitic': 200.0,    # dimensionless (penalises overshooting 40 %)
    'g_depth_site':  0.01,   # m  (lower weight — depth_CW is large)
    'g_p_net_sign':  0.001,  # kW (the magnitude is huge; tiny weight)
}


def _penalty(g_value: float, weight: float) -> float:
    """Quadratic outside-only penalty: ``λ · max(0, g)²``."""
    v = max(0.0, g_value)
    return weight * v * v


def build_objective(
    site: SiteContext,
    penalty_weights: Optional[Dict[str, float]] = None,
    evaluator: Callable[[DesignVector, SiteContext], DesignResult] = evaluate,
    user_constraints: Optional[UserConstraints] = None,
) -> Callable[[np.ndarray], float]:
    """Return ``J(x) -> float`` ready to hand to ``scipy.optimize``.

    Includes both the built-in physical constraints and any
    user-supplied exogenous limits (:class:`UserConstraints`).
    Without user constraints the LCOE surface is monotone in
    ``p_gross`` over OTEX's modelled range, so the optimum
    degenerates to the upper box bound — pass an :class:`UserConstraints`
    with at least one limit to obtain a genuine interior optimum.
    """
    weights = dict(DEFAULT_PENALTY_WEIGHTS)
    if penalty_weights:
        weights.update(penalty_weights)

    user = user_constraints or UserConstraints()

    def J(x_arr: np.ndarray) -> float:
        x = DesignVector.from_array(x_arr)
        result = evaluator(x, site)

        # Non-finite LCOE → strong but bounded penalty; the optimiser
        # can still see a gradient via the constraint penalties below
        # because those use the (possibly NaN-tainted) intermediates.
        base = result.lcoe if np.isfinite(result.lcoe) and result.lcoe > 0 else 1e6

        cons = evaluate_constraints(x, result, site)
        penalty = 0.0
        for name in CONSTRAINT_NAMES:
            g = cons.values.get(name, 0.0)
            if not np.isfinite(g):
                penalty += weights.get(name, 1.0) * 1e6
                continue
            penalty += _penalty(g, weights.get(name, 1.0))

        # User-supplied exogenous constraints (path B). Normalised
        # violations → single uniform weight works for all of them.
        for u_name, u_val in evaluate_user_constraints(result, site, user).items():
            if not np.isfinite(u_val):
                penalty += USER_CONSTRAINT_DEFAULT_WEIGHT * 1e6
                continue
            penalty += _penalty(u_val, USER_CONSTRAINT_DEFAULT_WEIGHT)

        return float(base + penalty)

    return J
