# -*- coding: utf-8 -*-
"""SLSQP per-site optimisation + regional orchestration.

The single-site optimiser runs ``scipy.optimize.minimize`` with the
SLSQP method on the penalty-method objective from
:mod:`otex.optimization.objective`. SLSQP handles bounded NLP
problems with 4 continuous variables cheaply; the inner
:func:`evaluate` runs the full forward pipeline so a typical solve
converges in 30-100 evaluations (~1-3 seconds per site).

For an entire region we loop over sites sequentially. Each site is
independent — joining them only matters for portfolio-level
constraints, which is a follow-up feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .constraints import (
    CONSTRAINT_NAMES,
    ConstraintResult,
    evaluate_constraints,
)
from .design_vector import Bounds, DEFAULT_BOUNDS, DesignVector
from .evaluator import DesignResult, SiteContext, evaluate
from .objective import DEFAULT_PENALTY_WEIGHTS, build_objective
from .user_constraints import UserConstraints, evaluate_user_constraints


@dataclass
class OptimizationResult:
    """Outcome of a single-site optimisation."""

    site_id: int
    longitude: float
    latitude: float
    success: bool
    n_evaluations: int
    message: str

    # Best design found
    x: DesignVector
    lcoe: float
    p_net: float
    capex_total: float
    opex: float

    # Constraint snapshot at the optimum
    max_violation: float
    constraint_values: Dict[str, float] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        # 5e-2 covers numerical drift of the *active* physical
        # constraints (pinch evap/cond sit on the boundary by
        # thermodynamic design) plus the slack we allow on user
        # constraints. Below this threshold every cap the user
        # actually set is respected.
        return self.max_violation <= 5e-2


def _default_x0(
    bounds: Bounds,
    site: SiteContext,
    user: Optional[UserConstraints] = None,
) -> np.ndarray:
    """Pick a starting point that respects any active user cap.

    Starting from the box centre is a poor choice when the user
    has imposed a tight cap (e.g. ``max_p_gross_MW = 120``): the
    centre is deep inside the infeasible region and L-BFGS-B
    spends most of its budget marching back to feasibility. A
    warm start at (or near) the user-set cap converges 3-10x
    faster.
    """
    # Sensible interior defaults for dT and depth.
    dT_WW0 = max(bounds.dT_WW[0],
                  min(bounds.dT_WW[1], 3.0))
    dT_CW0 = max(bounds.dT_CW[0],
                  min(bounds.dT_CW[1], 3.0))
    depth0 = max(bounds.depth_CW[0],
                  min(bounds.depth_CW[1], 1000.0))

    # Pick p_gross at the strictest of (user max_p_gross, user
    # max_p_net translated upwards by typical eff_net, user
    # max_aep / hours, user max_capex scaled, box upper bound).
    p_candidates = [bounds.p_gross[1]]   # box upper bound (closest to 0)
    if user is not None:
        if user.max_p_gross_MW is not None:
            p_candidates.append(-user.max_p_gross_MW * 1000.0)
        if user.max_p_net_MW is not None:
            # p_net ≈ 0.6 * p_gross for a typical OTEC plant
            p_candidates.append(-user.max_p_net_MW * 1000.0 / 0.6)
        if user.max_aep_MWh is not None:
            # AEP ≈ |p_net| * 0.914 * 8760 / 1000
            # → |p_net| ≈ AEP / 8 → p_gross ≈ AEP / 8 / 0.6
            p_candidates.append(-user.max_aep_MWh / 8.0 / 0.6 * 1000.0)
        if user.max_capex_MUSD is not None:
            # CAPEX (USD) ≈ 8000 USD/kW for OTEC at 100 MW.
            p_candidates.append(-user.max_capex_MUSD * 1e6 / 8000.0)
    # Most negative candidate is the binding one (closest to lower bound).
    p0 = max(p_candidates)   # max because all negative → largest is the cap
    p0 = max(bounds.p_gross[0], min(bounds.p_gross[1], p0))

    return np.array([p0, dT_WW0, dT_CW0, depth0], dtype=np.float64)


def _bounds_arrays(bounds: Bounds):
    """Return (lo, hi) numpy arrays from a :class:`Bounds`."""
    scipy = bounds.as_scipy()
    lo = np.array([b[0] for b in scipy], dtype=np.float64)
    hi = np.array([b[1] for b in scipy], dtype=np.float64)
    return lo, hi


def _normalize(x_phys: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (x_phys - lo) / (hi - lo)


def _denormalize(x_norm: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return lo + x_norm * (hi - lo)


def optimize_site(
    site: SiteContext,
    *,
    bounds: Bounds = DEFAULT_BOUNDS,
    x0: Optional[np.ndarray] = None,
    penalty_weights: Optional[Dict[str, float]] = None,
    user_constraints: Optional[UserConstraints] = None,
    options: Optional[Dict[str, Any]] = None,
) -> OptimizationResult:
    """Solve ``min J(x)`` for a single site via SLSQP.

    The four design variables (``p_gross``, ``dT_WW``, ``dT_CW``,
    ``depth_CW``) live on wildly different scales (~10⁵ kW vs ~5 K vs
    ~10³ m), so we **normalise to [0, 1]** before passing to SLSQP.
    The default finite-difference step then yields a well-conditioned
    gradient in every dimension — without this the optimiser barely
    moves ``p_gross`` from its starting point because the perturbation
    is swallowed by CoolProp's numerical noise.

    Parameters
    ----------
    site : SiteContext
        Site-specific design temperatures + base inputs template.
    bounds : Bounds, optional
        Box bounds on the design vector (physical units).
    x0 : array-like, optional
        Starting guess in physical units; defaults to the box centre.
    penalty_weights : dict, optional
        Override built-in constraint penalty weights.
    user_constraints : UserConstraints, optional
        Exogenous caps (max AEP, CAPEX, p_net, etc.) — typically the
        binding ones that produce an interior optimum.
    options : dict, optional
        Forwarded to ``scipy.optimize.minimize`` (e.g. ``maxiter``,
        ``ftol``).
    """
    J_phys = build_objective(
        site,
        penalty_weights=penalty_weights,
        user_constraints=user_constraints,
    )

    lo, hi = _bounds_arrays(bounds)
    span = hi - lo

    # Wrap the physical objective so SLSQP sees unit-cube inputs.
    def J_norm(x_norm: np.ndarray) -> float:
        return J_phys(lo + x_norm * span)

    if x0 is None:
        x0_phys = _default_x0(bounds, site, user_constraints)
    else:
        x0_phys = np.asarray(x0, dtype=np.float64)
    x0_norm = _normalize(x0_phys, lo, hi)
    x0_norm = np.clip(x0_norm, 0.0, 1.0)

    # L-BFGS-B is the right choice for penalty-method NLPs: it handles
    # box bounds natively, builds its own Hessian approximation
    # iteratively (so it stays robust through the curvature changes
    # introduced by quadratic penalties), and unlike SLSQP it does not
    # declare premature convergence on this specific class of problem
    # (SLSQP returned success after 5 evaluations with J = 38493 while
    # L-BFGS-B and trust-constr both reached J ≈ 22 in 300-700 evals
    # on the same instance). eps gives a finite-difference step large
    # enough to see through CoolProp's numerical noise on every axis
    # of the normalised cube.
    opts = {'maxiter': 200, 'ftol': 1e-9, 'eps': 1e-3, 'disp': False}
    if options:
        opts.update(options)

    res = minimize(
        J_norm, x0_norm, method='L-BFGS-B',
        bounds=[(0.0, 1.0)] * 4,
        options=opts,
    )

    x_opt = DesignVector.from_array(_denormalize(res.x, lo, hi))
    result = evaluate(x_opt, site)
    cons = evaluate_constraints(x_opt, result, site)

    # Merge in user-constraint snapshot so it's visible in the output
    # dataframe alongside the physical constraint values.
    all_constraints = dict(cons.values)
    user = user_constraints or UserConstraints()
    if user.any_active:
        all_constraints.update(evaluate_user_constraints(result, site, user))
    max_viol = max([0.0] + [v for v in all_constraints.values()
                            if np.isfinite(v) and v > 0])

    return OptimizationResult(
        site_id=site.site_id,
        longitude=site.longitude,
        latitude=site.latitude,
        success=bool(res.success),
        n_evaluations=int(res.nfev),
        message=str(res.message),
        x=x_opt,
        lcoe=result.lcoe,
        p_net=result.p_net,
        capex_total=result.capex_total,
        opex=result.opex,
        max_violation=max_viol,
        constraint_values=all_constraints,
    )


def run_regional_optimization(
    studied_region: str,
    *,
    cost_level: str = 'low_cost',
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    cycle_type: str = 'rankine_closed',
    fluid_type: str = 'ammonia',
    use_coolprop: bool = True,
    data_source: str = 'CMEMS',
    bounds: Bounds = DEFAULT_BOUNDS,
    penalty_weights: Optional[Dict[str, float]] = None,
    user_constraints: Optional[UserConstraints] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Optimise every feasible site in ``studied_region`` and return a DataFrame.

    Builds the data layer (download, CMEMS load, site catalog) exactly
    as :func:`otex.regional.run_regional_analysis` does, then loops
    over sites running :func:`optimize_site` on each. The output is
    one DataFrame row per site with the optimal design vector, LCOE,
    constraint snapshot, and solver diagnostics.
    """
    # Late imports — avoid pulling cmems/copernicusmarine at package
    # import time and keep this module testable in isolation.
    import os
    import time as _time
    from ..config import parameters_and_constants
    from ..data.resources import load_sites
    from ..data.cmems import data_processing, load_temperatures

    if year_start is None:
        year_start = 2020
    if year_end is None:
        year_end = year_start

    inputs = parameters_and_constants(
        cost_level=cost_level,
        fluid_type=fluid_type,
        cycle_type=cycle_type,
        use_coolprop=use_coolprop,
        data=data_source,
        year_start=year_start,
        year_end=year_end,
        # p_gross is irrelevant here — overwritten per-evaluation.
    )

    sites_df = load_sites(
        studied_region,
        min_depth=abs(inputs['min_depth']),
        max_depth=abs(inputs['max_depth']),
    )
    sites_df = sites_df[
        (sites_df['water_depth'] <= inputs['min_depth'])
        & (sites_df['water_depth'] >= inputs['max_depth'])
    ].sort_values(['longitude', 'latitude']).reset_index(drop=True)

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'Data_Results')
    region_dir = os.path.join(output_dir, studied_region.replace(' ', '_'))
    os.makedirs(region_dir, exist_ok=True)

    if inputs['data'] == 'HYCOM':
        from ..data.hycom import download_data
    else:
        from ..data.cmems import download_data
    files = download_data(cost_level, inputs, studied_region, region_dir + os.sep)

    # Reuse OTEX's data-processing path to get the design temperatures
    # at the actual oceanographic grid points.
    T_CW_profiles, T_CW_design, coordinates_CW, id_sites, _ts, inputs, nan_cw = (
        data_processing(files[len(files) // 2:], sites_df, inputs,
                        studied_region, region_dir + os.sep, 'CW')
    )
    T_WW_profiles, T_WW_design, coordinates_WW, _id, _ts, inputs, _nan_ww = (
        data_processing(files[:len(files) // 2], sites_df, inputs,
                        studied_region, region_dir + os.sep, 'WW', nan_cw)
    )

    n_sites = T_CW_design.shape[1]
    if verbose:
        print(f'  [optimize] {n_sites} sites to evaluate.', flush=True)

    rows: List[Dict[str, Any]] = []
    t0 = _time.time()
    for i in range(n_sites):
        # Use the median (axis index 1) as the design operating point.
        site_id = int(np.squeeze(id_sites)[i])
        site = SiteContext(
            site_id=site_id,
            longitude=float(coordinates_CW[i, 0]),
            latitude=float(coordinates_CW[i, 1]),
            T_WW_in=float(T_WW_design[1, i]),
            T_CW_in=float(T_CW_design[1, i]),
            dist_shore=float(np.squeeze(inputs['dist_shore'])[i]),
            eff_trans=float(np.squeeze(inputs['eff_trans'])[i]),
            inputs_template=inputs,
            cost_level=cost_level,
        )

        result = optimize_site(
            site, bounds=bounds, penalty_weights=penalty_weights,
            user_constraints=user_constraints,
        )

        row = {
            'id': site_id,
            'longitude': site.longitude,
            'latitude': site.latitude,
            'T_WW_design': site.T_WW_in,
            'T_CW_design': site.T_CW_in,
            'p_gross_opt_kW': result.x.p_gross,
            'p_gross_opt_MW': -result.x.p_gross / 1000.0,
            'dT_WW_opt': result.x.dT_WW,
            'dT_CW_opt': result.x.dT_CW,
            'depth_CW_opt': result.x.depth_CW,
            'lcoe_min': result.lcoe,
            'p_net_kW': result.p_net,
            'capex_total_MUSD': result.capex_total / 1e6,
            'opex_MUSDyr': result.opex / 1e6,
            'max_violation': result.max_violation,
            'feasible': result.feasible,
            'success': result.success,
            'n_evaluations': result.n_evaluations,
        }
        for name in CONSTRAINT_NAMES:
            row[name] = result.constraint_values.get(name, np.nan)
        rows.append(row)

        if verbose and ((i + 1) % 25 == 0 or i + 1 == n_sites):
            elapsed = _time.time() - t0
            print(f'  [optimize] {i + 1}/{n_sites} sites '
                  f'({elapsed:.1f}s, {elapsed / (i + 1):.2f}s/site)',
                  flush=True)

    return pd.DataFrame(rows)
