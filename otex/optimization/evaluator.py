# -*- coding: utf-8 -*-
"""Pure evaluator: design vector ``x`` → (LCOE, intermediates).

Wraps the existing forward pipeline (``otec_sizing`` +
``capex_opex_lcoe``) so the optimiser sees a single callable
``f(x) -> result``. Per-site context (design temperatures, distance
to shore, base ``inputs`` dict) is captured once via
:class:`SiteContext` and reused across all evaluations during one
optimisation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..plant.sizing import otec_sizing
from ..economics.costs import capex_opex_lcoe
from .design_vector import DesignVector


@dataclass
class SiteContext:
    """All site-specific information the evaluator needs.

    ``inputs_template`` is treated as immutable; the evaluator makes a
    shallow copy for every call so concurrent optimisations on
    different sites don't clobber each other's ``p_gross``,
    ``length_CW_inlet`` etc.
    """

    site_id: int
    longitude: float
    latitude: float
    T_WW_in: float                       # °C, design (typically median)
    T_CW_in: float                       # °C, design (typically median)
    dist_shore: float                    # km
    eff_trans: float                     # transmission efficiency at this site
    inputs_template: Dict[str, Any]      # template (will be copied per-call)
    cost_level: str = 'low_cost'
    # Outlet pipe lengths are fixed per template; CW inlet length is
    # set from the design variable depth_CW.
    pipe_outlet_extra_m: float = 60.0    # matches SeawaterPipes.cw_outlet_length


@dataclass
class DesignResult:
    """What the evaluator returns for one ``(x, site)`` call."""

    x: DesignVector

    # Headline numbers
    lcoe: float                          # ¢/kWh
    p_net: float                         # kW, negative = output
    p_pump_total: float                  # kW
    capex_total: float                   # USD
    opex: float                          # USD/yr
    eff_net: float                       # dimensionless

    # Heat-exchanger / pipe intermediates the constraints look at
    T_evap: float                        # °C
    T_cond: float                        # °C
    D_pipe_WW: float                     # m, pipe inner diameter (warm)
    D_pipe_CW: float                     # m, pipe inner diameter (cold)
    p_pump_WW: float                     # kW, warm pipe pumping power
    p_pump_CW: float                     # kW, cold pipe pumping power
    # Site-derived limit fields the constraints need but the evaluator
    # has access to via the context.
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        """Quick sanity flag — the optimisation will look at the
        constraint vector for the rigorous answer."""
        return (
            np.isfinite(self.lcoe) and self.lcoe > 0
            and np.isfinite(self.p_net) and self.p_net < 0
        )


def _scalar(arr_like):
    """Robust scalar conversion (handles 0-d arrays / 1-element vectors)."""
    a = np.asarray(arr_like).ravel()
    if a.size == 0:
        return float('nan')
    return float(a[0])


def evaluate(x: DesignVector, site: SiteContext) -> DesignResult:
    """Run the forward pipeline once at the design vector ``x``.

    The function is **pure** in the sense that it does not mutate
    ``site.inputs_template``; it operates on a local copy and
    returns a :class:`DesignResult`.
    """
    inputs = copy.copy(site.inputs_template)

    # Inject the design vector. p_gross uses OTEX's negative-output
    # convention; sizing.py reads inputs['p_gross'] directly.
    inputs['p_gross'] = float(x.p_gross)
    inputs['length_CW_inlet'] = float(x.depth_CW)
    inputs['length_CW'] = float(x.depth_CW) + site.pipe_outlet_extra_m

    # Site-derived numbers expected by capex_opex_lcoe.
    inputs['dist_shore'] = np.array([[site.dist_shore]], dtype=np.float64)
    inputs['eff_trans'] = np.array([[site.eff_trans]], dtype=np.float64)

    # Siting hazard multipliers default to zero in the optimiser; users
    # who want the siting penalty included can set them in the template.
    for k in ('ais_density_pct', 'pga_475', 'cyclone_freq_per_yr'):
        inputs.setdefault(k, np.zeros((1, 1), dtype=np.float64))

    # Run the design-point pipeline. otec_sizing is the existing
    # OTEX call; we feed it scalar temperatures so it produces a
    # scalar plant.
    T_WW_in = np.array([site.T_WW_in], dtype=np.float64)
    T_CW_in = np.array([site.T_CW_in], dtype=np.float64)

    plant = otec_sizing(T_WW_in, T_CW_in, x.dT_WW, x.dT_CW,
                        inputs, site.cost_level)

    capex_opex_dict, capex_total, opex, lcoe_nom = capex_opex_lcoe(
        plant, inputs, cost_level=site.cost_level,
    )

    return DesignResult(
        x=x,
        lcoe=_scalar(lcoe_nom),
        p_net=_scalar(plant['p_net_nom']),
        p_pump_total=_scalar(plant['p_pump_total_nom']),
        capex_total=_scalar(capex_total),
        opex=_scalar(opex),
        eff_net=_scalar(plant.get('eff_net_nom', plant.get('eff_net', np.nan))),
        T_evap=_scalar(plant.get('T_evap_nom', np.nan)),
        T_cond=_scalar(plant.get('T_cond_nom', np.nan)),
        D_pipe_WW=_scalar(plant.get('d_pipes_WW', np.nan)),
        D_pipe_CW=_scalar(plant.get('d_pipes_CW', np.nan)),
        p_pump_WW=_scalar(plant.get('p_pump_WW_nom', np.nan)),
        p_pump_CW=_scalar(plant.get('p_pump_CW_nom', np.nan)),
        extras={
            'capex_components': capex_opex_dict,
            'plant_raw': plant,
        },
    )
