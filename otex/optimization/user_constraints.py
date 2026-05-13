# -*- coding: utf-8 -*-
"""User-defined exogenous constraints for the inverse optimisation.

Without an exogenous cap from the user, the OTEC LCOE function is
monotonically decreasing in ``p_gross`` over any plausible range
(economies of scale dominate; OTEX's internal physics doesn't impose
a hard upper limit). The "optimum" therefore degenerates to the
upper box bound and the optimiser is reduced to a box-bound oracle.

To get a genuine *interior* optimum, the decision-maker must specify
what limits *their* plant size — budget, footprint, grid take-off
capacity, etc. Every field of :class:`UserConstraints` is optional;
the user enables only the ones that apply to their case study.

Each constraint is encoded as a **normalised** violation
``(value − limit) / limit`` so the penalty weights work uniformly
across wildly different units (MWh, USD, kW).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .evaluator import DesignResult, SiteContext


@dataclass
class UserConstraints:
    """Optional exogenous caps imposed by the decision-maker.

    Leave any field as ``None`` to disable that constraint. At least
    one constraint should be active for the inverse optimisation to
    yield an interior optimum in ``p_gross``; otherwise the solver
    walks to the upper box bound.
    """

    # Energetic
    max_aep_MWh: Optional[float] = None       # per-plant annual energy production
    max_p_net_MW: Optional[float] = None      # net delivered power capacity

    # Economic
    max_capex_MUSD: Optional[float] = None    # total capital expenditure

    # Engineering / operational
    max_p_gross_MW: Optional[float] = None    # gross plant size (mass / pipe footprint proxy)
    max_parasitic_ratio: Optional[float] = None  # P_pump_total / |P_gross|

    @property
    def any_active(self) -> bool:
        return any(getattr(self, f) is not None for f in (
            'max_aep_MWh', 'max_p_net_MW', 'max_capex_MUSD',
            'max_p_gross_MW', 'max_parasitic_ratio',
        ))

    def active_names(self) -> List[str]:
        names = []
        if self.max_aep_MWh is not None:        names.append('u_max_aep')
        if self.max_p_net_MW is not None:       names.append('u_max_p_net')
        if self.max_capex_MUSD is not None:     names.append('u_max_capex')
        if self.max_p_gross_MW is not None:     names.append('u_max_p_gross')
        if self.max_parasitic_ratio is not None: names.append('u_max_parasitic')
        return names


# Penalty weights are uniform across user constraints because they
# all return *normalised* violations (dimensionless). A 1.0
# normalised violation (i.e. value = 2 × limit) therefore incurs the
# same penalty cost regardless of which constraint is active.
USER_CONSTRAINT_DEFAULT_WEIGHT = 1.0e4


def evaluate_user_constraints(
    result: DesignResult,
    site: SiteContext,
    user: UserConstraints,
) -> Dict[str, float]:
    """Return ``{name: normalised_violation}`` for every active user limit.

    A non-positive value means the constraint is satisfied; a positive
    value is the fractional violation (1.0 = double the limit).
    """
    if not user.any_active:
        return {}

    # AEP in MWh: |p_net| (kW) * availability * hours_per_year / 1000
    availability = float(site.inputs_template.get('availability_factor', 0.914))
    # Use 8760 as the canonical hours-per-year for the LCOE calc consistency.
    aep_MWh = abs(result.p_net) * availability * 8760.0 / 1000.0
    capex_MUSD = result.capex_total / 1.0e6
    p_net_MW = abs(result.p_net) / 1000.0
    p_gross_MW = abs(result.x.p_gross) / 1000.0
    parasitic = (
        result.p_pump_total / abs(result.x.p_gross)
        if result.x.p_gross != 0 else 1.0
    )

    out: Dict[str, float] = {}
    if user.max_aep_MWh is not None:
        out['u_max_aep'] = (aep_MWh - user.max_aep_MWh) / user.max_aep_MWh
    if user.max_p_net_MW is not None:
        out['u_max_p_net'] = (p_net_MW - user.max_p_net_MW) / user.max_p_net_MW
    if user.max_capex_MUSD is not None:
        out['u_max_capex'] = (capex_MUSD - user.max_capex_MUSD) / user.max_capex_MUSD
    if user.max_p_gross_MW is not None:
        out['u_max_p_gross'] = (p_gross_MW - user.max_p_gross_MW) / user.max_p_gross_MW
    if user.max_parasitic_ratio is not None:
        out['u_max_parasitic'] = (
            parasitic - user.max_parasitic_ratio
        ) / user.max_parasitic_ratio
    return out
