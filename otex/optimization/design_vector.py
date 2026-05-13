# -*- coding: utf-8 -*-
"""The four continuous decision variables of the OTEC design problem.

Anything categorical (cycle type, working fluid, installation type)
is **specified by the user** before the optimisation runs — those
choices change the entire model structure and would require a
mixed-integer formulation if included as decision variables. To
compare alternatives, run the optimiser several times with each
configuration.

Sign conventions match the rest of OTEX:

* ``p_gross`` is NEGATIVE (gross power output, e.g. ``-100_000`` kW).
* ``dT_WW`` and ``dT_CW`` are positive temperature differences in °C
  across the warm/cold side of the heat exchangers.
* ``depth_CW`` is positive depth in metres of the cold-water intake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class DesignVector:
    """Four-dimensional continuous design vector ``x = (p_gross, dT_WW, dT_CW, depth_CW)``."""

    p_gross: float        # kW, negative (gross power OUTPUT)
    dT_WW: float          # °C, warm-side temperature drop in evaporator HX
    dT_CW: float          # °C, cold-side temperature rise in condenser HX
    depth_CW: float       # m, cold-water intake depth (positive)

    def as_array(self) -> np.ndarray:
        """Return ``[p_gross, dT_WW, dT_CW, depth_CW]`` as a 1-D array."""
        return np.array([self.p_gross, self.dT_WW, self.dT_CW, self.depth_CW],
                        dtype=np.float64)

    @classmethod
    def from_array(cls, x) -> 'DesignVector':
        """Inverse of :meth:`as_array`."""
        a = np.asarray(x, dtype=np.float64).ravel()
        if a.size != 4:
            raise ValueError(f"DesignVector requires 4 elements, got {a.size}")
        return cls(p_gross=float(a[0]), dT_WW=float(a[1]),
                   dT_CW=float(a[2]), depth_CW=float(a[3]))


@dataclass(frozen=True)
class Bounds:
    """Box bounds on the design vector. Lower/upper, per variable."""

    p_gross: Tuple[float, float]      # kW, negative; e.g. (-500_000, -1_000)
    dT_WW:   Tuple[float, float]      # °C
    dT_CW:   Tuple[float, float]      # °C
    depth_CW: Tuple[float, float]     # m

    def as_scipy(self) -> list:
        """Convert to the ``[(lo, hi), ...]`` form scipy.optimize wants."""
        return [self.p_gross, self.dT_WW, self.dT_CW, self.depth_CW]


# Sensible default bounds. The cost-coefficient correlations in
# economics/costs.py are calibrated around 100 MW, so we cap the
# upper bound at 500 MW (mild extrapolation) and the lower bound at
# 1 MW (below this, OTEC component costs lose meaning).
DEFAULT_BOUNDS = Bounds(
    p_gross=(-500_000.0, -1_000.0),
    dT_WW=(1.0, 6.0),
    dT_CW=(1.0, 6.0),
    depth_CW=(600.0, 3_000.0),
)
