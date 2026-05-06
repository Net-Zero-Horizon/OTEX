# -*- coding: utf-8 -*-
"""Time-series aggregation helpers for the economics module.

Multi-year simulations produce ``p_net`` arrays with shape
``(n_timesteps, n_sites)`` covering several calendar years. The NPV-based
LCOE expects per-year quantities, so this module provides the bridge
between raw time-series output and annual cashflow inputs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..config import hours_in_year


def aggregate_p_net_by_year(p_net: np.ndarray, timestamp) -> tuple[np.ndarray, list[int]]:
    """Average ``p_net`` over each calendar year.

    Parameters
    ----------
    p_net : np.ndarray
        Power profile, shape ``(n_timesteps, n_sites)``. Negative values
        represent power output (consistent with the rest of OTEX).
    timestamp : pd.DatetimeIndex or sequence of datetimes
        Time index aligned with ``p_net`` axis 0.

    Returns
    -------
    p_net_by_year : np.ndarray
        Mean power per (year, site), shape ``(n_years, n_sites)``.
    years : list[int]
        The calendar years in the same order as the rows of ``p_net_by_year``.
    """
    if p_net.ndim != 2:
        raise ValueError(
            f"p_net must be 2-D (n_timesteps, n_sites); got shape {p_net.shape}"
        )
    ts = pd.DatetimeIndex(timestamp)
    if len(ts) != p_net.shape[0]:
        raise ValueError(
            f"timestamp length ({len(ts)}) does not match p_net rows "
            f"({p_net.shape[0]})"
        )

    df = pd.DataFrame(p_net, index=ts)
    grouped = df.groupby(ts.year).mean()
    years = [int(y) for y in grouped.index]
    return grouped.to_numpy(dtype=np.float64), years


def annual_energy_kwh(
    p_net_by_year: np.ndarray,
    years: Sequence[int],
    availability_factor: float,
) -> np.ndarray:
    """Convert mean annual power (kW) into delivered energy (kWh/yr).

    Accounts for leap years via :func:`otex.config.hours_in_year`.

    Parameters
    ----------
    p_net_by_year : np.ndarray
        Mean power per (year, site), shape ``(n_years, n_sites)``. Sign
        convention matches ``p_net`` (negative = output).
    years : sequence of int
        Calendar years aligned with rows of ``p_net_by_year``.
    availability_factor : float
        Fraction of hours the plant is online (e.g. 0.914).

    Returns
    -------
    energy : np.ndarray
        Annual energy in kWh, shape ``(n_years, n_sites)``. Sign flipped so
        positive values represent energy delivered.
    """
    if len(years) != p_net_by_year.shape[0]:
        raise ValueError(
            f"years length ({len(years)}) does not match p_net_by_year rows "
            f"({p_net_by_year.shape[0]})"
        )
    hours = np.array([hours_in_year(int(y)) for y in years], dtype=np.float64)
    return -p_net_by_year * availability_factor * hours[:, None]
