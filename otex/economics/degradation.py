# -*- coding: utf-8 -*-
"""Degradation and OPEX escalation models for multi-year NPV economics.

Three degradation models are supported:

* ``constant``  — exponential decline at a fixed annual rate.
* ``logistic``  — sigmoid decline that accelerates and saturates around an
  inflection year, capturing accelerated wear once a damage threshold is
  crossed.
* ``step``      — discrete drops at scheduled major-maintenance milestones.

Three OPEX escalation models are supported:

* ``flat``        — no escalation (legacy behaviour).
* ``fixed_rate``  — geometric escalation at a fixed annual rate.
* ``indexed``     — user-supplied per-year multipliers (length must equal
  the project lifetime).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np


# ─── Degradation ──────────────────────────────────────────────────────


@dataclass
class DegradationConfig:
    """Power-output degradation model.

    Use ``model`` to select which of the three families is active. Only the
    parameters relevant to the selected model are read.
    """

    model: Literal['constant', 'logistic', 'step'] = 'constant'

    # constant: rate applied each year (fraction lost per year).
    rate: float = 0.005                              # 0.5%/yr

    # logistic: P(t) = 1 - L / (1 + exp(-k * (t - t0)))
    logistic_L: float = 0.30                         # asymptotic loss
    logistic_k: float = 0.30                         # slope
    logistic_t0: float = 15.0                        # inflection year

    # step: discrete drops at scheduled years
    step_years: List[int] = field(default_factory=lambda: [10, 20])
    step_drops: List[float] = field(default_factory=lambda: [0.05, 0.05])


def degradation_factor(lifetime_years: int, cfg: DegradationConfig) -> np.ndarray:
    """Return per-year multiplicative power factor for the project lifetime.

    Year 0 (commissioning) has factor 1.0; subsequent years apply the chosen
    model. Output shape is ``(lifetime_years,)``.
    """
    if lifetime_years <= 0:
        raise ValueError(f"lifetime_years must be positive, got {lifetime_years}")

    t = np.arange(lifetime_years)

    if cfg.model == 'constant':
        if not (0.0 <= cfg.rate < 1.0):
            raise ValueError(
                f"degradation rate must be in [0, 1); got {cfg.rate}"
            )
        return (1.0 - cfg.rate) ** t

    if cfg.model == 'logistic':
        if not (0.0 <= cfg.logistic_L <= 1.0):
            raise ValueError(
                f"logistic_L must be in [0, 1]; got {cfg.logistic_L}"
            )
        return 1.0 - cfg.logistic_L / (1.0 + np.exp(-cfg.logistic_k * (t - cfg.logistic_t0)))

    if cfg.model == 'step':
        if len(cfg.step_years) != len(cfg.step_drops):
            raise ValueError(
                f"step_years ({len(cfg.step_years)}) and step_drops "
                f"({len(cfg.step_drops)}) must have the same length"
            )
        factor = np.ones(lifetime_years, dtype=np.float64)
        for y, drop in zip(cfg.step_years, cfg.step_drops):
            if not (0.0 <= drop < 1.0):
                raise ValueError(
                    f"step drop must be in [0, 1); got {drop} at year {y}"
                )
            if 0 <= y < lifetime_years:
                factor[y:] *= (1.0 - drop)
        return factor

    raise ValueError(f"Unknown degradation model: {cfg.model!r}")


# ─── OPEX escalation ──────────────────────────────────────────────────


@dataclass
class OpexEscalationConfig:
    """Year-on-year OPEX escalation model."""

    model: Literal['flat', 'fixed_rate', 'indexed'] = 'flat'

    # fixed_rate: e.g. 0.02 = 2% per year
    rate: float = 0.0

    # indexed: per-year multipliers (length must equal lifetime_years).
    # First entry applies to year 0 (typically 1.0).
    index: Optional[List[float]] = None


def opex_escalation_factor(lifetime_years: int, cfg: OpexEscalationConfig) -> np.ndarray:
    """Return per-year multiplicative OPEX factor for the project lifetime.

    Year 0 has factor 1.0 for ``flat`` and ``fixed_rate`` (escalation begins
    in year 1). For ``indexed``, the user controls year 0 explicitly.
    """
    if lifetime_years <= 0:
        raise ValueError(f"lifetime_years must be positive, got {lifetime_years}")

    if cfg.model == 'flat':
        return np.ones(lifetime_years, dtype=np.float64)

    if cfg.model == 'fixed_rate':
        if cfg.rate <= -1.0:
            raise ValueError(
                f"opex escalation rate must be > -1.0; got {cfg.rate}"
            )
        return (1.0 + cfg.rate) ** np.arange(lifetime_years)

    if cfg.model == 'indexed':
        if cfg.index is None:
            raise ValueError("indexed escalation requires `index` to be set")
        if len(cfg.index) != lifetime_years:
            raise ValueError(
                f"indexed escalation length ({len(cfg.index)}) must equal "
                f"lifetime_years ({lifetime_years})"
            )
        return np.asarray(cfg.index, dtype=np.float64)

    raise ValueError(f"Unknown OPEX escalation model: {cfg.model!r}")


# ─── Cyclic extrapolation ─────────────────────────────────────────────


def extrapolate_cyclic(p_net_by_year: np.ndarray, lifetime_years: int) -> np.ndarray:
    """Replicate the simulated years cyclically to fill ``lifetime_years``.

    If ``n_simulated < lifetime_years``, the simulated pattern repeats from
    year 0 onwards. If ``n_simulated >= lifetime_years``, the array is
    truncated to ``lifetime_years`` rows.

    Parameters
    ----------
    p_net_by_year : np.ndarray
        Shape ``(n_simulated_years, n_sites)``.
    lifetime_years : int
        Project lifetime to fill.

    Returns
    -------
    np.ndarray
        Shape ``(lifetime_years, n_sites)``.
    """
    if p_net_by_year.ndim != 2:
        raise ValueError(
            f"p_net_by_year must be 2-D; got shape {p_net_by_year.shape}"
        )
    n_sim = p_net_by_year.shape[0]
    if n_sim == 0:
        raise ValueError("p_net_by_year has zero simulated years")

    n_repeat = int(np.ceil(lifetime_years / n_sim))
    tiled = np.tile(p_net_by_year, (n_repeat, 1))
    return tiled[:lifetime_years, :]
