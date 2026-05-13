# -*- coding: utf-8 -*-
"""OTEX formal design-optimization mode.

The package wraps the existing forward simulation pipeline
(``otec_sizing`` + ``capex_opex_lcoe``) inside a continuous
non-linear program that minimises LCOE plus quadratic penalties on
physical/technical constraint violations. Categorical design
choices (cycle type, working fluid, installation type) are taken as
**exogenous** — the user picks them and runs the optimisation;
comparing alternatives means running it several times.

Public API:

* :class:`DesignVector` — the four continuous decision variables.
* :class:`SiteContext`  — site temperatures and ``inputs`` template.
* :func:`evaluate`      — pure function ``x → DesignResult``.
* :func:`optimize_site` — local NLP solve for a single site.
* :func:`run_regional_optimization` — loop over every site in a region.
"""

from .design_vector import DesignVector, DEFAULT_BOUNDS, Bounds
from .evaluator import SiteContext, DesignResult, evaluate
from .constraints import ConstraintResult, evaluate_constraints, CONSTRAINT_NAMES
from .objective import build_objective, DEFAULT_PENALTY_WEIGHTS
from .user_constraints import UserConstraints, evaluate_user_constraints
from .optimize import (
    OptimizationResult,
    optimize_site,
    run_regional_optimization,
)

__all__ = [
    'DesignVector',
    'DEFAULT_BOUNDS',
    'Bounds',
    'SiteContext',
    'DesignResult',
    'evaluate',
    'ConstraintResult',
    'evaluate_constraints',
    'CONSTRAINT_NAMES',
    'UserConstraints',
    'evaluate_user_constraints',
    'build_objective',
    'DEFAULT_PENALTY_WEIGHTS',
    'OptimizationResult',
    'optimize_site',
    'run_regional_optimization',
]
