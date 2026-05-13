# -*- coding: utf-8 -*-
"""Tests for the formal design-optimisation mode (otex.optimization).

The evaluator + constraints are unit-tested against a hand-picked
design vector at a synthetic Caribbean-like site (T_WW=28, T_CW=5).
The full SLSQP solve is run end-to-end to confirm convergence and
that the optimum LCOE is no worse than the starting guess.
"""

from __future__ import annotations

import numpy as np
import pytest

from otex.config import parameters_and_constants


@pytest.fixture
def caribbean_inputs():
    """Polynomial-fluid (CoolProp-free) inputs for fast tests."""
    return parameters_and_constants(
        p_gross=-100_000,
        cycle_type='rankine_closed',
        fluid_type='ammonia',
        use_coolprop=False,
    )


@pytest.fixture
def caribbean_site(caribbean_inputs):
    """A typical Caribbean OTEC site with median T_WW=28, T_CW=5."""
    from otex.optimization import SiteContext
    return SiteContext(
        site_id=1,
        longitude=-77.0, latitude=18.0,
        T_WW_in=28.0, T_CW_in=5.0,
        dist_shore=30.0, eff_trans=0.97,
        inputs_template=caribbean_inputs,
    )


class TestDesignVector:

    def test_round_trip_through_array(self):
        from otex.optimization import DesignVector
        x = DesignVector(p_gross=-50_000, dT_WW=3.0, dT_CW=2.5, depth_CW=900.0)
        np.testing.assert_array_equal(
            DesignVector.from_array(x.as_array()).as_array(),
            x.as_array(),
        )

    def test_rejects_wrong_size(self):
        from otex.optimization import DesignVector
        with pytest.raises(ValueError):
            DesignVector.from_array([1.0, 2.0, 3.0])

    def test_default_bounds_shape(self):
        from otex.optimization import DEFAULT_BOUNDS
        scipy_bounds = DEFAULT_BOUNDS.as_scipy()
        assert len(scipy_bounds) == 4
        # All bounds are (lo, hi) tuples and lo < hi (p_gross is negative).
        for lo, hi in scipy_bounds:
            assert lo < hi


class TestEvaluator:

    def test_evaluate_runs_and_returns_finite_lcoe(self, caribbean_site):
        from otex.optimization import DesignVector, evaluate
        x = DesignVector(p_gross=-100_000, dT_WW=3.0, dT_CW=3.0, depth_CW=1000.0)
        res = evaluate(x, caribbean_site)
        assert np.isfinite(res.lcoe) and res.lcoe > 0
        assert res.p_net < 0   # net output negative by OTEX convention
        assert res.capex_total > 0

    def test_evaluator_does_not_mutate_template(self, caribbean_site):
        """Two consecutive evaluations with different p_gross must
        produce different LCOEs — confirms the template is copied."""
        from otex.optimization import DesignVector, evaluate
        x1 = DesignVector(p_gross=-50_000, dT_WW=3, dT_CW=3, depth_CW=1000)
        x2 = DesignVector(p_gross=-200_000, dT_WW=3, dT_CW=3, depth_CW=1000)
        before = dict(caribbean_site.inputs_template)
        r1 = evaluate(x1, caribbean_site)
        r2 = evaluate(x2, caribbean_site)
        assert r1.lcoe != r2.lcoe
        # Template still has its original p_gross.
        assert caribbean_site.inputs_template['p_gross'] == before['p_gross']


class TestConstraints:

    def test_default_design_satisfies_most_constraints(self, caribbean_site):
        from otex.optimization import (
            DesignVector, evaluate, evaluate_constraints,
        )
        x = DesignVector(p_gross=-100_000, dT_WW=3.0, dT_CW=3.0, depth_CW=1000.0)
        res = evaluate(x, caribbean_site)
        cons = evaluate_constraints(x, res, caribbean_site)

        # All declared constraint keys present.
        from otex.optimization import CONSTRAINT_NAMES
        for name in CONSTRAINT_NAMES:
            assert name in cons.values

        # Default Caribbean design should have all hard constraints
        # satisfied (negative g_i values).
        assert cons.max_violation == 0.0
        assert cons.feasible

    def test_pinch_violation_when_dT_too_large(self, caribbean_site):
        """dT_WW + dT_CW > available ΔT must trigger g_dT_total > 0."""
        from otex.optimization import (
            DesignVector, evaluate, evaluate_constraints,
        )
        # ΔT = 23 K; ask for 15 + 10 = 25 K of dT consumption — infeasible.
        x = DesignVector(p_gross=-100_000, dT_WW=15.0, dT_CW=10.0, depth_CW=1000.0)
        res = evaluate(x, caribbean_site)
        cons = evaluate_constraints(x, res, caribbean_site)
        assert cons.values['g_dT_total'] > 0


class TestObjective:

    def test_objective_value_increases_with_violation(self, caribbean_site):
        """A clearly infeasible x must have higher J than a feasible one."""
        from otex.optimization import DesignVector, build_objective
        J = build_objective(caribbean_site)

        x_ok = DesignVector(p_gross=-100_000, dT_WW=3, dT_CW=3, depth_CW=1000)
        x_bad = DesignVector(p_gross=-100_000, dT_WW=20, dT_CW=20, depth_CW=1000)
        assert J(x_bad.as_array()) > J(x_ok.as_array())


class TestOptimizeSite:

    def test_solver_converges_and_improves_lcoe(self, caribbean_site):
        from otex.optimization import (
            DesignVector, evaluate, optimize_site,
        )
        # LCOE at a sensible starting point.
        x0 = DesignVector(p_gross=-100_000, dT_WW=3.0, dT_CW=3.0, depth_CW=1000.0)
        lcoe_start = evaluate(x0, caribbean_site).lcoe

        result = optimize_site(caribbean_site, x0=x0.as_array())
        # L-BFGS-B should converge in fewer than the configured max iter.
        assert result.success
        # The optimum must be at least as good as the starting LCOE
        # (penalty method shouldn't make it strictly worse).
        assert result.lcoe <= lcoe_start + 1e-6
        # The optimum must be approximately feasible (penalty-method
        # tolerance is loose, but well within reason).
        assert result.max_violation < 0.1


class TestUserConstraints:
    """Path B — exogenous user-imposed caps generate interior optima."""

    def test_no_user_constraint_hits_upper_box_bound(self, caribbean_site):
        """Without any user cap, the LCOE function is monotone in
        ``p_gross`` (economies of scale dominate over OTEX's modelled
        range), so the optimum walks to the upper box bound."""
        from otex.optimization import (
            Bounds, UserConstraints, optimize_site,
        )
        bounds = Bounds(p_gross=(-500_000, -1_000), dT_WW=(1, 6),
                        dT_CW=(1, 6), depth_CW=(600, 3000))
        r = optimize_site(caribbean_site, bounds=bounds,
                           user_constraints=UserConstraints())
        # |p_gross_opt| should be very close to the upper bound (500 MW)
        assert abs(r.x.p_gross) > 450_000

    def test_max_p_gross_respected(self, caribbean_site):
        from otex.optimization import (
            Bounds, UserConstraints, optimize_site,
        )
        bounds = Bounds(p_gross=(-500_000, -1_000), dT_WW=(1, 6),
                        dT_CW=(1, 6), depth_CW=(600, 3000))
        cap_MW = 100.0
        r = optimize_site(caribbean_site, bounds=bounds,
                           user_constraints=UserConstraints(
                               max_p_gross_MW=cap_MW,
                           ))
        # Optimum should sit on the user cap (within 2 %).
        assert abs(r.x.p_gross / 1000.0) <= cap_MW * 1.02
        assert r.feasible

    def test_max_capex_respected(self, caribbean_site):
        from otex.optimization import (
            Bounds, UserConstraints, optimize_site,
        )
        bounds = Bounds(p_gross=(-500_000, -1_000), dT_WW=(1, 6),
                        dT_CW=(1, 6), depth_CW=(600, 3000))
        cap_MUSD = 800.0
        r = optimize_site(caribbean_site, bounds=bounds,
                           user_constraints=UserConstraints(
                               max_capex_MUSD=cap_MUSD,
                           ))
        assert r.capex_total / 1e6 <= cap_MUSD * 1.02
        assert r.feasible

    def test_max_p_net_respected(self, caribbean_site):
        from otex.optimization import (
            Bounds, UserConstraints, optimize_site,
        )
        bounds = Bounds(p_gross=(-500_000, -1_000), dT_WW=(1, 6),
                        dT_CW=(1, 6), depth_CW=(600, 3000))
        cap_MW = 50.0
        r = optimize_site(caribbean_site, bounds=bounds,
                           user_constraints=UserConstraints(
                               max_p_net_MW=cap_MW,
                           ))
        assert abs(r.p_net) / 1000.0 <= cap_MW * 1.05
        assert r.feasible

    def test_tighter_cap_yields_higher_lcoe(self, caribbean_site):
        """Tightening the user cap should monotonically raise LCOE
        (less plant size → diseconomies of scale → costlier per kWh)."""
        from otex.optimization import (
            Bounds, UserConstraints, optimize_site,
        )
        bounds = Bounds(p_gross=(-500_000, -1_000), dT_WW=(1, 6),
                        dT_CW=(1, 6), depth_CW=(600, 3000))
        r_big = optimize_site(caribbean_site, bounds=bounds,
                               user_constraints=UserConstraints(max_p_gross_MW=200))
        r_small = optimize_site(caribbean_site, bounds=bounds,
                                 user_constraints=UserConstraints(max_p_gross_MW=50))
        assert r_small.lcoe > r_big.lcoe
