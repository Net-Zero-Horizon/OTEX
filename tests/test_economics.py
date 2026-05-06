# -*- coding: utf-8 -*-
"""
Tests for otex.economics module.
"""

import pytest
import numpy as np
from dataclasses import replace
from otex.config import parameters_and_constants, Economics, OTEXConfig
from otex.economics import CostScheme, LOW_COST, HIGH_COST, get_cost_scheme


class TestEconomicsConfig:
    """Tests for Economics configuration."""

    def test_default_lifetime(self):
        """Default lifetime should be 30 years."""
        econ = Economics()
        assert econ.lifetime_years == 30

    def test_default_discount_rate(self):
        """Default discount rate should be 10%."""
        econ = Economics()
        assert econ.discount_rate == 0.10

    def test_crf_calculation(self):
        """CRF should be correctly calculated."""
        econ = Economics(lifetime_years=30, discount_rate=0.10)
        crf = econ.crf

        # Manual calculation: r*(1+r)^n / ((1+r)^n - 1)
        r = 0.10
        n = 30
        expected_crf = r * (1 + r)**n / ((1 + r)**n - 1)

        assert abs(crf - expected_crf) < 1e-10

    def test_crf_different_parameters(self):
        """CRF should change with different parameters."""
        econ1 = Economics(lifetime_years=20, discount_rate=0.08)
        econ2 = Economics(lifetime_years=30, discount_rate=0.10)

        assert econ1.crf != econ2.crf

    def test_availability_factor(self):
        """Default availability should be ~91.4% (8000/8760 hours)."""
        econ = Economics()
        assert 0.90 < econ.availability < 0.92


class TestCostLevel:
    """Tests for cost level configurations."""

    def test_low_cost_pipe_material(self):
        """Low cost should use HDPE pipes."""
        inputs = parameters_and_constants(cost_level='low_cost')

        # HDPE density is ~995 kg/m³
        assert inputs['rho_pipe'] < 1000

    def test_high_cost_pipe_material(self):
        """High cost should use FRP pipes."""
        inputs = parameters_and_constants(cost_level='high_cost')

        # FRP density is ~1016 kg/m³
        assert inputs['rho_pipe'] > 1000

    def test_cost_level_in_inputs(self):
        """Cost level should be accessible in inputs dict."""
        inputs_low = parameters_and_constants(cost_level='low_cost')
        inputs_high = parameters_and_constants(cost_level='high_cost')

        assert inputs_low['cost_level'] == 'low_cost'
        assert inputs_high['cost_level'] == 'high_cost'


class TestCostScheme:
    """Tests for custom CostScheme support."""

    def test_get_cost_scheme_returns_low_cost(self):
        """get_cost_scheme('low_cost') should return the LOW_COST singleton."""
        assert get_cost_scheme('low_cost') is LOW_COST

    def test_get_cost_scheme_returns_high_cost(self):
        """get_cost_scheme('high_cost') should return the HIGH_COST singleton."""
        assert get_cost_scheme('high_cost') is HIGH_COST

    def test_get_cost_scheme_passthrough(self):
        """get_cost_scheme(scheme) should return the same CostScheme object."""
        custom = CostScheme(turbine_coeff=400.0)
        assert get_cost_scheme(custom) is custom

    def test_get_cost_scheme_invalid_string(self):
        """An unknown string should raise ValueError with a helpful message."""
        with pytest.raises(ValueError, match='CostScheme'):
            get_cost_scheme('nonexistent_scheme')

    def test_get_cost_scheme_invalid_type(self):
        """A non-string, non-CostScheme argument should raise TypeError."""
        with pytest.raises(TypeError):
            get_cost_scheme(42)

    def test_builtin_schemes_differ(self):
        """LOW_COST and HIGH_COST must have different parameters."""
        assert LOW_COST.turbine_coeff != HIGH_COST.turbine_coeff
        assert LOW_COST.pipes_coeff != HIGH_COST.pipes_coeff
        assert LOW_COST.opex_fraction != HIGH_COST.opex_fraction
        assert LOW_COST.pipe_density != HIGH_COST.pipe_density

    def test_custom_scheme_from_scratch(self):
        """A CostScheme defined from scratch should be accepted by Economics."""
        custom = CostScheme(
            turbine_coeff=400.0,
            opex_fraction=0.04,
            pipe_density=1000.0,
        )
        econ = Economics(cost_level=custom)
        assert econ.cost_level is custom

    def test_derived_scheme_with_replace(self):
        """replace() should produce a distinct CostScheme with the modified field."""
        modified = replace(LOW_COST, turbine_coeff=999.0)
        assert modified.turbine_coeff == 999.0
        assert modified.opex_fraction == LOW_COST.opex_fraction
        assert modified is not LOW_COST

    def test_custom_scheme_pipe_density_in_legacy_dict(self):
        """A CostScheme with a custom pipe_density should propagate to inputs dict."""
        custom = replace(LOW_COST, pipe_density=1050.0)
        inputs = parameters_and_constants(cost_level=custom)
        assert inputs['rho_pipe'] == pytest.approx(1050.0)

    def test_custom_scheme_stored_in_inputs(self):
        """The CostScheme object itself should be stored under 'cost_level' in the dict."""
        custom = CostScheme(turbine_coeff=500.0)
        inputs = parameters_and_constants(cost_level=custom)
        assert inputs['cost_level'] is custom

    def test_otexconfig_accepts_cost_scheme(self):
        """OTEXConfig should accept a CostScheme as economics.cost_level."""
        custom = replace(HIGH_COST, opex_fraction=0.06)
        config = OTEXConfig(economics=Economics(cost_level=custom))
        legacy = config.to_legacy_dict()
        assert legacy['rho_pipe'] == pytest.approx(HIGH_COST.pipe_density)


class TestEconomicInputs:
    """Tests for economic parameters in inputs dictionary."""

    def test_economic_inputs_array(self):
        """economic_inputs array should be present."""
        inputs = parameters_and_constants()

        assert 'economic_inputs' in inputs
        assert len(inputs['economic_inputs']) == 4

    def test_crf_in_inputs(self):
        """CRF should be in inputs."""
        inputs = parameters_and_constants()

        assert 'crf' in inputs
        assert inputs['crf'] > 0

    def test_lifetime_in_inputs(self):
        """Lifetime should be in inputs."""
        inputs = parameters_and_constants()

        assert 'lifetime' in inputs
        assert inputs['lifetime'] == 30

    def test_discount_rate_in_inputs(self):
        """Discount rate should be in inputs."""
        inputs = parameters_and_constants()

        assert 'discount_rate' in inputs
        assert inputs['discount_rate'] == 0.10


class TestTransmissionThreshold:
    """Tests for AC/DC transmission threshold."""

    def test_threshold_exists(self):
        """AC/DC threshold should be in inputs."""
        inputs = parameters_and_constants()

        assert 'threshold_AC_DC' in inputs

    def test_threshold_value(self):
        """Default threshold should be 50 km."""
        inputs = parameters_and_constants()

        assert inputs['threshold_AC_DC'] == 50.0


class TestLCOECalculation:
    """Conceptual tests for LCOE calculation logic."""

    def test_lcoe_components_available(self):
        """All components needed for LCOE should be available."""
        inputs = parameters_and_constants()

        # Required for LCOE calculation
        required_keys = [
            'crf',
            'availability_factor',
            'lifetime',
        ]

        for key in required_keys:
            assert key in inputs, f"Missing key: {key}"

    def test_higher_crf_means_higher_cost(self):
        """Higher CRF (shorter lifetime or higher discount) should increase annual cost."""
        # Shorter lifetime = higher CRF
        econ_short = Economics(lifetime_years=15, discount_rate=0.10)
        econ_long = Economics(lifetime_years=30, discount_rate=0.10)

        assert econ_short.crf > econ_long.crf

        # Higher discount rate = higher CRF
        econ_low_dr = Economics(lifetime_years=30, discount_rate=0.05)
        econ_high_dr = Economics(lifetime_years=30, discount_rate=0.15)

        assert econ_high_dr.crf > econ_low_dr.crf


class TestTimeSeriesAggregation:
    """Tests for the per-year aggregation helpers used by NPV economics."""

    def test_aggregate_p_net_by_year_3_years(self):
        import pandas as pd
        from otex.economics.timeseries import aggregate_p_net_by_year

        ts = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        n = len(ts)
        p_net = np.zeros((n, 2))
        for i, year in enumerate([2020, 2021, 2022]):
            mask = ts.year == year
            p_net[mask, 0] = -100.0 * (i + 1)
            p_net[mask, 1] = -50.0 * (i + 1)

        p_by_year, years = aggregate_p_net_by_year(p_net, ts)
        assert years == [2020, 2021, 2022]
        assert p_by_year.shape == (3, 2)
        np.testing.assert_allclose(p_by_year[:, 0], [-100, -200, -300])
        np.testing.assert_allclose(p_by_year[:, 1], [-50, -100, -150])

    def test_aggregate_p_net_rejects_1d(self):
        import pandas as pd
        from otex.economics.timeseries import aggregate_p_net_by_year

        with pytest.raises(ValueError, match="2-D"):
            aggregate_p_net_by_year(
                np.zeros(365),
                pd.date_range('2020-01-01', periods=365),
            )

    def test_annual_energy_kwh_accounts_for_leap_year(self):
        from otex.economics.timeseries import annual_energy_kwh

        # Constant -100 kW. 2020 is leap, 2021 is not.
        p = np.array([[-100.0], [-100.0]])
        e = annual_energy_kwh(p, [2020, 2021], availability_factor=1.0)
        assert e[0, 0] == 100 * 8784
        assert e[1, 0] == 100 * 8760
        assert e[0, 0] - e[1, 0] == 100 * 24

    def test_annual_energy_applies_availability(self):
        from otex.economics.timeseries import annual_energy_kwh

        p = np.array([[-100.0]])
        e_full = annual_energy_kwh(p, [2021], availability_factor=1.0)
        e_partial = annual_energy_kwh(p, [2021], availability_factor=0.9)
        np.testing.assert_allclose(e_partial, 0.9 * e_full)


class TestDegradationModels:
    """Tests for the three degradation models."""

    def test_constant_degradation_year_zero_is_one(self):
        from otex.economics.degradation import DegradationConfig, degradation_factor
        cfg = DegradationConfig(model='constant', rate=0.005)
        f = degradation_factor(30, cfg)
        assert f.shape == (30,)
        assert f[0] == 1.0
        np.testing.assert_allclose(f[1], 0.995)
        np.testing.assert_allclose(f[10], 0.995 ** 10)

    def test_constant_zero_rate_means_no_degradation(self):
        from otex.economics.degradation import DegradationConfig, degradation_factor
        cfg = DegradationConfig(model='constant', rate=0.0)
        f = degradation_factor(30, cfg)
        np.testing.assert_array_equal(f, np.ones(30))

    def test_constant_rejects_invalid_rate(self):
        from otex.economics.degradation import DegradationConfig, degradation_factor
        with pytest.raises(ValueError, match="rate"):
            degradation_factor(30, DegradationConfig(model='constant', rate=1.0))
        with pytest.raises(ValueError, match="rate"):
            degradation_factor(30, DegradationConfig(model='constant', rate=-0.1))

    def test_logistic_monotonically_decreasing(self):
        from otex.economics.degradation import DegradationConfig, degradation_factor
        cfg = DegradationConfig(model='logistic', logistic_L=0.3,
                                logistic_k=0.3, logistic_t0=15.0)
        f = degradation_factor(30, cfg)
        assert np.all(np.diff(f) <= 0), "logistic must be monotonically non-increasing"
        # At t0, factor should be close to 1 - L/2.
        assert abs(f[15] - (1 - 0.3 / 2)) < 0.05

    def test_logistic_saturates_to_1_minus_L(self):
        from otex.economics.degradation import DegradationConfig, degradation_factor
        cfg = DegradationConfig(model='logistic', logistic_L=0.3,
                                logistic_k=1.0, logistic_t0=10.0)
        f = degradation_factor(50, cfg)
        # By year 50, the sigmoid should be very close to its asymptote.
        np.testing.assert_allclose(f[-1], 0.7, atol=0.01)

    def test_step_drops_apply_at_scheduled_years(self):
        from otex.economics.degradation import DegradationConfig, degradation_factor
        cfg = DegradationConfig(model='step',
                                step_years=[10, 20], step_drops=[0.05, 0.05])
        f = degradation_factor(30, cfg)
        # Years 0..9 are unaffected.
        np.testing.assert_array_equal(f[:10], np.ones(10))
        # Years 10..19: 5% loss.
        np.testing.assert_allclose(f[10:20], 0.95)
        # Years 20+: another 5% loss compounded.
        np.testing.assert_allclose(f[20:], 0.95 * 0.95)

    def test_step_lengths_must_match(self):
        from otex.economics.degradation import DegradationConfig, degradation_factor
        cfg = DegradationConfig(model='step',
                                step_years=[10, 20], step_drops=[0.05])
        with pytest.raises(ValueError, match="same length"):
            degradation_factor(30, cfg)

    def test_unknown_model_raises(self):
        from otex.economics.degradation import DegradationConfig, degradation_factor
        cfg = DegradationConfig()
        cfg.model = 'wibble'  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown degradation model"):
            degradation_factor(30, cfg)


class TestOpexEscalation:
    """Tests for OPEX escalation models."""

    def test_flat_returns_ones(self):
        from otex.economics.degradation import (
            OpexEscalationConfig, opex_escalation_factor,
        )
        f = opex_escalation_factor(30, OpexEscalationConfig(model='flat'))
        np.testing.assert_array_equal(f, np.ones(30))

    def test_fixed_rate_geometric(self):
        from otex.economics.degradation import (
            OpexEscalationConfig, opex_escalation_factor,
        )
        f = opex_escalation_factor(
            30, OpexEscalationConfig(model='fixed_rate', rate=0.02)
        )
        assert f[0] == 1.0
        np.testing.assert_allclose(f[1], 1.02)
        np.testing.assert_allclose(f[10], 1.02 ** 10)

    def test_indexed_uses_user_array(self):
        from otex.economics.degradation import (
            OpexEscalationConfig, opex_escalation_factor,
        )
        custom = [1.0, 1.1, 1.2, 1.3, 1.4]
        f = opex_escalation_factor(
            5, OpexEscalationConfig(model='indexed', index=custom)
        )
        np.testing.assert_array_equal(f, np.array(custom))

    def test_indexed_length_must_match_lifetime(self):
        from otex.economics.degradation import (
            OpexEscalationConfig, opex_escalation_factor,
        )
        cfg = OpexEscalationConfig(model='indexed', index=[1.0, 1.1])
        with pytest.raises(ValueError, match="length"):
            opex_escalation_factor(5, cfg)

    def test_indexed_requires_index(self):
        from otex.economics.degradation import (
            OpexEscalationConfig, opex_escalation_factor,
        )
        with pytest.raises(ValueError, match="requires"):
            opex_escalation_factor(5, OpexEscalationConfig(model='indexed'))


class TestCyclicExtrapolation:
    """Tests for the cyclic extrapolation helper."""

    def test_extends_short_simulation_to_full_lifetime(self):
        from otex.economics.degradation import extrapolate_cyclic
        p = np.array([[-100, -200], [-110, -210], [-120, -220]])
        out = extrapolate_cyclic(p, lifetime_years=10)
        assert out.shape == (10, 2)
        # First 3 rows match the input.
        np.testing.assert_array_equal(out[:3], p)
        # Year 3 should wrap back to year 0.
        np.testing.assert_array_equal(out[3], p[0])
        np.testing.assert_array_equal(out[6], p[0])

    def test_truncates_long_simulation(self):
        from otex.economics.degradation import extrapolate_cyclic
        p = np.arange(40).reshape(20, 2).astype(np.float64)
        out = extrapolate_cyclic(p, lifetime_years=5)
        assert out.shape == (5, 2)
        np.testing.assert_array_equal(out, p[:5])


class TestLcoeNpv:
    """Tests for the multi-year NPV LCOE."""

    def _make_inputs(self, lifetime=30, discount=0.10, avail=1.0,
                     deg_cfg=None, esc_cfg=None):
        from otex.economics.degradation import (
            DegradationConfig, OpexEscalationConfig,
        )
        return {
            'lifetime': lifetime,
            'discount_rate': discount,
            'availability_factor': avail,
            'degradation_config': deg_cfg or DegradationConfig(rate=0.0),
            'opex_escalation_config': esc_cfg or OpexEscalationConfig(model='flat'),
        }

    def test_lcoe_npv_basic_shape(self):
        from otex.economics.costs import lcoe_npv
        plant = {'CAPEX': np.array([1e8]), 'OPEX': np.array([1e6])}
        p_by_year = np.array([[-10000.0]])  # 1 year, 1 site, 10 MW
        result = lcoe_npv(plant, self._make_inputs(), p_by_year, [2020])
        assert result.shape == (1,)
        assert result[0] > 0

    def test_lcoe_npv_matches_legacy_when_zero_degradation_zero_escalation(self):
        """With no degradation, no escalation, and no leap years in the cycle,
        the NPV LCOE should match the legacy single-rate CRF formula closely.
        """
        from otex.economics.costs import lcoe_npv
        from otex.config import Economics
        from otex.economics.degradation import (
            DegradationConfig, OpexEscalationConfig,
        )

        capex, opex = 1e8, 1e6
        plant = {'CAPEX': np.array([capex]), 'OPEX': np.array([opex])}
        p_by_year = np.array([[-10000.0]])

        inputs = {
            'lifetime': 30,
            'discount_rate': 0.10,
            'availability_factor': 1.0,
            'degradation_config': DegradationConfig(model='constant', rate=0.0),
            'opex_escalation_config': OpexEscalationConfig(model='flat'),
        }

        # Use a non-leap-year-only cycle (2021 only) to make the comparison
        # clean against the 8760-based legacy formula.
        npv_lcoe = lcoe_npv(plant, inputs, p_by_year, [2021])

        econ = Economics(lifetime_years=30, discount_rate=0.10)
        legacy_lcoe = (capex * econ.crf + opex) * 100 / (10000 * 1.0 * 8760)

        np.testing.assert_allclose(npv_lcoe[0], legacy_lcoe, rtol=1e-6)

    def test_lcoe_increases_with_degradation(self):
        from otex.economics.costs import lcoe_npv
        from otex.economics.degradation import DegradationConfig

        plant = {'CAPEX': np.array([1e8]), 'OPEX': np.array([1e6])}
        p_by_year = np.array([[-10000.0]])

        no_deg = self._make_inputs(deg_cfg=DegradationConfig(rate=0.0))
        with_deg = self._make_inputs(deg_cfg=DegradationConfig(rate=0.01))

        lcoe_no = lcoe_npv(plant, no_deg, p_by_year, [2021])
        lcoe_yes = lcoe_npv(plant, with_deg, p_by_year, [2021])

        assert lcoe_yes[0] > lcoe_no[0], \
            "Degradation should reduce delivered energy and raise LCOE"

    def test_lcoe_increases_with_opex_escalation(self):
        from otex.economics.costs import lcoe_npv
        from otex.economics.degradation import OpexEscalationConfig

        plant = {'CAPEX': np.array([1e8]), 'OPEX': np.array([1e6])}
        p_by_year = np.array([[-10000.0]])

        flat = self._make_inputs(esc_cfg=OpexEscalationConfig(model='flat'))
        rising = self._make_inputs(
            esc_cfg=OpexEscalationConfig(model='fixed_rate', rate=0.02)
        )

        lcoe_flat = lcoe_npv(plant, flat, p_by_year, [2021])
        lcoe_rising = lcoe_npv(plant, rising, p_by_year, [2021])

        assert lcoe_rising[0] > lcoe_flat[0]

    def test_lcoe_npv_uses_multiyear_average_correctly(self):
        """3 years with different power should give the same result as a
        single year at the cyclically-replicated mean — verified via energy
        conservation when degradation/escalation are off."""
        from otex.economics.costs import lcoe_npv
        from otex.economics.degradation import (
            DegradationConfig, OpexEscalationConfig,
        )

        plant = {'CAPEX': np.array([1e8]), 'OPEX': np.array([1e6])}
        # Three different years (all non-leap to keep hours equal).
        p_3yr = np.array([[-10000.0], [-10000.0], [-10000.0]])
        p_1yr = np.array([[-10000.0]])

        inputs = {
            'lifetime': 30,
            'discount_rate': 0.10,
            'availability_factor': 1.0,
            'degradation_config': DegradationConfig(rate=0.0),
            'opex_escalation_config': OpexEscalationConfig(model='flat'),
        }

        lcoe_3 = lcoe_npv(plant, inputs, p_3yr, [2021, 2022, 2023])
        lcoe_1 = lcoe_npv(plant, inputs, p_1yr, [2021])
        # With identical power per year and matching non-leap calendar, the
        # two paths must converge to the same LCOE.
        np.testing.assert_allclose(lcoe_3[0], lcoe_1[0], rtol=1e-6)

    def test_lcoe_npv_handles_2d_capex_opex_shape(self):
        """Regression: capex_opex_lcoe stores CAPEX/OPEX with shape
        (1, n_sites) because siting risk multipliers carry that extra axis.
        lcoe_npv must squeeze them to 1-D so the result is also 1-D.
        """
        from otex.economics.costs import lcoe_npv
        plant = {
            'CAPEX': np.array([[1e8, 1.5e8, 2e8]]),  # shape (1, 3)
            'OPEX': np.array([[1e6, 1.5e6, 2e6]]),   # shape (1, 3)
        }
        p_by_year = np.full((4, 3), -10000.0)        # 4 years, 3 sites
        result = lcoe_npv(plant, self._make_inputs(), p_by_year,
                          [2020, 2021, 2022, 2023])
        assert result.shape == (3,), \
            f"Expected (3,), got {result.shape} — broadcasting bug"

    def test_lcoe_npv_rejects_non_positive_energy(self):
        from otex.economics.costs import lcoe_npv

        plant = {'CAPEX': np.array([1e8]), 'OPEX': np.array([1e6])}
        # Positive p_net means the plant is consuming power → energy negative.
        p_by_year = np.array([[+10000.0]])

        with pytest.raises(ValueError, match="non-positive"):
            lcoe_npv(plant, self._make_inputs(), p_by_year, [2021])


class TestLcoeTimeSeriesDeprecation:
    """The legacy lcoe_time_series should warn when called without timestamp."""

    def test_warns_without_timestamp(self):
        import warnings
        from otex.economics.costs import lcoe_time_series

        plant = {'CAPEX': np.array([1e8]), 'OPEX': np.array([1e6])}
        inputs = {'crf': 0.106, 'availability_factor': 1.0}
        p_ts = np.full((365, 1), -10000.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            lcoe_time_series(plant, inputs, p_ts)
        assert any(issubclass(wi.category, DeprecationWarning) for wi in w), \
            f"Expected DeprecationWarning, got {[wi.category for wi in w]}"

    def test_no_warning_when_timestamp_provided(self):
        import warnings
        import pandas as pd
        from otex.economics.costs import lcoe_time_series

        plant = {'CAPEX': np.array([1e8]), 'OPEX': np.array([1e6])}
        from otex.economics.degradation import (
            DegradationConfig, OpexEscalationConfig,
        )
        inputs = {
            'lifetime': 30,
            'discount_rate': 0.10,
            'availability_factor': 1.0,
            'degradation_config': DegradationConfig(rate=0.0),
            'opex_escalation_config': OpexEscalationConfig(model='flat'),
        }
        ts = pd.date_range('2021-01-01', periods=365, freq='D')
        p_ts = np.full((365, 1), -10000.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            lcoe_time_series(plant, inputs, p_ts, timestamp=ts)
        assert not any(issubclass(wi.category, DeprecationWarning) for wi in w)
