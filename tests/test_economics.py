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
