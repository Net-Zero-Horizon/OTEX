# -*- coding: utf-8 -*-
"""
Tests for otex.economics module.
"""

import pytest
import numpy as np
from otex.config import parameters_and_constants, Economics


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
