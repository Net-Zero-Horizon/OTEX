# -*- coding: utf-8 -*-
"""
Tests for otex.plant module.
"""

import pytest
import numpy as np


class TestPlantComponents:
    """Tests for plant component calculations."""

    def test_imports(self):
        """Plant modules should be importable."""
        from otex.plant import components
        from otex.plant import sizing
        from otex.plant import operation
        from otex.plant import utils

    def test_components_module_functions(self):
        """Components module should have expected functions."""
        from otex.plant import components

        # Check for common component-related functions
        module_contents = dir(components)
        assert len(module_contents) > 0


class TestPlantSizing:
    """Tests for plant sizing calculations."""

    def test_sizing_module_exists(self):
        """Sizing module should exist."""
        from otex.plant import sizing
        assert sizing is not None


class TestPlantOperation:
    """Tests for plant operation calculations."""

    def test_operation_module_exists(self):
        """Operation module should exist."""
        from otex.plant import operation
        assert operation is not None


class TestOffDesignAnalysis:
    """Tests for off-design analysis module."""

    def test_off_design_module_exists(self):
        """Off-design module should exist."""
        from otex.plant import off_design_analysis
        assert off_design_analysis is not None

    def test_off_design_function_exists(self):
        """off_design_analysis function should exist."""
        from otex.plant.off_design_analysis import off_design_analysis
        assert callable(off_design_analysis)


class TestPipeCalculations:
    """Tests for pipe-related calculations."""

    def test_pipe_properties_in_config(self):
        """Pipe properties should be in config."""
        from otex.config import parameters_and_constants

        inputs = parameters_and_constants()

        # Check pipe-related keys
        assert 'length_WW' in inputs
        assert 'length_CW' in inputs
        assert 'SDR_ratio' in inputs
        assert 'rho_pipe' in inputs
        assert 'roughness_pipe' in inputs

    def test_pipe_lengths_positive(self):
        """Pipe lengths should be positive."""
        from otex.config import parameters_and_constants

        inputs = parameters_and_constants()

        assert inputs['length_WW'] > 0
        assert inputs['length_CW'] > 0

    def test_cw_pipe_longer_than_ww(self):
        """Cold water pipe should be longer than warm water pipe."""
        from otex.config import parameters_and_constants

        inputs = parameters_and_constants()

        # CW intake is typically at 1000m depth, WW at 20m
        assert inputs['length_CW'] > inputs['length_WW']


class TestHeatExchangerParameters:
    """Tests for heat exchanger parameters."""

    def test_heat_transfer_coefficients(self):
        """Heat transfer coefficients should be present."""
        from otex.config import parameters_and_constants

        inputs = parameters_and_constants()

        assert 'U_evap' in inputs
        assert 'U_cond' in inputs
        assert inputs['U_evap'] > 0
        assert inputs['U_cond'] > 0

    def test_pinch_temperatures(self):
        """Pinch point temperatures should be defined."""
        from otex.config import parameters_and_constants

        inputs = parameters_and_constants()

        assert 'T_pinch_WW' in inputs
        assert 'T_pinch_CW' in inputs
        assert inputs['T_pinch_WW'] > 0
        assert inputs['T_pinch_CW'] > 0


class TestEfficiencies:
    """Tests for component efficiencies."""

    def test_turbine_efficiencies(self):
        """Turbine efficiencies should be present and valid."""
        from otex.config import parameters_and_constants

        inputs = parameters_and_constants()

        assert 'eff_isen_turb' in inputs
        assert 0 < inputs['eff_isen_turb'] <= 1

        assert 'eff_turb_mech' in inputs
        assert 0 < inputs['eff_turb_mech'] <= 1

        assert 'eff_turb_el' in inputs
        assert 0 < inputs['eff_turb_el'] <= 1

    def test_pump_efficiencies(self):
        """Pump efficiencies should be present and valid."""
        from otex.config import parameters_and_constants

        inputs = parameters_and_constants()

        assert 'eff_isen_pump' in inputs
        assert 0 < inputs['eff_isen_pump'] <= 1

        assert 'eff_hyd' in inputs
        assert 0 < inputs['eff_hyd'] <= 1

    def test_efficiencies_array(self):
        """Efficiencies array should be present."""
        from otex.config import parameters_and_constants

        inputs = parameters_and_constants()

        assert 'efficiencies' in inputs
        assert len(inputs['efficiencies']) == 8


class TestDepthLimits:
    """Tests for depth limit parameters."""

    def test_depth_limits_present(self):
        """Depth limits should be in config."""
        from otex.config import parameters_and_constants

        inputs = parameters_and_constants()

        assert 'min_depth' in inputs
        assert 'max_depth' in inputs

    def test_depth_limits_negative(self):
        """Depth limits should be negative (below sea level convention)."""
        from otex.config import parameters_and_constants

        inputs = parameters_and_constants()

        # Convention: negative values for depth below sea level
        assert inputs['min_depth'] < 0
        assert inputs['max_depth'] < 0

    def test_max_depth_deeper_than_min(self):
        """Max depth should be deeper (more negative) than min depth."""
        from otex.config import parameters_and_constants

        inputs = parameters_and_constants()

        # More negative = deeper
        assert inputs['max_depth'] < inputs['min_depth']
