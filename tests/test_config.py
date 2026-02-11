# -*- coding: utf-8 -*-
"""
Tests for otex.config module.
"""

import pytest
from otex.config import (
    OTEXConfig,
    DataConfig,
    CycleConfig,
    PlantConfig,
    Economics,
    parameters_and_constants,
    get_default_config,
)


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_year(self):
        """Default year should be 2020."""
        config = DataConfig()
        assert config.year == 2020

    def test_auto_date_computation(self):
        """date_start and date_end should be auto-computed from year."""
        config = DataConfig(year=2021)
        assert config.date_start == '2021-01-01 00:00:00'
        assert config.date_end == '2021-12-31 21:00:00'

    def test_explicit_dates_override(self):
        """Explicit dates should not be overwritten."""
        config = DataConfig(
            year=2021,
            date_start='2021-06-01 00:00:00',
            date_end='2021-06-30 21:00:00'
        )
        assert config.date_start == '2021-06-01 00:00:00'
        assert config.date_end == '2021-06-30 21:00:00'

    def test_default_source(self):
        """Default data source should be CMEMS."""
        config = DataConfig()
        assert config.source == 'CMEMS'


class TestCycleConfig:
    """Tests for CycleConfig dataclass."""

    def test_default_cycle(self):
        """Default cycle should be rankine_closed."""
        config = CycleConfig()
        assert config.cycle_type == 'rankine_closed'

    def test_default_fluid(self):
        """Default fluid should be ammonia."""
        config = CycleConfig()
        assert config.fluid_type == 'ammonia'

    def test_ammonia_concentration(self):
        """Ammonia concentration should have a default value."""
        config = CycleConfig()
        assert config.ammonia_concentration == 0.7


class TestOTEXConfig:
    """Tests for OTEXConfig dataclass."""

    def test_default_config_creation(self):
        """Should create config with all default values."""
        config = OTEXConfig()
        assert config.plant.gross_power == -136000.0
        assert config.cycle.cycle_type == 'rankine_closed'
        assert config.data.year == 2020

    def test_to_legacy_dict_contains_dates(self):
        """Legacy dict should contain year, date_start, date_end."""
        config = OTEXConfig(data=DataConfig(year=2022))
        legacy = config.to_legacy_dict()

        assert legacy['year'] == 2022
        assert legacy['date_start'] == '2022-01-01 00:00:00'
        assert legacy['date_end'] == '2022-12-31 21:00:00'

    def test_to_legacy_dict_contains_working_fluid(self):
        """Legacy dict should contain working_fluid object."""
        config = OTEXConfig()
        legacy = config.to_legacy_dict()

        assert legacy['working_fluid'] is not None
        assert hasattr(legacy['working_fluid'], 'saturation_pressure')

    def test_to_legacy_dict_contains_cycle(self):
        """Legacy dict should contain thermodynamic_cycle object."""
        config = OTEXConfig()
        legacy = config.to_legacy_dict()

        assert legacy['thermodynamic_cycle'] is not None
        assert hasattr(legacy['thermodynamic_cycle'], 'calculate_cycle_states')

    def test_to_legacy_dict_open_cycle_no_fluid(self):
        """Open cycle should have None working_fluid."""
        config = OTEXConfig(cycle=CycleConfig(cycle_type='rankine_open'))
        legacy = config.to_legacy_dict()

        assert legacy['working_fluid'] is None

    def test_to_legacy_dict_kalina_no_external_fluid(self):
        """Kalina cycle should have None working_fluid (uses internal mixture)."""
        config = OTEXConfig(cycle=CycleConfig(cycle_type='kalina'))
        legacy = config.to_legacy_dict()

        assert legacy['working_fluid'] is None

    def test_config_strings_in_legacy_dict(self):
        """Legacy dict should contain config_cycle_type and config_fluid_type."""
        config = OTEXConfig(
            cycle=CycleConfig(cycle_type='kalina', fluid_type='ammonia')
        )
        legacy = config.to_legacy_dict()

        assert legacy['config_cycle_type'] == 'kalina'
        assert legacy['config_fluid_type'] == 'ammonia'


class TestParametersAndConstants:
    """Tests for parameters_and_constants function."""

    def test_default_parameters(self):
        """Should return dict with default parameters."""
        inputs = parameters_and_constants()

        assert inputs['p_gross'] == -136000
        assert inputs['cost_level'] == 'low_cost'
        assert inputs['cycle_type'] == 'rankine_closed'

    def test_year_parameter(self):
        """Year parameter should set dates correctly."""
        inputs = parameters_and_constants(year=2023)

        assert inputs['year'] == 2023
        assert inputs['date_start'] == '2023-01-01 00:00:00'
        assert inputs['date_end'] == '2023-12-31 21:00:00'

    def test_cycle_type_parameter(self):
        """Cycle type parameter should be passed through."""
        inputs = parameters_and_constants(cycle_type='kalina')
        assert inputs['cycle_type'] == 'kalina'

    def test_fluid_type_parameter(self):
        """Fluid type parameter should be passed through."""
        # Use ammonia since other fluids require CoolProp
        inputs = parameters_and_constants(fluid_type='ammonia')
        assert inputs['fluid_type'] == 'ammonia'

    @pytest.mark.requires_coolprop
    def test_fluid_type_r134a_with_coolprop(self):
        """R134a fluid type requires CoolProp."""
        try:
            inputs = parameters_and_constants(fluid_type='r134a', use_coolprop=True)
            assert inputs['fluid_type'] == 'r134a'
        except (ImportError, ValueError):
            pytest.skip("CoolProp not available")

    def test_working_fluid_created(self):
        """Working fluid should be auto-created."""
        inputs = parameters_and_constants()
        assert inputs['working_fluid'] is not None

    def test_thermodynamic_cycle_created(self):
        """Thermodynamic cycle should be auto-created."""
        inputs = parameters_and_constants()
        assert inputs['thermodynamic_cycle'] is not None


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_returns_otex_config(self):
        """Should return OTEXConfig instance."""
        config = get_default_config()
        assert isinstance(config, OTEXConfig)

    def test_year_alias(self):
        """Year alias should work."""
        config = get_default_config(year=2025)
        assert config.data.year == 2025
        assert config.data.date_start == '2025-01-01 00:00:00'

    def test_cycle_type_alias(self):
        """Cycle type alias should work."""
        config = get_default_config(cycle_type='uehara')
        assert config.cycle.cycle_type == 'uehara'

    def test_gross_power_alias(self):
        """Gross power alias should work."""
        config = get_default_config(gross_power=-50000)
        assert config.plant.gross_power == -50000
