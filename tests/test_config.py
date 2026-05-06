# -*- coding: utf-8 -*-
"""
Tests for otex.config module.
"""

import warnings

import pytest
from otex.config import (
    OTEXConfig,
    DataConfig,
    CycleConfig,
    PlantConfig,
    Economics,
    parameters_and_constants,
    get_default_config,
    hours_in_year,
    hours_in_span,
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
        assert config.plant.gross_power == -100000.0
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

        assert inputs['p_gross'] == -100000
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


class TestMultiYearConfig:
    """Tests for multi-year simulation configuration (added in 0.2.0)."""

    def test_default_is_single_year(self):
        config = DataConfig()
        assert config.year_start == 2020
        assert config.year_end == 2020
        assert config.n_years == 1
        assert config.years == [2020]
        assert config.year_label == '2020'

    def test_multi_year_range(self):
        config = DataConfig(year_start=2020, year_end=2022)
        assert config.n_years == 3
        assert config.years == [2020, 2021, 2022]
        assert config.year_label == '2020-2022'
        assert config.date_start == '2020-01-01 00:00:00'
        assert config.date_end == '2022-12-31 21:00:00'

    def test_year_alias_backcompat_emits_deprecation(self):
        with pytest.warns(DeprecationWarning, match="year_start/year_end"):
            config = DataConfig(year=2019)
        assert config.year_start == 2019
        assert config.year_end == 2019
        assert config.n_years == 1

    def test_year_with_explicit_end(self):
        # Edge case: legacy `year` plus a new `year_end` extends the range.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = DataConfig(year=2020, year_end=2022)
        assert config.year_start == 2020
        assert config.year_end == 2022

    def test_inverted_range_raises(self):
        with pytest.raises(ValueError, match="year_end"):
            DataConfig(year_start=2022, year_end=2020)

    def test_hours_in_year_leap(self):
        assert hours_in_year(2020) == 8784   # leap
        assert hours_in_year(2021) == 8760
        assert hours_in_year(2024) == 8784   # leap

    def test_hours_in_span_accounts_for_leap(self):
        # 2020 leap, 2021 non-leap, 2022 non-leap
        assert hours_in_span(2020, 2022) == 8784 + 8760 + 8760
        # Single non-leap year
        assert hours_in_span(2021, 2021) == 8760

    def test_hours_total_property(self):
        config = DataConfig(year_start=2020, year_end=2022)
        assert config.hours_total == 8784 + 8760 + 8760

    def test_legacy_dict_exposes_multiyear_fields(self):
        config = OTEXConfig(data=DataConfig(year_start=2020, year_end=2022))
        legacy = config.to_legacy_dict()
        assert legacy['year_start'] == 2020
        assert legacy['year_end'] == 2022
        assert legacy['n_years'] == 3
        assert legacy['years'] == [2020, 2021, 2022]
        assert legacy['year_label'] == '2020-2022'
        assert legacy['hours_total'] == 26304
        # Legacy `year` key still present, points to year_start
        assert legacy['year'] == 2020

    def test_parameters_and_constants_multiyear(self):
        inputs = parameters_and_constants(year_start=2020, year_end=2022)
        assert inputs['n_years'] == 3
        assert inputs['year_label'] == '2020-2022'
        assert inputs['date_start'].startswith('2020-')
        assert inputs['date_end'].startswith('2022-')

    def test_parameters_and_constants_legacy_year_still_works(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            inputs = parameters_and_constants(year=2023)
        assert inputs['year_start'] == 2023
        assert inputs['year_end'] == 2023
        assert inputs['n_years'] == 1

    def test_get_default_config_multiyear(self):
        config = get_default_config(year_start=2018, year_end=2020)
        assert config.data.n_years == 3
        assert config.data.year_start == 2018
        assert config.data.year_end == 2020
