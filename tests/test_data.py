# -*- coding: utf-8 -*-
"""
Tests for otex.data module.
"""

import pytest
import numpy as np


class TestDataModuleImports:
    """Tests for data module imports."""

    def test_cmems_module_exists(self):
        """CMEMS module should exist."""
        try:
            from otex.data import cmems
            assert cmems is not None
        except ImportError as e:
            if 'copernicusmarine' in str(e):
                pytest.skip("copernicusmarine not installed")
            raise

    def test_netcdf_module_exists(self):
        """NetCDF module should exist."""
        try:
            from otex.data import netcdf
            assert netcdf is not None
        except ImportError as e:
            if 'copernicusmarine' in str(e):
                pytest.skip("copernicusmarine not installed")
            raise
        except FileNotFoundError:
            pytest.skip("NetCDF module requires data files not present in test environment")

    def test_multi_depth_module_exists(self):
        """Multi-depth module should exist."""
        try:
            from otex.data import multi_depth
            assert multi_depth is not None
        except ImportError as e:
            if 'copernicusmarine' in str(e):
                pytest.skip("copernicusmarine not installed")
            raise


class TestDataConfig:
    """Tests for data configuration."""

    def test_cmems_config(self):
        """CMEMS configuration should be correct."""
        from otex.config import parameters_and_constants

        inputs = parameters_and_constants(data='CMEMS')

        assert inputs['data'] == 'CMEMS'
        assert 'time_origin' in inputs
        assert '1950' in inputs['time_origin']

    def test_hycom_config(self):
        """HYCOM configuration should be correct."""
        from otex.config import DataConfig, OTEXConfig

        data_config = DataConfig(source='HYCOM')
        config = OTEXConfig(data=data_config)
        legacy = config.to_legacy_dict()

        assert legacy['data'] == 'HYCOM'
        assert 'time_origin' in legacy
        assert '2000' in legacy['time_origin']

    def test_time_resolution(self):
        """Time resolution should be configurable."""
        from otex.config import parameters_and_constants

        inputs = parameters_and_constants()

        assert 't_resolution' in inputs


class TestDateConfiguration:
    """Tests for date configuration."""

    def test_year_in_inputs(self):
        """Year should be in inputs."""
        from otex.config import parameters_and_constants

        inputs = parameters_and_constants(year=2021)

        assert inputs['year'] == 2021

    def test_date_start_format(self):
        """date_start should have correct format."""
        from otex.config import parameters_and_constants

        inputs = parameters_and_constants(year=2021)

        assert inputs['date_start'] == '2021-01-01 00:00:00'

    def test_date_end_format(self):
        """date_end should have correct format."""
        from otex.config import parameters_and_constants

        inputs = parameters_and_constants(year=2021)

        assert inputs['date_end'] == '2021-12-31 21:00:00'

    def test_different_years(self):
        """Different years should produce different dates."""
        from otex.config import parameters_and_constants

        inputs_2020 = parameters_and_constants(year=2020)
        inputs_2021 = parameters_and_constants(year=2021)

        assert inputs_2020['date_start'] != inputs_2021['date_start']
        assert '2020' in inputs_2020['date_start']
        assert '2021' in inputs_2021['date_start']


class TestTemperatureDataStructures:
    """Tests for temperature data structures."""

    def test_sample_temperature_arrays(self, sample_temperatures):
        """Sample temperature arrays should have correct shape."""
        assert sample_temperatures['T_WW'].shape == (4,)
        assert sample_temperatures['T_CW'].shape == (4,)

    def test_temperature_difference(self, sample_temperatures):
        """Temperature difference should be positive for OTEC."""
        T_WW = sample_temperatures['T_WW']
        T_CW = sample_temperatures['T_CW']

        delta_T = T_WW - T_CW

        assert np.all(delta_T > 0)
        assert np.all(delta_T > 15)

    def test_design_temperatures_shape(self, sample_temperatures):
        """Design temperatures should have [min, med, max] structure."""
        T_WW_design = sample_temperatures['T_WW_design']
        T_CW_design = sample_temperatures['T_CW_design']

        assert T_WW_design.shape[0] == 3
        assert T_CW_design.shape[0] == 3


class TestCMEMSFunctions:
    """Tests for CMEMS-specific functions."""

    def test_download_data_function_exists(self):
        """download_data function should exist."""
        try:
            from otex.data.cmems import download_data
            assert callable(download_data)
        except ImportError as e:
            if 'copernicusmarine' in str(e):
                pytest.skip("copernicusmarine not installed")
            raise

    def test_data_processing_function_exists(self):
        """data_processing function should exist."""
        try:
            from otex.data.cmems import data_processing
            assert callable(data_processing)
        except ImportError as e:
            if 'copernicusmarine' in str(e):
                pytest.skip("copernicusmarine not installed")
            raise

    def test_load_temperatures_function_exists(self):
        """load_temperatures function should exist."""
        try:
            from otex.data.cmems import load_temperatures
            assert callable(load_temperatures)
        except ImportError as e:
            if 'copernicusmarine' in str(e):
                pytest.skip("copernicusmarine not installed")
            raise


class TestHYCOMModule:
    """Tests for HYCOM data module."""

    def test_hycom_module_exists(self):
        """HYCOM module should be importable."""
        from otex.data import hycom
        assert hycom is not None

    def test_hycom_download_data_function_exists(self):
        """download_data function should exist in HYCOM module."""
        from otex.data.hycom import download_data
        assert callable(download_data)

    def test_hycom_depth_levels(self):
        """HYCOM should have 40 standard depth levels."""
        from otex.data.hycom import HYCOM_DEPTHS
        assert len(HYCOM_DEPTHS) == 40
        assert HYCOM_DEPTHS[0] == 0.0
        assert HYCOM_DEPTHS[-1] == 5000.0

    def test_get_nearest_hycom_depth_surface(self):
        """Nearest depth to 22m should be 20m."""
        from otex.data.hycom import get_nearest_hycom_depth
        assert get_nearest_hycom_depth(22.0) == 20.0

    def test_get_nearest_hycom_depth_deep(self):
        """Nearest depth to 1062m should be 1000m."""
        from otex.data.hycom import get_nearest_hycom_depth
        assert get_nearest_hycom_depth(1062.0) == 1000.0

    def test_get_nearest_hycom_depth_exact(self):
        """Exact match should return the same value."""
        from otex.data.hycom import get_nearest_hycom_depth
        assert get_nearest_hycom_depth(500.0) == 500.0

    def test_get_nearest_hycom_depth_midpoint(self):
        """Midpoint between 600 and 700 should return one of them."""
        from otex.data.hycom import get_nearest_hycom_depth
        result = get_nearest_hycom_depth(650.0)
        assert result in (600.0, 700.0)

    def test_get_hycom_experiment_reanalysis(self):
        """Year 2010 should select reanalysis experiment."""
        from otex.data.hycom import get_hycom_experiment
        exp = get_hycom_experiment(2010)
        assert "expt_53.X" in exp["url"]

    def test_get_hycom_experiment_analysis(self):
        """Year 2020 should select analysis experiment."""
        from otex.data.hycom import get_hycom_experiment
        exp = get_hycom_experiment(2020)
        assert "expt_93.0" in exp["url"]

    def test_get_hycom_experiment_gap_raises(self):
        """Year 2017 should raise ValueError (data gap)."""
        from otex.data.hycom import get_hycom_experiment
        with pytest.raises(ValueError, match="not available"):
            get_hycom_experiment(2017)

    def test_hycom_experiments_have_required_keys(self):
        """Each experiment should have url, years, and time_origin."""
        from otex.data.hycom import HYCOM_EXPERIMENTS
        for name, exp in HYCOM_EXPERIMENTS.items():
            assert "url" in exp, f"{name} missing 'url'"
            assert "years" in exp, f"{name} missing 'years'"
            assert "time_origin" in exp, f"{name} missing 'time_origin'"

    def test_lazy_import_download_data_hycom(self):
        """download_data_hycom should be accessible via lazy import."""
        from otex.data import download_data_hycom
        assert callable(download_data_hycom)
