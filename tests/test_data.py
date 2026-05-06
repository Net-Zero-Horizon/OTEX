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

    def test_download_data_honours_new_path_argument(self, tmp_path, monkeypatch):
        """Regression: cmems.download_data must write into the caller's
        ``new_path`` and not a hard-coded ``Data_Results/<region>/`` prefix.

        We monkey-patch ``copernicusmarine.subset`` to record the
        ``output_directory`` it would have used; the test asserts that
        the recorded path equals the ``new_path`` argument.
        """
        import os
        import sys
        import types
        from pathlib import Path
        import otex.data.cmems as cmems_mod

        captured = []

        def fake_subset(**kwargs):
            captured.append(kwargs)
            # Touch a file at the expected location so the validity
            # check on the next iteration finds something.
            out_dir = kwargs['output_directory']
            os.makedirs(out_dir, exist_ok=True)
            (Path(out_dir) / kwargs['output_filename']).write_bytes(b'')

        # Inject a stub copernicusmarine module so the lazy import
        # inside download_data succeeds without the real SDK.
        stub = types.SimpleNamespace(subset=fake_subset)
        monkeypatch.setitem(sys.modules, 'copernicusmarine', stub)
        monkeypatch.setattr(cmems_mod, 'copernicusmarine', stub)
        # Skip the netCDF4 sanity-check on the dummy file we created.
        class _DummyNC:
            def __init__(self, *a, **kw): pass
            def close(self): pass
        monkeypatch.setattr(cmems_mod, 'netCDF4',
                             types.SimpleNamespace(Dataset=_DummyNC))

        custom_path = str(tmp_path / 'my_custom_run_dir') + os.sep

        inputs = {
            'length_WW_inlet': 21.6,
            'length_CW_inlet': 1062.4,
            'date_start': '2020-01-01 00:00:00',
            'date_end': '2020-12-31 21:00:00',
            'year_start': 2020,
            'year_end': 2020,
        }

        cmems_mod.download_data('low_cost', inputs, 'Jamaica', custom_path)

        assert captured, "fake copernicusmarine.subset was never called"
        for call in captured:
            # The bug regressed when output_directory was 'Data_Results/Jamaica'
            # regardless of new_path. The fix forwards new_path verbatim.
            assert call['output_directory'] == custom_path, (
                f"download_data ignored new_path: wrote to "
                f"{call['output_directory']!r} instead of {custom_path!r}"
            )


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


class TestMultiYearHelpers:
    """Tests for the year-extraction helpers used by multi-year data processing."""

    def test_year_from_filename_simple(self):
        from otex.data.cmems import _year_from_filename
        assert _year_from_filename('T_20.0m_2021_Jamaica_1.nc') == 2021

    def test_year_from_filename_with_path(self):
        from otex.data.cmems import _year_from_filename
        assert _year_from_filename('/tmp/runs/T_1062.0m_2018_New_Zealand_2.nc') == 2018

    def test_year_from_filename_int_depth(self):
        from otex.data.cmems import _year_from_filename
        # Older files may have integer depth (no decimal point)
        assert _year_from_filename('T_20m_2020_Hawaii_1.nc') == 2020

    def test_year_from_filename_invalid_raises(self):
        from otex.data.cmems import _year_from_filename
        with pytest.raises(ValueError, match="Cannot extract year"):
            _year_from_filename('not_a_temperature_file.nc')

    def test_group_files_by_year_orders_chronologically(self):
        from otex.data.cmems import _group_files_by_year
        files = [
            'T_20.0m_2022_X_1.nc',
            'T_20.0m_2020_X_1.nc',
            'T_20.0m_2021_X_1.nc',
            'T_20.0m_2020_X_2.nc',
        ]
        grouped = _group_files_by_year(files)
        assert list(grouped.keys()) == [2020, 2021, 2022]
        assert len(grouped[2020]) == 2
        assert len(grouped[2021]) == 1


class TestMultiYearDataProcessing:
    """End-to-end tests for data_processing with synthetic NetCDF files."""

    def _write_synthetic_nc(self, path, year, depth=20.0, n_days=365,
                            lats=None, lons=None, base_temp=28.0):
        """Create a minimal CMEMS-compatible NetCDF for one year."""
        import netCDF4
        import datetime

        if lats is None:
            lats = np.array([10.0, 10.5])
        if lons is None:
            lons = np.array([-80.0, -79.5])

        # Hours since 1950-01-01 to year-01-01
        t_origin = datetime.datetime(1950, 1, 1)
        t_year_start = datetime.datetime(year, 1, 1)
        hours_offset = (t_year_start - t_origin).total_seconds() / 3600.0
        time_vals = hours_offset + np.arange(n_days) * 24.0

        ds = netCDF4.Dataset(path, 'w', format='NETCDF3_CLASSIC')
        ds.createDimension('time', n_days)
        ds.createDimension('depth', 1)
        ds.createDimension('latitude', len(lats))
        ds.createDimension('longitude', len(lons))

        v_time = ds.createVariable('time', 'f8', ('time',))
        v_time[:] = time_vals
        v_depth = ds.createVariable('depth', 'f4', ('depth',))
        v_depth[:] = [depth]
        v_lat = ds.createVariable('latitude', 'f4', ('latitude',))
        v_lat[:] = lats
        v_lon = ds.createVariable('longitude', 'f4', ('longitude',))
        v_lon[:] = lons

        v_T = ds.createVariable('thetao', 'f4',
                                ('time', 'depth', 'latitude', 'longitude'))
        # Simple seasonal signal + per-year offset to verify concatenation works.
        season = 2.0 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
        T = base_temp + season + (year - 2020) * 0.1
        # Broadcast to (time, depth=1, lat, lon)
        v_T[:] = np.broadcast_to(
            T[:, None, None, None],
            (n_days, 1, len(lats), len(lons)),
        ).astype('f4')
        ds.close()

    def test_synthetic_helper_writes_readable_file(self, tmp_path):
        """Sanity check: the synthetic NetCDF helper produces a valid file."""
        try:
            import netCDF4  # noqa: F401
        except ImportError:
            pytest.skip("netCDF4 not available")
        path = str(tmp_path / 'T_20.0m_2020_TestRegion_1.nc')
        self._write_synthetic_nc(path, 2020, n_days=10)
        import netCDF4
        nc = netCDF4.Dataset(path, 'r')
        assert nc.variables['thetao'].shape == (10, 1, 2, 2)
        nc.close()

    def _make_sites_df(self, lats, lons):
        import pandas as pd
        rows = []
        sid = 1
        for lat in lats:
            for lon in lons:
                rows.append({
                    'longitude': float(lon),
                    'latitude': float(lat),
                    'dist_shore': 10.0,
                    'id': sid,
                })
                sid += 1
        return pd.DataFrame(rows)

    def test_data_processing_multiyear_concatenates_time_axis(self, tmp_path):
        """data_processing with files from 3 years should produce a 3x-long time series."""
        try:
            import netCDF4  # noqa: F401
        except ImportError:
            pytest.skip("netCDF4 not available")

        from otex.data.cmems import data_processing
        from otex.config import parameters_and_constants

        lats = np.array([10.0, 10.5])
        lons = np.array([-80.0, -79.5])

        # Write 3 yearly NetCDFs (use small n_days to keep test fast).
        n_days_per_year = 30
        files = []
        for year in (2020, 2021, 2022):
            p = tmp_path / f'T_20.0m_{year}_TestRegion_1.nc'
            self._write_synthetic_nc(str(p), year, depth=20.0,
                                      n_days=n_days_per_year,
                                      lats=lats, lons=lons)
            files.append(str(p))

        sites_df = self._make_sites_df(lats, lons)
        inputs = parameters_and_constants(year_start=2020, year_end=2022)

        T_profiles, T_design, coords, ids, ts, inputs_out, nan_cols = data_processing(
            files, sites_df, inputs, 'TestRegion', str(tmp_path) + '/', 'WW'
        )

        # Time axis: 3 years × 30 days = 90 timesteps after asfreq + interp.
        # asfreq('24H') may produce 89 days because the spans are non-contiguous.
        # We accept anything >= 3*30 - 2 to allow for boundary effects.
        assert T_profiles.shape[0] >= 3 * n_days_per_year - 2
        assert T_profiles.shape[1] == 4  # 2 lats × 2 lons
        assert ts[0].year == 2020
        assert ts[-1].year == 2022

        # Cache file should use the multi-year label.
        cache_files = list(tmp_path.glob('T_20*m_2020-2022_TestRegion.h5'))
        assert len(cache_files) == 1, \
            f"Expected one multi-year H5 file, found: {list(tmp_path.glob('*.h5'))}"

    def test_data_processing_single_year_unchanged(self, tmp_path):
        """Single-year invocation must still produce the legacy filename."""
        try:
            import netCDF4  # noqa: F401
        except ImportError:
            pytest.skip("netCDF4 not available")

        from otex.data.cmems import data_processing
        from otex.config import parameters_and_constants

        lats = np.array([10.0, 10.5])
        lons = np.array([-80.0, -79.5])

        p = tmp_path / 'T_20.0m_2020_TestRegion_1.nc'
        self._write_synthetic_nc(str(p), 2020, depth=20.0,
                                  n_days=30, lats=lats, lons=lons)

        sites_df = self._make_sites_df(lats, lons)
        inputs = parameters_and_constants(year_start=2020, year_end=2020)

        data_processing(
            [str(p)], sites_df, inputs, 'TestRegion', str(tmp_path) + '/', 'WW'
        )

        cache_files = list(tmp_path.glob('T_20*m_2020_TestRegion.h5'))
        assert len(cache_files) == 1, \
            f"Expected single-year H5 file, found: {list(tmp_path.glob('*.h5'))}"

    def test_data_processing_mismatched_sites_across_years_raises(self, tmp_path):
        """If two yearly files have different site grids, processing must error."""
        try:
            import netCDF4  # noqa: F401
        except ImportError:
            pytest.skip("netCDF4 not available")

        from otex.data.cmems import data_processing
        from otex.config import parameters_and_constants

        # Year 2020 has 2x2 grid; year 2021 has a SHIFTED grid → mismatch.
        f20 = tmp_path / 'T_20.0m_2020_TestRegion_1.nc'
        f21 = tmp_path / 'T_20.0m_2021_TestRegion_1.nc'
        self._write_synthetic_nc(str(f20), 2020, n_days=30,
                                  lats=np.array([10.0, 10.5]),
                                  lons=np.array([-80.0, -79.5]))
        self._write_synthetic_nc(str(f21), 2021, n_days=30,
                                  lats=np.array([20.0, 20.5]),  # different!
                                  lons=np.array([-80.0, -79.5]))

        # sites_df must overlap with BOTH grids for the bug to surface as
        # "different sites per year" rather than "no sites at all".
        sites_df = self._make_sites_df(
            lats=np.array([10.0, 10.5, 20.0, 20.5]),
            lons=np.array([-80.0, -79.5]),
        )
        inputs = parameters_and_constants(year_start=2020, year_end=2021)

        with pytest.raises(ValueError, match="do not match"):
            data_processing(
                [str(f20), str(f21)], sites_df, inputs,
                'TestRegion', str(tmp_path) + '/', 'WW'
            )


# Tests for the on-demand region/site/bathymetry catalog (0.2.0).
# These exercise pure-Python helpers that don't require the network
# (network-dependent tests live in TestOnDemandCatalogNetwork below).

class TestRegionsHelpers:
    """Pure-Python helpers in otex.data.regions and sites."""

    def test_bbox_dataclass_basic(self):
        from otex.data.regions import BBox
        b = BBox(north=20, south=10, east=-70, west=-80)
        assert b.as_tuple() == (20, 10, -70, -80)
        assert not b.crosses_antimeridian

    def test_bbox_antimeridian_detection(self):
        from otex.data.regions import BBox
        b = BBox(north=10, south=-10, east=-170, west=170)
        assert b.crosses_antimeridian

    def test_split_at_antimeridian_simple_geometry(self):
        from shapely.geometry import box
        from otex.data.regions import _split_at_antimeridian
        # Simple polygon entirely on one side: returns single bbox.
        geom = box(-80, 10, -70, 20)
        parts = _split_at_antimeridian(geom)
        assert len(parts) == 1
        assert parts[0].east == -70
        assert parts[0].west == -80

    def test_expand_bbox_clips_to_global_range(self):
        from otex.data.regions import BBox
        from otex.data.sites import _expand_bbox
        # bbox near pole; buffer must not exceed 90.
        b = BBox(north=88, south=-88, east=170, west=-170)
        out = _expand_bbox(b, 5.0)
        assert out.north == 90.0
        assert out.south == -90.0
        assert out.east == 175.0
        assert out.west == -175.0

    def test_make_grid_resolution(self):
        from otex.data.regions import BBox
        from otex.data.sites import _make_grid
        b = BBox(north=10, south=0, east=10, west=0)
        lon, lat = _make_grid(b, step=1.0)
        # Inclusive on both ends after snapping → 11x11 grid.
        assert lon.size == 11 * 11
        assert lat.size == 11 * 11
        assert float(lon.min()) == 0.0
        assert float(lon.max()) == 10.0


class TestCoastlineHelpers:
    """Math-only helpers in otex.data.coastline."""

    def test_lonlat_to_ecef_unit_sphere(self):
        import numpy as np
        from otex.data.coastline import _lonlat_to_ecef
        # All ECEF unit-sphere points must have norm 1.
        lons = np.array([0.0, 90.0, -45.0, 180.0])
        lats = np.array([0.0, 0.0, 30.0, -60.0])
        pts = _lonlat_to_ecef(lons, lats)
        norms = np.linalg.norm(pts, axis=-1)
        np.testing.assert_allclose(norms, np.ones(4), atol=1e-12)

    def test_chord_to_arc_km_zero(self):
        import numpy as np
        from otex.data.coastline import _chord_to_arc_km
        np.testing.assert_allclose(_chord_to_arc_km(np.array(0.0)), 0.0)

    def test_chord_to_arc_km_full_diameter(self):
        import numpy as np
        from otex.data.coastline import _chord_to_arc_km, _EARTH_RADIUS_KM
        # Chord = 2 (diameter) → arc = pi * R (half circumference).
        result = float(_chord_to_arc_km(np.array(2.0)))
        np.testing.assert_allclose(result, np.pi * _EARTH_RADIUS_KM, rtol=1e-9)


class TestLoadRegionsLegacyAPI:
    """The 0.2.0 load_regions/load_sites preserve the legacy entry points
    for code that imported them, but reroute through Natural Earth /
    ETOPO. These tests assert the API surface (callable, schema) without
    requiring the network — they hit the on-process cache populated by
    a small monkey-patch.
    """

    def test_load_sites_requires_region_now(self):
        import pytest
        from otex.data.resources import load_sites
        with pytest.raises(ValueError, match="requires a region"):
            load_sites()

    def test_demand_module_exposes_multi_source_api(self):
        from otex.data.demand import (
            fetch_demand_TWh, _fetch_owid, _fetch_world_bank,
        )
        assert callable(fetch_demand_TWh)
        assert callable(_fetch_owid)
        assert callable(_fetch_world_bank)

    def test_demand_falls_back_when_first_provider_fails(self):
        """Multi-source dispatch must try each provider and stop on success."""
        from otex.data.demand import fetch_demand_TWh
        calls = []

        def failing(_iso):
            calls.append('failing')
            return None

        def succeeding(_iso):
            calls.append('succeeding')
            return 12.34, 2024

        twh, src, year = fetch_demand_TWh(
            'XYZ',
            providers=[('first', failing), ('second', succeeding)],
        )
        assert twh == 12.34
        assert src == 'second'
        assert year == 2024
        assert calls == ['failing', 'succeeding']

    def test_demand_returns_none_when_all_providers_fail(self):
        from otex.data.demand import fetch_demand_TWh
        twh, src, year = fetch_demand_TWh(
            'XYZ',
            providers=[('a', lambda _: None), ('b', lambda _: None)],
        )
        assert twh is None and src is None and year is None
