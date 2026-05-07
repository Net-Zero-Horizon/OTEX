# -*- coding: utf-8 -*-
"""Unit tests for the CMIP6 climate-scenario module.

Network access is mocked everywhere — these tests fabricate small
xarray.Dataset objects and patch ``xr.open_zarr`` so the code paths
that fetch from Pangeo Zarr exercise the same logic without ever
touching the network.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from otex.config import ClimateConfig, OTEXConfig
from otex.data.regions import BBox


def _synthetic_thetao_ds(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    base_temp_K: float = 301.0,    # ~28°C
    warming_K: float = 0.0,
    start_year: int = 1990,
    n_months: int = 12 * 120,       # 120 years — covers historical + SSPs
    n_levels: int = 5,
) -> xr.Dataset:
    """Build a tiny dataset that mimics CMIP6 monthly thetao.

    Curvilinear grid (j, i) with 2-D latitude/longitude coords, depth
    levels in metres, time as monthly DatetimeIndex spanning enough
    years to cover both the baseline window (1995-2014) and any
    plausible target year (2030-2080+). ``warming_K`` is added
    uniformly to make deltas easy to predict.
    """
    n_j, n_i = lat_grid.shape
    times = pd.date_range(f'{start_year}-01-15', periods=n_months, freq='MS')
    lev = np.array([5, 50, 200, 500, 1500])[:n_levels]
    rng = np.random.default_rng(0)
    noise = 0.05 * rng.standard_normal((n_months, n_levels, n_j, n_i))
    thetao = base_temp_K + warming_K + noise

    return xr.Dataset(
        {'thetao': (('time', 'lev', 'j', 'i'), thetao)},
        coords={
            'time': times,
            'lev': ('lev', lev),
            'latitude': (('j', 'i'), lat_grid),
            'longitude': (('j', 'i'), lon_grid),
        },
    )


def _make_grid(lat_range=(10.0, 20.0), lon_range=(-85.0, -70.0), n=8):
    lats_1d = np.linspace(lat_range[0], lat_range[1], n)
    lons_1d = np.linspace(lon_range[0], lon_range[1], n)
    lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
    return lat_grid, lon_grid


class TestClimateConfig:

    def test_default_is_historical_no_op(self):
        cfg = ClimateConfig()
        assert cfg.scenario == 'historical'
        assert cfg.target_year is None
        assert cfg.enabled is False
        assert cfg.label == 'historical'

    def test_enabled_when_scenario_and_year_set(self):
        cfg = ClimateConfig(scenario='ssp245', target_year=2050)
        assert cfg.enabled
        assert cfg.label == 'ssp245_2050'

    def test_otexconfig_carries_climate_into_legacy_dict(self):
        c = OTEXConfig()
        c.climate = ClimateConfig(scenario='ssp585', target_year=2080)
        legacy = c.to_legacy_dict()
        assert legacy['climate_enabled'] is True
        assert legacy['climate_label'] == 'ssp585_2080'
        assert legacy['climate_scenario'] == 'ssp585'
        assert legacy['climate_target_year'] == 2080
        assert legacy['climate_models'] == list(ClimateConfig().models)


class TestClimateDelta:

    def _patch_open(self, lat_grid, lon_grid, *, historical_K=0.0, future_K=0.0):
        """Patch ``_open_zarr_anonymous`` so historical and SSP URLs
        return synthetic datasets with the chosen warming offsets."""

        def fake_open(url):
            warming = historical_K if 'historical' in url else future_K
            return _synthetic_thetao_ds(
                lat_grid, lon_grid,
                base_temp_K=301.0, warming_K=warming,
            )

        return patch('otex.data.climate._open_zarr_anonymous',
                     side_effect=fake_open)

    def test_historical_delta_is_zero(self):
        from otex.data.climate import compute_delta_field

        lat_grid, lon_grid = _make_grid()
        with self._patch_open(lat_grid, lon_grid):
            df = compute_delta_field(
                model='CanESM5', scenario='historical', target_year=2050,
                depth_m=20.0, bbox=BBox(north=20, south=10, east=-70, west=-85),
            )
        assert (df['delta_C'] == 0).all()

    def test_uniform_warming_yields_uniform_delta(self):
        """If the GCM is +2 K everywhere in the future relative to the
        baseline, the per-cell delta must be ≈ +2."""
        from otex.data.climate import compute_delta_field

        lat_grid, lon_grid = _make_grid()
        target_K = 2.0

        with self._patch_open(lat_grid, lon_grid,
                               historical_K=0.0, future_K=target_K):
            df = compute_delta_field(
                model='CanESM5', scenario='ssp245', target_year=2050,
                depth_m=20.0, bbox=BBox(north=20, south=10, east=-70, west=-85),
            )

        assert df['delta_C'].notna().all()
        # Allow numerical noise from the time-mean of the synthetic noise.
        np.testing.assert_allclose(df['delta_C'].mean(), target_K, atol=0.05)
        assert df['delta_C'].std() < 0.1

    def test_delta_at_points_matches_field_mean_for_uniform_delta(self):
        """When the delta field is uniform, the interpolated values at
        any target points must equal that uniform value."""
        from otex.data.climate import EnsembleDelta, delta_at_points

        # Synthetic ensemble result: 100 grid pts, delta = +1.5 °C uniform.
        rng = np.random.default_rng(1)
        lon = rng.uniform(-85, -70, 100)
        lat = rng.uniform(10, 20, 100)
        delta_mean = np.full(100, 1.5)
        delta_std = np.zeros(100)
        ens = EnsembleDelta(lon=lon, lat=lat,
                             delta_mean=delta_mean, delta_std=delta_std,
                             models=['fake'])

        target_lons = np.array([-78.0, -76.0, -74.0])
        target_lats = np.array([15.0, 16.0, 17.0])
        out = delta_at_points(target_lons, target_lats, ens)
        np.testing.assert_allclose(out, [1.5, 1.5, 1.5], atol=1e-9)


class TestSliceBbox:
    """Defensive checks on the bbox slicer used internally."""

    def test_returns_subset_of_curvilinear_grid(self):
        from otex.data.climate import _slice_bbox

        lat_grid, lon_grid = _make_grid(lat_range=(-30, 30),
                                          lon_range=(-150, -50), n=20)
        ds = _synthetic_thetao_ds(lat_grid, lon_grid)
        sub = _slice_bbox(ds, BBox(north=15, south=10, east=-70, west=-80))

        # The sliced dataset must be smaller than the global one.
        assert sub.sizes['j'] < ds.sizes['j']
        assert sub.sizes['i'] < ds.sizes['i']
        # And cover the requested bbox interior at least.
        lats_in = sub['latitude'].values
        lons_in = np.where(
            sub['longitude'].values > 180,
            sub['longitude'].values - 360,
            sub['longitude'].values,
        )
        assert lats_in.min() <= 15 and lats_in.max() >= 10
        assert lons_in.min() <= -70 and lons_in.max() >= -80
