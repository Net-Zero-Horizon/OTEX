# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MPL-2.0
"""HYCOM grid mask utilities for site pre-filtering.

Symmetric counterpart to :mod:`otex.data.cmems_mask`. HYCOM GLBv/GLBy0.08
uses a different bathymetry and grid than CMEMS (asymmetric ~0.04° lat ×
0.08° lon, longitudes in 0-360), so the CMEMS mask does not transfer.
This module downloads a single-time, single-depth snapshot of
``water_temp`` from HYCOM's OPeNDAP endpoint (no authentication
required), extracts the valid-cell mask, and provides
:func:`filter_sites_by_hycom_mask` used by
:func:`otex.data.sites.build_sites` when ``data_source='HYCOM'``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .hycom import (
    HYCOM_EXPERIMENTS,
    get_hycom_experiment,
    get_nearest_hycom_depth,
    _lon_to_360,
)


def _cache_dir() -> Path:
    base = os.environ.get("OTEX_CACHE_DIR")
    d = (Path(base) if base else Path.home() / ".otex" / "cache") / "hycom_mask"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(west: float, east: float, south: float, north: float,
               depth_m: float, experiment_url: str) -> str:
    payload = repr((round(west, 4), round(east, 4),
                    round(south, 4), round(north, 4),
                    round(depth_m, 2), experiment_url)).encode()
    return hashlib.sha1(payload).hexdigest()[:14]


def _download_snapshot(nc_path: Path, west: float, east: float,
                       south: float, north: float,
                       depth_m: float, experiment: dict) -> None:
    """Fetch a single-time, single-depth HYCOM ``water_temp`` slice.

    Uses xarray + OPeNDAP against the experiment URL. HYCOM ships in
    the 0-360 longitude convention, so the bbox is translated before
    the ``.sel(...)`` call and the depth is snapped to the nearest of
    HYCOM's 40 fixed levels.
    """
    import xarray as xr

    w360 = _lon_to_360(west)
    e360 = _lon_to_360(east)
    if e360 < w360:
        # bbox straddles the antimeridian — HYCOM's monotonic 0-360
        # axis needs two sub-slices concatenated. OTEC regions rarely
        # trigger this (only Fiji, Kiribati, Tonga).
        raise ValueError(
            f"HYCOM bbox crosses the antimeridian (west={west}, east={east}). "
            "Split the region into two sub-bboxes east and west of 180°E."
        )

    hycom_depth = get_nearest_hycom_depth(depth_m)

    ds = xr.open_dataset(experiment['url'], decode_times=False)
    try:
        sub = (
            ds['water_temp']
            .sel(lon=slice(w360, e360),
                 lat=slice(south, north),
                 depth=hycom_depth)
            .isel(time=0)
        )
        # Wrap in a Dataset so ``.to_netcdf`` preserves the coordinate
        # arrays (including ``depth`` as a scalar coord).
        sub.load().to_netcdf(nc_path)
    finally:
        ds.close()


def hycom_valid_mask(west: float, east: float, south: float, north: float,
                     depth_m: float,
                     year: int = 2023,
                     refresh: bool = False,
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the HYCOM valid-cell mask for a bbox at ``depth_m``.

    Parameters
    ----------
    west, east, south, north : float
        Bounding box in the -180..180 longitude convention. Converted
        to 0-360 internally for the HYCOM query.
    depth_m : float
        Depth in metres (positive downward). Snapped to the nearest of
        the 40 HYCOM standard levels via
        :func:`otex.data.hycom.get_nearest_hycom_depth`.
    year : int
        Year used to select the HYCOM experiment (``GLBv0.08/expt_53.X``
        for 1994-2015, ``GLBy0.08/expt_93.0`` for 2019-2024). Default
        2023 picks the current analysis product. The mask is
        bathymetric and effectively time-invariant, but experiments do
        occasionally revise their land mask.
    refresh : bool
        If True, ignore any cached mask and re-download.

    Returns
    -------
    (lons, lats, mask) : (ndarray, ndarray, ndarray)
        1-D longitude (in -180..180) and latitude arrays and a 2-D
        boolean mask indexed as ``mask[lat_i, lon_i]``. ``True`` means
        HYCOM has valid ocean data at that cell + depth.
    """
    import xarray as xr

    experiment = get_hycom_experiment(int(year))

    key = _cache_key(west, east, south, north, depth_m, experiment['url'])
    nc_path = _cache_dir() / f"hycom_mask_{key}.nc"

    if refresh or not nc_path.exists():
        _download_snapshot(nc_path, west, east, south, north,
                           depth_m, experiment)

    with xr.open_dataset(nc_path) as ds:
        # After .sel + .isel(time=0) the array has dims (lat, lon).
        water_temp = ds['water_temp'].values
        lons_360 = np.asarray(ds['lon'].values, dtype=np.float64)
        lats = np.asarray(ds['lat'].values, dtype=np.float64)

    # Convert lons to -180..180 and sort ascending so index lookups are
    # monotonic (HYCOM 0-360 puts the antimeridian at the array edge;
    # a wrapped bbox would break the sort, but we rejected that above).
    lons = np.where(lons_360 > 180.0, lons_360 - 360.0, lons_360)
    order = np.argsort(lons)
    lons = lons[order]
    mask = ~np.isnan(water_temp)
    mask = mask[:, order]

    return lons, lats, mask


def filter_sites_by_hycom_mask(
    sites,  # pandas.DataFrame with 'longitude' and 'latitude'
    west: float, east: float, south: float, north: float,
    depth_m: float,
    year: int = 2023,
    refresh: bool = False,
):
    """Return the subset of ``sites`` whose nearest HYCOM cell is valid.

    Coordinates of surviving sites are snapped to the HYCOM cell centre
    (rounded to 3 decimals to match the downstream lookup in
    :func:`otex.data.cmems._extract_year_data`, which reads HYCOM files
    that :func:`otex.data.hycom.download_data` has renamed to
    CMEMS-style variable/dim names). Snapping is essential because the
    HYCOM grid is asymmetric (~0.04° lat × 0.08° lon), so a uniform
    ``_make_grid`` in ``sites.py`` cannot land on HYCOM centres directly.
    """
    if sites is None or len(sites) == 0:
        return sites

    lons, lats, mask = hycom_valid_mask(
        west, east, south, north, depth_m,
        year=year, refresh=refresh,
    )
    if lons.size == 0 or lats.size == 0:
        return sites.iloc[0:0].copy()

    # HYCOM lat step ≈ 0.04°, lon step ≈ 0.08°. Use the actual arrays
    # to compute per-dim steps rather than hardcoding.
    lat_step = float(lats[1] - lats[0]) if len(lats) > 1 else 0.04
    lon_step = float(lons[1] - lons[0]) if len(lons) > 1 else 0.08
    lon0 = float(lons[0])
    lat0 = float(lats[0])
    n_lat, n_lon = mask.shape

    site_lon = sites['longitude'].to_numpy(dtype=np.float64)
    site_lat = sites['latitude'].to_numpy(dtype=np.float64)
    lo_idx = np.round((site_lon - lon0) / lon_step).astype(np.int64)
    la_idx = np.round((site_lat - lat0) / lat_step).astype(np.int64)

    in_range = ((lo_idx >= 0) & (lo_idx < n_lon)
                & (la_idx >= 0) & (la_idx < n_lat))
    keep = np.zeros(len(sites), dtype=bool)
    keep[in_range] = mask[la_idx[in_range], lo_idx[in_range]]

    filtered = sites[keep].reset_index(drop=True).copy()
    if len(filtered) == 0:
        return filtered

    # Snap coordinates to HYCOM cell centres, rounded to 3 decimals.
    # Same rationale as the CMEMS version — see cmems_mask.py.
    kept_lo = lo_idx[keep]
    kept_la = la_idx[keep]
    filtered['longitude'] = np.round(lons[kept_lo], 3)
    filtered['latitude'] = np.round(lats[kept_la], 3)
    filtered = filtered.drop_duplicates(
        subset=['longitude', 'latitude'], keep='first'
    ).reset_index(drop=True)
    return filtered
