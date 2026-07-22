# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MPL-2.0
"""CMEMS grid mask utilities for site pre-filtering.

The OTEX site catalog is built from GEBCO 2022 bathymetry (~500 m).
CMEMS GLORYS12v1 uses ETOPO2 (~4 km) for its own bathymetry, so many
cells that GEBCO reports as ocean at 600-3000 m land in CMEMS cells
marked as land or below the model seafloor — the result is that
downstream ``data_processing`` drops them as NaN. On regions with
complex bank/channel bathymetry (Bahamas, Maldives, Marshall Islands…)
this can be > 90 % of the candidate pool.

This module downloads a **single-time, single-depth** CMEMS snapshot
per region, extracts the boolean valid-cell mask, and caches it. The
mask is then used by :func:`otex.data.sites.build_sites` to drop
candidates that CMEMS will never populate, so the returned catalog
matches the resolution actually usable by the framework.

Compute nodes without internet: warm the cache once on the login
node — every subsequent :func:`build_sites` call reads the cached
mask offline.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# CMEMS product identifiers (same as the ones used by
# ``otex.data.cmems.download_data``).
_CMEMS_DATASET = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
_CMEMS_VERSION = "202311"

# Grid step (must match _DEFAULT_GRID_RES_DEG in ``otex.data.sites``).
_GRID_STEP = 1.0 / 12.0

# Snapshot date used to sample the mask. Any date within the CMEMS
# reanalysis window (1993-01-01 -> present) works — the mask is
# static up to CMEMS bathymetry revisions.
_MASK_DATE = "2023-01-01"


def _cache_dir() -> Path:
    base = os.environ.get("OTEX_CACHE_DIR")
    d = (Path(base) if base else Path.home() / ".otex" / "cache") / "cmems_mask"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(west: float, east: float, south: float, north: float,
               depth_m: float) -> str:
    payload = repr((round(west, 4), round(east, 4),
                    round(south, 4), round(north, 4),
                    round(depth_m, 2))).encode()
    return hashlib.sha1(payload).hexdigest()[:14]


def _download_snapshot(nc_path: Path, west: float, east: float,
                       south: float, north: float,
                       depth_m: float) -> None:
    """Fetch a 1-day, 1-depth CMEMS thetao subset covering the bbox.

    Uses ``copernicusmarine.subset`` — same client and dataset that
    ``otex.data.cmems.download_data`` uses for the production year-long
    downloads, so the mask is derived from the identical model grid.
    """
    import copernicusmarine

    copernicusmarine.subset(
        dataset_id=_CMEMS_DATASET,
        dataset_version=_CMEMS_VERSION,
        variables=['thetao'],
        minimum_longitude=float(west),
        maximum_longitude=float(east),
        minimum_latitude=float(south),
        maximum_latitude=float(north),
        minimum_depth=float(depth_m),
        maximum_depth=float(depth_m),
        start_datetime=_MASK_DATE,
        end_datetime=_MASK_DATE,
        output_directory=str(nc_path.parent),
        output_filename=nc_path.name,
        netcdf3_compatible=True,
        force_download=True,
    )


def cmems_valid_mask(west: float, east: float, south: float, north: float,
                     depth_m: float,
                     refresh: bool = False,
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the CMEMS-grid valid-cell mask for a bbox at ``depth_m``.

    Parameters
    ----------
    west, east, south, north : float
        Bounding box in degrees.
    depth_m : float
        Depth at which to sample the mask (positive metres below sea
        surface). Cells where CMEMS returns NaN at this depth are
        marked ``False`` (typically because CMEMS bathymetry places
        the seafloor above ``depth_m`` at that cell).
    refresh : bool
        If True, ignore any cached mask and re-download.

    Returns
    -------
    (lons, lats, mask) : (ndarray, ndarray, ndarray)
        1-D longitude and latitude arrays and a 2-D boolean mask
        indexed as ``mask[lat_i, lon_i]``. ``True`` means CMEMS has
        valid ocean data at that cell + depth.

    Notes
    -----
    Requires ``copernicusmarine`` and (on the first call per bbox)
    an active internet connection with CMEMS credentials configured.
    Subsequent calls read the on-disk cache and are offline.
    """
    import xarray as xr

    key = _cache_key(west, east, south, north, depth_m)
    nc_path = _cache_dir() / f"cmems_mask_{key}.nc"

    if refresh or not nc_path.exists():
        _download_snapshot(nc_path, west, east, south, north, depth_m)

    with xr.open_dataset(nc_path) as ds:
        # thetao dims: (time, depth, latitude, longitude). Squeeze the
        # length-1 time and depth axes to a 2-D (lat, lon) grid.
        thetao = ds['thetao'].isel(time=0, depth=0).values
        lons = np.asarray(ds['longitude'].values, dtype=np.float64)
        lats = np.asarray(ds['latitude'].values, dtype=np.float64)

    mask = ~np.isnan(thetao)
    return lons, lats, mask


def filter_sites_by_cmems_mask(
    sites,  # pandas.DataFrame with 'longitude' and 'latitude' columns
    west: float, east: float, south: float, north: float,
    depth_m: float,
    refresh: bool = False,
):
    """Return the subset of ``sites`` whose nearest CMEMS cell is valid.

    A "valid" cell is one where CMEMS thetao is not NaN at ``depth_m``
    on the snapshot date. Sites outside the mask bbox are dropped.

    This is the entry point used by :func:`otex.data.sites.build_sites`
    when ``cmems_verify=True``.
    """
    if sites is None or len(sites) == 0:
        return sites

    lons, lats, mask = cmems_valid_mask(
        west, east, south, north, depth_m, refresh=refresh,
    )
    if lons.size == 0 or lats.size == 0:
        return sites.iloc[0:0].copy()

    # Nearest-cell lookup by rounding to the CMEMS grid step. The grid
    # origin comes from the downloaded mask itself (CMEMS grid is not
    # exactly on a multiple of 1/12°, so use the actual coordinates).
    lon0 = float(lons[0])
    lat0 = float(lats[0])
    n_lat, n_lon = mask.shape

    site_lon = sites['longitude'].to_numpy(dtype=np.float64)
    site_lat = sites['latitude'].to_numpy(dtype=np.float64)
    lo_idx = np.round((site_lon - lon0) / _GRID_STEP).astype(np.int64)
    la_idx = np.round((site_lat - lat0) / _GRID_STEP).astype(np.int64)

    in_range = ((lo_idx >= 0) & (lo_idx < n_lon)
                & (la_idx >= 0) & (la_idx < n_lat))
    keep = np.zeros(len(sites), dtype=bool)
    ok = in_range
    keep[ok] = mask[la_idx[ok], lo_idx[ok]]
    filtered = sites[keep].reset_index(drop=True).copy()
    if len(filtered) == 0:
        return filtered

    # Snap each surviving site's coordinates to its nearest CMEMS grid
    # center, rounded to 3 decimals to match the ``np.round(..., 3)``
    # applied in ``otex.data.cmems._extract_year_data``. This is
    # essential: OTEX's ``_make_grid`` places candidates on multiples
    # of 1/12° starting from the region bbox origin, but the CMEMS
    # native grid has a global half-cell offset (~9 km at the equator).
    # Without snapping, ``_extract_year_data``'s exact ``(lon, lat)``
    # lookup misses ~90 % of the mask-verified sites even though CMEMS
    # actually has data at their nearest cells.
    kept_lo = lo_idx[keep]
    kept_la = la_idx[keep]
    filtered['longitude'] = np.round(lons[kept_lo], 3)
    filtered['latitude'] = np.round(lats[kept_la], 3)
    # Multiple GEBCO candidates can snap to the same CMEMS cell —
    # dedupe deterministically (first wins, preserving the input
    # sort order of build_sites).
    filtered = filtered.drop_duplicates(
        subset=['longitude', 'latitude'], keep='first'
    ).reset_index(drop=True)
    return filtered
