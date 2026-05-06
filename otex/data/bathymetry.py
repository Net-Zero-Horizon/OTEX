# -*- coding: utf-8 -*-
"""ETOPO 2022 bathymetry loader with on-demand OPeNDAP fetching and caching.

OTEX uses ETOPO 2022 v2.0 at 60-arcsecond resolution (~1.85 km) as the
authoritative bathymetry source for site selection. The full global grid
is ~250 MB; this module fetches **only the bbox subset needed for the
current region** via NOAA NCEI's THREDDS OPeNDAP server, then caches the
subset locally so subsequent calls are instant.

The 0.083° (~9 km) resolution of CMEMS makes finer bathymetry unnecessary
for site classification, so 60-arcsec is the right tradeoff between
accuracy and download size.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

# Default NOAA NCEI THREDDS endpoint for ETOPO 2022, 60-arcsecond surface
# elevation. The "surface" variant treats ice sheets as the surface (vs
# bedrock). For OTEC site classification we only care about ocean depth
# below sea level, so either variant gives the same answer at our points.
_DEFAULT_OPENDAP_URL = (
    "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/"
    "60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"
)

# Override via env var, e.g. for offline mirror or testing.
ETOPO_URL = os.environ.get("OTEX_ETOPO_URL", _DEFAULT_OPENDAP_URL)


def _cache_dir() -> Path:
    base = os.environ.get("OTEX_CACHE_DIR")
    if base:
        d = Path(base) / "bathymetry"
    else:
        d = Path.home() / ".otex" / "cache" / "bathymetry"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _bbox_key(north: float, south: float, east: float, west: float) -> str:
    # Round to ETOPO's 60-arcsec grid (1/60 deg) so callers asking for
    # nearby bboxes hit the same cache file.
    rounded = tuple(round(v * 60) / 60 for v in (north, south, east, west))
    h = hashlib.sha1(repr(rounded).encode()).hexdigest()[:12]
    return f"etopo_{h}.nc"


@dataclass
class BathymetrySubset:
    """A regional bathymetry slice with helpers for site classification."""

    elevation: xr.DataArray  # negative = below sea level (m)

    @property
    def lon(self) -> np.ndarray:
        return self.elevation['lon'].values

    @property
    def lat(self) -> np.ndarray:
        return self.elevation['lat'].values

    def water_depth_at(self, longitudes, latitudes) -> np.ndarray:
        """Water depth in metres at the given coordinates.

        Negative values indicate ocean depth below sea level (e.g. -1500
        means 1500 m below surface). Land returns positive elevation.
        Uses nearest-neighbour interpolation onto the ETOPO grid; for
        OTEC-scale grids (CMEMS 0.083°) this is sufficient.
        """
        lons = np.asarray(longitudes, dtype=np.float64)
        lats = np.asarray(latitudes, dtype=np.float64)
        result = self.elevation.sel(
            lon=xr.DataArray(lons, dims="point"),
            lat=xr.DataArray(lats, dims="point"),
            method="nearest",
        ).values
        return result.astype(np.float64)


def fetch_bathymetry(
    north: float,
    south: float,
    east: float,
    west: float,
    *,
    refresh: bool = False,
    url: Optional[str] = None,
) -> BathymetrySubset:
    """Fetch (and cache) the ETOPO 2022 bathymetry subset for a bbox.

    Parameters
    ----------
    north, south, east, west : float
        Bounding box in degrees (north > south; east/west in -180..180).
    refresh : bool
        If True, ignore cache and re-fetch from OPeNDAP.
    url : str, optional
        Override the OPeNDAP URL (defaults to NOAA NCEI THREDDS).
    """
    if north <= south:
        raise ValueError(f"north ({north}) must be > south ({south})")

    cache_path = _cache_dir() / _bbox_key(north, south, east, west)
    if cache_path.exists() and not refresh:
        ds = xr.open_dataset(cache_path)
        return BathymetrySubset(elevation=ds['z'])

    src_url = url or ETOPO_URL
    remote = xr.open_dataset(src_url, decode_times=False)
    try:
        # Handle bboxes that cross the antimeridian (east < west).
        if east >= west:
            sub = remote.sel(
                lat=slice(south, north),
                lon=slice(west, east),
            )
            z = sub['z'].load()
        else:
            west_part = remote.sel(lat=slice(south, north),
                                    lon=slice(west, 180.0))['z'].load()
            east_part = remote.sel(lat=slice(south, north),
                                    lon=slice(-180.0, east))['z'].load()
            z = xr.concat([west_part, east_part], dim='lon')
    finally:
        remote.close()

    # Persist a thin slice (~1-5 MB per region) so subsequent calls skip
    # the network round-trip.
    z.to_netcdf(cache_path)
    return BathymetrySubset(elevation=z)
