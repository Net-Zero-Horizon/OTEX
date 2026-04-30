# -*- coding: utf-8 -*-
"""
OTEX siting layers: protected areas, shipping lanes, seismic and cyclone hazards.

Two responsibilities:
  * Download global geospatial layers on first use, then cache them locally.
  * Enrich the OTEX sites DataFrame with per-site columns:
        in_mpa_strict       (bool)   - inside WDPA IUCN I-IV (with buffer)
        ais_density_pct     (float)  - vessel density percentile [0, 100]
        pga_475             (float)  - peak ground acceleration [g], 475-yr
        cyclone_freq_per_yr (float)  - cyclone tracks/yr within 100 km

Geospatial dependencies (geopandas, rasterio, shapely, pyproj, requests) are
imported lazily — install via the `siting` extra:

    pip install otex[siting]
"""

from .download import (
    DEFAULT_CACHE_DIR,
    SitingDownloadError,
    ensure_layers,
    get_cache_dir,
)
from .enrich import enrich_sites

__all__ = [
    "DEFAULT_CACHE_DIR",
    "SitingDownloadError",
    "ensure_layers",
    "get_cache_dir",
    "enrich_sites",
]
