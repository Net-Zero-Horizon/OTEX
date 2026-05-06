# -*- coding: utf-8 -*-
"""
OTEX Data Module
Oceanographic data processing (CMEMS, HYCOM, NetCDF) and bundled reference data.
"""

# Lazy import for cmems/hycom to avoid copernicusmarine dependency at package import
# from .cmems import (
#     download_data,
#     data_processing,
#     load_temperatures,
# )

from .resources import (
    get_data_path,
    load_regions,
    load_sites,
)
from .regions import (
    BBox,
    Region,
    resolve_region,
    list_regions,
)
from .sites import build_sites
from .bathymetry import fetch_bathymetry, BathymetrySubset
from .coastline import distance_to_shore, get_coastline_index
from .demand import fetch_demand_TWh

__all__ = [
    "download_data",
    "data_processing",
    "load_temperatures",
    "download_data_hycom",
    "get_data_path",
    "load_regions",
    "load_sites",
    # New on-demand catalog (0.2.0+)
    "BBox",
    "Region",
    "resolve_region",
    "list_regions",
    "build_sites",
    "fetch_bathymetry",
    "BathymetrySubset",
    "distance_to_shore",
    "get_coastline_index",
    "fetch_demand_TWh",
]


def __getattr__(name):
    """Lazy import for CMEMS/HYCOM functions."""
    if name in ("download_data", "data_processing", "load_temperatures"):
        from . import cmems
        return getattr(cmems, name)
    if name == "download_data_hycom":
        from .hycom import download_data
        return download_data
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
