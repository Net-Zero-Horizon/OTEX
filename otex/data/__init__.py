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

__all__ = [
    "download_data",
    "data_processing",
    "load_temperatures",
    "download_data_hycom",
    "get_data_path",
    "load_regions",
    "load_sites",
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
