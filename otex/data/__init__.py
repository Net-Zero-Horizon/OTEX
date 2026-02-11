# -*- coding: utf-8 -*-
"""
OTEX Data Module
Oceanographic data processing (CMEMS, NetCDF) and bundled reference data.
"""

# Lazy import for cmems to avoid copernicusmarine dependency at package import
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
    "get_data_path",
    "load_regions",
    "load_sites",
]


def __getattr__(name):
    """Lazy import for CMEMS functions to avoid copernicusmarine dependency."""
    if name in ("download_data", "data_processing", "load_temperatures"):
        from . import cmems
        return getattr(cmems, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
