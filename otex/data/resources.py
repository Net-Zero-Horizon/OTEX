# -*- coding: utf-8 -*-
"""
Utilities for loading bundled reference data files.

Provides access to region definitions and CMEMS site properties
that are distributed with the otex package.
"""

from importlib import resources
from pathlib import Path

import pandas as pd


def get_data_path(filename):
    """Get the path to a bundled data file in the otex.data package.

    Args:
        filename: Name of the file (e.g., 'download_ranges_per_region.csv').

    Returns:
        Path object pointing to the file.
    """
    return resources.files("otex.data").joinpath(filename)


def load_regions():
    """Load the regions database as a DataFrame.

    Returns a DataFrame with columns: region, north, east, south, west, demand.
    Each row defines a geographic region with OTEC potential.

    Returns:
        pd.DataFrame: Regions database.
    """
    path = get_data_path("download_ranges_per_region.csv")
    return pd.read_csv(path, delimiter=";")


def load_sites():
    """Load the CMEMS site properties database as a DataFrame.

    Returns a DataFrame with site coordinates, region assignment,
    water depth, and distance to shore for all pre-screened OTEC sites.

    Returns:
        pd.DataFrame: Sites database.
    """
    path = get_data_path("CMEMS_points_with_properties.csv")
    return pd.read_csv(path, delimiter=";")
