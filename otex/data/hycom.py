# -*- coding: utf-8 -*-
"""
OTEX HYCOM Data Module

Download ocean temperature data from the HYCOM (Hybrid Coordinate Ocean
Model) via OPeNDAP.  No authentication is required.

Two experiments are available:
  - GLBv0.08/expt_53.X  (reanalysis, 1994-2015)
  - GLBy0.08/expt_93.0  (analysis,   Dec 2018 - Sep 2024)

The download function produces NetCDF files with CMEMS-compatible variable
names so that ``data_processing()`` in ``cmems.py`` can read them without
changes.
"""

import logging
import os
from time import time as _time

import numpy as np

logger = logging.getLogger(__name__)

# ── HYCOM standard depth levels (40 layers) ──────────────────────────

HYCOM_DEPTHS = [
    0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0,
    30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
    125.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 500.0, 600.0, 700.0,
    800.0, 900.0, 1000.0, 1250.0, 1500.0, 2000.0, 2500.0, 3000.0,
    4000.0, 5000.0,
]

# ── HYCOM experiments ─────────────────────────────────────────────────

HYCOM_EXPERIMENTS = {
    "reanalysis": {
        "url": "https://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_53.X",
        "years": (1994, 2015),
        "time_origin": "2000-01-01 00:00:00",
    },
    "analysis": {
        "url": "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0",
        "years": (2019, 2024),
        "time_origin": "2000-01-01 00:00:00",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────


def get_nearest_hycom_depth(target_depth: float) -> float:
    """Return the nearest HYCOM standard depth level to *target_depth*.

    Parameters
    ----------
    target_depth : float
        Requested depth in metres (positive downward).

    Returns
    -------
    float
        Nearest available HYCOM depth.
    """
    return min(HYCOM_DEPTHS, key=lambda d: abs(d - target_depth))


def get_hycom_experiment(year: int) -> dict:
    """Select the appropriate HYCOM experiment for *year*.

    Raises
    ------
    ValueError
        If the requested year falls in the 2016-2018 gap between
        the reanalysis and analysis products.
    """
    for exp in HYCOM_EXPERIMENTS.values():
        y_min, y_max = exp["years"]
        if y_min <= year <= y_max:
            return exp
    raise ValueError(
        f"HYCOM data is not available for year {year}.  "
        f"Available ranges: reanalysis 1994-2015, analysis 2019-2024."
    )


def _lon_to_360(lon: float) -> float:
    """Convert longitude from -180..180 to 0..360 (HYCOM convention)."""
    return lon % 360.0


# ── Main download function ───────────────────────────────────────────


def download_data(cost_level, inputs, studied_region, new_path):
    """Download HYCOM ocean temperature data for a region.

    Has the **same signature and return type** as
    ``otex.data.cmems.download_data`` so it can be used as a drop-in
    replacement.

    The downloaded NetCDF files use CMEMS-compatible variable and
    dimension names (``thetao``, ``latitude``, ``longitude``, ``depth``,
    ``time``) so that ``data_processing()`` reads them without changes.

    Parameters
    ----------
    cost_level : str
        Cost assumption (``'low_cost'`` or ``'high_cost'``).
    inputs : dict
        Configuration dictionary (from ``parameters_and_constants``).
    studied_region : str
        Region name matching the bundled regions database.
    new_path : str
        Output directory for downloaded NetCDF files.

    Returns
    -------
    list[str]
        Paths to downloaded NetCDF files (WW files first, then CW).
    """
    import xarray as xr

    from otex.data.resources import load_regions

    print(f"    [HYCOM download] Reading regions CSV...", flush=True)
    regions = load_regions()

    if not np.any(regions["region"] == studied_region):
        raise ValueError(
            f"Region '{studied_region}' not found.  "
            "Check for typos or whether it is in download_ranges_per_region.csv"
        )

    parts = regions["region"].value_counts()[studied_region]

    depth_WW = inputs["length_WW_inlet"]
    depth_CW = inputs["length_CW_inlet"]
    date_start = inputs["date_start"]
    date_end = inputs["date_end"]
    year = int(date_start[:4])

    experiment = get_hycom_experiment(year)
    opendap_url = experiment["url"]
    print(
        f"    [HYCOM download] Using experiment: {opendap_url} "
        f"(covers {experiment['years'][0]}-{experiment['years'][1]})",
        flush=True,
    )

    files: list[str] = []

    for depth in [depth_WW, depth_CW]:
        nearest = get_nearest_hycom_depth(depth)
        print(
            f"    [HYCOM download] Requested depth {depth}m → "
            f"nearest HYCOM level {nearest}m",
            flush=True,
        )

        for part in range(parts):
            region_rows = regions[regions["region"] == studied_region]
            north = float(region_rows["north"].iloc[part])
            south = float(region_rows["south"].iloc[part])
            west = float(region_rows["west"].iloc[part])
            east = float(region_rows["east"].iloc[part])

            filename = (
                f"T_{round(depth, 0)}m_{date_start[:4]}"
                f"_{studied_region}_{part + 1}.nc"
            ).replace(" ", "_")
            filepath = os.path.join(new_path, filename)
            files.append(filepath)

            # Skip if already downloaded and valid
            if os.path.exists(filepath):
                try:
                    import netCDF4

                    nc = netCDF4.Dataset(filepath, "r")
                    nc.close()
                    print(
                        f"    [HYCOM download] {filename} already exists and is valid.",
                        flush=True,
                    )
                    continue
                except Exception:
                    print(
                        f"    [HYCOM download] {filename} is corrupted, re-downloading...",
                        flush=True,
                    )
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass

            start_time = _time()
            print(
                f"    [HYCOM download] Downloading {filename} "
                f"(depth={nearest}m, part {part + 1}/{parts})...",
                flush=True,
            )

            # Convert longitude to HYCOM 0-360 convention
            west_360 = _lon_to_360(west)
            east_360 = _lon_to_360(east)

            # Convert dates to hours since time origin for OPeNDAP slicing
            import datetime

            t_origin = datetime.datetime.strptime(
                experiment["time_origin"], "%Y-%m-%d %H:%M:%S"
            )
            t_start = datetime.datetime.strptime(date_start, "%Y-%m-%d %H:%M:%S")
            t_end = datetime.datetime.strptime(date_end, "%Y-%m-%d %H:%M:%S")

            hours_start = (t_start - t_origin).total_seconds() / 3600.0
            hours_end = (t_end - t_origin).total_seconds() / 3600.0

            # Open OPeNDAP dataset (lazy — no data transferred yet)
            ds = xr.open_dataset(opendap_url, decode_times=False)

            try:
                # Select depth (exact match since we mapped to nearest)
                ds_depth = ds.sel(depth=nearest, method="nearest")

                # Select time range
                ds_time = ds_depth.sel(
                    time=slice(hours_start, hours_end),
                )

                # Select spatial extent
                # Handle the case where west > east in 0-360 (crossing 0° meridian)
                if west_360 <= east_360:
                    ds_sub = ds_time.sel(
                        lat=slice(south, north),
                        lon=slice(west_360, east_360),
                    )
                else:
                    # Region crosses 0° meridian — select both sides
                    ds_west = ds_time.sel(
                        lat=slice(south, north),
                        lon=slice(west_360, 360.0),
                    )
                    ds_east = ds_time.sel(
                        lat=slice(south, north),
                        lon=slice(0.0, east_360),
                    )
                    ds_sub = xr.concat([ds_west, ds_east], dim="lon")

                # Extract only temperature variable
                temp = ds_sub["water_temp"].load()

                # Build a CMEMS-compatible dataset:
                #   variable  : thetao
                #   dimensions: time, depth, latitude, longitude
                #   depth is a scalar coordinate expanded to a length-1 dim
                out_ds = xr.Dataset(
                    {
                        "thetao": xr.DataArray(
                            temp.values[:, np.newaxis, :, :]
                            if temp.ndim == 3
                            else temp.values,
                            dims=["time", "depth", "latitude", "longitude"],
                        ),
                    },
                    coords={
                        "time": (
                            "time",
                            ds_sub["time"].values,
                            {"units": f"hours since {experiment['time_origin']}"},
                        ),
                        "depth": ("depth", [nearest]),
                        "latitude": ("latitude", ds_sub["lat"].values),
                        "longitude": (
                            "longitude",
                            # Convert back to -180..180 for CMEMS compatibility
                            np.where(
                                ds_sub["lon"].values > 180,
                                ds_sub["lon"].values - 360,
                                ds_sub["lon"].values,
                            ),
                        ),
                    },
                )

                # Ensure output directory exists
                os.makedirs(os.path.dirname(filepath) or new_path, exist_ok=True)

                out_ds.to_netcdf(filepath, format="NETCDF3_CLASSIC")
                elapsed = (_time() - start_time) / 60
                print(
                    f"    [HYCOM download] {filename} saved. "
                    f"Time: {elapsed:.2f} minutes.",
                    flush=True,
                )

            finally:
                ds.close()

    print(f"    [HYCOM download] Returning {len(files)} files", flush=True)
    return files
