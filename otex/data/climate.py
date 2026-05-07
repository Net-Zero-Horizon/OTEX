# -*- coding: utf-8 -*-
"""CMIP6 climate-scenario support for OTEC analyses.

Implements **delta-method downscaling** — the IPCC-standard approach
for regional resource assessment under future climate scenarios:

    T_future(lon, lat, t) = T_CMEMS_historical(lon, lat, t) + Δ(lon, lat)

where Δ is the time-mean anomaly between a future and a baseline period
of a CMIP6 GCM. Anomalies are typically much more robust between models
than absolute temperatures because systematic GCM biases cancel.

OTEX pulls thetao fields directly from the **Pangeo CMIP6 Zarr archive**
(public Google Cloud bucket, anonymous access). The first call per
(model, scenario, period, depth) fetches a regional slice (~1-5 MB)
and caches it locally; subsequent calls are sub-millisecond.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .regions import BBox


# ─── Curated CMIP6 catalog ───────────────────────────────────────────
#
# Three GCMs spanning equilibrium climate sensitivity (ECS) range:
#
#   * MPI-ESM1-2-LR  — low ECS  (~3.0 K)
#   * EC-Earth3      — mid ECS  (~4.3 K)
#   * CanESM5        — high ECS (~5.6 K)
#
# Pinning specific versions guarantees reproducibility even if Pangeo
# republishes newer catalog entries. Override via env var
# ``OTEX_CMIP6_CATALOG`` to point at a JSON file with the same shape.

# Key: (model, experiment) → relative Zarr path under gs://cmip6/CMIP6/
_DEFAULT_CATALOG = {
    # MPI-ESM1-2-LR
    ('MPI-ESM1-2-LR', 'historical'):
        'CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Omon/thetao/gn/v20190710',
    ('MPI-ESM1-2-LR', 'ssp126'):
        'ScenarioMIP/MPI-M/MPI-ESM1-2-LR/ssp126/r1i1p1f1/Omon/thetao/gn/v20190710',
    ('MPI-ESM1-2-LR', 'ssp245'):
        'ScenarioMIP/MPI-M/MPI-ESM1-2-LR/ssp245/r1i1p1f1/Omon/thetao/gn/v20190710',
    ('MPI-ESM1-2-LR', 'ssp370'):
        'ScenarioMIP/MPI-M/MPI-ESM1-2-LR/ssp370/r1i1p1f1/Omon/thetao/gn/v20190710',
    ('MPI-ESM1-2-LR', 'ssp585'):
        'ScenarioMIP/MPI-M/MPI-ESM1-2-LR/ssp585/r1i1p1f1/Omon/thetao/gn/v20190710',

    # EC-Earth3
    ('EC-Earth3', 'historical'):
        'CMIP/EC-Earth-Consortium/EC-Earth3/historical/r1i1p1f1/Omon/thetao/gn/v20200918',
    ('EC-Earth3', 'ssp126'):
        'ScenarioMIP/EC-Earth-Consortium/EC-Earth3/ssp126/r1i1p1f1/Omon/thetao/gn/v20200918',
    ('EC-Earth3', 'ssp245'):
        'ScenarioMIP/EC-Earth-Consortium/EC-Earth3/ssp245/r1i1p1f1/Omon/thetao/gn/v20200918',
    ('EC-Earth3', 'ssp370'):
        'ScenarioMIP/EC-Earth-Consortium/EC-Earth3/ssp370/r1i1p1f1/Omon/thetao/gn/v20200918',
    ('EC-Earth3', 'ssp585'):
        'ScenarioMIP/EC-Earth-Consortium/EC-Earth3/ssp585/r1i1p1f1/Omon/thetao/gn/v20200918',

    # CanESM5
    ('CanESM5', 'historical'):
        'CMIP/CCCma/CanESM5/historical/r1i1p1f1/Omon/thetao/gn/v20190429',
    ('CanESM5', 'ssp126'):
        'ScenarioMIP/CCCma/CanESM5/ssp126/r1i1p1f1/Omon/thetao/gn/v20190429',
    ('CanESM5', 'ssp245'):
        'ScenarioMIP/CCCma/CanESM5/ssp245/r1i1p1f1/Omon/thetao/gn/v20190429',
    ('CanESM5', 'ssp370'):
        'ScenarioMIP/CCCma/CanESM5/ssp370/r1i1p1f1/Omon/thetao/gn/v20190429',
    ('CanESM5', 'ssp585'):
        'ScenarioMIP/CCCma/CanESM5/ssp585/r1i1p1f1/Omon/thetao/gn/v20190429',
}

DEFAULT_MODELS: Tuple[str, ...] = ('MPI-ESM1-2-LR', 'EC-Earth3', 'CanESM5')
SCENARIOS: Tuple[str, ...] = ('historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585')

# IPCC AR6 reference baseline period.
DEFAULT_BASELINE = (1995, 2014)
DEFAULT_FUTURE_WINDOW_YEARS = 30


def _gcs_url(rel_path: str) -> str:
    """Return the public Google Cloud Storage URL for a Pangeo path."""
    return f'gs://cmip6/CMIP6/{rel_path}'


def _cache_dir() -> Path:
    base = os.environ.get('OTEX_CACHE_DIR')
    d = Path(base) / 'cmip6' if base else Path.home() / '.otex' / 'cache' / 'cmip6'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _bbox_hash(bbox: BBox) -> str:
    rounded = tuple(round(v, 2) for v in bbox.as_tuple())
    return hashlib.sha1(repr(rounded).encode()).hexdigest()[:10]


def _slice_bbox(ds: xr.Dataset, bbox: BBox) -> xr.Dataset:
    """Mask a CMIP6 dataset to a lat/lon bounding box.

    CMIP6 ocean grids are typically curvilinear (``i``, ``j`` index
    dims) with 2-D ``latitude``/``longitude`` coords. We build a
    boolean mask on the 2-D coords and use ``xr.Dataset.where`` so
    the spatial dims survive (we still need them to interpolate
    onto target points later).
    """
    lat = ds['latitude']
    lon = ds['longitude']
    # Normalise lon to -180..180 for comparison.
    lon180 = xr.where(lon > 180, lon - 360, lon)

    # Pad bbox by 1° so the interpolation has neighbours.
    pad = 1.0
    mask = (
        (lat >= bbox.south - pad) & (lat <= bbox.north + pad)
        & (lon180 >= bbox.west - pad) & (lon180 <= bbox.east + pad)
    )
    # Keep only i/j ranges where any cell matches; drops huge global
    # slabs. Boolean masks coming from dask must be computed first
    # (xarray refuses lazy-mask isel because the result shape is
    # not known until the mask is materialised).
    j_any = mask.any(dim=[d for d in mask.dims if d != 'j']).compute()
    i_any = mask.any(dim=[d for d in mask.dims if d != 'i']).compute()
    return ds.isel(j=j_any, i=i_any)


def _interp_to_depth(ds_or_da, depth_m: float):
    """Linear interpolation of a depth-stratified field onto ``depth_m``."""
    # CMIP6 ocean datasets use ``lev`` for depth in metres.
    return ds_or_da.interp(lev=depth_m)


def _open_zarr_anonymous(url: str) -> xr.Dataset:
    """Open a Zarr store on Google Cloud anonymously."""
    return xr.open_zarr(
        url, consolidated=True, storage_options={'token': 'anon'}
    )


# ─── Public API ──────────────────────────────────────────────────────


def fetch_thetao_mean(
    model: str,
    scenario: str,
    period: Tuple[int, int],
    depth_m: float,
    bbox: BBox,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Time-mean thetao at ``depth_m`` over ``period`` for a region.

    Parameters
    ----------
    model : str
        GCM name (must be a key in ``_DEFAULT_CATALOG``).
    scenario : str
        ``'historical'`` or one of the SSPs.
    period : (year_start, year_end)
        Inclusive year range to average over.
    depth_m : float
        Target depth in metres (interpolated from GCM levels).
    bbox : :class:`BBox`
        Spatial extent of the slice.

    Returns
    -------
    pandas.DataFrame
        Columns: ``lon``, ``lat``, ``thetao_C`` (degrees Celsius). One
        row per GCM grid cell within the padded bbox.
    """
    if (model, scenario) not in _DEFAULT_CATALOG:
        raise ValueError(
            f"({model!r}, {scenario!r}) not in CMIP6 catalog. "
            f"Available models: {sorted({m for m, _ in _DEFAULT_CATALOG})}; "
            f"scenarios: {SCENARIOS}"
        )

    cache_path = _cache_dir() / (
        f'thetao_{model}_{scenario}_{period[0]}-{period[1]}'
        f'_d{int(round(depth_m))}_{_bbox_hash(bbox)}.parquet'
    )
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    rel = _DEFAULT_CATALOG[(model, scenario)]
    ds = _open_zarr_anonymous(_gcs_url(rel))

    # Restrict to the requested years before any reduction happens.
    t_start = f'{period[0]}-01-01'
    t_end = f'{period[1]}-12-31'
    ds = ds.sel(time=slice(t_start, t_end))
    if ds.sizes.get('time', 0) == 0:
        raise ValueError(
            f"No timesteps in {model}/{scenario} for {period}. "
            f"This experiment may not cover that range."
        )

    # Limit to bbox (drops huge tracts of ocean we don't need).
    ds = _slice_bbox(ds, bbox)

    # Vertical interp + temporal mean. Order matters: depth interp on
    # the lazy array first keeps memory bounded.
    field = _interp_to_depth(ds['thetao'], depth_m).mean(dim='time')

    # Materialise (this is when the Zarr download actually happens).
    field = field.load()

    # CMIP6 thetao is in kelvin; convert to Celsius.
    field_C = field - 273.15

    # Flatten 2-D (j, i) curvilinear field into a (lon, lat, thetao)
    # frame so the rest of OTEX (which works on irregular site lists)
    # can interpolate freely.
    lat2d = ds['latitude'].values.ravel()
    lon2d = ds['longitude'].values.ravel()
    lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)
    vals = field_C.values.ravel()

    out = pd.DataFrame({
        'lon': lon2d,
        'lat': lat2d,
        'thetao_C': vals,
    }).dropna()

    out.to_parquet(cache_path, index=False)
    return out


def compute_delta_field(
    model: str,
    scenario: str,
    target_year: int,
    depth_m: float,
    bbox: BBox,
    *,
    baseline_period: Tuple[int, int] = DEFAULT_BASELINE,
    future_window_years: int = DEFAULT_FUTURE_WINDOW_YEARS,
    refresh: bool = False,
) -> pd.DataFrame:
    """Delta field (future minus baseline) for a single model.

    Returns
    -------
    pandas.DataFrame
        Columns: ``lon``, ``lat``, ``delta_C`` (kelvin == degrees C
        difference). Same row order as :func:`fetch_thetao_mean`.
    """
    if scenario == 'historical':
        # Explicit no-op scenario: zero delta everywhere.
        base = fetch_thetao_mean(model, 'historical', baseline_period,
                                  depth_m, bbox, refresh=refresh)
        return pd.DataFrame({'lon': base['lon'], 'lat': base['lat'],
                             'delta_C': np.zeros(len(base))})

    half = future_window_years // 2
    future_period = (target_year - half, target_year - half + future_window_years - 1)

    base = fetch_thetao_mean(model, 'historical', baseline_period,
                              depth_m, bbox, refresh=refresh)
    fut = fetch_thetao_mean(model, scenario, future_period,
                             depth_m, bbox, refresh=refresh)

    # Both come from the same model on the same native grid — rows
    # should align by (lon, lat). Defensive merge anyway.
    merged = base.merge(fut, on=['lon', 'lat'], suffixes=('_base', '_fut'))
    return pd.DataFrame({
        'lon': merged['lon'],
        'lat': merged['lat'],
        'delta_C': merged['thetao_C_fut'] - merged['thetao_C_base'],
    })


@dataclass
class EnsembleDelta:
    """Multi-model delta field with mean and spread."""

    lon: np.ndarray
    lat: np.ndarray
    delta_mean: np.ndarray   # mean across models, °C
    delta_std: np.ndarray    # 1-sigma spread across models, °C
    models: List[str]


def ensemble_delta(
    scenario: str,
    target_year: int,
    depth_m: float,
    bbox: BBox,
    *,
    models: Tuple[str, ...] = DEFAULT_MODELS,
    baseline_period: Tuple[int, int] = DEFAULT_BASELINE,
    future_window_years: int = DEFAULT_FUTURE_WINDOW_YEARS,
    refresh: bool = False,
) -> EnsembleDelta:
    """Aggregate delta across an ensemble of CMIP6 models.

    Each model contributes its own (lon, lat) grid; we interpolate
    every model's delta onto the union of all model points so that
    ensemble mean and spread are well-defined per location.
    """
    per_model = []
    for model in models:
        per_model.append(compute_delta_field(
            model, scenario, target_year, depth_m, bbox,
            baseline_period=baseline_period,
            future_window_years=future_window_years,
            refresh=refresh,
        ))

    # Stack onto the first model's points (assumes models broadly
    # cover the region; for the OTEX bboxes used here this is true).
    from scipy.interpolate import griddata

    ref = per_model[0]
    target_pts = np.column_stack([ref['lon'].values, ref['lat'].values])

    deltas_at_ref = np.empty((len(models), len(ref)), dtype=np.float64)
    deltas_at_ref[0] = ref['delta_C'].values
    for k, m_df in enumerate(per_model[1:], start=1):
        src_pts = np.column_stack([m_df['lon'].values, m_df['lat'].values])
        deltas_at_ref[k] = griddata(
            src_pts, m_df['delta_C'].values, target_pts, method='linear',
        )

    # Linear can leave NaNs at the edges — fall back to nearest there.
    nan_mask = np.isnan(deltas_at_ref)
    if nan_mask.any():
        for k in range(1, len(models)):
            if nan_mask[k].any():
                src_pts = np.column_stack([
                    per_model[k]['lon'].values,
                    per_model[k]['lat'].values,
                ])
                fill = griddata(
                    src_pts, per_model[k]['delta_C'].values,
                    target_pts[nan_mask[k]], method='nearest',
                )
                deltas_at_ref[k, nan_mask[k]] = fill

    return EnsembleDelta(
        lon=ref['lon'].values,
        lat=ref['lat'].values,
        delta_mean=np.nanmean(deltas_at_ref, axis=0),
        delta_std=np.nanstd(deltas_at_ref, axis=0, ddof=0),
        models=list(models),
    )


def delta_at_points(
    target_lons: np.ndarray,
    target_lats: np.ndarray,
    delta: EnsembleDelta,
) -> np.ndarray:
    """Interpolate an ensemble delta onto OTEX site coordinates.

    Returns a 1-D numpy array of length ``len(target_lons)``, in K.
    Linear interpolation, with nearest-neighbour fallback at the
    extrapolation edges.
    """
    from scipy.interpolate import griddata

    src_pts = np.column_stack([delta.lon, delta.lat])
    tgt_pts = np.column_stack([target_lons, target_lats])

    out = griddata(src_pts, delta.delta_mean, tgt_pts, method='linear')
    nan = np.isnan(out)
    if nan.any():
        out[nan] = griddata(
            src_pts, delta.delta_mean, tgt_pts[nan], method='nearest',
        )
    return out
