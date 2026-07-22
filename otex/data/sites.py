# -*- coding: utf-8 -*-
"""On-demand site builder for OTEC regional analyses.

Produces the same DataFrame schema that earlier versions read from the
bundled ``CMEMS_points_with_properties.csv`` (id, longitude, latitude,
region, water_depth, dist_shore), but computes every column from
authoritative sources at run time:

* coordinates → uniform grid at the oceanographic-data resolution
  (default 0.083° to match CMEMS / HYCOM native cells);
* water_depth → ETOPO 2022 1-arcmin bathymetry (`bathymetry.py`);
* dist_shore  → great-circle distance to Natural Earth coastline
  (`coastline.py`);
* region      → label propagated from the caller.

Sites are cached on disk in ``~/.otex/cache/sites/<key>.parquet`` so
repeat calls take milliseconds. The cache key encodes bbox, depth
filters, grid resolution and the buffer; changing any of those
invalidates the cache automatically.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .bathymetry import fetch_bathymetry
from .coastline import distance_to_shore
from .regions import BBox, Region, resolve_region

# Default native CMEMS / HYCOM grid resolution in degrees.
_DEFAULT_GRID_RES_DEG = 1.0 / 12.0   # 0.0833°

# Buffer applied to a region's tight admin-0 bbox to capture nearby
# offshore waters. ~2° corresponds to ~220 km, which covers any
# reasonable OTEC site (typical max distance ~200 km offshore).
_DEFAULT_OFFSHORE_BUFFER_DEG = 2.0


def _cache_dir() -> Path:
    base = os.environ.get("OTEX_CACHE_DIR")
    if base:
        d = Path(base) / "sites"
    else:
        d = Path.home() / ".otex" / "cache" / "sites"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(
    bboxes: List[BBox],
    region_label: str,
    min_depth: float,
    max_depth: float,
    grid_resolution: float,
    offshore_buffer: float,
    lat_max: Optional[float] = None,
    cmems_verify: bool = False,
    cmems_verify_depth: Optional[float] = None,
) -> str:
    payload = repr((
        [b.as_tuple() for b in bboxes],
        region_label, min_depth, max_depth,
        grid_resolution, offshore_buffer, lat_max,
        cmems_verify, cmems_verify_depth,
    )).encode()
    return hashlib.sha1(payload).hexdigest()[:14]


def _expand_bbox(bbox: BBox, buffer_deg: float) -> BBox:
    """Pad a bbox by `buffer_deg` on each side, clipping to global range."""
    return BBox(
        north=min(90.0, bbox.north + buffer_deg),
        south=max(-90.0, bbox.south - buffer_deg),
        east=min(180.0, bbox.east + buffer_deg),
        west=max(-180.0, bbox.west - buffer_deg),
    )


def _make_grid(bbox: BBox, step: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a uniform lon/lat grid covering bbox at `step` resolution."""
    # Snap edges to the grid origin so multi-region runs share cells.
    lon_min = np.floor(bbox.west / step) * step
    lon_max = np.ceil(bbox.east / step) * step
    lat_min = np.floor(bbox.south / step) * step
    lat_max = np.ceil(bbox.north / step) * step
    lons = np.arange(lon_min, lon_max + step / 2, step)
    lats = np.arange(lat_min, lat_max + step / 2, step)
    LON, LAT = np.meshgrid(lons, lats)
    return LON.ravel(), LAT.ravel()


def _build_part(
    bbox: BBox,
    region_label: str,
    *,
    min_depth: float,
    max_depth: float,
    grid_resolution: float,
    offshore_buffer: float,
) -> pd.DataFrame:
    """Build sites for a single bbox part (used per-part for multi-bbox regions)."""
    expanded = _expand_bbox(bbox, offshore_buffer)
    lon, lat = _make_grid(expanded, grid_resolution)

    # 1. Bathymetry: keep only ocean cells with depth in [min_depth, max_depth].
    bat = fetch_bathymetry(
        north=expanded.north, south=expanded.south,
        east=expanded.east, west=expanded.west,
    )
    elev = bat.water_depth_at(lon, lat)            # negative = below sea level
    # OTEX convention (matches the legacy CSV): water_depth is stored as
    # the *elevation* — negative for ocean cells, e.g. -1500 m means 1500 m
    # below sea level. Downstream filters in regional.py use this sign.
    depth_abs = -elev                               # positive depth in metres
    mask = (depth_abs >= min_depth) & (depth_abs <= max_depth)
    if not np.any(mask):
        # No ocean cells (landlocked country or fully outside depth range).
        # Don't include 'id' here — build_sites() adds it after concatenation.
        return pd.DataFrame(columns=[
            'longitude', 'latitude', 'region',
            'water_depth', 'dist_shore',
        ])

    lon_keep = lon[mask]
    lat_keep = lat[mask]
    elev_keep = elev[mask]                          # already negative for ocean

    # 2. Distance to nearest coast (km).
    dist = distance_to_shore(lon_keep, lat_keep)

    # Sort by (lon, lat) so site IDs are deterministic across runs.
    order = np.lexsort((lat_keep, lon_keep))
    return pd.DataFrame({
        'longitude': lon_keep[order],
        'latitude': lat_keep[order],
        'region': region_label,
        'water_depth': elev_keep[order],            # elevation, m (negative)
        'dist_shore': dist[order],
    })


def build_sites(
    region: Optional[Union[str, Region]] = None,
    *,
    bbox: Optional[Union[BBox, Tuple[float, float, float, float]]] = None,
    polygon: Optional[object] = None,
    min_depth: float = 600.0,
    max_depth: float = 3000.0,
    grid_resolution: float = _DEFAULT_GRID_RES_DEG,
    offshore_buffer_deg: float = _DEFAULT_OFFSHORE_BUFFER_DEG,
    lat_max: Optional[float] = None,
    cmems_verify: bool = True,
    cmems_verify_depth: float = 1062.44,
    refresh: bool = False,
) -> pd.DataFrame:
    """Build a feasible OTEC sites DataFrame on demand.

    Exactly one of ``region``, ``bbox``, or ``polygon`` must be given.

    Parameters
    ----------
    region : str or Region
        Country / territory name or ISO code. Resolved via Natural
        Earth admin-0 boundaries (see :func:`resolve_region`).
    bbox : BBox or (north, south, east, west) tuple
        Direct bounding box, in degrees.
    polygon : shapely Polygon or MultiPolygon
        Custom area. Sites are generated within the polygon's bounds
        and then masked by the polygon (set membership).
    min_depth : float
        Minimum ocean water depth (m, positive). Cells shallower than
        this are excluded. Default 600 m matches OTEX's default
        ``DepthLimits.min_depth``.
    max_depth : float
        Maximum ocean water depth (m, positive). Default 3000 m.
    grid_resolution : float
        Site grid step in degrees. Default 1/12 ≈ 0.0833° to match
        CMEMS (``cmems_mod_glo_phy_my_0.083deg_P1D-m``) and HYCOM
        (``GLBy0.08``).
    offshore_buffer_deg : float
        Buffer applied to the region's tight admin-0 bbox to capture
        nearby offshore waters. Default 2° (~220 km), which is
        sufficient for any practical OTEC distance to shore.
    cmems_verify : bool
        If True (default since 0.5.0), each candidate site is checked
        against a CMEMS thetao snapshot at ``cmems_verify_depth`` and
        dropped when the model returns NaN. This removes GEBCO cells
        that CMEMS does not populate (typically ETOPO2 disagreements
        near bank/channel bathymetry) *before* the year-long CMEMS
        download in ``data_processing``, avoiding wasted I/O and
        reflecting the truly usable pool of sites. Requires
        ``copernicusmarine`` and a CMEMS-configured environment on the
        first call per bbox; the mask is cached under
        ``$OTEX_CACHE_DIR/cmems_mask/`` so subsequent (and offline)
        calls are free. On failure a warning is logged and the
        unverified pool is returned.
    cmems_verify_depth : float
        Depth (m below sea surface) at which to sample the CMEMS mask.
        Default 1062.44 m matches OTEX's default cold-water inlet
        (``SeawaterPipes.cw_inlet_length``) and pyOTEC's
        ``length_CW_inlet``. ``regional.run_regional_analysis`` passes
        the run-time ``inputs['length_CW_inlet']`` instead of relying on
        the default, so single-depth runs always mask at the same depth
        CMEMS will be queried on. For the formal per-site optimiser
        (``optimization`` package), which sweeps ``depth_CW`` inside
        ``DepthLimits`` (typically 600-3000 m), pass this argument set
        to the deepest depth the search will visit — or disable the
        pre-filter with ``cmems_verify=False`` and let each candidate
        depth's NaN mask be applied downstream on the fly.
    refresh : bool
        If True, ignore on-disk cache and re-compute.

    Returns
    -------
    pandas.DataFrame
        Columns: ``id``, ``longitude``, ``latitude``, ``region``,
        ``water_depth`` (m), ``dist_shore`` (km).
    """
    if sum(x is not None for x in (region, bbox, polygon)) != 1:
        raise ValueError(
            "build_sites: pass exactly one of region=, bbox=, polygon="
        )

    # Resolve the input into a list of BBox parts and a region label.
    if region is not None:
        if isinstance(region, str):
            region = resolve_region(region)
        bboxes = list(region.bboxes)
        region_label = region.name
        polygon_geom = region.geometry
    elif bbox is not None:
        if isinstance(bbox, tuple):
            n, s, e, w = bbox
            bbox = BBox(north=n, south=s, east=e, west=w)
        bboxes = [bbox]
        region_label = "custom_bbox"
        polygon_geom = None
    else:
        polygon_geom = polygon
        minx, miny, maxx, maxy = polygon.bounds
        bboxes = [BBox(north=maxy, south=miny, east=maxx, west=minx)]
        region_label = "custom_polygon"

    # Cache lookup. lat_max, cmems_verify and cmems_verify_depth are part
    # of the key so runs with the filter active don't share their cache
    # with unfiltered runs.
    cache_path = _cache_dir() / (_cache_key(
        bboxes, region_label, min_depth, max_depth,
        grid_resolution, offshore_buffer_deg, lat_max,
        cmems_verify, cmems_verify_depth if cmems_verify else None,
    ) + '.parquet')
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    # Build each part, then concatenate. Sites in different parts
    # share a global ID space.
    frames = []
    for part in bboxes:
        df = _build_part(
            part, region_label,
            min_depth=min_depth, max_depth=max_depth,
            grid_resolution=grid_resolution,
            offshore_buffer=offshore_buffer_deg,
        )
        frames.append(df)
    sites = pd.concat(frames, ignore_index=True)

    # Optional polygon mask (only for `polygon=`; for region= the
    # offshore buffer means polygon containment would clip too tight).
    if polygon is not None and not sites.empty:
        from shapely.geometry import Point
        keep = sites.apply(
            lambda r: polygon_geom.contains(Point(r['longitude'], r['latitude'])),
            axis=1,
        )
        sites = sites[keep].reset_index(drop=True)

    # Optional latitude cap (e.g. lat_max=40 restricts to ±40° for OTEC's
    # tropical-and-subtropical operating envelope — outside that band the
    # thermal gradient drops below ~18 °C).
    if lat_max is not None and not sites.empty:
        sites = sites[sites['latitude'].abs() <= float(lat_max)].reset_index(drop=True)

    # CMEMS-mask pre-filter (0.5.0+): drop GEBCO candidates that CMEMS
    # bathymetry marks as land or above-seafloor at cmems_verify_depth,
    # so the returned pool matches what data_processing can actually
    # populate. Falls back gracefully to the unverified pool if the
    # CMEMS client / credentials are unavailable.
    if cmems_verify and not sites.empty:
        try:
            from .cmems_mask import filter_sites_by_cmems_mask
            n_before = len(sites)
            # Union bbox covering every part, padded by the offshore
            # buffer — mirrors the extent that _build_part scans.
            wests, easts, souths, norths = zip(*[
                (_expand_bbox(b, offshore_buffer_deg).west,
                 _expand_bbox(b, offshore_buffer_deg).east,
                 _expand_bbox(b, offshore_buffer_deg).south,
                 _expand_bbox(b, offshore_buffer_deg).north)
                for b in bboxes
            ])
            sites = filter_sites_by_cmems_mask(
                sites,
                west=min(wests), east=max(easts),
                south=min(souths), north=max(norths),
                depth_m=cmems_verify_depth,
                refresh=refresh,
            )
            print(
                f"  [build_sites] CMEMS mask filter kept "
                f"{len(sites)}/{n_before} sites at {cmems_verify_depth:.0f} m."
            )
        except Exception as exc:                     # pragma: no cover
            import warnings
            warnings.warn(
                f"[build_sites] CMEMS mask verification skipped "
                f"({type(exc).__name__}: {exc}). Returning unverified "
                f"pool — downstream data_processing will drop CMEMS-NaN "
                f"sites as before.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Assign deterministic integer site IDs (lon-lat sorted within each part).
    sites = sites.sort_values(['longitude', 'latitude']).reset_index(drop=True)
    sites.insert(0, 'id', np.arange(1, len(sites) + 1, dtype=np.int64))

    sites.to_parquet(cache_path, index=False)
    return sites
