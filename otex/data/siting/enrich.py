# -*- coding: utf-8 -*-
"""
Enrich a sites DataFrame with siting attributes derived from global layers.

Adds these columns (silently zeroed/False if the source layer is unavailable):

    in_mpa_strict        bool   - inside WDPA IUCN cat I-IV polygon (with buffer)
    ais_density_pct      float  - vessel-density percentile at site, 0..100
    pga_475              float  - peak ground acceleration [g] at site
    cyclone_freq_per_yr  float  - count of IBTrACS tracks within 100 km / years

The function imports geopandas/rasterio lazily and falls back gracefully when
the `siting` extra is not installed: in that case columns are still added but
filled with zeros/False, so downstream code can rely on their presence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .download import SitingDownloadError, ensure_layers


_STRICT_IUCN_CATEGORIES = {"Ia", "Ib", "II", "III", "IV"}
_CYCLONE_RADIUS_KM = 100.0
# Seismic hazard maps (GEM) are defined on land only. OTEC platforms are
# offshore, so we sample within a coastal window and take the max — the
# nearest coastal fault dominates the platform's seismic exposure.
_PGA_WINDOW_KM = 50.0


def enrich_sites(
    sites_df: pd.DataFrame,
    *,
    mpa_buffer_km: float = 5.0,
    ais_buffer_km: float = 5.0,
    cache_dir: Optional[str] = None,
    refresh: bool = False,
    layers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Return a copy of sites_df with siting columns appended.

    Args:
        sites_df: must contain 'longitude' and 'latitude' columns (EPSG:4326).
        mpa_buffer_km: buffer applied to MPA polygons before point-in-polygon test.
        ais_buffer_km: buffer applied around the site when sampling AIS density.
        cache_dir: override default cache location.
        refresh: force re-download of layers.
        layers: subset of layers to process (default: all four).

    Notes:
        - Sites outside the source raster footprint receive NaN for the
          corresponding column; callers should treat NaN as "no data" (zero risk).
        - This function is intentionally tolerant: a missing layer logs a warning
          and the column is filled with the neutral default (False / 0.0).
    """
    out = sites_df.copy()
    out["in_mpa_strict"] = False
    out["ais_density_pct"] = 0.0
    out["pga_475"] = 0.0
    out["cyclone_freq_per_yr"] = 0.0

    if layers is None:
        layers = ["wdpa", "ais", "pga", "ibtracs"]

    try:
        paths = ensure_layers(layers, cache_dir=cache_dir, refresh=refresh)
    except SitingDownloadError as exc:
        print(f"[siting] WARNING: {exc}. Returning neutral siting columns.")
        return out

    if "wdpa" in paths:
        try:
            out["in_mpa_strict"] = _flag_in_mpa(
                out, paths["wdpa"], buffer_km=mpa_buffer_km
            )
        except Exception as exc:
            print(f"[siting] WARNING: WDPA enrichment failed: {exc}")

    if "ais" in paths:
        try:
            out["ais_density_pct"] = _sample_ais_percentile(
                out, paths["ais"], buffer_km=ais_buffer_km
            )
        except Exception as exc:
            print(f"[siting] WARNING: AIS enrichment failed: {exc}")

    if "pga" in paths:
        try:
            vals = _sample_raster_window(out, paths["pga"], buffer_km=_PGA_WINDOW_KM)
            out["pga_475"] = np.nan_to_num(vals, nan=0.0)
        except Exception as exc:
            print(f"[siting] WARNING: PGA enrichment failed: {exc}")

    if "ibtracs" in paths:
        try:
            out["cyclone_freq_per_yr"] = _cyclone_frequency(
                out, paths["ibtracs"], radius_km=_CYCLONE_RADIUS_KM
            )
        except Exception as exc:
            print(f"[siting] WARNING: IBTrACS enrichment failed: {exc}")

    return out


# ---------------------------------------------------------------------------
# Layer-specific helpers (lazy imports to keep core install lean)
# ---------------------------------------------------------------------------

def _require(module_name: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise ImportError(
            f"'{module_name}' is required for siting enrichment. "
            "Install with: pip install otex[siting]"
        ) from exc


def _flag_in_mpa(sites: pd.DataFrame, wdpa_path: Path, buffer_km: float) -> np.ndarray:
    """Boolean mask: site lies inside a strictly-protected (IUCN I-IV) MPA.

    `wdpa_path` may be either a single vector file (.gpkg, .shp, .parquet)
    or a directory containing the WDPA shapefile distribution. The official
    WDPA shapefile distribution is split into multiple parts (shp_0, shp_1,
    shp_2) due to the 2 GB shapefile size limit; we read every polygon file
    found, clipped by the sites' bounding box (with the buffer expanded into
    it) so we never load the full ~5 GB into memory.
    """
    gpd = _require("geopandas")
    _require("shapely")

    wdpa_path = Path(wdpa_path)
    if wdpa_path.is_dir():
        # All polygon shapefiles in the distribution. Excluding *points* keeps
        # protected-area centroids (which WDPA also publishes) out of the join.
        files = sorted(
            p for p in wdpa_path.rglob("*.shp")
            if "point" not in p.name.lower()
        )
        if not files:
            # Last-ditch fallback: take whatever shapefile is present
            files = sorted(wdpa_path.rglob("*.shp"))
        if not files:
            raise FileNotFoundError(f"No .shp file found under {wdpa_path}")
    else:
        files = [wdpa_path]

    # Bounding box around sites with buffer expansion. Reading WDPA with a bbox
    # mask is dramatically faster than loading full polygons globally.
    deg_buf = (buffer_km / 111.0) + 0.5  # extra slack for high-latitude widening
    minx = sites["longitude"].min() - deg_buf
    maxx = sites["longitude"].max() + deg_buf
    miny = sites["latitude"].min() - deg_buf
    maxy = sites["latitude"].max() + deg_buf
    bbox = (minx, miny, maxx, maxy)

    parts = []
    for f in files:
        try:
            gdf = gpd.read_file(f, bbox=bbox)
        except Exception as exc:
            print(f"[siting] WARNING: skipping {f.name}: {exc}")
            continue
        if gdf.empty:
            continue
        if "IUCN_CAT" in gdf.columns:
            gdf = gdf[gdf["IUCN_CAT"].isin(_STRICT_IUCN_CATEGORIES)]
        # Some WDPA *.prj files in this archive are zero-byte — assume EPSG:4326
        if gdf.crs is None:
            gdf = gdf.set_crs(4326, allow_override=True)
        if not gdf.empty:
            parts.append(gdf[["geometry"]])

    if not parts:
        return np.zeros(len(sites), dtype=bool)

    import pandas as pd  # local — pandas already imported at module top
    strict = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)

    # Buffer in metric CRS (Web Mercator is fine for 5 km within mid-latitudes)
    strict = strict.to_crs(3857)
    strict["geometry"] = strict.geometry.buffer(buffer_km * 1000.0)
    strict = strict.to_crs(4326)

    pts = gpd.GeoDataFrame(
        sites[["longitude", "latitude"]].copy(),
        geometry=gpd.points_from_xy(sites["longitude"], sites["latitude"]),
        crs=4326,
    )
    joined = gpd.sjoin(pts, strict[["geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined["index_right"].notna().values


def _sample_raster(sites: pd.DataFrame, raster_path: Path) -> np.ndarray:
    """Sample raster values at site coordinates (EPSG:4326). NaN if out of bounds."""
    rio = _require("rasterio")
    coords = list(zip(sites["longitude"].values, sites["latitude"].values))
    with rio.open(raster_path) as src:
        # Reproject coords to raster CRS if needed
        if src.crs and src.crs.to_epsg() != 4326:
            from rasterio.warp import transform
            xs, ys = zip(*coords)
            xs_t, ys_t = transform("EPSG:4326", src.crs, list(xs), list(ys))
            coords = list(zip(xs_t, ys_t))
        values = np.array(
            [v[0] if v is not None else np.nan for v in src.sample(coords)],
            dtype=float,
        )
    nodata = src.nodata if hasattr(src, "nodata") else None
    if nodata is not None:
        values = np.where(values == nodata, np.nan, values)
    return values


def _sample_ais_percentile(
    sites: pd.DataFrame, ais_path: Path, buffer_km: float
) -> np.ndarray:
    """Return AIS vessel-density percentile [0, 100] at each site.

    The buffer is applied conceptually as a search radius: we sample a small
    window around each point and take the max, which is robust to a site
    sitting one cell off a busy lane.
    """
    rio = _require("rasterio")
    raw = _sample_raster_window(sites, ais_path, buffer_km)
    # Compute a global percentile rank using the raster's own value distribution
    with rio.open(ais_path) as src:
        sample = src.read(1, masked=True).compressed()
    if sample.size == 0:
        return np.zeros(len(sites))
    # Use only positive density to define the rank distribution
    positive = sample[sample > 0]
    if positive.size == 0:
        return np.zeros(len(sites))
    sorted_pos = np.sort(positive)
    pct = np.searchsorted(sorted_pos, raw, side="right") / sorted_pos.size * 100.0
    pct = np.where(np.isnan(raw), 0.0, pct)
    return np.clip(pct, 0.0, 100.0)


def _sample_raster_window(
    sites: pd.DataFrame, raster_path: Path, buffer_km: float
) -> np.ndarray:
    """Take the max raster value in a small window around each site."""
    rio = _require("rasterio")
    from rasterio.windows import from_bounds

    out = np.full(len(sites), np.nan, dtype=float)
    deg_per_km_lat = 1.0 / 111.0
    with rio.open(raster_path) as src:
        for i, (lon, lat) in enumerate(
            zip(sites["longitude"].values, sites["latitude"].values)
        ):
            deg_per_km_lon = 1.0 / (111.0 * max(np.cos(np.radians(lat)), 0.1))
            dlon = buffer_km * deg_per_km_lon
            dlat = buffer_km * deg_per_km_lat
            try:
                win = from_bounds(
                    lon - dlon, lat - dlat, lon + dlon, lat + dlat, src.transform
                )
                arr = src.read(1, window=win, masked=True)
                # Some rasters (notably GEM PGA) encode missing values as NaN
                # in addition to a nodata sentinel — masked array max() then
                # propagates NaN. Coerce to plain float (NaN doesn't fit in
                # int dtypes, so cast first) and use nanmax.
                data = arr.astype(float).filled(np.nan)
                # Treat extreme sentinels (DBL_MAX-style nodata) as NaN
                data = np.where(np.abs(data) > 1e30, np.nan, data)
                if data.size and np.any(np.isfinite(data)):
                    out[i] = float(np.nanmax(data))
                else:
                    out[i] = 0.0
            except Exception:
                out[i] = np.nan
    return out


def _cyclone_frequency(
    sites: pd.DataFrame, ibtracs_path: Path, radius_km: float
) -> np.ndarray:
    """Tracks per year passing within `radius_km` of each site (IBTrACS v4)."""
    xr = _require("xarray")
    ds = xr.open_dataset(ibtracs_path)
    # IBTrACS variables of interest: lat, lon, season; shape (storm, time)
    if "lat" not in ds or "lon" not in ds or "season" not in ds:
        raise ValueError("IBTrACS file missing expected variables (lat/lon/season)")

    storm_lat = ds["lat"].values  # (n_storms, n_obs)
    storm_lon = ds["lon"].values
    seasons = ds["season"].values
    seasons = seasons[~pd.isna(seasons)]
    n_years = max(1, int(np.ptp(seasons)) + 1)

    # Reduce each storm to its track points (drop NaN)
    pts = np.column_stack([storm_lat.ravel(), storm_lon.ravel()])
    pts = pts[~np.isnan(pts).any(axis=1)]
    storm_idx = np.repeat(np.arange(storm_lat.shape[0]), storm_lat.shape[1])[
        ~np.isnan(np.column_stack([storm_lat.ravel(), storm_lon.ravel()])).any(axis=1)
    ]

    out = np.zeros(len(sites), dtype=float)
    site_lat = sites["latitude"].values
    site_lon = sites["longitude"].values

    for i in range(len(sites)):
        d = _haversine_km(site_lat[i], site_lon[i], pts[:, 0], pts[:, 1])
        within = d <= radius_km
        if not np.any(within):
            continue
        unique_storms = np.unique(storm_idx[within])
        out[i] = unique_storms.size / n_years
    return out


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    rad = np.radians
    dlat = rad(lat2 - lat1)
    dlon = rad(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(rad(lat1)) * np.cos(rad(lat2)) * np.sin(dlon / 2) ** 2
    # Clamp to [0, 1] — float rounding can push 'a' microscopically above 1
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
