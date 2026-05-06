# -*- coding: utf-8 -*-
"""Region resolver backed by Natural Earth admin-0 boundaries.

Replaces the hand-curated ``download_ranges_per_region.csv`` shipped
with earlier OTEX versions. Users now refer to a country or dependency
by its NE ``NAME``, ``NAME_LONG``, ``ADMIN`` field, or by ISO 3166-1
alpha-2 / alpha-3 code; OTEX returns a :class:`Region` dataclass with
the geometry, ISO codes, and a list of :class:`BBox` ranges suitable
for downloading rectangular slices of CMEMS / HYCOM data.

Multi-part regions that cross the antimeridian (e.g. Fiji) are split
into multiple :class:`BBox` objects so each download stays in a
contiguous longitude range.
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


_DEFAULT_NE_COUNTRIES_URL = (
    "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
)
NE_COUNTRIES_URL = os.environ.get("OTEX_COUNTRIES_URL", _DEFAULT_NE_COUNTRIES_URL)


def _cache_dir() -> Path:
    base = os.environ.get("OTEX_CACHE_DIR")
    if base:
        d = Path(base) / "regions"
    else:
        d = Path.home() / ".otex" / "cache" / "regions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download_countries(target_dir: Path, url: Optional[str] = None) -> Path:
    shp = target_dir / "ne_50m_admin_0_countries.shp"
    if shp.exists():
        return shp

    src = url or NE_COUNTRIES_URL
    with urllib.request.urlopen(src, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(target_dir)

    if not shp.exists():
        raise RuntimeError(
            f"Downloaded {src} but ne_50m_admin_0_countries.shp not found"
        )
    return shp


@dataclass
class BBox:
    """Inclusive lat/lon bounding box. west/east in -180..180."""

    north: float
    south: float
    east: float
    west: float

    def as_tuple(self):
        return (self.north, self.south, self.east, self.west)

    @property
    def crosses_antimeridian(self) -> bool:
        return self.east < self.west


@dataclass
class Region:
    """Resolved geographic region with geometry and bbox(es).

    A region has multiple :class:`BBox` ranges if its polygon crosses
    the antimeridian (e.g. Fiji), so callers can issue a contiguous
    longitude download per part.
    """

    name: str                          # canonical name (NE ``NAME``)
    iso_a2: str                        # ISO 3166-1 alpha-2
    iso_a3: str                        # ISO 3166-1 alpha-3
    geometry: object                   # shapely Polygon or MultiPolygon
    bboxes: List[BBox] = field(default_factory=list)

    @property
    def n_parts(self) -> int:
        return len(self.bboxes)


_cached_gdf = None


def _load_countries(refresh: bool = False):
    """Load (and cache in-process) the Natural Earth countries layer."""
    global _cached_gdf
    if _cached_gdf is not None and not refresh:
        return _cached_gdf

    import geopandas as gpd
    shp = _download_countries(_cache_dir())
    _cached_gdf = gpd.read_file(shp)
    return _cached_gdf


def _split_at_antimeridian(geometry) -> List[BBox]:
    """Return one or more BBox ranges covering the geometry.

    If the geometry crosses the antimeridian (longitude range wider
    than 180°), split it into a western part (west .. +180) and an
    eastern part (-180 .. east).
    """
    minx, miny, maxx, maxy = geometry.bounds
    if (maxx - minx) <= 180.0:
        return [BBox(north=maxy, south=miny, east=maxx, west=minx)]

    # Geometry wraps the antimeridian. Identify the gap by inspecting
    # individual longitudes; the simplest heuristic is to split the
    # bbox at lon=0 and report the two halves.
    geoms = list(getattr(geometry, "geoms", [geometry]))
    east_lons, west_lons = [], []
    east_lats, west_lats = [], []
    for g in geoms:
        gminx, gminy, gmaxx, gmaxy = g.bounds
        if gminx >= 0:
            east_lons.extend([gminx, gmaxx])
            east_lats.extend([gminy, gmaxy])
        elif gmaxx <= 0:
            west_lons.extend([gminx, gmaxx])
            west_lats.extend([gminy, gmaxy])
        else:
            # Mixed-sign part: split into two halves.
            east_lons.extend([0.0, gmaxx])
            east_lats.extend([gminy, gmaxy])
            west_lons.extend([gminx, 0.0])
            west_lats.extend([gminy, gmaxy])

    bboxes = []
    if east_lons:
        bboxes.append(BBox(
            north=max(east_lats), south=min(east_lats),
            east=max(east_lons), west=min(east_lons),
        ))
    if west_lons:
        bboxes.append(BBox(
            north=max(west_lats), south=min(west_lats),
            east=max(west_lons), west=min(west_lons),
        ))
    return bboxes


def _match_row(gdf, query: str):
    """Find the row matching `query` against any of NE's name fields."""
    q = query.strip().lower()
    if not q:
        return None

    # Exact match on ISO alpha-2 / alpha-3 first (3 letters → ISO_A3,
    # 2 letters → ISO_A2). NE also has "ADM0_A3" which is more
    # populated (no -99 placeholders) for dependencies.
    if len(q) == 2:
        for col in ("ISO_A2", "ISO_A2_EH"):
            if col in gdf.columns:
                hit = gdf[gdf[col].astype(str).str.lower() == q]
                if not hit.empty:
                    return hit.iloc[0]
    if len(q) == 3:
        for col in ("ISO_A3", "ISO_A3_EH", "ADM0_A3"):
            if col in gdf.columns:
                hit = gdf[gdf[col].astype(str).str.lower() == q]
                if not hit.empty:
                    return hit.iloc[0]

    # Name fields: try exact case-insensitive match first across the
    # canonical name fields, then fall back to substring containment.
    name_cols = [c for c in ("NAME", "NAME_LONG", "ADMIN", "SOVEREIGNT",
                              "NAME_EN", "FORMAL_EN") if c in gdf.columns]
    for col in name_cols:
        hit = gdf[gdf[col].astype(str).str.lower() == q]
        if not hit.empty:
            return hit.iloc[0]
    for col in name_cols:
        mask = gdf[col].astype(str).str.lower().str.contains(q, na=False)
        if mask.any():
            return gdf[mask].iloc[0]
    return None


def resolve_region(name: str, *, refresh: bool = False) -> Region:
    """Resolve a country / territory name or ISO code to a :class:`Region`.

    Lookup order:

    1. ISO 3166-1 alpha-2 (2 chars).
    2. ISO 3166-1 alpha-3 (3 chars), including NE's ``ADM0_A3`` field.
    3. Exact case-insensitive match against ``NAME``, ``NAME_LONG``,
       ``ADMIN``, ``SOVEREIGNT``.
    4. Substring containment against the same fields.

    Raises ``ValueError`` if no match is found.
    """
    gdf = _load_countries(refresh=refresh)
    row = _match_row(gdf, name)
    if row is None:
        raise ValueError(
            f"Region '{name}' not found in Natural Earth admin-0 "
            f"countries. Try the ISO 3166-1 alpha-3 code, or call "
            f"otex.data.list_regions() for available names."
        )

    geom = row["geometry"]
    bboxes = _split_at_antimeridian(geom)
    return Region(
        name=str(row.get("NAME", name)),
        iso_a2=str(row.get("ISO_A2", row.get("ISO_A2_EH", ""))),
        iso_a3=str(row.get("ISO_A3", row.get("ADM0_A3", ""))),
        geometry=geom,
        bboxes=bboxes,
    )


def list_regions(*, refresh: bool = False) -> "list[str]":
    """Return all NE region names available for ``resolve_region``."""
    gdf = _load_countries(refresh=refresh)
    return sorted(gdf["NAME"].dropna().astype(str).tolist())
