# -*- coding: utf-8 -*-
"""Backwards-compatible loaders for the region and site catalogs.

Since 0.2.0 OTEX no longer ships any large bundled data file:

* The 9.7 MB ``CMEMS_points_with_properties.csv`` is replaced by
  :func:`otex.data.sites.build_sites` (ETOPO 2022 + Natural Earth,
  on demand).
* The bbox-only ``download_ranges_per_region.csv`` is replaced by
  :func:`otex.data.regions.resolve_region` (Natural Earth admin-0).
* Annual electricity demand is fetched from a multi-source provider
  in :mod:`otex.data.demand` (Our World in Data + World Bank).

The two thin wrappers below preserve the legacy entry points for any
external code that imported them, but reroute through the new
modules. ``load_regions`` no longer carries a demand column — call
``otex.data.demand.fetch_demand_TWh(iso_a3)`` when you need it.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Optional

import pandas as pd


def get_data_path(filename: str) -> Path:
    """Resolve a bundled data filename to an absolute path."""
    return resources.files("otex.data").joinpath(filename)


def load_regions() -> pd.DataFrame:
    """Return a DataFrame describing every known region.

    Columns:

    * ``region`` — display name (Natural Earth ``NAME``).
    * ``iso_a3`` — ISO 3166-1 alpha-3 code where available.
    * ``north``, ``south``, ``east``, ``west`` — bounding box of the
      region's full geometry. Multi-part regions (e.g. Fiji crossing
      the antimeridian) only show the global bounds here; call
      :func:`otex.data.regions.resolve_region` for the per-part
      breakdown that download routines use.

    Annual electricity demand is no longer included — fetch it on
    demand via :func:`otex.data.demand.fetch_demand_TWh`.
    """
    from .regions import _load_countries

    gdf = _load_countries()
    rows = []
    for _, r in gdf.iterrows():
        geom = r["geometry"]
        if geom is None or geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        rows.append({
            "region": str(r.get("NAME", "")),
            "iso_a3": str(r.get("ISO_A3", r.get("ADM0_A3", ""))),
            "north": maxy,
            "south": miny,
            "east": maxx,
            "west": minx,
        })
    return pd.DataFrame(rows)


def load_sites(region: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """Build the OTEC site DataFrame for ``region`` on demand.

    The legacy 0.1.x signature took no arguments and returned the full
    global site catalogue (~218k rows from a bundled CSV). That CSV is
    no longer shipped (it was 9.7 MB and went stale quickly), so this
    function now requires a region argument and forwards extra kwargs
    to :func:`otex.data.sites.build_sites` (e.g. ``min_depth``,
    ``max_depth``, ``grid_resolution``).

    Parameters
    ----------
    region : str
        Country / territory name or ISO code (resolved via Natural
        Earth). See :func:`otex.data.regions.resolve_region`.

    Raises
    ------
    ValueError
        If ``region`` is None — required since 0.2.0.
    """
    if region is None:
        raise ValueError(
            "load_sites() requires a region argument since 0.2.0. "
            "The bundled CMEMS_points_with_properties.csv was removed "
            "in favour of on-demand site building. Pass a region name "
            "or ISO code, or use otex.data.sites.build_sites() directly."
        )
    from .sites import build_sites
    return build_sites(region, **kwargs)
