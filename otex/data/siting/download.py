# -*- coding: utf-8 -*-
"""
Download and cache global siting layers (WDPA, EMODnet AIS, GEM PGA, IBTrACS).

Layers are downloaded once and cached at ~/.otex/siting_cache/. Subsequent runs
reuse the cache. Pass refresh=True to force re-download.

Each source has a default URL that can be overridden either by editing
LAYER_SOURCES below or by setting an environment variable per layer:

    OTEX_WDPA_URL          - WDPA polygons (zipped shapefile or GeoPackage)
    OTEX_AIS_URL           - EMODnet vessel density GeoTIFF
    OTEX_PGA_URL           - GEM PGA 475-yr GeoTIFF
    OTEX_IBTRACS_URL       - NOAA IBTrACS NetCDF

If a default URL becomes stale, the user can supply a new one without changing
code. The contents/format requirements per layer are documented in enrich.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


DEFAULT_CACHE_DIR = Path.home() / ".otex" / "siting_cache"


class SitingDownloadError(RuntimeError):
    """Raised when a siting layer cannot be downloaded or located."""


# Default URLs. Override via env var or by passing urls= to ensure_layers().
# Keep these documented; if a host changes, the user has a single place to
# patch (env var) rather than editing source code.
LAYER_SOURCES: Dict[str, Dict[str, str]] = {
    "wdpa": {
        # WDPA April 2026 snapshot, mirrored on Zenodo with direct download
        # (no click-through, CC-BY 4.0). DOI: 10.5281/zenodo.19873142.
        # Archive is ~4.2 GB containing the full WDPA shapefile distribution;
        # we extract everything into <cache_dir>/wdpa/ and let enrich.py find
        # the polygon shapefile by globbing for the largest *.shp inside.
        "url_env": "OTEX_WDPA_URL",
        "default_url": (
            "https://zenodo.org/api/records/19873142/files/"
            "WDPA_04_2026.zip/content"
        ),
        "filename": "wdpa",
        "archive": "zip_all",
    },
    "ais": {
        # World Bank / IMF Global Shipping Traffic Density - all vessel types,
        # AIS 2015-2021 cumulative, ~500 m global resolution. Direct HTTPS, no
        # auth, CC-BY 4.0, 510 MB zipped GeoTIFF. Dataset ID 0037580.
        # The unit is AIS ping count per cell (proxy for vessel-hours).
        # Override OTEX_AIS_URL to use EMODnet (EU-only, monthly) or another
        # source. The "all vessels" variant is the default; per-category
        # variants (commercial, fishing, passenger) live at sibling URLs.
        "url_env": "OTEX_AIS_URL",
        "default_url": (
            "https://datacatalogfiles.worldbank.org/"
            "ddh-published/0037580/5/DR0045406/shipdensity_global.zip"
        ),
        "filename": "global_vessel_density.tif",
        "archive": "zip",
    },
    "pga": {
        # GEM Global Seismic Hazard Map v2023.1 - PGA at 10% probability of
        # exceedance in 50 years (475-yr return period), reference rock site.
        # Zenodo deposit, pinned version DOI 10.5281/zenodo.8409647.
        # License: CC-BY-NC-SA 4.0 (non-commercial, share-alike).
        # File is a ~33 MB ZIP containing the GeoTIFF; we auto-extract on save.
        "url_env": "OTEX_PGA_URL",
        "default_url": (
            "https://zenodo.org/api/records/8409647/files/"
            "GEM-GSHM_PGA-475y-rock_v2023.zip/content"
        ),
        "filename": "gem_pga_475.tif",
        "archive": "zip",  # signals _download_file to extract
    },
    "ibtracs": {
        # NOAA NCEI IBTrACS v04r01 ALL basins, NetCDF
        "url_env": "OTEX_IBTRACS_URL",
        "default_url": (
            "https://www.ncei.noaa.gov/data/"
            "international-best-track-archive-for-climate-stewardship-ibtracs/"
            "v04r01/access/netcdf/IBTrACS.ALL.v04r01.nc"
        ),
        "filename": "ibtracs_all.nc",
    },
}


def get_cache_dir(cache_dir: Optional[str | os.PathLike] = None) -> Path:
    """Return the resolved cache directory, creating it if needed."""
    path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_url(layer: str) -> str:
    src = LAYER_SOURCES[layer]
    return os.environ.get(src["url_env"], src["default_url"])


def _download_file(url: str, dest: Path, archive: Optional[str] = None) -> None:
    """Stream a URL to disk. If archive='zip', extract the first matching member.

    For archive='zip' we look inside the downloaded zip for a file whose suffix
    matches `dest` (e.g. dest=foo.tif -> extract the first .tif in the archive)
    and write that file to `dest`. This keeps callers simple: they always end
    up with a single file at the path they asked for.
    """
    try:
        import requests
    except ImportError as exc:
        raise SitingDownloadError(
            "The 'requests' package is required to download siting layers. "
            "Install OTEX with the siting extra: pip install otex[siting]"
        ) from exc

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        if archive == "zip":
            _extract_zip_member(tmp, dest)
            tmp.unlink(missing_ok=True)
        elif archive == "zip_all":
            _extract_zip_all(tmp, dest)
            tmp.unlink(missing_ok=True)
        else:
            tmp.replace(dest)
    except Exception as exc:  # network, HTTP, IO
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise SitingDownloadError(
            f"Failed to download {url} -> {dest}: {exc}"
        ) from exc


def _extract_zip_all(zip_path: Path, dest_dir: Path) -> None:
    """Extract all members into dest_dir (created if missing)."""
    import zipfile

    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def _extract_zip_member(zip_path: Path, dest: Path) -> None:
    """Extract the first archive member matching dest's suffix into dest."""
    import zipfile

    target_suffix = dest.suffix.lower()
    with zipfile.ZipFile(zip_path) as zf:
        candidates = [n for n in zf.namelist() if n.lower().endswith(target_suffix)]
        if not candidates:
            raise SitingDownloadError(
                f"No '{target_suffix}' member found in archive {zip_path}"
            )
        # Prefer top-level entries over nested ones for predictability
        candidates.sort(key=lambda n: (n.count("/"), len(n)))
        with zf.open(candidates[0]) as src, open(dest, "wb") as out:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)


def ensure_layers(
    layers: Optional[list[str]] = None,
    cache_dir: Optional[str | os.PathLike] = None,
    refresh: bool = False,
) -> Dict[str, Path]:
    """
    Make sure each requested layer exists on disk; download if missing.

    Args:
        layers: subset of {"wdpa", "ais", "pga", "ibtracs"}; default = all.
        cache_dir: where to store files (default ~/.otex/siting_cache).
        refresh: force re-download even if cached.

    Returns:
        Mapping {layer_name: local_path}.

    Raises:
        SitingDownloadError if a layer is missing AND no URL is configured.
    """
    if layers is None:
        layers = list(LAYER_SOURCES.keys())

    cache = get_cache_dir(cache_dir)
    out: Dict[str, Path] = {}
    for layer in layers:
        if layer not in LAYER_SOURCES:
            raise SitingDownloadError(f"Unknown siting layer: {layer}")
        spec = LAYER_SOURCES[layer]
        dest = cache / spec["filename"]
        # A cached directory counts only if it has at least one entry;
        # an empty dir from a previous failed extract should trigger re-download.
        already_cached = dest.exists() and (
            dest.is_file() or any(dest.iterdir())
        )
        if already_cached and not refresh:
            out[layer] = dest
            continue
        url = _resolve_url(layer)
        if not url:
            raise SitingDownloadError(
                f"No URL configured for layer '{layer}'. Set the {spec['url_env']} "
                f"environment variable or place the file at {dest} manually."
            )
        print(f"[siting] downloading {layer} from {url} -> {dest}")
        _download_file(url, dest, archive=spec.get("archive"))
        out[layer] = dest
    return out
