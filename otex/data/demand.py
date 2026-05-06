# -*- coding: utf-8 -*-
"""Multi-source electricity demand lookup.

Replaces the bundled ``country_demand.csv`` shipped with earlier OTEX
releases. Annual electricity demand (TWh/yr) for a country is queried,
in order, from the following public sources:

1. **Our World in Data** — bulk
   ``owid-energy-data.csv`` download, cached once locally. Covers
   ~214 ISO 3166-1 alpha-3 codes through year-1.
2. **World Bank Open Data API** — derived as
   ``EG.USE.ELEC.KH.PC * SP.POP.TOTL / 1e9`` (per-capita consumption
   times total population). Slower but covers some territories that
   OWID misses.

Both providers are free and require no authentication. Each result is
cached in ``~/.otex/cache/demand/`` so subsequent calls are
sub-millisecond.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import pandas as pd

_DEFAULT_OWID_URL = (
    "https://github.com/owid/energy-data/raw/master/owid-energy-data.csv"
)
OWID_URL = os.environ.get("OTEX_OWID_URL", _DEFAULT_OWID_URL)

_DEFAULT_WB_BASE = "https://api.worldbank.org/v2"
WB_BASE = os.environ.get("OTEX_WORLDBANK_URL", _DEFAULT_WB_BASE)


def _cache_dir() -> Path:
    base = os.environ.get("OTEX_CACHE_DIR")
    d = Path(base) / "demand" if base else Path.home() / ".otex" / "cache" / "demand"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Provider 1: Our World in Data (bulk CSV) ──────────────────────────

_owid_df_cache: Optional[pd.DataFrame] = None


def _owid_df(refresh: bool = False) -> pd.DataFrame:
    """Download (once) and cache the OWID energy-data CSV."""
    global _owid_df_cache
    if _owid_df_cache is not None and not refresh:
        return _owid_df_cache

    cache_path = _cache_dir() / "owid-energy-data.csv"
    if not cache_path.exists() or refresh:
        with urllib.request.urlopen(OWID_URL, timeout=120) as resp:
            cache_path.write_bytes(resp.read())

    _owid_df_cache = pd.read_csv(cache_path)
    return _owid_df_cache


def _fetch_owid(iso_a3: str) -> Optional[Tuple[float, int]]:
    """Latest non-null ``electricity_demand`` (TWh) for the ISO3 code."""
    df = _owid_df()
    sub = df[
        (df["iso_code"] == iso_a3) & (df["electricity_demand"].notna())
    ]
    if sub.empty:
        return None
    latest = sub.sort_values("year").iloc[-1]
    return float(latest["electricity_demand"]), int(latest["year"])


# ── Provider 2: World Bank Open Data API ──────────────────────────────


def _wb_get(iso_a3: str, indicator: str) -> Optional[Tuple[float, int]]:
    """Fetch the most recent value for a World Bank indicator."""
    url = f"{WB_BASE}/country/{iso_a3}/indicator/{indicator}?format=json&mrv=1"
    with urllib.request.urlopen(url, timeout=20) as resp:
        payload = json.loads(resp.read())
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        return None
    rec = payload[1][0]
    if rec.get("value") is None:
        return None
    return float(rec["value"]), int(rec["date"])


def _fetch_world_bank(iso_a3: str) -> Optional[Tuple[float, int]]:
    """Total demand (TWh) = per-capita kWh × population / 1e9.

    Result is cached as JSON to avoid hitting the API twice per country.
    """
    cache_file = _cache_dir() / f"wb_{iso_a3}.json"
    if cache_file.exists():
        twh, year = json.loads(cache_file.read_text())
        return float(twh), int(year)

    pc = _wb_get(iso_a3, "EG.USE.ELEC.KH.PC")     # per capita, kWh
    if pc is None:
        return None
    pop = _wb_get(iso_a3, "SP.POP.TOTL")          # total population
    if pop is None:
        return None

    pc_kwh, pc_year = pc
    pop_count, _ = pop
    twh = pc_kwh * pop_count / 1.0e9
    year = pc_year
    cache_file.write_text(json.dumps([twh, year]))
    return twh, year


# ── Public multi-source API ───────────────────────────────────────────


_DEFAULT_PROVIDERS: Tuple[Tuple[str, Callable[[str], Optional[Tuple[float, int]]]], ...] = (
    ("owid", _fetch_owid),
    ("world_bank", _fetch_world_bank),
)


def fetch_demand_TWh(
    iso_a3: str,
    *,
    providers: Optional[Sequence[Tuple[str, Callable[[str], Optional[Tuple[float, int]]]]]] = None,
) -> Tuple[Optional[float], Optional[str], Optional[int]]:
    """Return ``(demand_TWh, source_name, year)`` or ``(None, None, None)``.

    Tries each provider in order; the first to return a value wins.

    Parameters
    ----------
    iso_a3 : str
        ISO 3166-1 alpha-3 country code (e.g. ``'JAM'``).
    providers : sequence, optional
        Iterable of ``(name, callable)`` pairs to try in order.
        Defaults to OWID followed by World Bank.
    """
    if providers is None:
        providers = _DEFAULT_PROVIDERS
    for name, fn in providers:
        try:
            result = fn(iso_a3)
        except Exception:
            result = None
        if result is not None:
            twh, year = result
            return twh, name, year
    return None, None, None
