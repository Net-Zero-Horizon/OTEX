#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-compute siting attributes for the bundled OTEX sites database.

Runs the same enrichment used at runtime (otex.data.siting.enrich_sites) and
writes the result as a CSV alongside the original CMEMS_points_with_properties
file. Useful when:
  * You want to inspect the enriched table.
  * You want to ship a baked-in version of the enriched CSV with a fork.
  * You want to refresh the cached layers ahead of a batch run.

Usage:
    python scripts/build_siting_layers.py
    python scripts/build_siting_layers.py --refresh --mpa-buffer 10
    python scripts/build_siting_layers.py --output /tmp/sites_enriched.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from otex.data.resources import get_data_path, load_sites
from otex.data.siting import enrich_sites


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpa-buffer", type=float, default=5.0,
                        help="Buffer (km) around protected areas. Default: 5.0")
    parser.add_argument("--ais-buffer", type=float, default=5.0,
                        help="Buffer (km) for AIS density sampling. Default: 5.0")
    parser.add_argument("--cache-dir", default=None,
                        help="Override siting cache directory.")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-download layers even if cached.")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: alongside the source CSV).")
    parser.add_argument("--layers", nargs="+", default=None,
                        choices=["wdpa", "ais", "pga", "ibtracs"],
                        help="Restrict to a subset of layers.")
    args = parser.parse_args()

    sites = load_sites()
    print(f"[siting] loaded {len(sites)} sites; running enrichment...")
    enriched = enrich_sites(
        sites,
        mpa_buffer_km=args.mpa_buffer,
        ais_buffer_km=args.ais_buffer,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
        layers=args.layers,
    )

    if args.output:
        out_path = Path(args.output)
    else:
        src = Path(str(get_data_path("CMEMS_points_with_properties.csv")))
        out_path = src.with_name("CMEMS_points_with_siting.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, sep=";", index=False)
    print(f"[siting] wrote {out_path}")
    print(enriched.head())
    return 0


if __name__ == "__main__":
    sys.exit(main())
