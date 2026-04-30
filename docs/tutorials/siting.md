# Site Screening: Protected Areas, Shipping Lanes, and Hazards

OTEX can screen candidate OTEC sites against four global geospatial layers
before running the techno-economic analysis. The siting layer was designed
around two questions:

1. **Where can we *not* build?** — protected areas and busy shipping lanes
   are treated as **hard exclusions**: a site flagged by either is removed
   from the candidate list.
2. **Where is it *more expensive* to build?** — seismic and tropical-cyclone
   exposure are treated as **soft penalties**: a per-site multiplier is
   applied to CAPEX (seismic) and OPEX (cyclone), with a shared shipping
   density multiplier on both.

The feature is **off by default**. Existing scripts and notebooks behave
exactly as before unless siting flags are turned on.

## Quick Start

```python
from otex.config import OTEXConfig, PlantConfig, DataConfig

cfg = OTEXConfig(
    plant=PlantConfig(gross_power=-50_000),       # 50 MW
    data=DataConfig(source='HYCOM', year=2020),
)

# Hard exclusions
cfg.siting.enable_mpa_filter = True
cfg.siting.enable_ais_filter = True

# Soft cost multipliers
cfg.siting.enable_hazard_costs = True
```

The first call that needs a layer triggers a download into
`~/.otex/siting_cache/`. Subsequent runs reuse the cache. To force a refresh,
set `cfg.siting.refresh = True` for one run.

## Data Sources and Licences

| Layer | Source | License | Default cache file |
|-------|--------|---------|--------------------|
| WDPA (April 2026) | [Zenodo 19873142](https://doi.org/10.5281/zenodo.19873142) | CC-BY 4.0 | `wdpa/` (~5 GB extracted) |
| Vessel density | [World Bank Global Shipping Traffic Density](https://datacatalog.worldbank.org/search/dataset/0037580) | CC-BY 4.0 | `global_vessel_density.tif` (~5 GB) |
| Seismic hazard | [GEM Global Seismic Hazard Map v2023.1](https://doi.org/10.5281/zenodo.8409647) | CC-BY-NC-SA 4.0 | `gem_pga_475.tif` (~165 MB) |
| Cyclone tracks | [NOAA IBTrACS v04r01](https://www.ncei.noaa.gov/products/international-best-track-archive) | Public domain | `ibtracs_all.nc` (~50 MB) |

The default URLs are encoded in `otex/data/siting/download.py` and can be
overridden per layer with environment variables:

```bash
export OTEX_WDPA_URL=...
export OTEX_AIS_URL=...
export OTEX_PGA_URL=...
export OTEX_IBTRACS_URL=...
```

This is useful if a host changes, you want to pin a different snapshot, or
you have the file already on disk and prefer not to re-download.

## What Goes Into Each Decision

### Hard exclusion: WDPA

Sites whose buffered position falls inside a polygon classified
**IUCN Ia / Ib / II / III / IV** are flagged `in_mpa_strict = True` and
removed. Categories V and VI (sustainable-use areas) are not flagged.

The buffer (`mpa_buffer_km`, default **5 km**) accounts for the platform
footprint plus mooring/cable spread.

The WDPA shapefile distribution is split into three files (`shp_0`, `shp_1`,
`shp_2`) due to the 2 GB shapefile limit; OTEX reads each one with a bounding
box clip around the site set, so memory stays bounded even for global runs.

### Hard exclusion: shipping lanes

For each site, OTEX samples the World Bank vessel-density raster within
`ais_buffer_km` (default **5 km**) and computes the percentile rank of the
window-max relative to all positive cells in the raster. Sites whose
percentile is **≥ `ais_exclusion_pct` (default 95)** are excluded.

The choice of "max within window" is deliberate: shipping lanes are narrow,
and a candidate site one cell off-axis from a busy lane should still inherit
the risk.

### Soft cost multiplier: shipping intensity

Below the exclusion percentile, the same density value is normalised to
[0, 1] via `pct / ais_exclusion_pct` and multiplied by `w_ais` (default
**0.20**). This term is added to **both** CAPEX and OPEX:

- CAPEX rationale — riser/cable strike risk increases with traffic
  density, raising the contingency budget.
- OPEX rationale — insurance premiums and downtime expectations scale
  with collision probability.

### Soft cost multiplier: seismic hazard (CAPEX only)

GEM publishes peak ground acceleration at 10% probability of exceedance in
50 years (≈475-year return). The map is defined on land only, so we sample
within a **50 km coastal window** and take the max — for an offshore
platform, the dominant fault is whatever is nearest the coast.

The multiplier is `1 + w_seismic · clip(pga / pga_ref_g, 0, 1)`, with
`w_seismic = 0.15` and `pga_ref_g = 0.4 g` by default. Sites farther than
50 km from any land receive zero (no coastal seismic exposure).

### Soft cost multiplier: cyclone frequency (OPEX only)

For each site, OTEX counts unique IBTrACS storm tracks that pass within
**100 km** and divides by the number of years in the catalogue. The result
is normalised against `cyclone_ref_per_yr` (default **0.5 tracks/yr**) and
weighted by `w_cyclone` (default **0.25**), producing the OPEX-only term.

## Configuration Reference

```python
from otex.config import SitingConfig

s = SitingConfig(
    enable_mpa_filter=False,         # Hard MPA exclusion (IUCN I-IV)
    enable_ais_filter=False,         # Hard P95 AIS exclusion
    enable_hazard_costs=False,       # Soft seismic + cyclone multipliers

    mpa_buffer_km=5.0,               # MPA polygon buffer
    ais_buffer_km=5.0,               # AIS sampling window radius
    ais_exclusion_pct=95.0,          # AIS exclusion percentile

    w_ais=0.20,                      # AIS weight, applied to CAPEX and OPEX
    w_seismic=0.15,                  # PGA weight, CAPEX only
    w_cyclone=0.25,                  # Cyclone weight, OPEX only

    pga_ref_g=0.4,                   # PGA normalisation reference [g]
    cyclone_ref_per_yr=0.5,          # Cyclone-track normalisation [/yr]

    cache_dir=None,                  # None = ~/.otex/siting_cache
    refresh=False,                   # True = re-download all layers once
)
```

## Pre-computing the Enriched Site Table

Instead of triggering downloads from inside a regional run, you can build a
flat enriched CSV once and inspect it:

```bash
python scripts/build_siting_layers.py --mpa-buffer 5 --ais-buffer 5
```

Output: `otex/data/CMEMS_points_with_siting.csv` containing the standard
columns plus `in_mpa_strict`, `ais_density_pct`, `pga_475`, and
`cyclone_freq_per_yr`. Subset with `--layers wdpa pga` to skip the heavier
ones.

## Caveats

- **Coverage gaps**: the World Bank vessel-density raster is global but
  AIS-only — small fishing boats with no AIS transponder are invisible. If
  artisanal fishing matters in your study area, add a custom layer.
- **GEM licence**: the seismic raster is **CC-BY-NC-SA 4.0**. If you publish
  derivative cost figures commercially, swap in another seismic source or
  contact GEM for a commercial licence.
- **WDPA snapshot**: pinned to April 2026. Override `OTEX_WDPA_URL` if you
  need a more recent or differently versioned dataset.
- **Cache size**: the full set of layers occupies ~5 GB on disk. Set
  `cache_dir` if `~/.otex/` is space-constrained.
