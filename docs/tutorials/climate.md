# Climate Scenario Analysis

Since 0.3.0, OTEX can project an OTEC site's performance under future
climate conditions using **CMIP6** ocean temperature scenarios. The
implementation follows the IPCC-standard *delta-method* downscaling:
historical CMEMS reanalysis is used as the high-resolution baseline,
and a multi-model ensemble of CMIP6 GCMs supplies the time-mean
warming (Δ) at each grid cell, which is then added to the observed
time series before the techno-economic analysis runs.

## Why delta-method?

```
T_future(lon, lat, t) = T_CMEMS_observed(lon, lat, t) + Δ(lon, lat)

           Δ(lon, lat) = ⟨T_GCM_future⟩ − ⟨T_GCM_baseline⟩
```

Two important properties:

1. The **spatial detail** of CMEMS (1/12°) and the **observed
   seasonality** are preserved — only the climate state is shifted.
2. **GCM biases cancel** in the difference, making Δ much more
   robust between models than absolute GCM temperatures.

The same mechanism is applied independently at the OTEC warm-water
intake depth (~20 m) and cold-water intake depth (~1062 m). Surface
waters typically warm faster than the deep ocean, so under climate
change the **ΔT thermal gradient widens**, which generally *reduces*
LCOE for OTEC.

## Available scenarios

| Scenario | Description |
|---|---|
| `historical` *(default)* | No delta applied — identical to the standard CMEMS pipeline. |
| `ssp126` | Strong mitigation, ~1.5 °C global warming target. |
| `ssp245` | Middle-of-the-road emissions. |
| `ssp370` | High emissions, no mitigation. |
| `ssp585` | Worst-case fossil-fuel-intensive trajectory. |

## Default ensemble

Three CMIP6 GCMs spanning the range of equilibrium climate sensitivity:

| Model | Institution | ECS (°C) |
|---|---|---|
| `MPI-ESM1-2-LR` | Max Planck Institute | ~3.0 |
| `EC-Earth3` | EC-Earth Consortium | ~4.3 |
| `CanESM5` | CCCma | ~5.6 |

Override via `--climate-models` (CLI) or `climate.models=(...)` (Python).

## Quick start

```bash
# Project Jamaica's 2020-2023 OTEC performance under SSP2-4.5 by 2050.
otex-regional Jamaica \
    --year-start 2020 --year-end 2023 \
    --climate-scenario ssp245 --climate-year 2050
```

```python
from otex.regional import run_regional_analysis

otec_plants, sites = run_regional_analysis(
    studied_region='Jamaica',
    year_start=2020,
    year_end=2023,
    climate_scenario='ssp245',
    climate_year=2050,
)
```

Output files include a `_<scenario>_<target_year>` suffix:

```
Data_Results/Jamaica/Jamaica_2020-2023_ssp245_2050_100.0_MW_low_cost/
├── OTEC_sites_Jamaica_2020-2023_ssp245_2050_100.0_MW_low_cost.csv
├── OTEC_sites_yearly_Jamaica_2020-2023_ssp245_2050_100.0_MW_low_cost.csv
└── net_power_profiles_per_day_Jamaica_2020-2023_ssp245_2050_100.0_MW_low_cost.csv
```

The schema is identical to the baseline run — same `LCOE`, `AEP`,
`AEP_min/p50/max/std` columns — but the values reflect the projected
future ocean state.

## Comparing baseline vs scenario

The simplest workflow is two runs (baseline and future) side by side:

```python
import pandas as pd

base = pd.read_csv("Data_Results/Jamaica/Jamaica_2020-2023_100.0_MW_low_cost/"
                    "OTEC_sites_Jamaica_2020-2023_100.0_MW_low_cost.csv", sep=';')
fut  = pd.read_csv("Data_Results/Jamaica/Jamaica_2020-2023_ssp245_2050_100.0_MW_low_cost/"
                    "OTEC_sites_Jamaica_2020-2023_ssp245_2050_100.0_MW_low_cost.csv", sep=';')

merged = base.merge(fut[['longitude', 'latitude', 'LCOE', 'AEP']],
                    on=['longitude', 'latitude'], suffixes=('_base', '_2050'))
merged['ΔLCOE_pct'] = 100 * (merged['LCOE_2050'] - merged['LCOE_base']) / merged['LCOE_base']
merged['ΔAEP_pct']  = 100 * (merged['AEP_2050']  - merged['AEP_base'])  / merged['AEP_base']
print(merged[['ΔLCOE_pct', 'ΔAEP_pct']].describe())
```

For the default Jamaica example, expect ΔLCOE ≈ −5 % and ΔAEP ≈ +1.8 %
under SSP2-4.5 / 2050.

## Caching

* CMIP6 monthly thetao subsets land in `~/.otex/cache/cmip6/` as
  Parquet files. Each ``(model, scenario, period, depth, bbox)``
  request becomes one ~1-5 MB file.
* The first call per region/scenario downloads ~10-50 MB (3 models ×
  baseline + future window × 2 depths) and takes 5-15 minutes.
* Subsequent calls — including the same scenario at different target
  years that fall within an already-fetched future window — are
  sub-millisecond.

## Reproducibility

OTEX pins specific CMIP6 dataset versions in
``otex/data/climate.py::_DEFAULT_CATALOG`` (e.g. CanESM5 `v20190429`,
EC-Earth3 `v20200918`, MPI-ESM1-2-LR `v20190710`). To use newer
versions or substitute models, override the catalog via the
``OTEX_CMIP6_CATALOG`` environment variable.

## Caveats

* **Delta-method assumes stationarity of GCM biases.** This is the
  standard assumption in regional climate impact studies (IPCC
  AR6 WGII) and is generally valid for ocean temperature deltas at
  the regional scale used here.
* **Models cover 2015-2100.** Target years before 2015 or after 2100
  raise an error from the Pangeo Zarr selection.
* **The future window is 30 years centred on the target year.** A
  request for `target_year=2050` averages 2036-2065 in each model.
  Override via ``ClimateConfig.future_window_years``.
