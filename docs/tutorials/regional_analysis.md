# Regional Analysis Tutorial

Learn how to analyze OTEC potential for specific geographic regions using real oceanographic data from CMEMS.

## Prerequisites

- OTEX installed (`pip install otex`)
- CMEMS credentials configured (see [Installation Guide](../installation.md#cmems-data-access))
- Internet connection for data download

## Overview

Regional analysis in OTEX:

1. Downloads temperature profiles from CMEMS for your region
2. Identifies feasible OTEC sites (adequate water depth)
3. Sizes plants for each site based on local conditions
4. Calculates LCOE considering distance to shore
5. Generates time-resolved power profiles

## Available Regions

OTEX includes a bundled database of pre-defined regions covering tropical areas worldwide:

```python
from otex.data import load_regions

regions = load_regions()
print(regions.head(20))
```

Popular regions include:
- Caribbean: Jamaica, Cuba, Dominican Republic, Puerto Rico, Bahamas
- Pacific: Hawaii, Philippines, Fiji, Guam, Samoa
- Indian Ocean: Mauritius, Maldives, Seychelles, Reunion
- Southeast Asia: Indonesia, Malaysia, Vietnam
- Africa: Kenya, Tanzania, Mozambique

## Basic Usage

### Command Line

After installing OTEX via pip, the `otex-regional` command is available:

```bash
# Analyze Jamaica with default settings (136 MW, low_cost, 2020)
otex-regional Jamaica

# Specify plant size and year
otex-regional Jamaica --power -50000 --year 2021

# Use different cycle and cost assumptions
otex-regional Philippines --cycle kalina --cost high_cost
```

### Python API

```python
from otex.regional import run_regional_analysis

# Run analysis
otec_plants, sites_df = run_regional_analysis(
    studied_region='Jamaica',
    p_gross=-50000,          # 50 MW
    cost_level='low_cost',
    year=2020,
    cycle_type='rankine_closed',
    fluid_type='ammonia',
    use_coolprop=True
)
```

## Step-by-Step Guide

### Step 1: Choose Your Region

First, verify your region exists in the database:

```python
from otex.data import load_regions

regions = load_regions()
print(regions[regions['region'].str.contains('Jam', case=False)])
```

Output:
```
         region   north    east  south    west    demand
Jamaica  19.358 -74.009 14.083 -80.833  3.092992
```

### Step 2: Check Available Sites

View potential OTEC sites in your region:

```python
from otex.data import load_sites

sites = load_sites()
jamaica_sites = sites[sites['region'] == 'Jamaica']

print(f"Total sites in Jamaica: {len(jamaica_sites)}")
print(f"Water depth range: {jamaica_sites['water_depth'].min():.0f} to {jamaica_sites['water_depth'].max():.0f} m")
print(f"Distance to shore: {jamaica_sites.iloc[:, 4].min():.1f} to {jamaica_sites.iloc[:, 4].max():.1f} km")
```

### Step 3: Run the Analysis

```python
from otex.regional import run_regional_analysis

otec_plants, sites_df = run_regional_analysis(
    studied_region='Jamaica',
    p_gross=-50000,
    year=2020,
)
```

This will:
1. Download CMEMS temperature data (~5-15 minutes first time)
2. Process and cache data locally
3. Run OTEC sizing for all valid sites
4. Calculate LCOE for each site
5. Save results to `Data_Results/Jamaica/`

### Step 4: Examine Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
results = pd.read_csv(
    'Data_Results/Jamaica/Jamaica_2020_50.0_MW_low_cost/OTEC_sites_Jamaica_2020_50.0_MW_low_cost.csv',
    sep=';',
    index_col='id'
)

print(results.head())
print(f"\nNumber of feasible sites: {len(results)}")
print(f"LCOE range: {results['LCOE'].min():.2f} - {results['LCOE'].max():.2f} ct/kWh")
print(f"Best site LCOE: {results['LCOE'].min():.2f} ct/kWh")
```

### Step 5: Visualize Results

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# LCOE map
ax1 = axes[0, 0]
scatter = ax1.scatter(
    results['longitude'],
    results['latitude'],
    c=results['LCOE'],
    cmap='RdYlGn_r',
    s=50
)
plt.colorbar(scatter, ax=ax1, label='LCOE (ct/kWh)')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.set_title('LCOE by Location')

# LCOE histogram
ax2 = axes[0, 1]
ax2.hist(results['LCOE'], bins=20, edgecolor='white')
ax2.axvline(results['LCOE'].median(), color='red', linestyle='--', label='Median')
ax2.set_xlabel('LCOE (ct/kWh)')
ax2.set_ylabel('Number of sites')
ax2.set_title('LCOE Distribution')
ax2.legend()

# Net power vs LCOE
ax3 = axes[1, 0]
ax3.scatter(results['p_net_nom'], results['LCOE'], alpha=0.6)
ax3.set_xlabel('Net Power (MW)')
ax3.set_ylabel('LCOE (ct/kWh)')
ax3.set_title('Net Power vs LCOE')

# Temperature difference
ax4 = axes[1, 1]
delta_T = results['T_WW_med'] - results['T_CW_med']
ax4.scatter(delta_T, results['LCOE'], alpha=0.6)
ax4.set_xlabel('Temperature Difference (°C)')
ax4.set_ylabel('LCOE (ct/kWh)')
ax4.set_title('ΔT vs LCOE')

plt.tight_layout()
plt.savefig('jamaica_analysis.png', dpi=150)
plt.show()
```

## Understanding the Output Files

### OTEC Sites CSV

| Column | Description | Unit |
|--------|-------------|------|
| id | Site identifier | - |
| longitude | Site longitude | degrees |
| latitude | Site latitude | degrees |
| p_net_nom | Nominal net power | MW |
| AEP | Annual Energy Production | GWh |
| CAPEX | Capital expenditure | $M |
| LCOE | Levelized cost of energy | ct/kWh |
| Configuration | Optimal ΔT configuration | - |
| T_WW_min/med/max | Warm water temperature stats | °C |
| T_CW_min/med/max | Cold water temperature stats | °C |

### Power Profiles CSV

Daily average net power output over the year:

```python
profiles = pd.read_csv(
    'Data_Results/Jamaica/.../net_power_profiles_per_day_Jamaica_2020_50.0_MW_low_cost.csv',
    sep=';',
    index_col=0,
    parse_dates=True
)

# Plot annual profile
profiles.plot(figsize=(12, 4))
plt.ylabel('Net Power (kW)')
plt.title('Average Daily Net Power Output')
plt.show()
```

## Advanced Options

### Custom Plant Size

```python
from otex.regional import run_regional_analysis

# Analyze different plant sizes
for size_mw in [20, 50, 100, 200]:
    run_regional_analysis(
        studied_region='Jamaica',
        p_gross=-size_mw * 1000,
        year=2020
    )
```

### Different Thermodynamic Cycles

```python
from otex.regional import run_regional_analysis

# Compare cycles
cycles = ['rankine_closed', 'kalina', 'uehara']
for cycle in cycles:
    run_regional_analysis(
        studied_region='Jamaica',
        p_gross=-50000,
        cycle_type=cycle
    )
```

### Multi-Year Analysis

```bash
# Analyze multiple years via CLI
for year in 2018 2019 2020 2021; do
    otex-regional Jamaica --year $year
done
```

## Combining with Uncertainty Analysis

After regional analysis, run uncertainty analysis on the best site:

```python
import pandas as pd
from otex.analysis import MonteCarloAnalysis, UncertaintyConfig

# Load regional results
results = pd.read_csv('...OTEC_sites_Jamaica_2020_50.0_MW_low_cost.csv', sep=';')

# Find best site
best_site = results.loc[results['LCOE'].idxmin()]
print(f"Best site: ({best_site['longitude']}, {best_site['latitude']})")
print(f"T_WW: {best_site['T_WW_med']:.1f}°C, T_CW: {best_site['T_CW_med']:.1f}°C")

# Run uncertainty analysis for this site
config = UncertaintyConfig(n_samples=500, seed=42)
mc = MonteCarloAnalysis(
    T_WW=best_site['T_WW_med'],
    T_CW=best_site['T_CW_med'],
    config=config,
    p_gross=-50000
)
ua_results = mc.run()

stats = ua_results.compute_statistics()
print(f"\nLCOE with uncertainty:")
print(f"Mean: {stats['lcoe']['lcoe_mean']:.2f} ct/kWh")
print(f"90% CI: [{stats['lcoe']['lcoe_p5']:.2f}, {stats['lcoe']['lcoe_p95']:.2f}]")
```

## Performance Tips

1. **First run is slower**: Data download and processing takes 5-15 minutes
2. **Subsequent runs are faster**: Processed data is cached in HDF5 files
3. **Reduce memory usage**: Use smaller regions or reduce spatial resolution
4. **Parallel processing**: Enabled by default for Monte Carlo

## Troubleshooting

### "No valid sites found"

Check that your region has sufficient water depth:
```python
from otex.data import load_sites

sites = load_sites()
region_sites = sites[sites['region'] == 'YourRegion']
print(f"Depths: {region_sites['water_depth'].describe()}")
```
Sites need water depth of at least 600-1000m.

### Download failures

1. Verify CMEMS credentials: `copernicusmarine login --check`
2. Check internet connection
3. Try again later (CMEMS servers may be busy)

### Memory errors

Reduce plant size or use a smaller region.

## Next Steps

- [Uncertainty Analysis Tutorial](uncertainty_analysis.md)
- [API Reference](../api/README.md)
