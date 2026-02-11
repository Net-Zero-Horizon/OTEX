"""
Create NetCDF file from existing CSV results
"""
import pandas as pd
import xarray as xr
from datetime import datetime
import os

# Read the CSV file
csv_file = './Global_OTEC_CMEMS/Rankine_Closed_Ammonia_low_cost.csv'
df = pd.read_csv(csv_file)

print(f"Loaded {len(df)} regions from CSV")
print(f"Columns: {list(df.columns)}")

# Build dataset with only columns that exist
data_vars = {}

# Required columns
required_cols = ['p_net_avg_MW', 'p_net_max_MW', 'p_net_min_MW',
                'lcoe_avg', 'lcoe_min', 'T_WW_avg', 'T_CW_avg', 'deltaT_avg']

for col in required_cols:
    if col in df.columns:
        data_vars[col] = (['region'], df[col].values)
        print(f"  Added: {col}")

# Optional columns
optional_cols = ['n_sites']
for col in optional_cols:
    if col in df.columns:
        data_vars[col] = (['region'], df[col].values)
        print(f"  Added: {col}")

# Create xarray Dataset
ds = xr.Dataset(
    data_vars,
    coords={
        'region': df['region'].values,
    },
    attrs={
        'title': 'Global OTEC Analysis - Rankine_Closed_Ammonia',
        'configuration': 'Rankine_Closed_Ammonia',
        'cycle': 'Rankine_Closed',
        'fluid': 'Ammonia',
        'cost_level': 'low_cost',
        'year': 2020,
        'creation_date': datetime.now().isoformat(),
        'description': 'OTEC feasibility analysis using CMEMS oceanographic data',
        'data_source': 'CMEMS Global Ocean Physics Analysis',
    }
)

# Save NetCDF
nc_filename = './Global_OTEC_CMEMS/Rankine_Closed_Ammonia_low_cost.nc'
ds.to_netcdf(nc_filename)
print(f"\n✓ Successfully created: {nc_filename}")

# Print dataset info
print(f"\nDataset dimensions: {dict(ds.dims)}")
print(f"Data variables: {list(ds.data_vars)}")
