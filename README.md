![logo](/img/logo.png)
# OTEX - Ocean Thermal Energy eXchange
[![CI](https://github.com/msotocalvo/OTEX/actions/workflows/workflow.yml/badge.svg)](https://github.com/msotocalvo/OTEX/actions/workflows/workflow.yml)

OTEX is a Python library for Ocean Thermal Energy Conversion (OTEC) plant design, simulation, and techno-economic analysis.

## Features

- **Multiple thermodynamic cycles**: Rankine (closed, open, hybrid), Kalina, Uehara
- **Working fluids**: Ammonia, R134a, R245fa, Propane, Isobutane (with CoolProp)
- **Global analysis**: Integration with CMEMS oceanographic data
- **Economic modeling**: LCOE calculation with configurable cost levels
- **Regional analysis**: Site-specific OTEC potential assessment

## Installation

```bash
pip install otex
```

For high-accuracy fluid properties with CoolProp:

```bash
pip install otex[coolprop]
```

## Quick Start

```python
from otex.config import parameters_and_constants

# Get default configuration for a 136 MW OTEC plant
inputs = parameters_and_constants(
    p_gross=-136000,      # kW (negative = power output)
    cost_level='low_cost',
    cycle_type='rankine_closed',
    fluid_type='ammonia',
    year=2020
)

print(f"Cycle: {inputs['cycle_type']}")
print(f"Working fluid: {inputs['working_fluid']}")
print(f"Year: {inputs['year']}")
```

## Regional Analysis

```bash
# Analyze a specific region
python scripts/regional_analysis.py Philippines --year 2021 --cycle kalina

# Batch analysis for multiple regions
python scripts/regional_batch.py --regions Philippines Jamaica Hawaii
```

## Configuration Options

| Parameter | Values | Default |
|-----------|--------|---------|
| `cycle_type` | `rankine_closed`, `rankine_open`, `rankine_hybrid`, `kalina`, `uehara` | `rankine_closed` |
| `fluid_type` | `ammonia`, `r134a`, `r245fa`, `propane`, `isobutane` | `ammonia` |
| `cost_level` | `low_cost`, `high_cost` | `low_cost` |
| `year` | 1993-2023 | 2020 |

## Requirements

- Python >= 3.9
- NumPy, Pandas, SciPy, Matplotlib
- xarray, netCDF4 (for oceanographic data)
- CoolProp (optional, for additional working fluids)

## Aknowledgments
OTEX is built uppon the basis of a repository called pyOTEC (https://github.com/JKALanger/pyOTEC), read more about it here:
Langer, J., Blok, K. The global techno-economic potential of floating, closed-cycle ocean thermal energy conversion. J. Ocean Eng. Mar. Energy (2023). https://doi.org/10.1007/s40722-023-00301-1

## Citation

If you use OTEX in your research, please cite:

## License

MIT License

<!-- trigger workflow -->
