# OTEX Documentation

**OTEX** (Ocean Thermal Energy eXchange) is a Python library for designing, simulating, and analyzing Ocean Thermal Energy Conversion (OTEC) power plants.

## Features

- **Multiple thermodynamic cycles**: Rankine (closed/open/hybrid), Kalina, Uehara
- **Various working fluids**: Ammonia, R134a, R245fa, Propane, Isobutane
- **Global analysis**: Integration with CMEMS oceanographic data
- **Uncertainty analysis**: Monte Carlo, Sobol sensitivity, Tornado diagrams
- **Techno-economic assessment**: CAPEX, OPEX, and LCOE calculations

## Quick Links

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
tutorials/quickstart
```

```{toctree}
:maxdepth: 2
:caption: User Guide

tutorials/regional_analysis
tutorials/uncertainty_analysis
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/README
```

## Installation

```bash
pip install otex
```

For additional features:

```bash
# High-accuracy fluid properties
pip install otex[coolprop]

# Uncertainty analysis (Sobol indices)
pip install otex[uncertainty]

# All optional dependencies
pip install otex[all]
```

## Basic Usage

```python
from otex.config import parameters_and_constants
from otex.plant.sizing import otec_sizing
from otex.economics.costs import capex_opex_lcoe

# Configure a 100 MW OTEC plant
inputs = parameters_and_constants(
    p_gross=-100000,
    cycle_type='rankine_closed',
    fluid_type='ammonia'
)

# Size the plant for given ocean conditions
T_WW, T_CW = 28.0, 5.0  # Warm and cold water temperatures
plant = otec_sizing([T_WW], [T_CW], 3.0, 3.0, inputs, 'low_cost')

# Calculate costs
costs, capex, opex, lcoe = capex_opex_lcoe(plant, inputs, 'low_cost')
print(f"LCOE: {lcoe[0]:.2f} ct/kWh")
```

## Citation

If you use OTEX in your research, please cite:

```bibtex
@software{otex2024,
  author = {Soto-Calvo, Manuel},
  title = {OTEX: Ocean Thermal Energy eXchange},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/msotocalvo/OTEX},
  doi = {10.5281/zenodo.18428742}
}
```

## License

OTEX is released under the MIT License.
