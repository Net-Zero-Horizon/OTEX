<p align="center">
  <img src="img/logo.png" alt="OTEX Logo" width="400"/>
</p>

<h1 align="center">OTEX - Ocean Thermal Energy eXchange</h1>

<p align="center">
  <strong>A Python library for OTEC plant design, simulation, and techno-economic analysis</strong>
</p>

<p align="center">
  <a href="https://github.com/msotocalvo/OTEX/actions/workflows/workflow.yml">
    <img src="https://github.com/msotocalvo/OTEX/actions/workflows/workflow.yml/badge.svg" alt="CI">
  </a>
  <a href="https://codecov.io/gh/msotocalvo/OTEX">
    <img src="https://codecov.io/gh/msotocalvo/OTEX/branch/main/graph/badge.svg" alt="codecov">
  </a>
  <a href="https://otex.readthedocs.io">
    <img src="https://readthedocs.org/projects/otex/badge/?version=latest" alt="Documentation">
  </a>
  <a href="https://doi.org/10.5281/zenodo.18428742">
    <img src="https://zenodo.org/badge/1145581288.svg" alt="DOI">
  </a>
</p>

<p align="center">
  <a href="https://pypi.org/project/otex/">
    <img src="https://img.shields.io/pypi/v/otex.svg" alt="PyPI">
  </a>
  <a href="https://pypi.org/project/otex/">
    <img src="https://img.shields.io/pypi/pyversions/otex.svg" alt="Python">
  </a>
  <a href="https://pepy.tech/project/otex">
    <img src="https://static.pepy.tech/badge/otex" alt="Downloads">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

**OTEX** (Ocean Thermal Energy eXchange) is a Python library for designing, simulating, and analyzing Ocean Thermal Energy Conversion (OTEC) power plants. It integrates with global oceanographic databases to enable site-specific techno-economic assessments anywhere in the tropical oceans.

OTEX enables researchers and engineers to:

- **Design OTEC plants** with multiple thermodynamic cycles and working fluids
- **Analyze regional and global potential** using CMEMS oceanographic data
- **Perform uncertainty analysis** with Monte Carlo simulations and sensitivity studies
- **Compare scenarios** across different locations, plant sizes, and configurations

## Features

### Thermodynamic Cycles
| Cycle | Description | Status |
|-------|-------------|--------|
| Rankine Closed | Ammonia/organic fluid closed loop | ✅ Stable |
| Rankine Open | Flash evaporation of seawater | ✅ Stable |
| Rankine Hybrid | Combined closed/open cycle | ✅ Stable |
| Kalina | Ammonia-water mixture | ✅ Stable |
| Uehara | Advanced ammonia-water cycle | ✅ Stable |

### Working Fluids
- **Ammonia** (NH₃) - Default, polynomial or CoolProp
- **R134a** - Requires CoolProp
- **R245fa** - Requires CoolProp
- **Propane** - Requires CoolProp
- **Isobutane** - Requires CoolProp

### Analysis Capabilities
- **Regional Analysis**: Site-specific LCOE maps and power profiles
- **Uncertainty Analysis**: Monte Carlo with Latin Hypercube Sampling
- **Sensitivity Analysis**: Sobol indices and Tornado diagrams
- **Off-design Performance**: Time-resolved power output profiles

## Installation

### Basic Installation

```bash
pip install otex
```

### With Optional Dependencies

```bash
# High-accuracy fluid properties
pip install otex[coolprop]

# Uncertainty analysis (Sobol indices)
pip install otex[uncertainty]

# All optional dependencies
pip install otex[all]
```

### Development Installation

```bash
git clone https://github.com/msotocalvo/OTEX.git
cd OTEX
pip install -e ".[dev]"
```

### CMEMS Data Access

For downloading oceanographic data, you need Copernicus Marine credentials:

1. Create account at [Copernicus Marine](https://data.marine.copernicus.eu/)
2. Configure credentials:
   ```bash
   copernicusmarine login
   ```

See [Installation Guide](docs/installation.md) for detailed instructions.

## Quick Start

### Basic Plant Configuration

```python
from otex.config import parameters_and_constants

# Configure a 100 MW OTEC plant
inputs = parameters_and_constants(
    p_gross=-100000,           # 100 MW (negative = power output)
    cost_level='low_cost',
    cycle_type='rankine_closed',
    fluid_type='ammonia',
    year=2020
)

print(f"Cycle: {inputs['cycle_type']}")
print(f"Discount rate: {inputs['discount_rate']:.1%}")
print(f"Plant lifetime: {inputs['lifetime']} years")
```

### Regional Analysis

```bash
# Analyze Cuba for 2020 with a 50 MW plant
python scripts/regional_analysis.py Cuba --year 2020 --power -50000

# Analyze with Kalina cycle
python scripts/regional_analysis.py Philippines --cycle kalina --year 2021
```

### Uncertainty Analysis

```python
from otex.analysis import (
    MonteCarloAnalysis,
    UncertaintyConfig,
    TornadoAnalysis,
    plot_histogram,
    plot_tornado
)

# Monte Carlo analysis
config = UncertaintyConfig(n_samples=1000, seed=42)
mc = MonteCarloAnalysis(T_WW=28.0, T_CW=5.0, config=config)
results = mc.run()

# Get statistics
stats = results.compute_statistics()
print(f"LCOE: {stats['lcoe']['lcoe_mean']:.2f} ± {stats['lcoe']['lcoe_std']:.2f} ct/kWh")
print(f"90% CI: [{stats['lcoe']['lcoe_p5']:.2f}, {stats['lcoe']['lcoe_p95']:.2f}]")

# Tornado diagram
tornado = TornadoAnalysis(T_WW=28.0, T_CW=5.0)
tornado_results = tornado.run()
plot_tornado(tornado_results)
```

### Command Line Interface

```bash
# Tornado analysis
python scripts/uncertainty_analysis.py --T_WW 28 --T_CW 5 --method tornado

# Monte Carlo with 500 samples
python scripts/uncertainty_analysis.py --T_WW 28 --T_CW 5 --method monte-carlo --samples 500

# Full analysis with plots
python scripts/uncertainty_analysis.py --T_WW 28 --T_CW 5 --method all --samples 200 --save-plots
```

## Documentation

| Document | Description |
|----------|-------------|
| [Installation Guide](docs/installation.md) | Detailed setup instructions |
| [Quick Start Tutorial](docs/tutorials/quickstart.md) | Get started in 10 minutes |
| [Regional Analysis](docs/tutorials/regional_analysis.md) | Analyze specific regions |
| [Uncertainty Analysis](docs/tutorials/uncertainty_analysis.md) | Monte Carlo and sensitivity |
| [API Reference](docs/api/README.md) | Complete API documentation |
| [01 - Quick Start](docs/examples/01_quickstart.ipynb) | Basic plant sizing and cost analysis |
| [02 - Regional Analysis](docs/examples/02_regional_analysis.ipynb) | Analyze OTEC potential for a region |
| [03 - Uncertainty Analysis](docs/examples/03_uncertainty_analysis.ipynb) | Monte Carlo, Tornado, Sobol |

## Project Structure

```
OTEX/
├── otex/                    # Main package
│   ├── core/               # Thermodynamic cycles and fluids
│   ├── plant/              # Plant sizing and operation
│   ├── economics/          # Cost models and LCOE
│   ├── analysis/           # Uncertainty and sensitivity
│   ├── data/               # Data loading (CMEMS, NetCDF)
│   └── config.py           # Configuration management
├── scripts/                 # CLI scripts
│   ├── regional_analysis.py
│   ├── global_analysis.py
│   └── uncertainty_analysis.py
├── tests/                   # Test suite
├── docs/                    # Documentation
└── data/                    # Reference data files
```

## Configuration Options

| Parameter | Options | Default |
|-----------|---------|---------|
| `cycle_type` | `rankine_closed`, `rankine_open`, `rankine_hybrid`, `kalina`, `uehara` | `rankine_closed` |
| `fluid_type` | `ammonia`, `r134a`, `r245fa`, `propane`, `isobutane` | `ammonia` |
| `cost_level` | `'low_cost'`, `'high_cost'`, or a `CostScheme` object | `'low_cost'` |
| `p_gross` | Any negative value (kW) | `-136000` |
| `year` | 1993-2023 | `2020` |

### Custom Cost Schemes

Beyond the two built-in scenarios you can define your own cost parameters with `CostScheme` and Python's standard `dataclasses.replace()`:

```python
from otex.economics import CostScheme, LOW_COST
from dataclasses import replace

# Modify specific parameters of an existing scheme
my_scheme = replace(LOW_COST, turbine_coeff=400, opex_fraction=0.04)

# Use it everywhere cost_level is accepted
inputs = parameters_and_constants(p_gross=-100000, cost_level=my_scheme)
costs, capex, opex, lcoe = capex_opex_lcoe(plant, inputs, my_scheme)
```

All existing code that uses `cost_level='low_cost'` or `cost_level='high_cost'` continues to work unchanged.

## Requirements

- Python >= 3.9
- NumPy, Pandas, SciPy, Matplotlib
- xarray, netCDF4 (oceanographic data)
- tqdm (progress bars)

**Optional:**
- CoolProp (additional working fluids)
- SALib (Sobol sensitivity analysis)

## Acknowledgments

OTEX builds upon [pyOTEC](https://github.com/JKALanger/pyOTEC) by Langer et al. For the original methodology, see:

> Langer, J., Blok, K. *The global techno-economic potential of floating, closed-cycle ocean thermal energy conversion.* J. Ocean Eng. Mar. Energy (2023). https://doi.org/10.1007/s40722-023-00301-1

## Citation

If you use OTEX in your research, please cite:

```bibtex
@software{otex2024,
  author = {Soto-Calvo, Manuel and OTEX Development Team},
  title = {OTEX: Ocean Thermal Energy eXchange},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/msotocalvo/OTEX},
  doi = {10.5281/zenodo.18428742}
}
```
## Studies usign OTEX:
- Soto Calvo M, and Lee HS., 2025. Ocean Thermal Energy Conversion (OTEC) Potential in Central American and Caribbean Regions: A Multicriteria Analysis for Optimal Sites. Applied Energy. 394: 126182. https://doi.org/10.1016/j.apenergy.2025.126182

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for ocean energy research
</p>

