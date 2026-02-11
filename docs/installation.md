# Installation Guide

This guide covers all installation options for OTEX, including optional dependencies and data access configuration.

## Table of Contents

- [Requirements](#requirements)
- [Basic Installation](#basic-installation)
- [Optional Dependencies](#optional-dependencies)
- [Development Installation](#development-installation)
- [CMEMS Data Access](#cmems-data-access)
- [Verifying Installation](#verifying-installation)
- [Troubleshooting](#troubleshooting)

## Requirements

### System Requirements

- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 4 GB RAM minimum (8 GB recommended for large analyses)
- **Disk Space**: 500 MB for installation, additional space for data downloads

### Python Dependencies

Core dependencies (installed automatically):

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.20 | Numerical computing |
| pandas | ≥1.3 | Data manipulation |
| scipy | ≥1.7 | Scientific computing |
| matplotlib | ≥3.4 | Visualization |
| xarray | ≥0.19 | N-dimensional arrays |
| netCDF4 | ≥1.5 | NetCDF file support |
| tables | ≥3.6 | HDF5 file support |
| tqdm | ≥4.60 | Progress bars |

## Basic Installation

### From PyPI (Recommended)

```bash
pip install otex
```

### From GitHub

```bash
pip install git+https://github.com/msotocalvo/OTEX.git
```

## Optional Dependencies

### CoolProp (Recommended)

CoolProp provides high-accuracy thermodynamic properties for multiple working fluids. Without CoolProp, OTEX uses polynomial correlations for ammonia only.

```bash
pip install otex[coolprop]
```

Or install separately:

```bash
pip install CoolProp>=6.4
```

**Working fluids requiring CoolProp:**
- R134a
- R245fa
- Propane
- Isobutane

### SALib (Uncertainty Analysis)

SALib is required for Sobol sensitivity analysis:

```bash
pip install otex[uncertainty]
```

Or install separately:

```bash
pip install SALib>=1.4.0
```

### All Optional Dependencies

```bash
pip install otex[all]
```

This includes:
- CoolProp
- SALib
- pytest and pytest-cov (for testing)

## Development Installation

For contributing to OTEX or modifying the source code:

```bash
# Clone the repository
git clone https://github.com/msotocalvo/OTEX.git
cd OTEX

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install in development mode with all dependencies
pip install -e ".[dev,all]"

# Run tests to verify installation
pytest tests/ -v
```

## CMEMS Data Access

OTEX uses oceanographic data from Copernicus Marine Environment Monitoring Service (CMEMS). To download data, you need to configure access credentials.

### Step 1: Create Account

1. Go to [Copernicus Marine](https://data.marine.copernicus.eu/)
2. Click "Register" and create an account
3. Verify your email address

### Step 2: Configure Credentials

**Option A: Using copernicusmarine CLI (Recommended)**

```bash
# Install the CLI tool
pip install copernicusmarine

# Login (stores credentials securely)
copernicusmarine login
```

Follow the prompts to enter your username and password.

**Option B: Environment Variables**

```bash
export COPERNICUSMARINE_SERVICE_USERNAME="your_username"
export COPERNICUSMARINE_SERVICE_PASSWORD="your_password"
```

Add these to your `~/.bashrc` or `~/.zshrc` for persistence.

**Option C: Configuration File**

Create `~/.copernicusmarine/credentials` with:

```
username: your_username
password: your_password
```

### Step 3: Verify Access

```python
from otex.data.cmems import download_data
# If credentials are configured correctly, this won't raise an error
```

### Data Storage

Downloaded data is cached locally in the `Data_Results/` directory:

```
Data_Results/
├── Jamaica/
│   ├── Jamaica_2020_50.0_MW_low_cost/
│   │   ├── T_22.0m_2020_Jamaica.h5      # Warm water temperatures
│   │   ├── T_1062.0m_2020_Jamaica.h5    # Cold water temperatures
│   │   └── OTEC_sites_Jamaica_*.csv     # Results
│   └── raw_netcdf/                       # Original downloads
├── Philippines/
└── ...
```

## Verifying Installation

### Basic Verification

```python
# Test import
import otex
print(f"OTEX version: {otex.__version__}")

# Test configuration
from otex.config import parameters_and_constants
inputs = parameters_and_constants()
print(f"Default cycle: {inputs['cycle_type']}")
print(f"Default fluid: {inputs['fluid_type']}")
```

### CoolProp Verification

```python
from otex.core.fluids import get_working_fluid

# This will use CoolProp if available
fluid = get_working_fluid('ammonia', use_coolprop=True)
print(f"Fluid type: {type(fluid).__name__}")
```

If CoolProp is not installed, you'll see a warning and polynomial correlations will be used.

### Uncertainty Module Verification

```python
from otex.analysis import MonteCarloAnalysis, UncertaintyConfig

config = UncertaintyConfig(n_samples=10, parallel=False)
mc = MonteCarloAnalysis(T_WW=28.0, T_CW=5.0, config=config)
results = mc.run(show_progress=False)
print(f"LCOE samples: {len(results.lcoe)}")
```

### Full Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=otex --cov-report=html

# Skip slow tests
pytest tests/ -v -m "not slow"
```

## Troubleshooting

### Common Issues

#### CoolProp Installation Fails

On some systems, CoolProp requires compilation. Try:

```bash
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# macOS
xcode-select --install

# Then retry
pip install CoolProp
```

#### HDF5/netCDF4 Issues

```bash
# Ubuntu/Debian
sudo apt-get install libhdf5-dev libnetcdf-dev

# macOS
brew install hdf5 netcdf

# Then reinstall
pip install --force-reinstall h5py netCDF4
```

#### CMEMS Download Errors

1. Verify credentials:
   ```bash
   copernicusmarine login --check
   ```

2. Check internet connection and firewall settings

3. Verify data product availability:
   ```bash
   copernicusmarine describe --contains GLOBAL_MULTIYEAR_PHY
   ```

#### Memory Errors

For large analyses, increase available memory or reduce sample size:

```python
config = UncertaintyConfig(n_samples=500)  # Reduce from 1000
```

### Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/msotocalvo/OTEX/issues)
- **Discussions**: [Ask questions](https://github.com/msotocalvo/OTEX/discussions)

## Next Steps

- [Quick Start Tutorial](tutorials/quickstart.md)
- [Regional Analysis Guide](tutorials/regional_analysis.md)
- [Uncertainty Analysis Guide](tutorials/uncertainty_analysis.md)
