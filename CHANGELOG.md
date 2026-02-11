# Changelog

All notable changes to OTEX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Uncertainty analysis module (`otex.analysis`)
  - Monte Carlo analysis with Latin Hypercube Sampling
  - Sobol global sensitivity analysis (requires SALib)
  - Tornado diagram analysis
  - Visualization functions for all analysis types
- CLI script for uncertainty analysis (`scripts/uncertainty_analysis.py`)
- Comprehensive documentation
  - Installation guide with CMEMS setup
  - Quick start tutorial
  - Regional analysis tutorial
  - Uncertainty analysis tutorial
  - API reference structure
- CONTRIBUTING.md with development guidelines
- This CHANGELOG.md

### Changed
- Enhanced README.md with badges, better structure, and examples

## [0.1.0] - 2024-01-30

### Added
- Initial release of OTEX
- Core thermodynamic cycle models
  - Rankine closed cycle
  - Rankine open cycle
  - Rankine hybrid cycle
  - Kalina cycle
  - Uehara cycle
- Working fluid support
  - Ammonia (polynomial and CoolProp)
  - R134a, R245fa, Propane, Isobutane (CoolProp)
- Plant sizing module
  - Component sizing (turbine, heat exchangers, pumps)
  - Seawater pipe design
  - Off-design performance analysis
- Economic analysis
  - CAPEX calculation by component
  - OPEX estimation
  - LCOE calculation
  - Onshore vs offshore cost models
- Data integration
  - CMEMS oceanographic data download
  - NetCDF processing
  - Multi-depth temperature profiles
- Configuration management
  - Centralized configuration with dataclasses
  - Legacy compatibility layer
- Regional analysis script
- Global analysis script
- Test suite with pytest

### Based On
- Original pyOTEC methodology by Langer et al. (2023)

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2024-01-30 | Initial release |

---

## Upgrade Guide

### Upgrading to 0.2.0 (when released)

The uncertainty analysis module is fully backwards compatible. No breaking changes.

To use the new features:

```python
# New imports
from otex.analysis import (
    MonteCarloAnalysis,
    UncertaintyConfig,
    TornadoAnalysis,
    SobolAnalysis,
    plot_histogram,
    plot_tornado
)
```

Install optional dependency for Sobol analysis:
```bash
pip install SALib>=1.4.0
```

---

[Unreleased]: https://github.com/msotocalvo/OTEX/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/msotocalvo/OTEX/releases/tag/v0.1.0
