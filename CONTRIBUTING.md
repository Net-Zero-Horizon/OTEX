# Contributing to OTEX

Thank you for your interest in contributing to OTEX! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

Please be respectful and constructive in all interactions. We're building a welcoming community for ocean energy research.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/OTEX.git
   cd OTEX
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/msotocalvo/OTEX.git
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Types of Contributions

- **Bug fixes**: Fix issues in existing code
- **New features**: Add new functionality
- **Documentation**: Improve docs, tutorials, examples
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Refactoring**: Improve code quality

### Before Starting

1. Check existing [issues](https://github.com/msotocalvo/OTEX/issues) to avoid duplicates
2. For significant changes, open an issue first to discuss
3. For new features, consider the project scope and design

## Development Setup

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install in development mode with all dependencies
pip install -e ".[dev,all]"
```

### Project Structure

```
OTEX/
├── otex/                    # Main package
│   ├── core/               # Thermodynamic cycles and fluids
│   ├── plant/              # Plant sizing and operation
│   ├── economics/          # Cost models
│   ├── analysis/           # Uncertainty and sensitivity
│   ├── data/               # Data loading
│   └── config.py           # Configuration
├── scripts/                 # CLI scripts
├── tests/                   # Test suite
├── docs/                    # Documentation
└── pyproject.toml          # Project configuration
```

## Code Style

### General Guidelines

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable names
- Keep functions focused and reasonably sized
- Add docstrings to all public functions/classes

### Formatting

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for issues
ruff check otex/

# Auto-fix issues
ruff check --fix otex/

# Format code
ruff format otex/
```

### Docstring Style

Use NumPy-style docstrings:

```python
def calculate_lcoe(capex, opex, power, crf):
    """
    Calculate Levelized Cost of Energy.

    Parameters
    ----------
    capex : float
        Total capital expenditure in USD.
    opex : float
        Annual operating expenditure in USD/year.
    power : float
        Net power output in kW.
    crf : float
        Capital recovery factor.

    Returns
    -------
    float
        LCOE in ct/kWh.

    Examples
    --------
    >>> calculate_lcoe(1e8, 3e6, 50000, 0.106)
    24.5
    """
    ...
```

### Type Hints

Add type hints to function signatures:

```python
from typing import Dict, List, Optional, Tuple

def run_analysis(
    temperature: float,
    n_samples: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, float]:
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_uncertainty.py -v

# Run with coverage
pytest tests/ --cov=otex --cov-report=html

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for common setup
- Mark slow tests with `@pytest.mark.slow`

Example:

```python
import pytest
import numpy as np
from otex.analysis import MonteCarloAnalysis, UncertaintyConfig


class TestMonteCarloAnalysis:
    """Tests for MonteCarloAnalysis class."""

    def test_initialization(self):
        """Test that analysis initializes correctly."""
        config = UncertaintyConfig(n_samples=10)
        mc = MonteCarloAnalysis(T_WW=28.0, T_CW=5.0, config=config)

        assert mc.T_WW == 28.0
        assert mc.T_CW == 5.0
        assert mc.config.n_samples == 10

    @pytest.mark.slow
    def test_run_full_analysis(self):
        """Test running complete analysis (slow)."""
        config = UncertaintyConfig(n_samples=100, parallel=False)
        mc = MonteCarloAnalysis(T_WW=28.0, T_CW=5.0, config=config)
        results = mc.run(show_progress=False)

        assert len(results.lcoe) == 100
        assert not np.all(np.isnan(results.lcoe))
```

### Test Coverage

Aim for >80% coverage on new code. Check coverage report:

```bash
pytest tests/ --cov=otex --cov-report=html
open htmlcov/index.html
```

## Pull Request Process

### Before Submitting

1. **Update from upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

3. **Check code style**:
   ```bash
   ruff check otex/
   ```

4. **Update documentation** if needed

### Submitting

1. Push your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a Pull Request on GitHub

3. Fill out the PR template with:
   - Description of changes
   - Related issue (if any)
   - Testing performed
   - Documentation updates

### PR Review

- Maintainers will review your PR
- Address feedback promptly
- Keep PRs focused on single issues/features
- Squash commits if requested

### After Merge

```bash
git checkout main
git pull upstream main
git branch -d feature/your-feature-name
```

## Reporting Issues

### Bug Reports

Include:
- OTEX version (`python -c "import otex; print(otex.__version__)"`)
- Python version (`python --version`)
- Operating system
- Minimal code to reproduce
- Full error traceback
- Expected vs actual behavior

### Feature Requests

Include:
- Use case description
- Proposed solution (if any)
- Alternatives considered

### Security Issues

For security vulnerabilities, please email directly rather than opening a public issue.

## Questions?

- Open a [Discussion](https://github.com/msotocalvo/OTEX/discussions)
- Check existing [Issues](https://github.com/msotocalvo/OTEX/issues)
- Read the [Documentation](docs/)

---

Thank you for contributing to OTEX!
