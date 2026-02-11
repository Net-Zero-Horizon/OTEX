# Uncertainty Analysis Tutorial

Learn how to quantify and analyze uncertainty in OTEC techno-economic assessments using Monte Carlo simulations and sensitivity analysis.

## Why Uncertainty Analysis?

OTEC cost estimates involve significant uncertainty in:

- **Thermodynamic parameters**: Heat transfer coefficients, efficiencies
- **Economic factors**: CAPEX, OPEX, discount rates
- **Operating conditions**: Temperature variations, availability

Uncertainty analysis helps you:
- Quantify confidence intervals for LCOE estimates
- Identify which parameters matter most
- Make robust investment decisions
- Communicate results with appropriate caveats

## Available Methods

| Method | Purpose | Speed | Requires SALib |
|--------|---------|-------|----------------|
| Monte Carlo | Full uncertainty propagation | Slow | No |
| Tornado | Quick sensitivity screening | Fast | No |
| Sobol | Global sensitivity indices | Medium | Yes |

## Monte Carlo Analysis

### Basic Usage

```python
from otex.analysis import MonteCarloAnalysis, UncertaintyConfig

# Configure the analysis
config = UncertaintyConfig(
    n_samples=1000,    # Number of samples (more = better accuracy)
    seed=42,           # Random seed for reproducibility
    parallel=True      # Use multiple CPU cores
)

# Create and run analysis
mc = MonteCarloAnalysis(
    T_WW=28.0,         # Warm water temperature (°C)
    T_CW=5.0,          # Cold water temperature (°C)
    config=config,
    p_gross=-50000,    # 50 MW plant
    cost_level='low_cost'
)

results = mc.run(show_progress=True)
```

### Interpreting Results

```python
# Get comprehensive statistics
stats = results.compute_statistics()

# LCOE statistics
lcoe = stats['lcoe']
print(f"LCOE Statistics")
print(f"{'='*40}")
print(f"Mean:     {lcoe['lcoe_mean']:.2f} ct/kWh")
print(f"Std Dev:  {lcoe['lcoe_std']:.2f} ct/kWh")
print(f"CV:       {lcoe['lcoe_cv']:.1%}")
print(f"Median:   {lcoe['lcoe_median']:.2f} ct/kWh")
print(f"Min:      {lcoe['lcoe_min']:.2f} ct/kWh")
print(f"Max:      {lcoe['lcoe_max']:.2f} ct/kWh")
print(f"P5:       {lcoe['lcoe_p5']:.2f} ct/kWh")
print(f"P95:      {lcoe['lcoe_p95']:.2f} ct/kWh")

# Confidence interval
low, high = results.get_confidence_interval('lcoe', confidence=0.90)
print(f"\n90% Confidence Interval: [{low:.2f}, {high:.2f}] ct/kWh")
```

### Latin Hypercube Sampling

OTEX uses Latin Hypercube Sampling (LHS) for efficient coverage of the parameter space:

```python
# Access the samples
samples = mc.samples
print(f"Sample shape: {samples.shape}")  # (n_samples, n_parameters)

# View parameter names
print(f"Parameters: {config.parameter_names}")
```

LHS ensures better coverage than simple random sampling, especially for small sample sizes.

## Tornado Analysis

Tornado diagrams show which parameters have the largest impact on results:

```python
from otex.analysis import TornadoAnalysis

tornado = TornadoAnalysis(
    T_WW=28.0,
    T_CW=5.0,
    variation_pct=10.0,   # ±10% variation (if not using bounds)
    p_gross=-50000,
    cost_level='low_cost'
)

# Run analysis
tornado_results = tornado.run(
    output='lcoe',        # Analyze LCOE
    use_bounds=True,      # Use parameter bounds, not percentage
    show_progress=True
)

# View results
print(f"Baseline LCOE: {tornado_results.baseline:.2f} ct/kWh")
print(f"\nParameter Rankings (by swing magnitude):")
for i, (name, swing) in enumerate(tornado_results.get_ranking(), 1):
    print(f"{i:2d}. {name:40s} {swing:+.2f} ct/kWh")
```

### Interpreting Tornado Results

- **Swing**: Total change from low to high parameter value
- **Positive swing**: Higher parameter value → higher LCOE
- **Negative swing**: Higher parameter value → lower LCOE
- **Large swings**: Focus on reducing uncertainty in these parameters

## Sobol Sensitivity Analysis

Sobol analysis provides rigorous, global sensitivity indices:

```python
from otex.analysis import SobolAnalysis

sobol = SobolAnalysis(
    T_WW=28.0,
    T_CW=5.0,
    n_samples=512,           # Base samples (total = n*(2d+2))
    calc_second_order=False, # Set True for interaction effects
    p_gross=-50000,
    cost_level='low_cost'
)

# Run analysis (requires SALib)
sobol_results = sobol.run(output='lcoe', show_progress=True)

# View results
print("Sobol Sensitivity Indices")
print("="*60)
print(f"{'Parameter':40s} {'S1':>8s} {'ST':>8s}")
print("-"*60)
for name, st in sobol_results.get_ranking('ST'):
    idx = sobol_results.parameter_names.index(name)
    s1 = sobol_results.S1[idx]
    print(f"{name:40s} {s1:8.3f} {st:8.3f}")
```

### Understanding Sobol Indices

| Index | Name | Interpretation |
|-------|------|----------------|
| S1 | First-order | Direct effect of parameter |
| ST | Total-order | Total effect including interactions |
| ST - S1 | Interactions | Effect through parameter interactions |

- **S1 close to ST**: Parameter acts independently
- **ST >> S1**: Strong interactions with other parameters
- **Sum of S1 ≈ 1**: Additive model (no interactions)
- **Sum of ST > 1**: Significant parameter interactions

## Visualization

### Histogram with Statistics

```python
from otex.analysis import plot_histogram
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
plot_histogram(results, output='lcoe', ax=ax, bins=50)
plt.savefig('lcoe_histogram.png', dpi=150)
plt.show()
```

### Tornado Diagram

```python
from otex.analysis import plot_tornado

fig, ax = plt.subplots(figsize=(12, 8))
plot_tornado(tornado_results, ax=ax, top_n=10)
plt.savefig('tornado_diagram.png', dpi=150)
plt.show()
```

### Sobol Indices Bar Chart

```python
from otex.analysis import plot_sobol_indices

fig, ax = plt.subplots(figsize=(10, 8))
plot_sobol_indices(sobol_results, ax=ax, top_n=10)
plt.savefig('sobol_indices.png', dpi=150)
plt.show()
```

### Summary Figure

```python
from otex.analysis import create_summary_figure

fig = create_summary_figure(
    mc_results=results,
    tornado_results=tornado_results,
    sobol_results=sobol_results,  # Optional
    output='lcoe'
)
fig.savefig('uncertainty_summary.png', dpi=150)
```

### Scatter Matrix

```python
from otex.analysis import plot_scatter_matrix

fig = plot_scatter_matrix(
    results,
    output='lcoe',
    max_params=5  # Show top 5 correlated parameters
)
fig.savefig('scatter_matrix.png', dpi=150)
```

## Customizing Parameters

### Default Uncertain Parameters

```python
from otex.analysis import get_default_parameters

params = get_default_parameters()
for p in params:
    print(f"{p.name}: {p.nominal} ({p.distribution}, {p.bounds})")
```

Default parameters include:
- Turbine isentropic efficiency
- Pump isentropic efficiency
- Heat transfer coefficients (U_evap, U_cond)
- CAPEX factors (turbine, HX, pump, structure)
- OPEX factor
- Discount rate

### Custom Parameters

```python
from otex.analysis import UncertainParameter, UncertaintyConfig

# Define custom parameters
custom_params = [
    UncertainParameter(
        name='discount_rate',
        nominal=0.08,
        distribution='uniform',
        bounds=(0.05, 0.12),
        category='economic'
    ),
    UncertainParameter(
        name='turbine_isentropic_efficiency',
        nominal=0.85,
        distribution='normal',
        bounds=(0.85, 0.03),  # mean, std for normal
        category='efficiency'
    ),
    UncertainParameter(
        name='capex_structure_factor',
        nominal=1.0,
        distribution='triangular',
        bounds=(0.8, 1.8),  # min, max (mode = nominal)
        category='economic'
    ),
]

config = UncertaintyConfig(
    parameters=custom_params,
    n_samples=500,
    seed=42
)
```

### Distribution Types

| Type | bounds meaning | Example |
|------|----------------|---------|
| `uniform` | (min, max) | `(0.8, 1.2)` |
| `normal` | (mean, std) | `(0.82, 0.04)` |
| `triangular` | (min, max), mode=nominal | `(0.7, 1.3)` |

## Command Line Interface

After installing OTEX via pip, the `otex-uncertainty` command is available:

```bash
# Tornado analysis (fast)
otex-uncertainty \
    --T_WW 28 --T_CW 5 \
    --method tornado

# Monte Carlo (comprehensive)
otex-uncertainty \
    --T_WW 28 --T_CW 5 \
    --method monte-carlo \
    --samples 1000

# Sobol analysis (requires SALib: pip install otex[uncertainty])
otex-uncertainty \
    --T_WW 28 --T_CW 5 \
    --method sobol \
    --samples 512

# All methods with saved plots
otex-uncertainty \
    --T_WW 28 --T_CW 5 \
    --method all \
    --samples 500 \
    --save-plots \
    --output-dir ./results/
```

## Best Practices

### Sample Size Guidelines

| Method | Minimum | Recommended | High Accuracy |
|--------|---------|-------------|---------------|
| Monte Carlo | 100 | 1,000 | 10,000 |
| Tornado | 2*n_params | 2*n_params | 2*n_params |
| Sobol | 64 | 512 | 2,048 |

### Convergence Check

```python
# Run with increasing sample sizes
sample_sizes = [100, 500, 1000, 2000]
means = []

for n in sample_sizes:
    config = UncertaintyConfig(n_samples=n, seed=42)
    mc = MonteCarloAnalysis(T_WW=28.0, T_CW=5.0, config=config)
    results = mc.run(show_progress=False)
    means.append(results.compute_statistics()['lcoe']['lcoe_mean'])

# Plot convergence
plt.plot(sample_sizes, means, 'o-')
plt.xlabel('Number of samples')
plt.ylabel('Mean LCOE (ct/kWh)')
plt.title('Convergence Check')
```

### Recommended Workflow

1. **Start with Tornado**: Quick identification of important parameters
2. **Run Monte Carlo**: Full uncertainty propagation (1000+ samples)
3. **Optional Sobol**: If you need rigorous sensitivity indices
4. **Visualize and report**: Create summary figures

## Exporting Results

```python
import pandas as pd

# Export Monte Carlo samples
df = pd.DataFrame(results.samples, columns=results.parameter_names)
df['lcoe'] = results.lcoe
df['net_power'] = results.net_power
df['capex'] = results.capex
df.to_csv('monte_carlo_results.csv', index=False)

# Export statistics
stats = results.compute_statistics()
stats_df = pd.DataFrame(stats).T
stats_df.to_csv('uncertainty_statistics.csv')

# Export tornado results
tornado_df = pd.DataFrame({
    'parameter': tornado_results.parameter_names,
    'low_value': tornado_results.low_values,
    'high_value': tornado_results.high_values,
    'swing': tornado_results.swings
})
tornado_df.to_csv('tornado_results.csv', index=False)
```

## Next Steps

- [API Reference](../api/README.md) - Complete function documentation
- [Examples](../examples/) - Jupyter notebooks with worked examples
