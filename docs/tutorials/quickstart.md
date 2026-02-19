# Quick Start Tutorial

Get started with OTEX in 10 minutes. This tutorial covers the essential concepts and basic usage.

## Prerequisites

- OTEX installed (`pip install otex`)
- Basic Python knowledge

## 1. Understanding OTEC Basics

Ocean Thermal Energy Conversion (OTEC) generates electricity from the temperature difference between warm surface water (~25-30°C) and cold deep water (~4-6°C). The key parameters are:

| Parameter | Typical Value | Unit |
|-----------|---------------|------|
| Warm water temperature (T_WW) | 26-30 | °C |
| Cold water temperature (T_CW) | 4-6 | °C |
| Temperature difference (ΔT) | 20-25 | °C |
| Carnot efficiency | 6-8 | % |
| Net efficiency | 2-3 | % |

## 2. Basic Configuration

```python
from otex.config import parameters_and_constants

# Create configuration for a 100 MW plant
inputs = parameters_and_constants(
    p_gross=-100000,      # 100 MW gross power (negative = output)
    cost_level='low_cost',
    cycle_type='rankine_closed',
    fluid_type='ammonia',
    year=2020
)

# View key parameters
print(f"Gross power: {-inputs['p_gross']/1000:.0f} MW")
print(f"Cycle type: {inputs['cycle_type']}")
print(f"Working fluid: {inputs['fluid_type']}")
print(f"Discount rate: {inputs['discount_rate']:.1%}")
print(f"Plant lifetime: {inputs['lifetime']} years")
print(f"Capacity factor: {inputs['availability_factor']:.1%}")
```

Output:
```
Gross power: 100 MW
Cycle type: rankine_closed
Working fluid: ammonia
Discount rate: 10.0%
Plant lifetime: 30 years
Capacity factor: 91.4%
```

## 3. Plant Sizing

Size an OTEC plant for specific water temperatures:

```python
import numpy as np
from otex.config import parameters_and_constants
from otex.plant.sizing import otec_sizing

# Configuration
inputs = parameters_and_constants(p_gross=-50000, cost_level='low_cost')

# Water temperatures (as arrays for vectorized computation)
T_WW = np.array([28.0])  # Warm water: 28°C
T_CW = np.array([5.0])   # Cold water: 5°C

# Temperature drops across heat exchangers
del_T_WW = 3.0  # °C drop in warm water
del_T_CW = 3.0  # °C rise in cold water

# Size the plant
plant = otec_sizing(T_WW, T_CW, del_T_WW, del_T_CW, inputs, 'low_cost')

# View results
print(f"Net power: {-plant['p_net_nom'][0]/1000:.1f} MW")
print(f"Net efficiency: {plant['eff_net_nom'][0]*100:.2f}%")
print(f"Evaporator area: {plant['A_evap'][0]:,.0f} m²")
print(f"Condenser area: {plant['A_cond'][0]:,.0f} m²")
print(f"Warm water flow: {plant['m_WW_nom'][0]:,.0f} kg/s")
print(f"Cold water flow: {plant['m_CW_nom'][0]:,.0f} kg/s")
```

Output:
```
Net power: 29.4 MW
Net efficiency: 2.45%
Evaporator area: 98,543 m²
Condenser area: 134,231 m²
Warm water flow: 89,234 kg/s
Cold water flow: 71,456 kg/s
```

## 4. Cost Analysis

Calculate CAPEX, OPEX, and LCOE:

```python
from otex.economics.costs import capex_opex_lcoe

# Add distance to shore for cable costs
inputs['dist_shore'] = np.array([20.0])  # 20 km offshore

# Calculate transmission efficiency
dist = 20.0
inputs['eff_trans'] = 0.979 - 1e-6 * dist**2 - 9e-5 * dist

# Get costs
costs_dict, capex_total, opex, lcoe = capex_opex_lcoe(plant, inputs, 'low_cost')

print(f"Total CAPEX: ${capex_total[0]/1e6:,.1f} M")
print(f"Annual OPEX: ${opex[0]/1e6:,.1f} M/year")
print(f"LCOE: {lcoe[0]:.2f} ct/kWh")
print()
print("CAPEX Breakdown:")
print(f"  Turbine: ${costs_dict['turbine_CAPEX']/1e6:,.1f} M")
print(f"  Heat exchangers: ${(costs_dict['evap_CAPEX']+costs_dict['cond_CAPEX'])/1e6:,.1f} M")
print(f"  Platform: ${costs_dict['platform_CAPEX']/1e6:,.1f} M")
print(f"  Mooring: ${costs_dict['mooring_CAPEX']/1e6:,.1f} M")
print(f"  Cables: ${costs_dict['cable_CAPEX']/1e6:,.1f} M")
```

## 5. Custom Cost Schemes

By default OTEX ships two built-in cost scenarios: `'low_cost'` and `'high_cost'`. You can also create fully custom schemes or derive one from an existing scenario using Python's `dataclasses.replace()`.

```python
from otex.economics import CostScheme, LOW_COST, HIGH_COST
from dataclasses import replace

# ── Option A: derive from an existing scheme ─────────────────────────────────
# Adjust turbine cost and OPEX rate, keep everything else from LOW_COST
my_scheme = replace(
    LOW_COST,
    turbine_coeff=400,      # $/kW (default 328)
    opex_fraction=0.04,     # 4% of CAPEX/yr (default 3%)
)

# ── Option B: build from scratch ─────────────────────────────────────────────
# Only specify what differs — unset fields use LOW_COST defaults
custom = CostScheme(
    turbine_coeff=380,
    hx_coeff=260,
    pipes_coeff=12.0,
    capex_extra_fraction=0.08,
    opex_fraction=0.035,
    pipe_density=1010.0,    # kg/m³ — affects pipe mass calculation
)

# ── Use anywhere cost_level is accepted ──────────────────────────────────────
from otex.config import parameters_and_constants
from otex.economics import capex_opex_lcoe

inputs = parameters_and_constants(p_gross=-50000, cost_level=my_scheme)
costs_dict, capex_total, opex, lcoe = capex_opex_lcoe(plant, inputs, my_scheme)
print(f"Custom LCOE: {lcoe[0]:.2f} ct/kWh")
```

The built-in schemes differ primarily in their reference coefficients and contingency rates:

| Parameter | `LOW_COST` | `HIGH_COST` |
|-----------|-----------|------------|
| `turbine_coeff` [$/kW] | 328 | 512 |
| `hx_coeff` [$/m²] | 226 | 916 |
| `pipes_coeff` [$/kg] | 9.0 | 30.1 |
| `capex_extra_fraction` | 5 % | 20 % |
| `opex_fraction` | 3 %/yr | 5 %/yr |
| Pipe material | HDPE (995 kg/m³) | FRP (1016 kg/m³) |

See [`otex.economics.CostScheme`](../api/README.md#costscheme) for a full list of fields.

## 6. Quick Uncertainty Analysis

Assess uncertainty in LCOE estimates:

```python
from otex.analysis import MonteCarloAnalysis, UncertaintyConfig

# Configure analysis
config = UncertaintyConfig(
    n_samples=100,    # Number of Monte Carlo samples
    seed=42,          # For reproducibility
    parallel=False    # Set True for faster execution
)

# Run analysis
mc = MonteCarloAnalysis(
    T_WW=28.0,
    T_CW=5.0,
    config=config,
    p_gross=-50000,
    cost_level='low_cost'
)
results = mc.run(show_progress=True)

# Get statistics
stats = results.compute_statistics()
lcoe = stats['lcoe']

print(f"\nLCOE Results (100 samples)")
print(f"{'='*40}")
print(f"Mean:   {lcoe['lcoe_mean']:.2f} ct/kWh")
print(f"Std:    {lcoe['lcoe_std']:.2f} ct/kWh")
print(f"CV:     {lcoe['lcoe_cv']:.1%}")
print(f"Min:    {lcoe['lcoe_min']:.2f} ct/kWh")
print(f"Max:    {lcoe['lcoe_max']:.2f} ct/kWh")
print(f"90% CI: [{lcoe['lcoe_p5']:.2f}, {lcoe['lcoe_p95']:.2f}] ct/kWh")
```

## 7. Sensitivity Analysis

Identify most influential parameters:

```python
from otex.analysis import TornadoAnalysis

tornado = TornadoAnalysis(
    T_WW=28.0,
    T_CW=5.0,
    p_gross=-50000,
    cost_level='low_cost'
)

results = tornado.run(output='lcoe', show_progress=True)

print(f"\nParameter Sensitivity (LCOE)")
print(f"Baseline: {results.baseline:.2f} ct/kWh")
print(f"{'='*50}")
for name, swing in results.get_ranking()[:5]:
    print(f"{name:40s} {swing:+.2f} ct/kWh")
```

Output:
```
Parameter Sensitivity (LCOE)
Baseline: 24.19 ct/kWh
==================================================
discount_rate                            +15.51 ct/kWh
capex_structure_factor                   +5.42 ct/kWh
turbine_isentropic_efficiency            -4.87 ct/kWh
capex_HX_factor                          +2.41 ct/kWh
opex_factor                              +2.13 ct/kWh
```

## 8. Visualization

```python
import matplotlib.pyplot as plt
from otex.analysis import plot_histogram, plot_tornado

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot LCOE distribution
plot_histogram(results, output='lcoe', ax=ax1)

# Plot tornado diagram
plot_tornado(tornado_results, ax=ax2, top_n=8)

plt.tight_layout()
plt.savefig('otex_quickstart.png', dpi=150)
plt.show()
```

## Next Steps

- **[Regional Analysis](regional_analysis.md)**: Analyze specific locations with real oceanographic data
- **[Uncertainty Analysis](uncertainty_analysis.md)**: Deep dive into Monte Carlo and Sobol analysis
- **[API Reference](../api/README.md)**: Complete documentation of all functions

## Key Takeaways

1. **Configuration**: Use `parameters_and_constants()` to set up plant parameters
2. **Sizing**: Use `otec_sizing()` to calculate component sizes and flows
3. **Economics**: Use `capex_opex_lcoe()` to get costs and LCOE
4. **Cost schemes**: Pass `'low_cost'`, `'high_cost'`, or a custom `CostScheme` to any function that accepts `cost_level`
5. **Uncertainty**: Use `MonteCarloAnalysis` for probabilistic assessment
6. **Sensitivity**: Use `TornadoAnalysis` to identify key parameters

Remember:
- LCOE typically ranges from 15-40 ct/kWh depending on conditions
- Discount rate is usually the most influential economic parameter
- Temperature difference (ΔT) strongly affects plant efficiency
- Larger plants have better economies of scale
- `dataclasses.replace(LOW_COST, ...)` is the quickest way to explore cost sensitivity
