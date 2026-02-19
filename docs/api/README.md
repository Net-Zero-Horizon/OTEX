# API Reference

Complete documentation of OTEX public API.

## Module Overview

| Module | Description |
|--------|-------------|
| [`otex.config`](#otexconfig) | Configuration management |
| [`otex.core`](#otexcore) | Thermodynamic cycles and fluids |
| [`otex.plant`](#otexplant) | Plant sizing and operation |
| [`otex.economics`](#otexeconomics) | Cost analysis and LCOE |
| [`otex.analysis`](#otexanalysis) | Uncertainty and sensitivity |
| [`otex.data`](#otexdata) | Data loading and processing |

---

## otex.config

Configuration management for OTEX analyses.

### `parameters_and_constants`

```python
def parameters_and_constants(
    p_gross: float = -136000,
    cost_level: Union[str, CostScheme] = 'low_cost',
    data: str = 'CMEMS',
    fluid_type: str = 'ammonia',
    cycle_type: str = 'rankine_closed',
    use_coolprop: bool = True,
    optimize_depth: bool = False,
    year: int = 2020
) -> Dict[str, Any]
```

Create configuration dictionary for OTEC analysis.

**Parameters:**
- `p_gross`: Gross power output in kW (negative = output)
- `cost_level`: `'low_cost'`, `'high_cost'`, or a [`CostScheme`](#costscheme) object
- `data`: Data source (`'CMEMS'` or `'HYCOM'`)
- `fluid_type`: Working fluid (`'ammonia'`, `'r134a'`, etc.)
- `cycle_type`: Thermodynamic cycle type
- `use_coolprop`: Use CoolProp for fluid properties
- `optimize_depth`: Optimize cold water intake depth
- `year`: Year for analysis

**Returns:**
- Dictionary with all configuration parameters

**Example:**
```python
from otex.config import parameters_and_constants

# Built-in scheme
inputs = parameters_and_constants(
    p_gross=-50000,
    cost_level='low_cost',
    cycle_type='rankine_closed'
)

# Custom scheme
from otex.economics import LOW_COST
from dataclasses import replace

my_scheme = replace(LOW_COST, turbine_coeff=400, opex_fraction=0.04)
inputs = parameters_and_constants(p_gross=-50000, cost_level=my_scheme)
```

### `OTEXConfig`

Dataclass-based configuration (modern API):

```python
from otex.config import OTEXConfig, get_default_config

config = get_default_config()
config.plant.gross_power = -50000
config.economics.discount_rate = 0.08

inputs = config.to_legacy_dict()
```

---

## otex.core

Thermodynamic cycles and working fluids.

### Cycles

Available cycles:
- `RankineClosedCycle` - Closed-loop Rankine with organic fluid
- `RankineOpenCycle` - Flash evaporation of seawater
- `RankineHybridCycle` - Combined closed/open cycle
- `KalinaCycle` - Ammonia-water mixture
- `UeharaCycle` - Advanced ammonia-water cycle

```python
from otex.core import get_thermodynamic_cycle

cycle = get_thermodynamic_cycle('rankine_closed')
cycle = get_thermodynamic_cycle('kalina', ammonia_concentration=0.7)
```

### Working Fluids

```python
from otex.core import get_working_fluid

# With CoolProp (recommended)
fluid = get_working_fluid('ammonia', use_coolprop=True)

# Without CoolProp (polynomial correlations)
fluid = get_working_fluid('ammonia', use_coolprop=False)
```

Available fluids: `'ammonia'`, `'r134a'`, `'r245fa'`, `'propane'`, `'isobutane'`

---

## otex.plant

Plant sizing and operation.

### `otec_sizing`

```python
def otec_sizing(
    T_WW_in: np.ndarray,
    T_CW_in: np.ndarray,
    del_T_WW: float,
    del_T_CW: float,
    inputs: Dict,
    cost_level: Union[str, CostScheme]
) -> Dict[str, np.ndarray]
```

Size OTEC plant components for given conditions.

**Parameters:**
- `T_WW_in`: Warm water inlet temperature(s) in °C
- `T_CW_in`: Cold water inlet temperature(s) in °C
- `del_T_WW`: Temperature drop in warm water (°C)
- `del_T_CW`: Temperature rise in cold water (°C)
- `inputs`: Configuration dictionary
- `cost_level`: `'low_cost'`, `'high_cost'`, or a `CostScheme` object

**Returns:**
- Dictionary with plant parameters:
  - `p_net_nom`: Net power output (kW)
  - `p_gross_nom`: Gross power output (kW)
  - `A_evap`: Evaporator area (m²)
  - `A_cond`: Condenser area (m²)
  - `m_WW_nom`: Warm water flow rate (kg/s)
  - `m_CW_nom`: Cold water flow rate (kg/s)
  - And many more...

**Example:**
```python
import numpy as np
from otex.config import parameters_and_constants
from otex.plant.sizing import otec_sizing

inputs = parameters_and_constants(p_gross=-50000)
plant = otec_sizing(
    np.array([28.0]),
    np.array([5.0]),
    3.0, 3.0,
    inputs, 'low_cost'
)
print(f"Net power: {-plant['p_net_nom'][0]/1000:.1f} MW")
```

---

## otex.economics

Cost analysis and LCOE calculation.

### `CostScheme`

```python
@dataclass
class CostScheme:
    # Turbine [$/kW]:   coeff * (ref_power / -p_gross) ** exp
    turbine_coeff: float = 328.0
    turbine_ref_power: float = 136000.0
    turbine_exp: float = 0.16

    # Heat exchangers [$/m²]: coeff * (ref_power / -p_gross) ** exp
    hx_coeff: float = 226.0
    hx_ref_power: float = 80000.0
    hx_exp: float = 0.16

    # Seawater pumps [$/kW]: coeff * (ref_power / p_pump_total) ** exp
    pump_coeff: float = 1674.0
    pump_ref_power: float = 5600.0
    pump_exp: float = 0.38

    # Pipes [$/kg of pipe mass]
    pipes_coeff: float = 9.0

    # Structure [$/kW]: coeff * (ref_power / -p_gross) ** exp
    structure_coeff: float = 4465.0
    structure_ref_power: float = 28100.0
    structure_exp: float = 0.35

    # Deployment [$/kW]
    deploy_coeff: float = 650.0

    # Controls & management [$/kW]: coeff * (ref_power / -p_gross) ** exp
    controls_coeff: float = 3113.0
    controls_ref_power: float = 3960.0
    controls_exp: float = 0.70

    # Contingency and OPEX (fractions)
    capex_extra_fraction: float = 0.05   # fraction of CAPEX subtotal
    opex_fraction: float = 0.03          # annual fraction of total CAPEX

    # Pipe material density [kg/m³]
    pipe_density: float = 995.0          # HDPE by default (995); FRP = 1016
```

Parametric cost scheme for OTEC plant economic analysis. All monetary values are in USD (2021).

Two built-in instances are provided as module-level constants:

| Constant | Description |
|----------|-------------|
| `LOW_COST` | Optimistic scenario — defaults shown above |
| `HIGH_COST` | Conservative scenario — higher coefficients, FRP pipes, 20% contingency |

**Creating a custom scheme:**

```python
from otex.economics import CostScheme, LOW_COST, HIGH_COST
from dataclasses import replace

# From scratch — only override what you need (all fields have defaults)
my_scheme = CostScheme(
    turbine_coeff=400,
    opex_fraction=0.04,
    pipe_density=1000.0,
)

# Derived from an existing scheme — modify specific parameters
optimistic = replace(LOW_COST, turbine_coeff=280, capex_extra_fraction=0.03)
pessimistic = replace(HIGH_COST, discount_rate=0.12)   # discount_rate via Economics
```

**Using a custom scheme anywhere `cost_level` is accepted:**

```python
costs, capex, opex, lcoe = capex_opex_lcoe(plant, inputs, cost_level=my_scheme)

config = OTEXConfig(economics=Economics(cost_level=my_scheme))

inputs = parameters_and_constants(cost_level=my_scheme)
```

### `get_cost_scheme`

```python
def get_cost_scheme(cost_level: Union[str, CostScheme]) -> CostScheme
```

Resolve a string identifier or `CostScheme` to a `CostScheme` instance.
Raises `ValueError` if the string is not a recognised built-in name.

### `capex_opex_lcoe`

```python
def capex_opex_lcoe(
    otec_plant_nom: Dict,
    inputs: Dict,
    cost_level: Union[str, CostScheme] = 'low_cost'
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]
```

Calculate CAPEX, OPEX, and LCOE for sized plant.

**Parameters:**
- `otec_plant_nom`: Plant design from `otec_sizing()`
- `inputs`: Configuration dictionary (must include `dist_shore`, `crf`)
- `cost_level`: `'low_cost'`, `'high_cost'`, or a `CostScheme` object

**Returns:**
- `CAPEX_OPEX_dict`: Component-wise costs
- `CAPEX_total`: Total CAPEX ($)
- `OPEX`: Annual OPEX ($/year)
- `LCOE_nom`: Levelized cost of energy (ct/kWh)

**Example:**
```python
from otex.economics import capex_opex_lcoe, LOW_COST
from dataclasses import replace

inputs['dist_shore'] = np.array([20.0])
inputs['eff_trans'] = 0.978

# Built-in scheme
costs, capex, opex, lcoe = capex_opex_lcoe(plant, inputs, 'low_cost')

# Custom scheme
my_scheme = replace(LOW_COST, turbine_coeff=400, opex_fraction=0.04)
costs, capex, opex, lcoe = capex_opex_lcoe(plant, inputs, my_scheme)
print(f"LCOE: {lcoe[0]:.2f} ct/kWh")
```

---

## otex.analysis

Uncertainty and sensitivity analysis.

### `UncertainParameter`

```python
@dataclass
class UncertainParameter:
    name: str
    nominal: float
    distribution: Literal['uniform', 'normal', 'triangular'] = 'uniform'
    bounds: Tuple[float, float] = (0.0, 1.0)
    category: Literal['thermodynamic', 'economic', 'efficiency'] = 'thermodynamic'
```

Define an uncertain parameter with its distribution.

### `UncertaintyConfig`

```python
@dataclass
class UncertaintyConfig:
    parameters: List[UncertainParameter]  # Default parameters if not specified
    n_samples: int = 1000
    seed: int = 42
    parallel: bool = True
    n_workers: Optional[int] = None
```

Configuration for uncertainty analysis.

### `MonteCarloAnalysis`

```python
class MonteCarloAnalysis:
    def __init__(
        self,
        T_WW: float,
        T_CW: float,
        config: Optional[UncertaintyConfig] = None,
        p_gross: float = -136000,
        cost_level: str = 'low_cost'
    ): ...

    def run(self, show_progress: bool = True) -> UncertaintyResults: ...
```

Monte Carlo analysis with Latin Hypercube Sampling.

### `UncertaintyResults`

```python
@dataclass
class UncertaintyResults:
    samples: np.ndarray      # (n_samples, n_params)
    lcoe: np.ndarray         # (n_samples,)
    net_power: np.ndarray    # (n_samples,)
    capex: np.ndarray        # (n_samples,)
    opex: np.ndarray         # (n_samples,)
    parameter_names: List[str]
    config: Optional[UncertaintyConfig]

    def compute_statistics(self) -> Dict[str, Dict[str, float]]: ...
    def get_confidence_interval(self, output: str, confidence: float) -> Tuple[float, float]: ...
    def to_dataframe(self) -> pd.DataFrame: ...
```

`compute_statistics()` returns per-output keys including `mean`, `std`, `median`, `cv`, `skewness`, `kurtosis`, `p5`, `p10`, `p25`, `p75`, `p90`, `p95`, `n_valid`, `n_invalid`.

`to_dataframe()` returns a tidy `pd.DataFrame` with one row per simulation run: parameter columns, output columns (`lcoe`, `net_power`, `capex`, `opex`), and a boolean `valid` column.

### `TornadoAnalysis`

```python
class TornadoAnalysis:
    def __init__(
        self,
        T_WW: float,
        T_CW: float,
        variation_pct: float = 10.0,
        config: Optional[UncertaintyConfig] = None,
        p_gross: float = -136000,
        cost_level: str = 'low_cost'
    ): ...

    def run(
        self,
        output: str = 'lcoe',
        use_bounds: bool = True,
        show_progress: bool = True
    ) -> TornadoResults: ...
```

### `SobolAnalysis`

```python
class SobolAnalysis:
    def __init__(
        self,
        T_WW: float,
        T_CW: float,
        n_samples: int = 1024,
        calc_second_order: bool = False,
        config: Optional[UncertaintyConfig] = None,
        p_gross: float = -136000,
        cost_level: str = 'low_cost'
    ): ...

    def run(self, output: str = 'lcoe', show_progress: bool = True) -> SobolResults: ...
```

Requires SALib package.

### Visualization Functions

```python
def plot_histogram(
    results: UncertaintyResults,
    output: str = 'lcoe',
    ax: Optional[Axes] = None,
    bins: int = 50,
    show_stats: bool = True
) -> Axes: ...

def plot_tornado(
    results: TornadoResults,
    ax: Optional[Axes] = None,
    top_n: int = 10
) -> Axes: ...

def plot_sobol_indices(
    results: SobolResults,
    ax: Optional[Axes] = None,
    top_n: int = 10
) -> Axes: ...

def plot_scatter_matrix(
    results: UncertaintyResults,
    output: str = 'lcoe',
    max_params: int = 5
) -> Figure: ...

def create_summary_figure(
    mc_results: UncertaintyResults,
    tornado_results: TornadoResults,
    sobol_results: Optional[SobolResults] = None,
    output: str = 'lcoe'
) -> Figure: ...
```

### Export Functions

```python
from otex.analysis import (
    export_analysis,
    make_samples_df,
    make_statistics_df,
    make_correlations_df,
    make_parameters_df,
    make_tornado_df,
    make_sobol_df,
)
```

#### `export_analysis`

```python
def export_analysis(
    output_dir: str | Path,
    mc_results: Optional[UncertaintyResults] = None,
    tornado_results: Optional[TornadoResults] = None,
    sobol_results: Optional[SobolResults] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path
```

Export a complete analysis bundle to `output_dir`. Creates the directory if it does not exist.

| File generated | Contents |
|----------------|----------|
| `metadata.json` | Run configuration: temperatures, cost level, sample count, seed, OTEX version, timestamp |
| `samples.csv` | Raw MC samples — parameter values + all outputs + `valid` flag, one row per run |
| `statistics.csv` | Descriptive statistics per output: mean, std, CV, skewness, kurtosis, min–max, P5–P95 |
| `correlations.csv` | Spearman ρ and p-value between each parameter and each output, ranked by `\|lcoe_rho\|` |
| `parameters.csv` | Uncertain parameter definitions: name, category, nominal, distribution, bounds |
| `tornado.csv` | OAT sensitivity: rank, swing (absolute and %), output at low/high bounds |
| `sobol.csv` | Variance-based indices: S1, ST, confidence intervals, interaction term, % of total variance |

`SobolResults.to_dataframe()` and `TornadoResults.to_dataframe()` provide per-object access to the same DataFrames.

**Example:**

```python
from otex.analysis import (
    MonteCarloAnalysis, TornadoAnalysis, SobolAnalysis,
    export_analysis,
)

mc_results      = MonteCarloAnalysis(T_WW=28, T_CW=5).run()
tornado_results = TornadoAnalysis(T_WW=28, T_CW=5).run()
sobol_results   = SobolAnalysis(T_WW=28, T_CW=5, n_samples=512).run()

export_analysis(
    output_dir='results/run_01',
    mc_results=mc_results,
    tornado_results=tornado_results,
    sobol_results=sobol_results,
    metadata={'T_WW': 28, 'T_CW': 5, 'cost_level': 'low_cost'},
)
# Exported 7 files to results/run_01
```

---

## otex.data

Data loading and processing.

### `download_data`

```python
def download_data(
    cost_level: str,
    inputs: Dict,
    studied_region: str,
    dl_path: str
) -> List[str]
```

Download CMEMS oceanographic data for a region.

### `data_processing`

```python
def data_processing(
    files: List[str],
    sites_df: pd.DataFrame,
    inputs: Dict,
    studied_region: str,
    new_path: str,
    water_type: str,
    nan_columns: Optional[np.ndarray] = None
) -> Tuple[...]
```

Process downloaded NetCDF files into temperature profiles.

### `load_temperatures`

```python
def load_temperatures(
    h5_file: str,
    inputs: Dict
) -> Tuple[...]
```

Load cached temperature data from HDF5 file.

---

## See Also

- [Tutorials](../tutorials/) - Step-by-step guides
- [Examples](../examples/) - Jupyter notebooks
- [GitHub Repository](https://github.com/msotocalvo/OTEX)
