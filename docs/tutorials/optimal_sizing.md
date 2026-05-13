# Mode Inverso — Per-Site Design Optimisation

Since 0.4.0 OTEX exposes a second mode of operation that flips the
question OTEX answers:

| Mode | Question | Input | Output |
|---|---|---|---|
| **Forward** (`otex-regional`) | What's the LCOE if I install P_gross MW at each site? | `p_gross` | LCOE, AEP per site |
| **Inverse** (`otex-regional-optimal`) | What plant design minimises LCOE at each site? | bounds on x | optimal `x = (p_gross, dT_WW, dT_CW, depth_CW)` per site |

Both modes coexist — the forward pipeline is untouched and keeps
serving the case where plant size comes from an external constraint
(demand, capital budget, regulatory cap). Use the inverse mode when
you want OTEX itself to recommend the design.

## The optimisation problem

Per site, OTEX solves a 4-D continuous non-linear program:

```
minimise   LCOE(x)
                                    ← continuous decision vector
   x = (p_gross,         kW, negative
        dT_WW,           °C, warm-side temperature drop
        dT_CW,           °C, cold-side temperature rise
        depth_CW)        m,  cold-water intake depth

subject to:
   physical constraints g_i(x) ≤ 0:
      • dT_WW + dT_CW + pinch margins ≤ available T_WW − T_CW
      • evaporator and condenser pinch points respected
      • pipe inner diameter ≤ max physical limit (~8 m)
      • parasitic ratio (P_pump / |P_gross|) ≤ 40 %
      • cold-water intake fits the site bathymetry
      • net power positive
   user-defined exogenous caps (any subset, all optional):
      • max AEP per plant (MWh/yr)
      • max net delivered power (MW)
      • max gross plant size (MW)
      • max total CAPEX (M USD)
      • max parasitic ratio
   box bounds on x.
```

Categorical design choices (cycle type, working fluid, installation
type) are **exogenous** — they change the entire model structure
entirely. Compare alternatives by running the optimiser several
times with each combination.

The constrained problem is collapsed to an unconstrained one via a
**quadratic penalty method** on every constraint. The solver is
`scipy.optimize.minimize(method='L-BFGS-B')` operating on a
**normalised [0,1] cube**: each variable is rescaled so SLSQP-style
finite-difference gradients work cleanly across the wildly different
physical magnitudes (p_gross ~10⁵ kW vs dT ~5 K vs depth ~10³ m).

> **Why is the user constraint mandatory in practice?**
> Without an exogenous cap, the LCOE function is monotonically
> decreasing in ``p_gross`` over OTEX's modelled range — the cost
> correlations ``C ∝ p^α`` are concave (economies of scale) and the
> internal physics doesn't impose a hard upper limit on plant size.
> The "optimum" degenerates to the upper box bound and the solver is
> reduced to a box-bound oracle. Setting at least one
> ``--max-…`` flag — whatever cap *your* decision context imposes —
> produces a genuine interior optimum.

A typical site converges in 200-600 L-BFGS-B evaluations
(0.3-1 s with polynomial fluid properties, 5-10 s with CoolProp).

## Quick start

Decide what limits *your* plant size, then pass it on the CLI:

```bash
# "I can build no more than 100 MW gross per plant"
otex-regional-optimal Jamaica --year-start 2020 --year-end 2023 \
    --max-p-gross-MW 100

# "Each plant must stay below USD 800 M CAPEX"
otex-regional-optimal Jamaica --year-start 2020 --year-end 2023 \
    --max-capex-MUSD 800

# "I want at most 50 MW of net delivered power per site"
otex-regional-optimal Jamaica --year-start 2020 --year-end 2023 \
    --max-p-net-MW 50
```

The Python API mirrors the CLI:

```python
from otex.optimization import (
    run_regional_optimization, Bounds, UserConstraints,
)

df = run_regional_optimization(
    studied_region='Jamaica',
    year_start=2020, year_end=2023,
    bounds=Bounds(
        p_gross=(-500_000, -1_000),
        dT_WW=(1.0, 6.0),
        dT_CW=(1.0, 6.0),
        depth_CW=(600.0, 3000.0),
    ),
    user_constraints=UserConstraints(
        max_p_gross_MW=120,        # cap any one of these
        # max_capex_MUSD=800,
        # max_p_net_MW=50,
        # max_aep_MWh=400_000,
        # max_parasitic_ratio=0.30,
    ),
)
print(df.head())
```

## Output columns

The optimiser writes
`OTEC_sites_optimal_<region>_<year_label>_<cost_level>.csv` to the
region directory with:

| Column | Meaning |
|---|---|
| `id`, `longitude`, `latitude` | Site identifier |
| `T_WW_design`, `T_CW_design` | Design-point temperatures used (median across the year range) |
| `p_gross_opt_MW` | Optimal gross plant size |
| `dT_WW_opt`, `dT_CW_opt` | Optimal heat-exchanger ΔT |
| `depth_CW_opt` | Optimal cold-water intake depth |
| `lcoe_min` | LCOE at the optimum (¢/kWh) |
| `p_net_kW`, `capex_total_MUSD`, `opex_MUSDyr` | Headline numbers at the optimum |
| `max_violation` | Maximum constraint violation magnitude |
| `feasible` | True if `max_violation ≤ 0.01` (1 % of pinch margin) |
| `success`, `n_evaluations`, `message` | SLSQP diagnostics |
| `g_dT_total`, `g_pinch_evap`, … | Per-constraint values at the optimum |

## How to read the results

* **`feasible = True`** plus a small `max_violation` (~0.003 K) is
  normal. The evaporator pinch constraint is *active* at every
  LCOE-minimum, so the optimum sits exactly on it modulo numerical
  noise. The 1 % threshold is well below the engineering uncertainty
  of the pinch margin itself.
* **`feasible = False`** at every site usually means the bounds are
  inconsistent with the site's available ΔT (T_WW − T_CW is too low
  for the requested dT_WW + dT_CW + pinch margin). Loosen the dT
  bounds or skip the marginal sites.
* **`dT_CW_opt` at the upper bound** is common — bigger cold-side ΔT
  extracts more energy per unit mass. If you can engineer a larger
  cold-water HX, raising `--dT-max` lets the optimum keep going.
* **`depth_CW_opt` at the lower bound (600 m)** indicates the
  optimum wants the shallowest cold-water intake the depth bracket
  allows. Real OTEC pilots have stayed at 1000 m for thermal-margin
  reasons; tighten `--depth-min-m` if you want to enforce a colder
  intake.

## Comparing forward and inverse outputs

```python
import pandas as pd
fwd = pd.read_csv("…/OTEC_sites_<region>_…_100.0_MW_low_cost.csv", sep=';')
opt = pd.read_csv("…/OTEC_sites_optimal_<region>_…_low_cost.csv", sep=';')

merged = fwd.merge(
    opt[['longitude','latitude','lcoe_min','p_gross_opt_MW']],
    on=['longitude','latitude'],
)
merged['ΔLCOE_pct'] = 100*(merged.LCOE - merged.lcoe_min) / merged.LCOE
print(merged[['p_gross_opt_MW', 'LCOE', 'lcoe_min', 'ΔLCOE_pct']].describe())
```

For Jamaica 2020-2023, the inverse mode delivers a **median 12.7 %
LCOE reduction** vs a fixed 100 MW design.

## Limitations of the MVP

* **Local optimiser, single start.** L-BFGS-B is a local solver. Per-site
  multi-start or a global optimiser (CMA-ES, differential evolution)
  is a planned follow-up; for now the warm-start that respects active
  user constraints gives reproducible and good — but possibly not
  globally optimal — results.
* **Cost correlations were calibrated around 100 MW.** Designs
  outside [10, 200] MW are mild extrapolations of OTEX's economic
  scheme; the optimisation itself is mathematically valid but the
  numerical LCOE for very small or very large optima carries extra
  uncertainty. Pin ``--max-p-gross-MW`` to a realistic upper bound to
  stay in the calibrated range.
* **No portfolio-level constraints.** Each site is optimised
  independently. Joint constraints (regional capacity target,
  shared transmission infrastructure, complementarity bonuses) are
  a follow-on feature.
