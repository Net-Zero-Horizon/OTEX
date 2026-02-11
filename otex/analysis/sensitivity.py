# -*- coding: utf-8 -*-
"""
Sensitivity Analysis for OTEC plants.

Implements:
- Sobol sensitivity analysis (global, variance-based)
- Tornado diagram analysis (local, one-at-a-time)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

from .distributions import UncertaintyConfig, get_default_parameters


@dataclass
class SobolResults:
    """
    Results from Sobol sensitivity analysis.

    Attributes:
        S1: First-order sensitivity indices
        ST: Total-order sensitivity indices
        S1_conf: Confidence intervals for S1
        ST_conf: Confidence intervals for ST
        parameter_names: Names of parameters
        output_name: Name of output variable analyzed
    """
    S1: np.ndarray
    ST: np.ndarray
    S1_conf: np.ndarray = field(default_factory=lambda: np.array([]))
    ST_conf: np.ndarray = field(default_factory=lambda: np.array([]))
    parameter_names: List[str] = field(default_factory=list)
    output_name: str = 'lcoe'

    def get_ranking(self, index: str = 'ST') -> List[Tuple[str, float]]:
        """
        Get parameters ranked by importance.

        Args:
            index: 'S1' for first-order or 'ST' for total-order

        Returns:
            List of (parameter_name, index_value) sorted by importance
        """
        indices = self.S1 if index == 'S1' else self.ST
        ranking = sorted(
            zip(self.parameter_names, indices),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return ranking

    def to_dict(self) -> Dict:
        """Convert results to dictionary."""
        return {
            'S1': dict(zip(self.parameter_names, self.S1)),
            'ST': dict(zip(self.parameter_names, self.ST)),
            'ranking_ST': self.get_ranking('ST'),
            'ranking_S1': self.get_ranking('S1'),
            'output': self.output_name
        }


@dataclass
class TornadoResults:
    """
    Results from Tornado diagram analysis.

    Attributes:
        parameter_names: Names of parameters
        low_values: Output values with parameter at low bound
        high_values: Output values with parameter at high bound
        baseline: Output value with all parameters at nominal
        swings: Total swing (high - low) for each parameter
        output_name: Name of output variable analyzed
    """
    parameter_names: List[str]
    low_values: np.ndarray
    high_values: np.ndarray
    baseline: float
    swings: np.ndarray = field(default_factory=lambda: np.array([]))
    output_name: str = 'lcoe'

    def __post_init__(self):
        """Compute swings if not provided."""
        if len(self.swings) == 0:
            self.swings = self.high_values - self.low_values

    def get_ranking(self) -> List[Tuple[str, float]]:
        """
        Get parameters ranked by swing magnitude.

        Returns:
            List of (parameter_name, swing) sorted by absolute swing
        """
        ranking = sorted(
            zip(self.parameter_names, self.swings),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return ranking

    def to_dict(self) -> Dict:
        """Convert results to dictionary."""
        return {
            'parameter_names': self.parameter_names,
            'low_values': self.low_values.tolist(),
            'high_values': self.high_values.tolist(),
            'baseline': self.baseline,
            'swings': self.swings.tolist(),
            'ranking': self.get_ranking(),
            'output': self.output_name
        }


def _run_model(
    T_WW: float,
    T_CW: float,
    param_overrides: Dict[str, float],
    p_gross: float = -136000,
    cost_level: str = 'low_cost'
) -> Dict[str, float]:
    """
    Run OTEC model with parameter overrides.

    Args:
        T_WW: Warm water temperature
        T_CW: Cold water temperature
        param_overrides: Dictionary of parameter names to values
        p_gross: Gross power output
        cost_level: Cost scenario

    Returns:
        Dictionary with 'lcoe', 'net_power', 'capex', 'opex'
    """
    from otex.config import parameters_and_constants
    from otex.plant.sizing import otec_sizing
    from otex.economics.costs import capex_opex_lcoe

    try:
        inputs = parameters_and_constants(p_gross=p_gross, cost_level=cost_level)

        # Apply parameter overrides
        for name, value in param_overrides.items():
            if name == 'turbine_isentropic_efficiency':
                inputs['eff_isen_turb'] = value
            elif name == 'pump_isentropic_efficiency':
                inputs['eff_isen_pump'] = value
            elif name == 'U_evap':
                inputs['U_evap'] = value
            elif name == 'U_cond':
                inputs['U_cond'] = value
            elif name == 'discount_rate':
                inputs['discount_rate'] = value
                r = value
                n = inputs['lifetime']
                inputs['crf'] = r * (1 + r)**n / ((1 + r)**n - 1)
            else:
                # For cost factors and other parameters, add directly
                inputs[name] = value

        del_T_WW = 3.0
        del_T_CW = 3.0

        # Set distance to shore and transmission efficiency
        dist_shore = 10.0  # Default distance in km
        inputs['dist_shore'] = np.array([dist_shore])

        # Calculate transmission efficiency based on distance
        threshold = inputs.get('threshold_AC_DC', 50.0)
        if dist_shore <= threshold:
            eff_trans = 0.979 - 1e-6 * dist_shore**2 - 9e-5 * dist_shore
        else:
            eff_trans = 0.964 - 8e-5 * dist_shore
        inputs['eff_trans'] = eff_trans

        T_WW_arr = np.array([T_WW])
        T_CW_arr = np.array([T_CW])

        plant = otec_sizing(T_WW_arr, T_CW_arr, del_T_WW, del_T_CW, inputs, cost_level)
        _, capex, opex, lcoe = capex_opex_lcoe(plant, inputs, cost_level)

        return {
            'lcoe': float(lcoe[0]),
            'net_power': float(plant['p_net_nom'][0]),
            'capex': float(capex[0]),
            'opex': float(opex[0])
        }
    except Exception:
        return {
            'lcoe': np.nan,
            'net_power': np.nan,
            'capex': np.nan,
            'opex': np.nan
        }


class SobolAnalysis:
    """
    Global sensitivity analysis using Sobol indices.

    Requires SALib library for Saltelli sampling and Sobol analysis.

    Example:
        >>> from otex.analysis import SobolAnalysis
        >>> sobol = SobolAnalysis(T_WW=28.0, T_CW=5.0, n_samples=512)
        >>> results = sobol.run(output='lcoe')
        >>> for name, val in results.get_ranking()[:5]:
        ...     print(f"{name}: ST={val:.3f}")
    """

    def __init__(
        self,
        T_WW: float,
        T_CW: float,
        n_samples: int = 1024,
        calc_second_order: bool = False,
        config: Optional[UncertaintyConfig] = None,
        p_gross: float = -136000,
        cost_level: str = 'low_cost'
    ):
        """
        Initialize Sobol analysis.

        Args:
            T_WW: Warm water temperature (°C)
            T_CW: Cold water temperature (°C)
            n_samples: Number of base samples (total = n_samples * (2*d + 2))
            calc_second_order: Whether to compute second-order indices
            config: Uncertainty configuration
            p_gross: Gross power output (kW)
            cost_level: Cost scenario
        """
        self.T_WW = T_WW
        self.T_CW = T_CW
        self.n_samples = n_samples
        self.calc_second_order = calc_second_order
        self.config = config or UncertaintyConfig()
        self.p_gross = p_gross
        self.cost_level = cost_level

    def run(self, output: str = 'lcoe', show_progress: bool = True) -> SobolResults:
        """
        Run Sobol sensitivity analysis.

        Args:
            output: Output variable to analyze ('lcoe', 'net_power', 'capex')
            show_progress: Whether to show progress bar

        Returns:
            SobolResults with sensitivity indices
        """
        try:
            from SALib.sample import saltelli
            from SALib.analyze import sobol
        except ImportError:
            raise ImportError(
                "SALib is required for Sobol analysis. "
                "Install with: pip install SALib"
            )

        problem = self.config.get_problem_dict()

        # Generate Saltelli samples
        param_values = saltelli.sample(
            problem,
            self.n_samples,
            calc_second_order=self.calc_second_order
        )

        n_runs = param_values.shape[0]
        Y = np.zeros(n_runs)

        # Run model for each sample
        iterator = range(n_runs)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Sobol ({output})")

        for i in iterator:
            overrides = dict(zip(problem['names'], param_values[i]))
            result = _run_model(
                self.T_WW, self.T_CW, overrides,
                self.p_gross, self.cost_level
            )
            Y[i] = result[output]

        # Handle NaN values
        valid_mask = ~np.isnan(Y)
        if not np.all(valid_mask):
            # Replace NaN with mean of valid values for analysis
            Y[~valid_mask] = np.nanmean(Y)

        # Analyze
        Si = sobol.analyze(
            problem, Y,
            calc_second_order=self.calc_second_order,
            print_to_console=False
        )

        return SobolResults(
            S1=np.array(Si['S1']),
            ST=np.array(Si['ST']),
            S1_conf=np.array(Si.get('S1_conf', [])),
            ST_conf=np.array(Si.get('ST_conf', [])),
            parameter_names=problem['names'],
            output_name=output
        )


class TornadoAnalysis:
    """
    One-at-a-time sensitivity analysis for tornado diagrams.

    Varies each parameter individually while holding others at nominal.

    Example:
        >>> from otex.analysis import TornadoAnalysis
        >>> tornado = TornadoAnalysis(T_WW=28.0, T_CW=5.0, variation_pct=10.0)
        >>> results = tornado.run(output='lcoe')
        >>> for name, swing in results.get_ranking()[:5]:
        ...     print(f"{name}: swing={swing:.2f}")
    """

    def __init__(
        self,
        T_WW: float,
        T_CW: float,
        variation_pct: float = 10.0,
        config: Optional[UncertaintyConfig] = None,
        p_gross: float = -136000,
        cost_level: str = 'low_cost'
    ):
        """
        Initialize Tornado analysis.

        Args:
            T_WW: Warm water temperature (°C)
            T_CW: Cold water temperature (°C)
            variation_pct: Percentage variation from nominal (default 10%)
            config: Uncertainty configuration (uses parameter bounds if provided)
            p_gross: Gross power output (kW)
            cost_level: Cost scenario
        """
        self.T_WW = T_WW
        self.T_CW = T_CW
        self.variation_pct = variation_pct
        self.config = config or UncertaintyConfig()
        self.p_gross = p_gross
        self.cost_level = cost_level

    def _get_parameter_bounds(self, param) -> Tuple[float, float]:
        """Get low and high values for a parameter."""
        if param.distribution == 'uniform':
            return param.bounds[0], param.bounds[1]
        elif param.distribution == 'normal':
            mean, std = param.bounds
            return mean - 2*std, mean + 2*std
        else:
            return param.bounds[0], param.bounds[1]

    def run(
        self,
        output: str = 'lcoe',
        use_bounds: bool = True,
        show_progress: bool = True
    ) -> TornadoResults:
        """
        Run tornado analysis.

        Args:
            output: Output variable to analyze
            use_bounds: If True, use parameter bounds; if False, use ±variation_pct
            show_progress: Whether to show progress bar

        Returns:
            TornadoResults with swing values for each parameter
        """
        params = self.config.parameters
        n_params = len(params)
        param_names = [p.name for p in params]

        # Get baseline (all nominal values)
        nominal_overrides = {p.name: p.nominal for p in params}
        baseline_result = _run_model(
            self.T_WW, self.T_CW, nominal_overrides,
            self.p_gross, self.cost_level
        )
        baseline = baseline_result[output]

        low_values = np.zeros(n_params)
        high_values = np.zeros(n_params)

        iterator = enumerate(params)
        if show_progress:
            iterator = tqdm(list(iterator), desc=f"Tornado ({output})")

        for i, param in iterator:
            # Determine low and high values
            if use_bounds:
                low_val, high_val = self._get_parameter_bounds(param)
            else:
                pct = self.variation_pct / 100.0
                low_val = param.nominal * (1 - pct)
                high_val = param.nominal * (1 + pct)

            # Run with low value
            overrides = nominal_overrides.copy()
            overrides[param.name] = low_val
            result_low = _run_model(
                self.T_WW, self.T_CW, overrides,
                self.p_gross, self.cost_level
            )
            low_values[i] = result_low[output]

            # Run with high value
            overrides[param.name] = high_val
            result_high = _run_model(
                self.T_WW, self.T_CW, overrides,
                self.p_gross, self.cost_level
            )
            high_values[i] = result_high[output]

        return TornadoResults(
            parameter_names=param_names,
            low_values=low_values,
            high_values=high_values,
            baseline=baseline,
            output_name=output
        )
