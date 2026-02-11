# -*- coding: utf-8 -*-
"""
Monte Carlo Uncertainty Analysis for OTEC plants.

Implements Latin Hypercube Sampling for propagating parameter uncertainty
through the OTEC sizing and cost models.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

from .distributions import UncertaintyConfig, UncertainParameter


@dataclass
class UncertaintyResults:
    """
    Results from Monte Carlo uncertainty analysis.

    Attributes:
        samples: Parameter samples array (n_samples, n_params)
        lcoe: LCOE values for each sample (n_samples,)
        net_power: Net power output for each sample (n_samples,)
        capex: Total CAPEX for each sample (n_samples,)
        opex: Annual OPEX for each sample (n_samples,)
        parameter_names: Names of parameters in order
        config: Configuration used for the analysis
    """
    samples: np.ndarray
    lcoe: np.ndarray
    net_power: np.ndarray
    capex: np.ndarray
    opex: np.ndarray = field(default_factory=lambda: np.array([]))
    parameter_names: List[str] = field(default_factory=list)
    config: Optional[UncertaintyConfig] = None

    def compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for all output variables.

        Returns:
            Nested dictionary with statistics for each output variable
        """
        outputs = {
            'lcoe': self.lcoe,
            'net_power': self.net_power,
            'capex': self.capex,
            'opex': self.opex
        }

        stats = {}
        for name, values in outputs.items():
            if len(values) == 0:
                continue

            # Remove NaN values for statistics
            valid = values[~np.isnan(values)]
            if len(valid) == 0:
                stats[name] = {f'{name}_mean': np.nan}
                continue

            stats[name] = {
                f'{name}_mean': np.mean(valid),
                f'{name}_std': np.std(valid),
                f'{name}_median': np.median(valid),
                f'{name}_min': np.min(valid),
                f'{name}_max': np.max(valid),
                f'{name}_p5': np.percentile(valid, 5),
                f'{name}_p25': np.percentile(valid, 25),
                f'{name}_p75': np.percentile(valid, 75),
                f'{name}_p95': np.percentile(valid, 95),
                f'{name}_cv': np.std(valid) / np.mean(valid) if np.mean(valid) != 0 else np.nan,
                f'{name}_n_valid': len(valid),
                f'{name}_n_invalid': len(values) - len(valid)
            }

        return stats

    def get_confidence_interval(
        self, output: str = 'lcoe', confidence: float = 0.90
    ) -> tuple:
        """
        Get confidence interval for specified output.

        Args:
            output: Output variable ('lcoe', 'net_power', 'capex', 'opex')
            confidence: Confidence level (default 0.90 = 90%)

        Returns:
            Tuple of (lower, upper) bounds
        """
        values = getattr(self, output)
        valid = values[~np.isnan(values)]

        alpha = (1 - confidence) / 2
        lower = np.percentile(valid, alpha * 100)
        upper = np.percentile(valid, (1 - alpha) * 100)

        return lower, upper


def _run_single_simulation(args: tuple) -> Dict[str, float]:
    """
    Run a single OTEC simulation with given parameters.

    This is a standalone function for parallel execution.

    Args:
        args: Tuple of (sample_values, param_names, T_WW, T_CW, p_gross,
                        cost_level, base_inputs)

    Returns:
        Dictionary with simulation results
    """
    sample_values, param_names, T_WW, T_CW, p_gross, cost_level, base_inputs = args

    # Import here to avoid pickling issues in multiprocessing
    from otex.config import parameters_and_constants
    from otex.plant.sizing import otec_sizing
    from otex.economics.costs import capex_opex_lcoe

    try:
        # Get fresh inputs dict
        inputs = parameters_and_constants(p_gross=p_gross, cost_level=cost_level)

        # Apply sampled parameter values
        for name, value in zip(param_names, sample_values):
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
                # Recalculate CRF with new discount rate
                r = value
                n = inputs['lifetime']
                inputs['crf'] = r * (1 + r)**n / ((1 + r)**n - 1)
            else:
                # For cost factors and other parameters, add directly
                inputs[name] = value

        # Use fixed temperature deltas for sizing
        del_T_WW = 3.0
        del_T_CW = 3.0

        # Set distance to shore and transmission efficiency
        dist_shore = 10.0  # Default distance in km
        inputs['dist_shore'] = np.array([dist_shore])

        # Calculate transmission efficiency based on distance
        # AC for <= 50 km, DC for > 50 km
        threshold = inputs.get('threshold_AC_DC', 50.0)
        if dist_shore <= threshold:
            eff_trans = 0.979 - 1e-6 * dist_shore**2 - 9e-5 * dist_shore
        else:
            eff_trans = 0.964 - 8e-5 * dist_shore
        inputs['eff_trans'] = eff_trans

        # Convert temperatures to arrays for compatibility
        T_WW_arr = np.array([T_WW])
        T_CW_arr = np.array([T_CW])

        # Run plant sizing
        plant = otec_sizing(T_WW_arr, T_CW_arr, del_T_WW, del_T_CW, inputs, cost_level)
        _, capex, opex, lcoe = capex_opex_lcoe(plant, inputs, cost_level)

        return {
            'lcoe': float(lcoe[0]),
            'net_power': float(plant['p_net_nom'][0]),
            'capex': float(capex[0]),
            'opex': float(opex[0]),
            'valid': True
        }

    except Exception:
        return {
            'lcoe': np.nan,
            'net_power': np.nan,
            'capex': np.nan,
            'opex': np.nan,
            'valid': False
        }


class MonteCarloAnalysis:
    """
    Monte Carlo analysis with Latin Hypercube Sampling for OTEC uncertainty.

    Example:
        >>> from otex.analysis import MonteCarloAnalysis, UncertaintyConfig
        >>> config = UncertaintyConfig(n_samples=1000, seed=42)
        >>> mc = MonteCarloAnalysis(T_WW=28.0, T_CW=5.0, config=config)
        >>> results = mc.run()
        >>> stats = results.compute_statistics()
        >>> print(f"LCOE: {stats['lcoe']['lcoe_mean']:.2f} ct/kWh")
    """

    def __init__(
        self,
        T_WW: float,
        T_CW: float,
        config: Optional[UncertaintyConfig] = None,
        p_gross: float = -136000,
        cost_level: str = 'low_cost'
    ):
        """
        Initialize Monte Carlo analysis.

        Args:
            T_WW: Warm water temperature (°C)
            T_CW: Cold water temperature (°C)
            config: Uncertainty configuration (uses defaults if None)
            p_gross: Gross power output (kW, negative)
            cost_level: Cost scenario ('low_cost' or 'high_cost')
        """
        self.T_WW = T_WW
        self.T_CW = T_CW
        self.config = config or UncertaintyConfig()
        self.p_gross = p_gross
        self.cost_level = cost_level

        self._samples: Optional[np.ndarray] = None

    def _generate_samples(self) -> np.ndarray:
        """
        Generate parameter samples using Latin Hypercube Sampling.

        Returns:
            Array of shape (n_samples, n_params) with sampled values
        """
        n_samples = self.config.n_samples
        n_params = self.config.n_params
        seed = self.config.seed

        # Generate LHS samples in [0, 1]^d
        sampler = qmc.LatinHypercube(d=n_params, seed=seed)
        unit_samples = sampler.random(n=n_samples)

        # Transform to parameter space using inverse CDF
        samples = np.zeros((n_samples, n_params))
        for i, param in enumerate(self.config.parameters):
            samples[:, i] = param.ppf(unit_samples[:, i])

        return samples

    @property
    def samples(self) -> np.ndarray:
        """Get or generate LHS samples."""
        if self._samples is None:
            self._samples = self._generate_samples()
        return self._samples

    def run(self, show_progress: bool = True) -> UncertaintyResults:
        """
        Execute Monte Carlo simulation.

        Args:
            show_progress: Whether to show progress bar

        Returns:
            UncertaintyResults with all simulation outcomes
        """
        samples = self.samples
        n_samples = self.config.n_samples
        param_names = self.config.parameter_names

        # Initialize result arrays
        lcoe = np.zeros(n_samples)
        net_power = np.zeros(n_samples)
        capex = np.zeros(n_samples)
        opex = np.zeros(n_samples)

        # Prepare arguments for each simulation
        args_list = [
            (samples[i], param_names, self.T_WW, self.T_CW,
             self.p_gross, self.cost_level, None)
            for i in range(n_samples)
        ]

        if self.config.parallel and self.config.n_samples > 10:
            # Parallel execution
            n_workers = self.config.n_workers
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(_run_single_simulation, args): i
                    for i, args in enumerate(args_list)
                }

                iterator = as_completed(futures)
                if show_progress:
                    iterator = tqdm(iterator, total=n_samples, desc="Monte Carlo")

                for future in iterator:
                    i = futures[future]
                    result = future.result()
                    lcoe[i] = result['lcoe']
                    net_power[i] = result['net_power']
                    capex[i] = result['capex']
                    opex[i] = result['opex']
        else:
            # Sequential execution
            iterator = range(n_samples)
            if show_progress:
                iterator = tqdm(iterator, desc="Monte Carlo")

            for i in iterator:
                result = _run_single_simulation(args_list[i])
                lcoe[i] = result['lcoe']
                net_power[i] = result['net_power']
                capex[i] = result['capex']
                opex[i] = result['opex']

        return UncertaintyResults(
            samples=samples,
            lcoe=lcoe,
            net_power=net_power,
            capex=capex,
            opex=opex,
            parameter_names=param_names,
            config=self.config
        )

    def compute_correlations(
        self, results: UncertaintyResults, output: str = 'lcoe'
    ) -> Dict[str, float]:
        """
        Compute Spearman rank correlations between parameters and output.

        Args:
            results: Results from run()
            output: Output variable to correlate with

        Returns:
            Dictionary mapping parameter names to correlation coefficients
        """
        from scipy.stats import spearmanr

        output_values = getattr(results, output)
        valid = ~np.isnan(output_values)

        correlations = {}
        for i, name in enumerate(self.config.parameter_names):
            param_values = results.samples[valid, i]
            out_values = output_values[valid]
            corr, _ = spearmanr(param_values, out_values)
            correlations[name] = corr

        return correlations
