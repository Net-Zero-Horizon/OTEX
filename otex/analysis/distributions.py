# -*- coding: utf-8 -*-
"""
Uncertainty distributions for OTEC plant parameters.

Defines uncertain parameters with their nominal values and probability distributions
for use in Monte Carlo simulations and sensitivity analyses.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Tuple, Optional
import numpy as np
from scipy import stats


@dataclass
class UncertainParameter:
    """
    Defines an uncertain parameter with its distribution.

    Attributes:
        name: Parameter name (must match key in inputs dict)
        nominal: Nominal/central value
        distribution: Type of probability distribution
        bounds: For uniform/triangular: (min, max); for normal: (mean, std)
        category: Classification for grouping parameters
    """
    name: str
    nominal: float
    distribution: Literal['uniform', 'normal', 'triangular'] = 'uniform'
    bounds: Tuple[float, float] = (0.0, 1.0)
    category: Literal['thermodynamic', 'economic', 'efficiency'] = 'thermodynamic'

    def __post_init__(self):
        """Validate parameter configuration."""
        if self.distribution == 'uniform':
            if self.bounds[0] >= self.bounds[1]:
                raise ValueError(
                    f"Parameter '{self.name}': uniform bounds must satisfy min < max"
                )
        elif self.distribution == 'normal':
            if self.bounds[1] <= 0:
                raise ValueError(
                    f"Parameter '{self.name}': normal distribution requires positive std"
                )
        elif self.distribution == 'triangular':
            if self.bounds[0] >= self.bounds[1]:
                raise ValueError(
                    f"Parameter '{self.name}': triangular bounds must satisfy min < max"
                )

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Generate n samples from this parameter's distribution.

        Args:
            n: Number of samples
            rng: Random number generator (optional)

        Returns:
            Array of n samples
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.distribution == 'uniform':
            return rng.uniform(self.bounds[0], self.bounds[1], n)
        elif self.distribution == 'normal':
            mean, std = self.bounds
            return rng.normal(mean, std, n)
        elif self.distribution == 'triangular':
            left, right = self.bounds
            mode = self.nominal
            # Clip mode to be within bounds
            mode = np.clip(mode, left, right)
            return rng.triangular(left, mode, right, n)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """
        Percent point function (inverse CDF) for use with LHS.

        Args:
            q: Quantiles in [0, 1]

        Returns:
            Values corresponding to quantiles
        """
        if self.distribution == 'uniform':
            return stats.uniform.ppf(
                q, loc=self.bounds[0], scale=self.bounds[1] - self.bounds[0]
            )
        elif self.distribution == 'normal':
            mean, std = self.bounds
            return stats.norm.ppf(q, loc=mean, scale=std)
        elif self.distribution == 'triangular':
            left, right = self.bounds
            mode = np.clip(self.nominal, left, right)
            # scipy triangular uses c = (mode - left) / (right - left)
            c = (mode - left) / (right - left) if right != left else 0.5
            return stats.triang.ppf(q, c, loc=left, scale=right - left)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")


def get_default_parameters() -> List[UncertainParameter]:
    """
    Return the default set of uncertain parameters for OTEC uncertainty analysis.

    Returns:
        List of UncertainParameter objects with default configurations
    """
    return [
        # Efficiency parameters (normal distributions centered on nominal)
        UncertainParameter(
            name='turbine_isentropic_efficiency',
            nominal=0.82,
            distribution='normal',
            bounds=(0.82, 0.041),  # mean=0.82, std=0.041 (~5%)
            category='efficiency'
        ),
        UncertainParameter(
            name='pump_isentropic_efficiency',
            nominal=0.80,
            distribution='normal',
            bounds=(0.80, 0.04),  # mean=0.80, std=0.04 (~5%)
            category='efficiency'
        ),

        # Heat transfer coefficients (uniform ±10%)
        UncertainParameter(
            name='U_evap',
            nominal=4.5,
            distribution='uniform',
            bounds=(4.05, 4.95),  # ±10%
            category='thermodynamic'
        ),
        UncertainParameter(
            name='U_cond',
            nominal=3.5,
            distribution='uniform',
            bounds=(3.15, 3.85),  # ±10%
            category='thermodynamic'
        ),

        # CAPEX cost factors (uniform, asymmetric upward bias for uncertainty)
        UncertainParameter(
            name='capex_turbine_factor',
            nominal=1.0,
            distribution='uniform',
            bounds=(0.8, 1.5),
            category='economic'
        ),
        UncertainParameter(
            name='capex_HX_factor',
            nominal=1.0,
            distribution='uniform',
            bounds=(0.8, 1.5),
            category='economic'
        ),
        UncertainParameter(
            name='capex_pump_factor',
            nominal=1.0,
            distribution='uniform',
            bounds=(0.8, 1.5),
            category='economic'
        ),
        UncertainParameter(
            name='capex_structure_factor',
            nominal=1.0,
            distribution='uniform',
            bounds=(0.8, 1.5),
            category='economic'
        ),

        # OPEX factor (uniform, smaller range)
        UncertainParameter(
            name='opex_factor',
            nominal=1.0,
            distribution='uniform',
            bounds=(0.8, 1.2),
            category='economic'
        ),

        # Discount rate (uniform, wide range for economic scenarios)
        UncertainParameter(
            name='discount_rate',
            nominal=0.10,
            distribution='uniform',
            bounds=(0.05, 0.15),
            category='economic'
        ),
    ]


@dataclass
class UncertaintyConfig:
    """
    Configuration for uncertainty analysis.

    Attributes:
        parameters: List of uncertain parameters (uses defaults if not specified)
        n_samples: Number of Monte Carlo samples
        seed: Random seed for reproducibility
        parallel: Whether to use parallel processing
        n_workers: Number of parallel workers (None = auto)
    """
    parameters: List[UncertainParameter] = field(default_factory=get_default_parameters)
    n_samples: int = 1000
    seed: int = 42
    parallel: bool = True
    n_workers: Optional[int] = None

    @property
    def n_params(self) -> int:
        """Number of uncertain parameters."""
        return len(self.parameters)

    @property
    def parameter_names(self) -> List[str]:
        """List of parameter names."""
        return [p.name for p in self.parameters]

    def get_bounds_array(self) -> np.ndarray:
        """
        Get parameter bounds as 2D array for SALib.

        Returns:
            Array of shape (n_params, 2) with [min, max] for each parameter
        """
        bounds = np.zeros((self.n_params, 2))
        for i, p in enumerate(self.parameters):
            if p.distribution == 'uniform':
                bounds[i] = [p.bounds[0], p.bounds[1]]
            elif p.distribution == 'normal':
                # Use ±3 sigma as bounds for SALib
                mean, std = p.bounds
                bounds[i] = [mean - 3*std, mean + 3*std]
            elif p.distribution == 'triangular':
                bounds[i] = [p.bounds[0], p.bounds[1]]
        return bounds

    def get_problem_dict(self) -> dict:
        """
        Get SALib problem definition dictionary.

        Returns:
            Dictionary with 'num_vars', 'names', and 'bounds'
        """
        return {
            'num_vars': self.n_params,
            'names': self.parameter_names,
            'bounds': self.get_bounds_array().tolist()
        }
