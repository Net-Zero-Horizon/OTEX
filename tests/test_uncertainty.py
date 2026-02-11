# -*- coding: utf-8 -*-
"""
Tests for otex.analysis uncertainty module.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from otex.analysis.distributions import (
    UncertainParameter,
    UncertaintyConfig,
    get_default_parameters,
)
from otex.analysis.uncertainty import (
    MonteCarloAnalysis,
    UncertaintyResults,
)
from otex.analysis.sensitivity import (
    TornadoAnalysis,
    TornadoResults,
)


class TestUncertainParameter:
    """Tests for UncertainParameter dataclass."""

    def test_uniform_parameter_creation(self):
        """Test creating a uniform distribution parameter."""
        param = UncertainParameter(
            name='test_param',
            nominal=1.0,
            distribution='uniform',
            bounds=(0.8, 1.2)
        )
        assert param.name == 'test_param'
        assert param.nominal == 1.0
        assert param.distribution == 'uniform'
        assert param.bounds == (0.8, 1.2)

    def test_normal_parameter_creation(self):
        """Test creating a normal distribution parameter."""
        param = UncertainParameter(
            name='efficiency',
            nominal=0.82,
            distribution='normal',
            bounds=(0.82, 0.04),  # mean, std
            category='efficiency'
        )
        assert param.distribution == 'normal'
        assert param.category == 'efficiency'

    def test_uniform_bounds_validation(self):
        """Uniform bounds must have min < max."""
        with pytest.raises(ValueError, match="min < max"):
            UncertainParameter(
                name='bad',
                nominal=1.0,
                distribution='uniform',
                bounds=(1.2, 0.8)  # Invalid: min > max
            )

    def test_normal_std_validation(self):
        """Normal distribution requires positive std."""
        with pytest.raises(ValueError, match="positive std"):
            UncertainParameter(
                name='bad',
                nominal=1.0,
                distribution='normal',
                bounds=(1.0, -0.1)  # Invalid: negative std
            )

    def test_sample_uniform(self):
        """Test sampling from uniform distribution."""
        param = UncertainParameter(
            name='test',
            nominal=1.0,
            distribution='uniform',
            bounds=(0.5, 1.5)
        )
        rng = np.random.default_rng(42)
        samples = param.sample(1000, rng)

        assert len(samples) == 1000
        assert np.all(samples >= 0.5)
        assert np.all(samples <= 1.5)

    def test_sample_normal(self):
        """Test sampling from normal distribution."""
        param = UncertainParameter(
            name='test',
            nominal=0.82,
            distribution='normal',
            bounds=(0.82, 0.04)
        )
        rng = np.random.default_rng(42)
        samples = param.sample(10000, rng)

        # Check mean and std are approximately correct
        assert abs(np.mean(samples) - 0.82) < 0.01
        assert abs(np.std(samples) - 0.04) < 0.005

    def test_ppf_uniform(self):
        """Test percent point function for uniform."""
        param = UncertainParameter(
            name='test',
            nominal=1.0,
            distribution='uniform',
            bounds=(0.0, 2.0)
        )
        quantiles = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        values = param.ppf(quantiles)

        np.testing.assert_array_almost_equal(
            values, [0.0, 0.5, 1.0, 1.5, 2.0]
        )

    def test_ppf_normal(self):
        """Test percent point function for normal."""
        param = UncertainParameter(
            name='test',
            nominal=0.0,
            distribution='normal',
            bounds=(0.0, 1.0)  # mean=0, std=1
        )
        # Median of standard normal is 0
        assert abs(param.ppf(np.array([0.5]))[0]) < 1e-10


class TestUncertaintyConfig:
    """Tests for UncertaintyConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = UncertaintyConfig()
        assert config.n_samples == 1000
        assert config.seed == 42
        assert config.parallel is True
        assert len(config.parameters) > 0

    def test_n_params_property(self):
        """Test n_params property."""
        config = UncertaintyConfig()
        assert config.n_params == len(config.parameters)

    def test_parameter_names_property(self):
        """Test parameter_names property."""
        config = UncertaintyConfig()
        names = config.parameter_names
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_custom_parameters(self):
        """Test config with custom parameters."""
        params = [
            UncertainParameter('p1', 1.0, 'uniform', (0.5, 1.5)),
            UncertainParameter('p2', 2.0, 'uniform', (1.0, 3.0)),
        ]
        config = UncertaintyConfig(parameters=params, n_samples=500)

        assert config.n_params == 2
        assert config.n_samples == 500
        assert config.parameter_names == ['p1', 'p2']

    def test_get_bounds_array(self):
        """Test bounds array generation."""
        params = [
            UncertainParameter('p1', 1.0, 'uniform', (0.0, 2.0)),
            UncertainParameter('p2', 0.5, 'uniform', (0.3, 0.7)),
        ]
        config = UncertaintyConfig(parameters=params)
        bounds = config.get_bounds_array()

        assert bounds.shape == (2, 2)
        np.testing.assert_array_equal(bounds[0], [0.0, 2.0])
        np.testing.assert_array_equal(bounds[1], [0.3, 0.7])

    def test_get_problem_dict(self):
        """Test SALib problem dictionary generation."""
        config = UncertaintyConfig()
        problem = config.get_problem_dict()

        assert 'num_vars' in problem
        assert 'names' in problem
        assert 'bounds' in problem
        assert problem['num_vars'] == config.n_params
        assert len(problem['names']) == config.n_params
        assert len(problem['bounds']) == config.n_params


class TestDefaultParameters:
    """Tests for default parameter set."""

    def test_default_parameters_exist(self):
        """Test that default parameters are defined."""
        params = get_default_parameters()
        assert len(params) > 0

    def test_default_parameters_names(self):
        """Test that expected parameters are present."""
        params = get_default_parameters()
        names = [p.name for p in params]

        expected = [
            'turbine_isentropic_efficiency',
            'pump_isentropic_efficiency',
            'U_evap',
            'U_cond',
            'discount_rate',
        ]
        for exp in expected:
            assert exp in names, f"Missing parameter: {exp}"

    def test_default_parameters_categories(self):
        """Test that parameters have valid categories."""
        params = get_default_parameters()
        valid_categories = {'thermodynamic', 'economic', 'efficiency'}

        for p in params:
            assert p.category in valid_categories


class TestMonteCarloAnalysis:
    """Tests for MonteCarloAnalysis class."""

    def test_initialization(self):
        """Test MonteCarloAnalysis initialization."""
        config = UncertaintyConfig(n_samples=10)
        mc = MonteCarloAnalysis(
            T_WW=28.0,
            T_CW=5.0,
            config=config
        )
        assert mc.T_WW == 28.0
        assert mc.T_CW == 5.0
        assert mc.config.n_samples == 10

    def test_lhs_sampling(self):
        """Test that LHS sampling produces correct shape."""
        config = UncertaintyConfig(n_samples=100, seed=42)
        mc = MonteCarloAnalysis(T_WW=28.0, T_CW=5.0, config=config)

        samples = mc.samples
        assert samples.shape == (100, config.n_params)

    def test_lhs_reproducibility(self):
        """Test that LHS sampling is reproducible with seed."""
        config = UncertaintyConfig(n_samples=50, seed=123)

        mc1 = MonteCarloAnalysis(T_WW=28.0, T_CW=5.0, config=config)
        mc2 = MonteCarloAnalysis(T_WW=28.0, T_CW=5.0, config=config)

        np.testing.assert_array_equal(mc1.samples, mc2.samples)

    def test_lhs_coverage(self):
        """Test that LHS provides good coverage of parameter space."""
        params = [
            UncertainParameter('p1', 0.5, 'uniform', (0.0, 1.0)),
        ]
        config = UncertaintyConfig(parameters=params, n_samples=100, seed=42)
        mc = MonteCarloAnalysis(T_WW=28.0, T_CW=5.0, config=config)

        samples = mc.samples[:, 0]

        # LHS should have samples in each 1% bin
        # Check that we cover most of the range
        hist, _ = np.histogram(samples, bins=10, range=(0, 1))
        assert np.all(hist >= 5), "LHS should provide good coverage"

    @pytest.mark.slow
    def test_run_small_sample(self):
        """Test running Monte Carlo with small sample."""
        config = UncertaintyConfig(n_samples=5, seed=42, parallel=False)
        mc = MonteCarloAnalysis(
            T_WW=28.0,
            T_CW=5.0,
            config=config,
            p_gross=-136000,
            cost_level='low_cost'
        )

        results = mc.run(show_progress=False)

        assert isinstance(results, UncertaintyResults)
        assert len(results.lcoe) == 5
        assert len(results.net_power) == 5
        assert len(results.capex) == 5


class TestUncertaintyResults:
    """Tests for UncertaintyResults dataclass."""

    def test_compute_statistics(self):
        """Test statistics computation."""
        results = UncertaintyResults(
            samples=np.random.randn(100, 3),
            lcoe=np.random.uniform(10, 20, 100),
            net_power=np.random.uniform(-100000, -50000, 100),
            capex=np.random.uniform(1e8, 2e8, 100),
            opex=np.random.uniform(1e6, 2e6, 100),
            parameter_names=['p1', 'p2', 'p3']
        )

        stats = results.compute_statistics()

        assert 'lcoe' in stats
        assert 'lcoe_mean' in stats['lcoe']
        assert 'lcoe_std' in stats['lcoe']
        assert 'lcoe_p5' in stats['lcoe']
        assert 'lcoe_p95' in stats['lcoe']

    def test_compute_statistics_with_nan(self):
        """Test statistics handle NaN values."""
        lcoe = np.array([10.0, 15.0, np.nan, 20.0, 12.0])
        results = UncertaintyResults(
            samples=np.zeros((5, 1)),
            lcoe=lcoe,
            net_power=np.zeros(5),
            capex=np.zeros(5),
            parameter_names=['p1']
        )

        stats = results.compute_statistics()
        assert stats['lcoe']['lcoe_n_valid'] == 4
        assert stats['lcoe']['lcoe_n_invalid'] == 1

    def test_get_confidence_interval(self):
        """Test confidence interval calculation."""
        np.random.seed(42)
        lcoe = np.random.normal(15, 2, 1000)
        results = UncertaintyResults(
            samples=np.zeros((1000, 1)),
            lcoe=lcoe,
            net_power=np.zeros(1000),
            capex=np.zeros(1000),
            parameter_names=['p1']
        )

        low, high = results.get_confidence_interval('lcoe', confidence=0.90)
        assert low < 15 < high
        assert high - low < 10  # Should be reasonable interval


class TestTornadoAnalysis:
    """Tests for TornadoAnalysis class."""

    def test_initialization(self):
        """Test TornadoAnalysis initialization."""
        tornado = TornadoAnalysis(T_WW=28.0, T_CW=5.0)
        assert tornado.T_WW == 28.0
        assert tornado.T_CW == 5.0
        assert tornado.variation_pct == 10.0

    @pytest.mark.slow
    def test_run_tornado(self):
        """Test running tornado analysis."""
        # Use minimal parameters for speed
        params = [
            UncertainParameter('discount_rate', 0.10, 'uniform', (0.05, 0.15)),
        ]
        config = UncertaintyConfig(parameters=params)

        tornado = TornadoAnalysis(
            T_WW=28.0,
            T_CW=5.0,
            config=config,
            p_gross=-136000,
            cost_level='low_cost'
        )

        results = tornado.run(output='lcoe', show_progress=False)

        assert isinstance(results, TornadoResults)
        assert len(results.parameter_names) == 1
        assert results.baseline > 0
        assert len(results.swings) == 1


class TestTornadoResults:
    """Tests for TornadoResults dataclass."""

    def test_swing_calculation(self):
        """Test that swings are calculated correctly."""
        results = TornadoResults(
            parameter_names=['p1', 'p2'],
            low_values=np.array([10.0, 12.0]),
            high_values=np.array([20.0, 18.0]),
            baseline=15.0
        )

        np.testing.assert_array_equal(results.swings, [10.0, 6.0])

    def test_get_ranking(self):
        """Test parameter ranking by swing."""
        results = TornadoResults(
            parameter_names=['p1', 'p2', 'p3'],
            low_values=np.array([10.0, 14.0, 5.0]),
            high_values=np.array([20.0, 16.0, 25.0]),
            baseline=15.0
        )

        ranking = results.get_ranking()
        # p3 has swing 20, p1 has swing 10, p2 has swing 2
        assert ranking[0][0] == 'p3'
        assert ranking[1][0] == 'p1'
        assert ranking[2][0] == 'p2'

    def test_to_dict(self):
        """Test dictionary conversion."""
        results = TornadoResults(
            parameter_names=['p1'],
            low_values=np.array([10.0]),
            high_values=np.array([20.0]),
            baseline=15.0,
            output_name='lcoe'
        )

        d = results.to_dict()
        assert 'baseline' in d
        assert 'ranking' in d
        assert d['output'] == 'lcoe'


class TestVisualization:
    """Tests for visualization functions."""

    def test_plot_histogram_creation(self):
        """Test histogram plot creates without error."""
        from otex.analysis.visualization import plot_histogram

        results = UncertaintyResults(
            samples=np.zeros((100, 1)),
            lcoe=np.random.uniform(10, 20, 100),
            net_power=np.zeros(100),
            capex=np.zeros(100),
            parameter_names=['p1']
        )

        ax = plot_histogram(results, output='lcoe')
        assert ax is not None
        plt.close('all')

    def test_plot_tornado_creation(self):
        """Test tornado plot creates without error."""
        from otex.analysis.visualization import plot_tornado

        results = TornadoResults(
            parameter_names=['p1', 'p2', 'p3'],
            low_values=np.array([10.0, 12.0, 8.0]),
            high_values=np.array([20.0, 18.0, 22.0]),
            baseline=15.0
        )

        ax = plot_tornado(results)
        assert ax is not None
        plt.close('all')

    def test_plot_cumulative_distribution(self):
        """Test CDF plot creates without error."""
        from otex.analysis.visualization import plot_cumulative_distribution

        results = UncertaintyResults(
            samples=np.zeros((100, 1)),
            lcoe=np.random.uniform(10, 20, 100),
            net_power=np.zeros(100),
            capex=np.zeros(100),
            parameter_names=['p1']
        )

        ax = plot_cumulative_distribution(results, output='lcoe')
        assert ax is not None
        plt.close('all')


# Optional Sobol tests (require SALib)
class TestSobolAnalysis:
    """Tests for SobolAnalysis class."""

    @pytest.fixture
    def check_salib(self):
        """Check if SALib is available."""
        pytest.importorskip("SALib")

    def test_initialization(self, check_salib):
        """Test SobolAnalysis initialization."""
        from otex.analysis.sensitivity import SobolAnalysis

        sobol = SobolAnalysis(T_WW=28.0, T_CW=5.0, n_samples=64)
        assert sobol.T_WW == 28.0
        assert sobol.n_samples == 64

    def test_sobol_results_ranking(self):
        """Test SobolResults ranking."""
        from otex.analysis.sensitivity import SobolResults

        results = SobolResults(
            S1=np.array([0.1, 0.3, 0.2]),
            ST=np.array([0.15, 0.4, 0.25]),
            parameter_names=['p1', 'p2', 'p3']
        )

        ranking = results.get_ranking('ST')
        assert ranking[0][0] == 'p2'
        assert ranking[1][0] == 'p3'
        assert ranking[2][0] == 'p1'

    def test_sobol_indices_valid_range(self):
        """Test that Sobol indices are in valid range."""
        from otex.analysis.sensitivity import SobolResults

        results = SobolResults(
            S1=np.array([0.1, 0.3, 0.5]),
            ST=np.array([0.15, 0.4, 0.6]),
            parameter_names=['p1', 'p2', 'p3']
        )

        # S1 should sum to ~1 (or less with interactions)
        # ST should be >= S1
        assert np.all(results.ST >= results.S1 - 0.01)  # Allow small numerical error
