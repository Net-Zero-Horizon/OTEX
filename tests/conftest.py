# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures for OTEX tests.
"""

import pytest
import numpy as np


@pytest.fixture
def default_inputs():
    """Default input parameters for OTEC analysis."""
    from otex.config import parameters_and_constants
    return parameters_and_constants()


@pytest.fixture
def kalina_inputs():
    """Input parameters configured for Kalina cycle."""
    from otex.config import parameters_and_constants
    return parameters_and_constants(
        cycle_type='kalina',
        year=2020
    )


@pytest.fixture
def open_cycle_inputs():
    """Input parameters configured for Open Rankine cycle."""
    from otex.config import parameters_and_constants
    return parameters_and_constants(
        cycle_type='rankine_open',
        year=2020
    )


@pytest.fixture
def ammonia_fluid():
    """Ammonia working fluid instance."""
    from otex.core.fluids import get_working_fluid
    return get_working_fluid('ammonia', use_coolprop=False)


@pytest.fixture
def rankine_cycle(ammonia_fluid):
    """Rankine closed cycle instance with ammonia."""
    from otex.core.cycles import get_thermodynamic_cycle
    return get_thermodynamic_cycle('rankine_closed', working_fluid=ammonia_fluid)


@pytest.fixture
def sample_temperatures():
    """Sample temperature data for testing."""
    return {
        'T_WW': np.array([28.0, 27.5, 27.0, 26.5]),  # Warm water temperatures [°C]
        'T_CW': np.array([5.0, 5.5, 6.0, 6.5]),      # Cold water temperatures [°C]
        'T_WW_design': np.array([[26.0], [27.0], [28.0]]),  # min, med, max
        'T_CW_design': np.array([[7.0], [6.0], [5.0]]),     # max, med, min (inverted)
    }


@pytest.fixture
def typical_otec_conditions():
    """Typical OTEC operating conditions."""
    return {
        'T_evap': 24.0,      # Evaporation temperature [°C]
        'T_cond': 12.0,      # Condensation temperature [°C]
        'T_WW_in': 28.0,     # Warm water inlet [°C]
        'T_CW_in': 5.0,      # Cold water inlet [°C]
        'dT_WW': 3.0,        # Warm water temperature drop [°C]
        'dT_CW': 3.0,        # Cold water temperature rise [°C]
        'p_gross': -136000,  # Gross power [kW]
    }
