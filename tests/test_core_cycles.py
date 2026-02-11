# -*- coding: utf-8 -*-
"""
Tests for otex.core.cycles module.
"""

import pytest
import numpy as np
from otex.core.cycles import get_thermodynamic_cycle
from otex.core.fluids import get_working_fluid


class TestGetThermodynamicCycle:
    """Tests for get_thermodynamic_cycle factory function."""

    def test_rankine_closed_default(self):
        """Should return Rankine closed cycle by default."""
        wf = get_working_fluid('ammonia', use_coolprop=False)
        cycle = get_thermodynamic_cycle('rankine_closed', working_fluid=wf)

        assert cycle is not None
        assert hasattr(cycle, 'calculate_cycle_states')

    def test_rankine_open(self):
        """Should support Rankine open cycle."""
        cycle = get_thermodynamic_cycle('rankine_open')

        assert cycle is not None

    def test_kalina_cycle(self):
        """Should support Kalina cycle."""
        cycle = get_thermodynamic_cycle('kalina', ammonia_concentration=0.7)

        assert cycle is not None

    def test_uehara_cycle(self):
        """Should support Uehara cycle."""
        cycle = get_thermodynamic_cycle('uehara', ammonia_concentration=0.7)

        assert cycle is not None

    def test_invalid_cycle_type(self):
        """Should raise error for invalid cycle type."""
        with pytest.raises((ValueError, KeyError)):
            get_thermodynamic_cycle('invalid_cycle')


class TestRankineClosedCycle:
    """Tests for Rankine closed cycle calculations."""

    @pytest.fixture
    def cycle(self):
        """Create Rankine closed cycle with ammonia."""
        wf = get_working_fluid('ammonia', use_coolprop=False)
        return get_thermodynamic_cycle('rankine_closed', working_fluid=wf)

    def test_has_calculate_method(self, cycle):
        """Cycle should have calculate_cycle_states method."""
        assert hasattr(cycle, 'calculate_cycle_states')
        assert callable(cycle.calculate_cycle_states)

    def test_has_heat_transfer_method(self, cycle):
        """Cycle should have calculate_heat_transfer method."""
        assert hasattr(cycle, 'calculate_heat_transfer')
        assert callable(cycle.calculate_heat_transfer)

    def test_has_mass_flow_method(self, cycle):
        """Cycle should have calculate_mass_flow method."""
        assert hasattr(cycle, 'calculate_mass_flow')
        assert callable(cycle.calculate_mass_flow)

    def test_has_pump_power_method(self, cycle):
        """Cycle should have calculate_pump_power method."""
        assert hasattr(cycle, 'calculate_pump_power')
        assert callable(cycle.calculate_pump_power)

    def test_fluid_property(self, cycle):
        """Cycle should have reference to working fluid."""
        assert hasattr(cycle, 'fluid')
        assert cycle.fluid is not None


class TestRankineOpenCycle:
    """Tests for Rankine open (flash) cycle."""

    @pytest.fixture
    def cycle(self):
        """Create Rankine open cycle."""
        return get_thermodynamic_cycle('rankine_open')

    def test_cycle_exists(self, cycle):
        """Cycle should be created successfully."""
        assert cycle is not None

    def test_has_calculate_method(self, cycle):
        """Should have calculate_cycle_states method."""
        assert hasattr(cycle, 'calculate_cycle_states')


class TestKalinaCycle:
    """Tests for Kalina cycle."""

    @pytest.fixture
    def cycle(self):
        """Create Kalina cycle."""
        return get_thermodynamic_cycle('kalina', ammonia_concentration=0.7)

    def test_cycle_exists(self, cycle):
        """Cycle should be created successfully."""
        assert cycle is not None

    def test_has_calculate_method(self, cycle):
        """Should have calculate_cycle_states method."""
        assert hasattr(cycle, 'calculate_cycle_states')

    def test_has_mixture(self, cycle):
        """Kalina cycle should have mixture component."""
        # Kalina uses NH3-H2O mixture internally
        assert hasattr(cycle, 'mixture') or hasattr(cycle, 'x_basic')

    def test_ammonia_concentration_stored(self, cycle):
        """Ammonia concentration should be accessible via x_basic."""
        if hasattr(cycle, 'x_basic'):
            assert 0 < cycle.x_basic < 1


class TestUeharaCycle:
    """Tests for Uehara cycle."""

    @pytest.fixture
    def cycle(self):
        """Create Uehara cycle."""
        return get_thermodynamic_cycle('uehara', ammonia_concentration=0.7)

    def test_cycle_exists(self, cycle):
        """Cycle should be created successfully."""
        assert cycle is not None

    def test_has_calculate_method(self, cycle):
        """Should have calculate_cycle_states method."""
        assert hasattr(cycle, 'calculate_cycle_states')


class TestCycleNames:
    """Tests for cycle identification."""

    def test_rankine_closed_name(self):
        """Rankine closed cycle should have identifiable name."""
        wf = get_working_fluid('ammonia', use_coolprop=False)
        cycle = get_thermodynamic_cycle('rankine_closed', working_fluid=wf)

        assert hasattr(cycle, 'name')
        assert 'rankine' in cycle.name.lower() or 'closed' in cycle.name.lower()

    def test_rankine_open_name(self):
        """Rankine open cycle should have identifiable name."""
        cycle = get_thermodynamic_cycle('rankine_open')

        assert hasattr(cycle, 'name')

    def test_kalina_name(self):
        """Kalina cycle should have identifiable name."""
        cycle = get_thermodynamic_cycle('kalina', ammonia_concentration=0.7)

        assert hasattr(cycle, 'name')
        assert 'kalina' in cycle.name.lower()
