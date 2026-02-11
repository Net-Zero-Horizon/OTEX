# -*- coding: utf-8 -*-
"""
Tests for otex.core.fluids module.
"""

import pytest
import numpy as np
from otex.core.fluids import get_working_fluid


class TestGetWorkingFluid:
    """Tests for get_working_fluid factory function."""

    def test_ammonia_default(self):
        """Should return ammonia fluid by default."""
        fluid = get_working_fluid()
        assert fluid is not None
        assert 'ammonia' in fluid.__class__.__name__.lower() or 'nh3' in fluid.name.lower()

    def test_ammonia_explicit(self):
        """Should return ammonia when explicitly requested."""
        fluid = get_working_fluid('ammonia', use_coolprop=False)
        assert fluid is not None

    @pytest.mark.requires_coolprop
    @pytest.mark.parametrize('fluid_type', ['r134a', 'r245fa', 'propane', 'isobutane'])
    def test_other_fluids_with_coolprop(self, fluid_type):
        """Should support multiple fluid types with CoolProp."""
        try:
            fluid = get_working_fluid(fluid_type, use_coolprop=True)
            assert fluid is not None
        except (ImportError, ValueError) as e:
            pytest.skip(f"Fluid {fluid_type} not available: {e}")


class TestAmmoniaProperties:
    """Tests for ammonia fluid properties."""

    @pytest.fixture
    def ammonia(self):
        """Get ammonia fluid instance."""
        return get_working_fluid('ammonia', use_coolprop=False)

    def test_saturation_pressure_scalar(self, ammonia):
        """Saturation pressure should work with scalar input."""
        T = 25.0  # °C
        P = ammonia.saturation_pressure(T)

        assert P is not None
        P_val = float(np.atleast_1d(P)[0])
        assert P_val > 0
        # At 25°C, ammonia P_sat should be around 10 bar
        assert 8 < P_val < 12

    def test_saturation_pressure_array(self, ammonia):
        """Saturation pressure should work with array input."""
        T = np.array([20.0, 25.0, 30.0])
        P = ammonia.saturation_pressure(T)

        P = np.atleast_1d(P)
        assert len(P) == 3
        assert all(p > 0 for p in P)
        # Pressure should increase with temperature
        assert P[0] < P[1] < P[2]

    def test_saturation_temperature_scalar(self, ammonia):
        """Saturation temperature should work with scalar input."""
        P = 10.0  # bar (approx 1000 kPa)
        T = ammonia.saturation_temperature(P)

        T_val = float(np.atleast_1d(T)[0])
        assert T_val is not None
        # At 10 bar, T_sat should be around 25°C (298 K)
        # The polynomial returns Kelvin
        assert 290 < T_val < 310 or 17 < T_val < 37  # K or °C

    def test_enthalpy_liquid(self, ammonia):
        """Liquid enthalpy should be calculable."""
        T = 25.0
        h = ammonia.enthalpy_liquid(T)

        assert h is not None
        assert isinstance(h, (int, float, np.ndarray))

    def test_enthalpy_vapor(self, ammonia):
        """Vapor enthalpy should be calculable."""
        T = 25.0
        h = ammonia.enthalpy_vapor(T)

        assert h is not None
        # Vapor enthalpy should be greater than liquid enthalpy
        h_liq = ammonia.enthalpy_liquid(T)
        h_val = float(np.atleast_1d(h)[0])
        h_liq_val = float(np.atleast_1d(h_liq)[0])
        assert h_val > h_liq_val

    def test_entropy_liquid(self, ammonia):
        """Liquid entropy should be calculable."""
        T = 25.0
        s = ammonia.entropy_liquid(T)

        assert s is not None

    def test_entropy_vapor(self, ammonia):
        """Vapor entropy should be calculable."""
        T = 25.0
        s = ammonia.entropy_vapor(T)

        assert s is not None
        # Vapor entropy should be greater than liquid entropy
        s_liq = ammonia.entropy_liquid(T)
        s_val = float(np.atleast_1d(s)[0])
        s_liq_val = float(np.atleast_1d(s_liq)[0])
        assert s_val > s_liq_val

    def test_latent_heat_positive(self, ammonia):
        """Latent heat (h_vapor - h_liquid) should be positive."""
        T = 25.0
        h_vap = ammonia.enthalpy_vapor(T)
        h_liq = ammonia.enthalpy_liquid(T)

        L = float(np.atleast_1d(h_vap)[0]) - float(np.atleast_1d(h_liq)[0])

        assert L > 0
        # Latent heat of ammonia at 25°C should be around 1100-1200 kJ/kg
        assert 1000 < L < 1400


class TestFluidPhysicalConsistency:
    """Tests for physical consistency of fluid properties."""

    @pytest.fixture
    def ammonia(self):
        return get_working_fluid('ammonia', use_coolprop=False)

    def test_clausius_clapeyron_consistency(self, ammonia):
        """Pressure should increase with temperature (Clausius-Clapeyron)."""
        T1 = 20.0
        T2 = 30.0

        P1 = float(np.atleast_1d(ammonia.saturation_pressure(T1))[0])
        P2 = float(np.atleast_1d(ammonia.saturation_pressure(T2))[0])

        assert P2 > P1

    def test_enthalpy_temperature_relation(self, ammonia):
        """Enthalpy should generally increase with temperature."""
        T = np.array([15.0, 20.0, 25.0, 30.0])
        h_liq = np.array([float(np.atleast_1d(ammonia.enthalpy_liquid(t))[0]) for t in T])

        # Liquid enthalpy should increase with temperature
        for i in range(len(T) - 1):
            assert h_liq[i + 1] > h_liq[i]

    def test_latent_heat_decreases_with_temperature(self, ammonia):
        """Latent heat should decrease as temperature increases (approaching critical point)."""
        T = np.array([10.0, 20.0, 30.0])
        L = []
        for t in T:
            h_vap = float(np.atleast_1d(ammonia.enthalpy_vapor(t))[0])
            h_liq = float(np.atleast_1d(ammonia.enthalpy_liquid(t))[0])
            L.append(h_vap - h_liq)

        # Latent heat should decrease with temperature
        for i in range(len(T) - 1):
            assert L[i + 1] < L[i]
