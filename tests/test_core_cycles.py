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


# ---------------------------------------------------------------------------
# Carnot bound regression tests
# ---------------------------------------------------------------------------

# Both Kalina and Uehara previously violated the Carnot limit by 5-6x because
# the turbine outlet enthalpy was approximated as
#     h_out_isen = h_in - 0.5 * (h_in - h_liquid_at_outlet)
# which folds the latent heat of condensation (~1100 kJ/kg for NH3) into the
# turbine work. The fix uses the entropy-balance form
#     x_isen = (s_in - s_f) / (s_g - s_f);  h_isen = h_f + x_isen * (h_g - h_f)
# This regression test guards against the bogus approximation reappearing.

import numpy as np
import pytest


class TestCarnotBound:
    """Cycle thermal efficiency must respect the external Carnot limit."""

    @staticmethod
    def _carnot_external(T_WW_C, T_CW_C):
        T_h, T_c = T_WW_C + 273.15, T_CW_C + 273.15
        return (T_h - T_c) / T_h

    @staticmethod
    def _otec_inputs():
        # Realistic OTEC operating point (Caribbean-like)
        return {
            "T_evap": 24.0, "T_cond": 9.0,
            "p_evap": 9.7,  "p_cond": 6.1,
            "T_WW": 27.0,   "T_CW": 6.0,
            "kwargs": {
                "eff_isen_turb": 0.82,
                "eff_isen_pump": 0.80,
                "rho_NH3": 640,
                "eff_pump_NH3_mech": 0.95,
                "p_gross": -100000,
            },
        }

    def test_kalina_does_not_violate_carnot(self):
        """Faithful KCS-11 (separator + recuperator + absorber) must
        respect the external Carnot limit."""
        from otex.core.cycles import KalinaCycle
        from otex.core.mixtures import AmmoniaWaterMixture
        case = self._otec_inputs()
        eta_carnot = self._carnot_external(case["T_WW"], case["T_CW"])
        mix = AmmoniaWaterMixture()
        x_basic = 0.7
        p_evap = mix.saturation_pressure(case["T_evap"], x_basic)
        p_cond = mix.saturation_pressure(case["T_cond"], x_basic)
        kwargs = {**case["kwargs"], "kalina_split_ratio": 0.30, "kalina_regen_approach_K": 5.0}

        c = KalinaCycle(ammonia_concentration=x_basic)
        s = c.calculate_cycle_states(case["T_evap"], case["T_cond"], p_evap, p_cond, kwargs)
        m = c.calculate_mass_flow(-100000, s)
        Q_evap, _Q_cond, _Q_recup = c.calculate_heat_transfer(m, s)
        # Single-stage turbine on the rich-vapor branch:
        #   W_turb = m_rich * (h_5 - h_7)
        W_turb = m["m_rich"] * (s["h_5"] - s["h_7"])
        W_pump = c.calculate_pump_power(m, s, kwargs)
        eta = (abs(W_turb) - abs(W_pump)) / abs(Q_evap)
        assert eta < eta_carnot, f"Kalina eta={eta:.3f} exceeds Carnot {eta_carnot:.3f}"
        # Turbine must produce positive work
        assert s["h_5"] - s["h_7"] > 0, "Kalina turbine produced negative work"
        # And eta must be physically meaningful (>2.5%) at OTEC ΔT
        assert eta > 0.025, f"Kalina eta={eta:.3f} unrealistically low"

    def test_uehara_does_not_violate_carnot(self):
        """Faithful Uehara (separator + 2-stage turbine + regenerator +
        absorber) must respect the external Carnot limit."""
        from otex.core.cycles import UeharaCycle
        from otex.core.mixtures import AmmoniaWaterMixture
        case = self._otec_inputs()
        eta_carnot = self._carnot_external(case["T_WW"], case["T_CW"])
        # Mixture cycles need the bubble point of the basic solution, not
        # the pure-NH3 saturation pressure (this is what plant/utils.py
        # supplies in production code).
        mix = AmmoniaWaterMixture()
        x_basic = 0.7
        p_evap = mix.saturation_pressure(case["T_evap"], x_basic)
        p_cond = mix.saturation_pressure(case["T_cond"], x_basic)
        kwargs = {**case["kwargs"], "uehara_split_ratio": 0.20, "uehara_regen_approach_K": 5.0}

        c = UeharaCycle(ammonia_concentration=x_basic)
        s = c.calculate_cycle_states(case["T_evap"], case["T_cond"], p_evap, p_cond, kwargs)
        m = c.calculate_mass_flow(-100000, s)
        Q_evap, _Q_regen, _Q_cond = c.calculate_heat_transfer(m, s)
        W_turb = m["m_rich"] * (s["h_5"] - s["h_8"])
        W_pump = c.calculate_pump_power(m, s, kwargs)
        eta = (abs(W_turb) - abs(W_pump)) / abs(Q_evap)
        assert eta < eta_carnot, f"Uehara eta={eta:.3f} exceeds Carnot {eta_carnot:.3f}"
        # And it must not be trivially small either.
        assert eta > 0.025, f"Uehara eta={eta:.3f} unrealistically low"

    def test_uehara_in_literature_band(self):
        """At canonical OTEC ΔT (24/9 °C, eff_turb 0.82, x_basic=0.85, split
        f=0.20, 5 K regenerator approach) the Uehara cycle should land in
        the 3-5% net band reported by Uehara & Ikegami (1990) and later
        OTEC literature."""
        from otex.core.cycles import UeharaCycle
        from otex.core.mixtures import AmmoniaWaterMixture
        case = self._otec_inputs()
        mix = AmmoniaWaterMixture()
        x_basic = 0.85
        p_evap = mix.saturation_pressure(case["T_evap"], x_basic)
        p_cond = mix.saturation_pressure(case["T_cond"], x_basic)
        kwargs = {**case["kwargs"], "uehara_split_ratio": 0.20, "uehara_regen_approach_K": 5.0}
        c = UeharaCycle(ammonia_concentration=x_basic)
        s = c.calculate_cycle_states(case["T_evap"], case["T_cond"], p_evap, p_cond, kwargs)
        m = c.calculate_mass_flow(-100000, s)
        Q_evap, _, _ = c.calculate_heat_transfer(m, s)
        W_turb = m["m_rich"] * (s["h_5"] - s["h_8"])
        W_pump = c.calculate_pump_power(m, s, kwargs)
        eta = (abs(W_turb) - abs(W_pump)) / abs(Q_evap)
        assert 0.030 <= eta <= 0.055, (
            f"Uehara eta={eta:.4f} outside literature band [0.030, 0.055] "
            "for OTEC ΔT~18K with x_basic=0.85"
        )

    def test_uehara_turbines_produce_positive_work(self):
        """Both HP and LP turbines must produce positive work per kg of
        rich vapor. Previously a bad T_int calculation made the LP turbine
        receive less entropy than its outlet saturated-liquid envelope and
        the result clipped to a "compression" (negative work)."""
        from otex.core.cycles import UeharaCycle
        from otex.core.mixtures import AmmoniaWaterMixture
        case = self._otec_inputs()
        mix = AmmoniaWaterMixture()
        x_basic = 0.7
        p_evap = mix.saturation_pressure(case["T_evap"], x_basic)
        p_cond = mix.saturation_pressure(case["T_cond"], x_basic)
        c = UeharaCycle(ammonia_concentration=x_basic)
        s = c.calculate_cycle_states(case["T_evap"], case["T_cond"], p_evap, p_cond, case["kwargs"])
        h_5 = float(np.atleast_1d(s["h_5"])[0])
        h_7 = float(np.atleast_1d(s["h_7"])[0])
        h_8 = float(np.atleast_1d(s["h_8"])[0])
        assert h_5 > h_7, f"HP turbine drop must be positive: h_5={h_5}, h_7={h_7}"
        assert h_7 > h_8, f"LP turbine drop must be positive: h_7={h_7}, h_8={h_8}"
        # And the intermediate pressure must lie strictly between the bounds
        p_int = float(np.atleast_1d(s["p_int"])[0])
        p_evap_v = float(np.atleast_1d(p_evap)[0]) if hasattr(p_evap, '__iter__') else p_evap
        p_cond_v = float(np.atleast_1d(p_cond)[0]) if hasattr(p_cond, '__iter__') else p_cond
        assert p_cond_v < p_int < p_evap_v, (
            f"p_int={p_int} not between p_cond={p_cond_v} and p_evap={p_evap_v}"
        )

    def test_uehara_separator_mass_balance(self):
        """f * y_rich + (1-f) * x_lean must recover x_basic exactly."""
        from otex.core.cycles import UeharaCycle
        from otex.core.mixtures import AmmoniaWaterMixture
        case = self._otec_inputs()
        mix = AmmoniaWaterMixture()
        x_basic = 0.7
        p_evap = mix.saturation_pressure(case["T_evap"], x_basic)
        p_cond = mix.saturation_pressure(case["T_cond"], x_basic)
        c = UeharaCycle(ammonia_concentration=x_basic)
        s = c.calculate_cycle_states(case["T_evap"], case["T_cond"], p_evap, p_cond, case["kwargs"])
        f = s["split_ratio"]
        x_recovered = f * float(np.atleast_1d(s["y_rich"])[0]) + (1 - f) * float(np.atleast_1d(s["x_lean"])[0])
        assert abs(x_recovered - x_basic) < 1e-9, (
            f"separator mass balance broken: recovered x={x_recovered}, expected {x_basic}"
        )

    def test_kalina_isentropic_drop_is_physical(self):
        """The isentropic Δh in the Kalina turbine must be far smaller than
        the latent heat of condensation. The previous (h_in - h_f_out)/2
        approximation gave ~640 kJ/kg; a proper entropy-balance gives
        <250 kJ/kg at OTEC ΔT ~15°C."""
        from otex.core.cycles import KalinaCycle
        from otex.core.mixtures import AmmoniaWaterMixture
        case = self._otec_inputs()
        mix = AmmoniaWaterMixture()
        x_basic = 0.7
        p_evap = mix.saturation_pressure(case["T_evap"], x_basic)
        p_cond = mix.saturation_pressure(case["T_cond"], x_basic)
        c = KalinaCycle(ammonia_concentration=x_basic)
        s = c.calculate_cycle_states(case["T_evap"], case["T_cond"], p_evap, p_cond, case["kwargs"])
        # Single-stage turbine: inlet h_5 (separator vapor), outlet h_7.
        delta_h = float(np.atleast_1d(s["h_5"] - s["h_7"])[0])
        assert 0.0 < delta_h < 250.0, (
            f"Kalina turbine drop = {delta_h:.1f} kJ/kg; expected <250 kJ/kg "
            "at OTEC ΔT (the bogus approximation gave ~525 kJ/kg)"
        )

    def test_kalina_mass_and_energy_balance(self):
        """Faithful KCS-11 must close mass and energy balances exactly:
          - separator: x_basic = f * y_rich + (1-f) * x_lean
          - recuperator: m_basic * (h_3 - h_2) ≈ m_lean * (h_6 - h_8)
        The previous implementation used hard-coded approach temperatures
        and an x_lean formula independent of the split ratio, so neither
        balance held."""
        from otex.core.cycles import KalinaCycle
        from otex.core.mixtures import AmmoniaWaterMixture
        case = self._otec_inputs()
        mix = AmmoniaWaterMixture()
        x_basic = 0.85
        p_evap = mix.saturation_pressure(case["T_evap"], x_basic)
        p_cond = mix.saturation_pressure(case["T_cond"], x_basic)
        kwargs = {**case["kwargs"], "kalina_split_ratio": 0.30, "kalina_regen_approach_K": 5.0}
        c = KalinaCycle(ammonia_concentration=x_basic)
        s = c.calculate_cycle_states(case["T_evap"], case["T_cond"], p_evap, p_cond, kwargs)
        m = c.calculate_mass_flow(-100000, s)

        # Mass balance at separator
        f = s["split_ratio"]
        y_rich = float(np.atleast_1d(s["y_rich"])[0])
        x_lean = float(np.atleast_1d(s["x_lean"])[0])
        recovered = f * y_rich + (1 - f) * x_lean
        assert abs(recovered - x_basic) < 1e-9, (
            f"separator mass balance broken: {recovered} vs {x_basic}"
        )

        # Energy balance on recuperator
        Q_basic = m["m_basic"] * (s["h_3"] - s["h_2"])
        Q_lean = m["m_lean"] * (s["h_6"] - s["h_8"])
        rel_err = abs(float(np.atleast_1d(Q_basic - Q_lean)[0])) / max(abs(float(np.atleast_1d(Q_lean)[0])), 1.0)
        assert rel_err < 1e-3, (
            f"recuperator energy balance broken: Q_basic={Q_basic}, Q_lean={Q_lean}, "
            f"relative mismatch {rel_err:.4f}"
        )


class TestMixtureSaturation:
    """Antoine equations for pure-component sat pressures must be physical."""

    def test_nh3_saturation_pressure_at_otec_temps(self):
        from otex.core.mixtures import AmmoniaWaterMixture
        m = AmmoniaWaterMixture()
        # Real NH3 saturation: 0°C -> 4.30 bar, 20°C -> 8.57 bar, 30°C -> 11.67 bar
        p_0  = m._P_sat_NH3(273.15)
        p_20 = m._P_sat_NH3(293.15)
        p_30 = m._P_sat_NH3(303.15)
        assert 4.0 < p_0  < 5.0,  f"NH3 P_sat(0 C)={p_0:.2f} bar (expected ~4.3)"
        assert 8.0 < p_20 < 9.5,  f"NH3 P_sat(20 C)={p_20:.2f} bar (expected ~8.6)"
        assert 10.5 < p_30 < 13,  f"NH3 P_sat(30 C)={p_30:.2f} bar (expected ~11.7)"

    def test_nh3_saturation_temperature_inverse_round_trip(self):
        from otex.core.mixtures import AmmoniaWaterMixture
        m = AmmoniaWaterMixture()
        # Round trip: P -> T_sat -> P should be self-consistent
        for P in [4.3, 8.0, 10.0, 12.0]:
            T_K = m._T_sat_NH3_inverse(P)
            P_check = m._P_sat_NH3(T_K)
            assert abs(P - P_check) / P < 1e-3, (
                f"NH3 inverse round-trip failed at P={P}: got {P_check:.3f}"
            )

    def test_h2o_saturation_at_100c_is_atmospheric(self):
        """Water boils at 100°C at 1 atm = 1.013 bar."""
        from otex.core.mixtures import AmmoniaWaterMixture
        m = AmmoniaWaterMixture()
        p_100 = m._P_sat_H2O(373.15)
        assert 0.95 < p_100 < 1.05, f"H2O P_sat(100 C)={p_100:.3f} bar (expected ~1.013)"


class TestRankineOpenHeatTransfer:
    """The flash (open) cycle's Q_evap was previously inherited from the base
    class formula `m * (h_3 - h_2)`, which inflated the heat duty by ~190x
    because it treated the entire warm-seawater flow as if it were vapor."""

    def test_open_cycle_q_evap_is_not_575_gw(self):
        from otex.core.cycles import RankineOpenCycle
        c = RankineOpenCycle()
        inputs = {
            "eff_isen_turb": 0.82, "eff_isen_pump": 0.80,
            "rho_NH3": 640, "eff_pump_NH3_mech": 0.95, "p_gross": -100000,
        }
        # Pure-NH3 pressures used as a placeholder; the open cycle does not
        # reference them for the water/steam states.
        s = c.calculate_cycle_states(24.0, 9.0, 9.7, 6.1, inputs)
        m = c.calculate_mass_flow(-100000, s)
        Q_evap, Q_cond = c.calculate_heat_transfer(m, s)
        Q_evap_val = float(np.atleast_1d(Q_evap)[0])
        # For 100 MW gross at ~3% thermal efficiency we expect Q_evap on the
        # order of single-digit GW. Anything > 50 GW is the broken case.
        assert 1e6 < Q_evap_val < 5e6, (
            f"Q_evap={Q_evap_val/1e3:.0f} MW out of physical range "
            "for 100 MW open OTEC (expected 2-4 GW)"
        )

    def test_open_cycle_thermal_efficiency_is_physical(self):
        from otex.core.cycles import RankineOpenCycle
        c = RankineOpenCycle()
        inputs = {
            "eff_isen_turb": 0.82, "eff_isen_pump": 0.80,
            "rho_NH3": 640, "eff_pump_NH3_mech": 0.95, "p_gross": -100000,
        }
        s = c.calculate_cycle_states(24.0, 9.0, 9.7, 6.1, inputs)
        m = c.calculate_mass_flow(-100000, s)
        Q_evap, _ = c.calculate_heat_transfer(m, s)
        m_steam = float(np.atleast_1d(s["m_steam"])[0])
        h_3 = float(np.atleast_1d(s["h_3"])[0])
        h_4 = float(np.atleast_1d(s["h_4"])[0])
        W_turb = m_steam * (h_3 - h_4)
        eta_thermal = W_turb / float(np.atleast_1d(Q_evap)[0])
        # Open OTEC at ΔT~18K typically reports 2-4% thermal efficiency
        assert 0.02 < eta_thermal < 0.06, (
            f"Open cycle thermal eta={eta_thermal:.4f} outside literature band"
        )


class TestCyclePumpPowerConvention:
    """All cycle-level calculate_pump_power methods must return only the
    working-fluid pump (plus cycle-specific auxiliaries like vacuum), not
    seawater pumping. Seawater pumping is computed by plant/sizing.py from
    pipe friction, so duplicating it here would double-count."""

    def test_hybrid_only_includes_nh3_and_vacuum(self):
        from otex.core.cycles import RankineHybridCycle
        from otex.core.fluids import get_working_fluid
        nh3 = get_working_fluid("ammonia", use_coolprop=False)
        c = RankineHybridCycle(working_fluid=nh3)
        inputs = {
            "eff_isen_turb": 0.82, "eff_isen_pump": 0.80,
            "rho_NH3": 640, "eff_pump_NH3_mech": 0.95, "p_gross": -100000,
            "T_pinch_WW": 1.0,
        }
        p_evap = float(np.asarray(nh3.saturation_pressure(24.0)).reshape(-1)[0])
        p_cond = float(np.asarray(nh3.saturation_pressure(9.0)).reshape(-1)[0])
        s = c.calculate_cycle_states(24.0, 9.0, p_evap, p_cond, inputs)
        m = c.calculate_mass_flow(-100000, s)
        W_pump = float(np.atleast_1d(c.calculate_pump_power(m, s, inputs))[0])
        # Hybrid pump = NH3 (~1-2 MW) + vacuum (~0.3 MW for 12% flash share).
        # If WW+CW were included it would jump above 30 MW.
        assert W_pump < 10_000, (
            f"Hybrid calculate_pump_power={W_pump:.0f} kW is too high - "
            "likely double-counts seawater pumping that plant/sizing.py "
            "already accounts for from pipe friction analysis."
        )
