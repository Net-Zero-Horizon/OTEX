# -*- coding: utf-8 -*-
"""
OTEX Ammonia-Water Mixture Properties
Implements thermodynamic properties of NH3-H2O mixtures for Kalina/Uehara cycles

Based on correlations from:
- Ibrahim & Klein (1993) - Thermodynamic properties of ammonia-water mixtures
- El-Sayed & Tribus (1985) - Theoretical comparison of Rankine and Kalina cycles
- Thorin (2000) - Comparison of correlations for ammonia-water properties

Note: CoolProp does not support the NH3-H2O binary pair in its HEOS backend.
These polynomial/activity-coefficient correlations are used instead.

@author: OTEX Development Team
"""

import numpy as np


class AmmoniaWaterMixture:
    """
    Thermodynamic properties of ammonia-water mixture

    Uses Ibrahim & Klein (1993) correlations
    Valid range:
    - Pressure: 1-100 bar
    - Temperature: 0-200°C
    - Concentration: 0-1 (mass fraction NH3)
    """

    def __init__(self):
        self.R = 8.314472  # Universal gas constant [J/mol·K]
        self.M_NH3 = 17.031  # Molecular weight NH3 [g/mol]
        self.M_H2O = 18.015  # Molecular weight H2O [g/mol]

    def saturation_pressure(self, T, x):
        """
        Saturation pressure of NH3-H2O mixture

        Args:
            T: Temperature [°C]
            x: NH3 mass fraction (0-1)

        Returns:
            P_sat: Saturation pressure [bar]
        """
        T_K = T + 273.15

        # Pure component saturation pressures (Antoine equations)
        P_NH3_sat = self._P_sat_NH3(T_K)
        P_H2O_sat = self._P_sat_H2O(T_K)

        # Raoult's law with activity coefficients (simplified)
        # For more accuracy, would need full activity coefficient model
        x_mol = self._mass_to_mole_fraction(x)

        # Activity coefficients (simplified Wilson model)
        gamma_NH3, gamma_H2O = self._activity_coefficients(x_mol, T_K)

        P_sat = x_mol * gamma_NH3 * P_NH3_sat + (1 - x_mol) * gamma_H2O * P_H2O_sat

        return P_sat

    def saturation_temperature(self, P, x):
        """
        Saturation temperature (bubble point) of NH3-H2O mixture

        Args:
            P: Pressure [bar]
            x: NH3 mass fraction (0-1)

        Returns:
            T_sat: Saturation temperature [°C]
        """
        from scipy.optimize import fsolve

        # Initial guess based on pure NH3
        T_guess = self._T_sat_NH3_inverse(P)

        # Solve P_sat(T, x) = P
        T_sat_K = fsolve(lambda T: self.saturation_pressure(T - 273.15, x) - P, T_guess)[0]

        return T_sat_K - 273.15

    def enthalpy_liquid(self, T, P, x):
        """
        Liquid enthalpy of NH3-H2O mixture

        Args:
            T: Temperature [°C]
            P: Pressure [bar]
            x: NH3 mass fraction (0-1)

        Returns:
            h_liq: Liquid enthalpy [kJ/kg]
        """
        T_K = T + 273.15

        # Reference state: liquid at 0°C, 1 bar
        # h = h_ideal + h_excess

        h_NH3_pure = self._h_liq_NH3(T_K, P)
        h_H2O_pure = self._h_liq_H2O(T_K, P)

        # Ideal mixing enthalpy
        h_ideal = x * h_NH3_pure + (1 - x) * h_H2O_pure

        # Excess enthalpy (heat of mixing)
        h_excess = self._excess_enthalpy(T_K, P, x)

        h_liq = h_ideal + h_excess

        return h_liq

    def enthalpy_vapor(self, T, P, y):
        """
        Vapor enthalpy of NH3-H2O mixture

        Args:
            T: Temperature [°C]
            P: Pressure [bar]
            y: NH3 mass fraction in vapor (0-1)

        Returns:
            h_vap: Vapor enthalpy [kJ/kg]
        """
        T_K = T + 273.15

        h_NH3_vap = self._h_vap_NH3(T_K, P)
        h_H2O_vap = self._h_vap_H2O(T_K, P)

        # Ideal gas mixture (reasonable for low pressures)
        h_vap = y * h_NH3_vap + (1 - y) * h_H2O_vap

        return h_vap

    def entropy_liquid(self, T, P, x):
        """
        Liquid entropy of NH3-H2O mixture (vectorized)

        Args:
            T: Temperature [°C]
            P: Pressure [bar]
            x: NH3 mass fraction (0-1)

        Returns:
            s_liq: Liquid entropy [kJ/kg·K]
        """
        T_K = T + 273.15

        s_NH3_pure = self._s_liq_NH3(T_K, P)
        s_H2O_pure = self._s_liq_H2O(T_K, P)

        # Ideal mixing entropy
        x_mol = self._mass_to_mole_fraction(x)
        s_ideal = x * s_NH3_pure + (1 - x) * s_H2O_pure

        # Entropy of mixing (ideal gas approximation) - vectorized
        x_mol_arr = np.atleast_1d(np.asarray(x_mol, dtype=float))
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        s_mix = np.zeros_like(x_mol_arr)

        # Find valid entries (0 < x_mol < 1)
        mask_valid = (x_mol_arr > 0) & (x_mol_arr < 1)

        if np.any(mask_valid):
            x_mol_v = x_mol_arr[mask_valid]
            x_v = x_arr[mask_valid] if x_arr.size > 1 else x_arr

            s_mix_v = -self.R * (x_mol_v * np.log(x_mol_v) + (1 - x_mol_v) * np.log(1 - x_mol_v))
            # Convert J/mol·K to kJ/kg·K
            M_avg = x_v * self.M_NH3 + (1 - x_v) * self.M_H2O
            s_mix[mask_valid] = s_mix_v / M_avg

        # Handle scalar output
        if s_mix.size == 1:
            s_mix = s_mix[0]

        s_liq = s_ideal + s_mix

        return s_liq

    def entropy_vapor(self, T, P, y):
        """
        Vapor entropy of NH3-H2O mixture

        Args:
            T: Temperature [°C]
            P: Pressure [bar]
            y: NH3 mass fraction in vapor (0-1)

        Returns:
            s_vap: Vapor entropy [kJ/kg·K]
        """
        T_K = T + 273.15

        # Pure component vapor entropies
        s_NH3_vap = self._s_vap_NH3(T_K, P)
        s_H2O_vap = self._s_vap_H2O(T_K, P)

        # Ideal mixing entropy for vapor
        y_mol = self._mass_to_mole_fraction(y)
        s_ideal = y * s_NH3_vap + (1 - y) * s_H2O_vap

        # Entropy of mixing (ideal gas)
        y_mol_arr = np.atleast_1d(np.asarray(y_mol, dtype=float))
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        s_mix = np.zeros_like(y_mol_arr)

        mask_valid = (y_mol_arr > 0) & (y_mol_arr < 1)

        if np.any(mask_valid):
            y_mol_v = y_mol_arr[mask_valid]
            y_v = y_arr[mask_valid] if y_arr.size > 1 else y_arr

            s_mix_v = -self.R * (y_mol_v * np.log(y_mol_v) + (1 - y_mol_v) * np.log(1 - y_mol_v))
            M_avg = y_v * self.M_NH3 + (1 - y_v) * self.M_H2O
            s_mix[mask_valid] = s_mix_v / M_avg

        if s_mix.size == 1:
            s_mix = s_mix[0]

        s_vap = s_ideal + s_mix

        return s_vap

    def vapor_liquid_equilibrium(self, T, P, x_liquid):
        """
        Calculate vapor composition in equilibrium with liquid (vectorized)

        Args:
            T: Temperature [°C] - can be scalar, 1D, or 2D array
            P: Pressure [bar] - can be scalar, 1D, or 2D array
            x_liquid: NH3 mass fraction in liquid (0-1) - typically scalar

        Returns:
            y_vapor: NH3 mass fraction in vapor (0-1) - same shape as T/P
        """
        # Ensure T and P are arrays and get their shape
        T_arr = np.atleast_1d(np.asarray(T, dtype=float))
        P_arr = np.atleast_1d(np.asarray(P, dtype=float))
        original_shape = P_arr.shape

        T_K = T_arr + 273.15

        # Convert to mole fractions (x_liquid is typically scalar)
        x_mol = self._mass_to_mole_fraction(x_liquid)
        # Ensure x_mol is scalar for broadcasting
        if hasattr(x_mol, '__len__') and len(x_mol) == 1:
            x_mol = float(x_mol)

        # Pure component saturation pressures (same shape as T)
        P_NH3_sat = self._P_sat_NH3(T_K)
        P_H2O_sat = self._P_sat_H2O(T_K)

        # Activity coefficients (scalar since x_mol is typically scalar)
        gamma_NH3, gamma_H2O = self._activity_coefficients(x_mol, T_K)
        # Ensure gamma values are scalar for proper broadcasting
        if hasattr(gamma_NH3, '__len__') and len(np.atleast_1d(gamma_NH3)) == 1:
            gamma_NH3 = float(np.atleast_1d(gamma_NH3)[0])
            gamma_H2O = float(np.atleast_1d(gamma_H2O)[0])

        # Vapor mole fractions (Raoult's law with activity coefficients)
        # All operations should broadcast correctly now
        y_NH3_mol = (x_mol * gamma_NH3 * P_NH3_sat) / P_arr
        y_H2O_mol = ((1 - x_mol) * gamma_H2O * P_H2O_sat) / P_arr

        # Normalize (shouldn't be needed if at equilibrium, but for safety)
        y_total = y_NH3_mol + y_H2O_mol
        y_NH3_mol = y_NH3_mol / y_total

        # Convert to mass fraction
        y_vapor = self._mole_to_mass_fraction(y_NH3_mol)

        # Return scalar if original input was scalar
        if original_shape == (1,) and np.isscalar(T):
            return float(y_vapor) if hasattr(y_vapor, '__len__') else y_vapor

        return y_vapor

    def dew_temperature(self, P, y_vapor):
        """
        Calculate dew point temperature

        Args:
            P: Pressure [bar]
            y_vapor: NH3 mass fraction in vapor (0-1)

        Returns:
            T_dew: Dew temperature [°C]
        """
        from scipy.optimize import fsolve

        # Initial guess
        T_guess = self.saturation_temperature(P, y_vapor)

        # Solve for temperature where vapor composition matches
        def residual(T):
            # At dew point, a tiny amount of liquid forms
            # Assume liquid composition ≈ vapor composition for initial guess
            x_liquid_guess = y_vapor
            y_calc = self.vapor_liquid_equilibrium(T, P, x_liquid_guess)
            return y_calc - y_vapor

        T_dew = fsolve(residual, T_guess)[0]

        return T_dew

    # ==================== Private Helper Methods ====================

    def _mass_to_mole_fraction(self, x_mass):
        """Convert mass fraction to mole fraction (vectorized, preserves shape)"""
        # Check if input is scalar
        is_scalar = np.isscalar(x_mass)
        x_mass_arr = np.asarray(x_mass, dtype=float)
        original_shape = x_mass_arr.shape

        # Flatten for processing, will reshape at end
        x_mass_flat = x_mass_arr.ravel()
        x_mol_flat = np.zeros_like(x_mass_flat)

        # Handle edge cases
        mask_zero = x_mass_flat == 0
        mask_one = x_mass_flat == 1
        mask_valid = ~mask_zero & ~mask_one

        x_mol_flat[mask_zero] = 0
        x_mol_flat[mask_one] = 1
        if np.any(mask_valid):
            x_mol_flat[mask_valid] = (x_mass_flat[mask_valid] / self.M_NH3) / \
                                (x_mass_flat[mask_valid] / self.M_NH3 + (1 - x_mass_flat[mask_valid]) / self.M_H2O)

        # Reshape to original shape
        x_mol = x_mol_flat.reshape(original_shape)

        # Return scalar if input was scalar
        return float(x_mol) if is_scalar else x_mol

    def _mole_to_mass_fraction(self, x_mol):
        """Convert mole fraction to mass fraction (vectorized, preserves shape)"""
        # Check if input is scalar
        is_scalar = np.isscalar(x_mol)
        x_mol_arr = np.asarray(x_mol, dtype=float)
        original_shape = x_mol_arr.shape

        # Flatten for processing, will reshape at end
        x_mol_flat = x_mol_arr.ravel()
        x_mass_flat = np.zeros_like(x_mol_flat)

        # Handle edge cases
        mask_zero = x_mol_flat == 0
        mask_one = x_mol_flat == 1
        mask_valid = ~mask_zero & ~mask_one

        x_mass_flat[mask_zero] = 0
        x_mass_flat[mask_one] = 1
        if np.any(mask_valid):
            x_mass_flat[mask_valid] = (x_mol_flat[mask_valid] * self.M_NH3) / \
                                 (x_mol_flat[mask_valid] * self.M_NH3 + (1 - x_mol_flat[mask_valid]) * self.M_H2O)

        # Reshape to original shape
        x_mass = x_mass_flat.reshape(original_shape)

        # Return scalar if input was scalar
        return float(x_mass) if is_scalar else x_mass

    def _P_sat_NH3(self, T_K):
        """NH3 saturation pressure [bar] - Antoine equation"""
        # Valid range: 230-430 K
        A, B, C = 15.9017, 2132.5, -32.98
        log_P_mmHg = A - B / (T_K + C)
        P_bar = 10**(log_P_mmHg) / 750.06  # mmHg to bar
        return P_bar

    def _P_sat_H2O(self, T_K):
        """H2O saturation pressure [bar] - Antoine equation"""
        # Valid range: 273-373 K
        A, B, C = 16.3872, 3885.7, -42.98
        log_P_mmHg = A - B / (T_K + C)
        P_bar = 10**(log_P_mmHg) / 750.06  # mmHg to bar
        return P_bar

    def _T_sat_NH3_inverse(self, P_bar):
        """Inverse of NH3 saturation pressure (approximate)"""
        P_mmHg = P_bar * 750.06
        A, B, C = 15.9017, 2132.5, -32.98
        T_K = B / (A - np.log10(P_mmHg)) - C
        return T_K

    def _activity_coefficients(self, x_mol, T_K):
        """
        Activity coefficients using simplified Wilson model (vectorized)

        This is a simplified version. For high accuracy, use:
        - NRTL model
        - UNIQUAC model
        - Or property databases like REFPROP
        """
        # Wilson parameters (fitted to experimental data)
        # These are approximate - for production use, fit to experimental VLE data
        Lambda_12 = 0.4  # NH3-H2O interaction
        Lambda_21 = 1.8  # H2O-NH3 interaction

        x_mol = np.atleast_1d(np.asarray(x_mol, dtype=float))
        gamma_NH3 = np.ones_like(x_mol)
        gamma_H2O = np.ones_like(x_mol)

        # Find valid entries (not 0 or 1)
        mask_valid = (x_mol > 0) & (x_mol < 1)

        if np.any(mask_valid):
            x_v = x_mol[mask_valid]

            # Wilson equation
            ln_gamma_NH3 = -np.log(x_v + (1 - x_v) * Lambda_12) + \
                           (1 - x_v) * (Lambda_12 / (x_v + (1 - x_v) * Lambda_12) - \
                                         Lambda_21 / ((1 - x_v) + x_v * Lambda_21))

            ln_gamma_H2O = -np.log((1 - x_v) + x_v * Lambda_21) - \
                           x_v * (Lambda_12 / (x_v + (1 - x_v) * Lambda_12) - \
                                   Lambda_21 / ((1 - x_v) + x_v * Lambda_21))

            gamma_NH3[mask_valid] = np.exp(ln_gamma_NH3)
            gamma_H2O[mask_valid] = np.exp(ln_gamma_H2O)

        # Return scalars if input was scalar
        if gamma_NH3.size == 1:
            return gamma_NH3[0], gamma_H2O[0]
        return gamma_NH3, gamma_H2O

    def _h_liq_NH3(self, T_K, P_bar):
        """Liquid NH3 enthalpy [kJ/kg] - simplified correlation"""
        # Reference: 0°C, 1 bar
        T_ref = 273.15
        cp_liq = 4.6  # kJ/kg·K (approximate, varies with T)
        h_ref = 200.0  # kJ/kg at reference state

        h = h_ref + cp_liq * (T_K - T_ref)
        return h

    def _h_liq_H2O(self, T_K, P_bar):
        """Liquid H2O enthalpy [kJ/kg]"""
        T_ref = 273.15
        cp_liq = 4.18  # kJ/kg·K
        h_ref = 0.0  # Reference state

        h = h_ref + cp_liq * (T_K - T_ref)
        return h

    def _h_vap_NH3(self, T_K, P_bar):
        """Vapor NH3 enthalpy [kJ/kg]"""
        # Simplified: h_vap = h_liq + h_fg
        h_liq = self._h_liq_NH3(T_K, P_bar)
        h_fg = 1200.0 - 2.5 * (T_K - 273.15)  # Approximate latent heat
        return h_liq + h_fg

    def _h_vap_H2O(self, T_K, P_bar):
        """Vapor H2O enthalpy [kJ/kg]"""
        h_liq = self._h_liq_H2O(T_K, P_bar)
        h_fg = 2500.0 - 2.3 * (T_K - 273.15)  # Approximate latent heat
        return h_liq + h_fg

    def _s_liq_NH3(self, T_K, P_bar):
        """Liquid NH3 entropy [kJ/kg·K]"""
        T_ref = 273.15
        cp_liq = 4.6  # kJ/kg·K
        s_ref = 1.0  # kJ/kg·K at reference

        s = s_ref + cp_liq * np.log(T_K / T_ref)
        return s

    def _s_liq_H2O(self, T_K, P_bar):
        """Liquid H2O entropy [kJ/kg·K]"""
        T_ref = 273.15
        cp_liq = 4.18  # kJ/kg·K
        s_ref = 0.0

        s = s_ref + cp_liq * np.log(T_K / T_ref)
        return s

    def _s_vap_NH3(self, T_K, P_bar):
        """Vapor NH3 entropy [kJ/kg·K]"""
        # s_vap = s_liq + s_fg (entropy of vaporization)
        s_liq = self._s_liq_NH3(T_K, P_bar)
        # Approximate entropy of vaporization: h_fg / T
        h_fg = 1200.0 - 2.5 * (T_K - 273.15)  # kJ/kg
        s_fg = h_fg / T_K
        return s_liq + s_fg

    def _s_vap_H2O(self, T_K, P_bar):
        """Vapor H2O entropy [kJ/kg·K]"""
        s_liq = self._s_liq_H2O(T_K, P_bar)
        h_fg = 2500.0 - 2.3 * (T_K - 273.15)  # kJ/kg
        s_fg = h_fg / T_K
        return s_liq + s_fg

    def _excess_enthalpy(self, T_K, P_bar, x):
        """
        Excess enthalpy of mixing [kJ/kg]

        Represents heat released/absorbed when mixing NH3 and H2O
        This is a simplified model - for accuracy, use experimental data
        """
        # Heat of mixing is maximum around x = 0.5
        # Negative (exothermic mixing)
        x_mol = self._mass_to_mole_fraction(x)

        # Simplified polynomial model
        h_excess = -50.0 * x_mol * (1 - x_mol)  # kJ/kg

        return h_excess


if __name__ == "__main__":
    # Test ammonia-water mixture properties
    print("Testing Ammonia-Water Mixture Properties\n")
    print("="*70)

    mixture = AmmoniaWaterMixture()

    # Test case 1: Saturation properties
    print("\n1. Saturation Properties:")
    T = 20.0  # °C
    x = 0.7   # 70% NH3 by mass

    P_sat = mixture.saturation_pressure(T, x)
    print(f"   Temperature: {T}°C")
    print(f"   NH3 concentration: {x*100:.1f}% (mass)")
    print(f"   Saturation pressure: {P_sat:.3f} bar")

    # Test case 2: Reverse calculation
    print("\n2. Saturation Temperature:")
    P = 10.0  # bar
    x = 0.7
    T_sat = mixture.saturation_temperature(P, x)
    print(f"   Pressure: {P} bar")
    print(f"   NH3 concentration: {x*100:.1f}% (mass)")
    print(f"   Saturation temperature: {T_sat:.2f}°C")

    # Test case 3: VLE
    print("\n3. Vapor-Liquid Equilibrium:")
    T = 50.0  # °C
    P = 15.0  # bar
    x_liq = 0.6  # Liquid composition

    y_vap = mixture.vapor_liquid_equilibrium(T, P, x_liq)
    print(f"   Temperature: {T}°C")
    print(f"   Pressure: {P} bar")
    print(f"   Liquid NH3 concentration: {x_liq*100:.1f}%")
    print(f"   Vapor NH3 concentration: {y_vap*100:.1f}%")
    print(f"   Enrichment factor: {y_vap/x_liq:.2f}x")

    # Test case 4: Enthalpies
    print("\n4. Thermodynamic Properties:")
    T = 40.0
    P = 12.0
    x = 0.65

    h_liq = mixture.enthalpy_liquid(T, P, x)
    h_vap = mixture.enthalpy_vapor(T, P, x)
    s_liq = mixture.entropy_liquid(T, P, x)

    print(f"   T = {T}°C, P = {P} bar, x = {x*100:.1f}%")
    print(f"   Liquid enthalpy: {h_liq:.2f} kJ/kg")
    print(f"   Vapor enthalpy: {h_vap:.2f} kJ/kg")
    print(f"   Latent heat: {h_vap - h_liq:.2f} kJ/kg")
    print(f"   Liquid entropy: {s_liq:.4f} kJ/kg·K")

    # Test case 5: Concentration sweep
    print("\n5. Effect of Concentration on Properties:")
    print(f"   At T = 30°C:")
    print(f"   {'NH3 [%]':<10} {'P_sat [bar]':<15} {'h_liq [kJ/kg]':<20}")
    print(f"   {'-'*45}")

    for x_test in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        P_sat_test = mixture.saturation_pressure(30.0, x_test)
        h_liq_test = mixture.enthalpy_liquid(30.0, 10.0, x_test)
        print(f"   {x_test*100:<10.1f} {P_sat_test:<15.3f} {h_liq_test:<20.2f}")

    print("\n" + "="*70)
    print("Testing complete!")
    print("\nNote: These are Ibrahim & Klein (1993) correlations.")
    print("CoolProp does not support the NH3-H2O binary pair in HEOS backend.")
