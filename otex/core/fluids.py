# -*- coding: utf-8 -*-
"""
OTEX Working Fluids Module
Provides thermodynamic properties for different working fluids

Supported fluids:
- Ammonia (NH3) - original implementation
- R134a, R245fa - refrigerants
- Propane, Isobutane - hydrocarbons
- Ammonia-Water mixture (for Kalina/Uehara cycles)

Can use either:
1. CoolProp library (recommended, more accurate)
2. Polynomial correlations (fallback, faster)

@author: OTEX Development Team
"""

import numpy as np
from abc import ABC, abstractmethod

# Try to import CoolProp, but allow graceful fallback
try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False
    print("Warning: CoolProp not available. Using polynomial correlations instead.")
    print("For better accuracy, install CoolProp: pip install CoolProp")


class WorkingFluid(ABC):
    """
    Abstract base class for working fluids
    All working fluids must implement these methods
    """

    def __init__(self, name):
        self.name = name
        self.molecular_weight = None  # kg/kmol
        self.critical_temp = None     # K
        self.critical_pressure = None # Pa

    @abstractmethod
    def saturation_pressure(self, T):
        """
        Calculate saturation pressure from temperature

        Args:
            T: Temperature in Celsius (scalar or array)

        Returns:
            P_sat: Saturation pressure in bar (same shape as T)
        """
        pass

    @abstractmethod
    def saturation_temperature(self, p):
        """
        Calculate saturation temperature from pressure

        Args:
            p: Pressure in bar (scalar or array)

        Returns:
            T_sat: Saturation temperature in Celsius (same shape as p)
        """
        pass

    @abstractmethod
    def enthalpy_liquid(self, T=None, p=None):
        """
        Calculate enthalpy of saturated liquid

        Args:
            T: Temperature in Celsius (provide T or p, not both)
            p: Pressure in bar

        Returns:
            h_f: Liquid enthalpy in kJ/kg
        """
        pass

    @abstractmethod
    def enthalpy_vapor(self, T=None, p=None):
        """
        Calculate enthalpy of saturated vapor

        Args:
            T: Temperature in Celsius (provide T or p, not both)
            p: Pressure in bar

        Returns:
            h_g: Vapor enthalpy in kJ/kg
        """
        pass

    @abstractmethod
    def entropy_liquid(self, T=None, p=None):
        """
        Calculate entropy of saturated liquid

        Args:
            T: Temperature in Celsius (provide T or p, not both)
            p: Pressure in bar

        Returns:
            s_f: Liquid entropy in kJ/kgK
        """
        pass

    @abstractmethod
    def entropy_vapor(self, T=None, p=None):
        """
        Calculate entropy of saturated vapor

        Args:
            T: Temperature in Celsius (provide T or p, not both)
            p: Pressure in bar

        Returns:
            s_g: Vapor entropy in kJ/kgK
        """
        pass

    @abstractmethod
    def density_liquid(self, T=None, p=None):
        """
        Calculate density of saturated liquid

        Args:
            T: Temperature in Celsius
            p: Pressure in bar

        Returns:
            rho: Density in kg/m3
        """
        pass

    def enthalpy_two_phase(self, h_f, h_g, quality):
        """
        Calculate enthalpy of two-phase mixture

        Args:
            h_f: Liquid enthalpy in kJ/kg
            h_g: Vapor enthalpy in kJ/kg
            quality: Vapor quality (0 to 1)

        Returns:
            h: Mixture enthalpy in kJ/kg
        """
        return h_f * (1 - quality) + h_g * quality

    def entropy_two_phase(self, s_f, s_g, quality):
        """
        Calculate entropy of two-phase mixture

        Args:
            s_f: Liquid entropy in kJ/kgK
            s_g: Vapor entropy in kJ/kgK
            quality: Vapor quality (0 to 1)

        Returns:
            s: Mixture entropy in kJ/kgK
        """
        return s_f * (1 - quality) + s_g * quality

    def quality_from_enthalpy(self, h, h_f, h_g):
        """
        Calculate vapor quality from enthalpy

        Args:
            h: Mixture enthalpy in kJ/kg
            h_f: Liquid enthalpy in kJ/kg
            h_g: Vapor enthalpy in kJ/kg

        Returns:
            x: Vapor quality (0 to 1)
        """
        return (h - h_f) / (h_g - h_f)

    def quality_from_entropy(self, s, s_f, s_g):
        """
        Calculate vapor quality from entropy

        Args:
            s: Mixture entropy in kJ/kgK
            s_f: Liquid entropy in kJ/kgK
            s_g: Vapor entropy in kJ/kgK

        Returns:
            x: Vapor quality (0 to 1)
        """
        return (s - s_f) / (s_g - s_f)


class CoolPropFluid(WorkingFluid):
    """
    Working fluid using CoolProp library for accurate properties
    """

    def __init__(self, fluid_name, coolprop_name=None):
        """
        Args:
            fluid_name: Display name (e.g., 'Ammonia')
            coolprop_name: CoolProp identifier (e.g., 'NH3')
                          If None, uses fluid_name
        """
        super().__init__(fluid_name)

        if not COOLPROP_AVAILABLE:
            raise ImportError("CoolProp is not available. Cannot use CoolPropFluid.")

        self.coolprop_name = coolprop_name if coolprop_name else fluid_name

        # Get critical properties
        self.critical_temp = CP.PropsSI('Tcrit', self.coolprop_name)
        self.critical_pressure = CP.PropsSI('Pcrit', self.coolprop_name)
        self.molecular_weight = CP.PropsSI('M', self.coolprop_name) * 1000  # kg/kmol

    def _vectorize_coolprop(self, prop, input1_name, input1_val, input2_name, input2_val):
        """Helper to vectorize CoolProp calls which don't support arrays natively"""
        input1_arr = np.atleast_1d(input1_val)
        input2_arr = np.atleast_1d(input2_val)

        # Store original shape to restore later
        original_shape = None
        if input1_arr.size > 1:
            original_shape = input1_arr.shape
        elif input2_arr.size > 1:
            original_shape = input2_arr.shape

        # Convert numpy types to Python scalars
        def to_scalar(val):
            # Handle numpy arrays, scalars, and Python numbers
            if isinstance(val, np.ndarray):
                if val.size == 1:
                    return float(val.flat[0])
                else:
                    raise ValueError(f"Expected scalar, got array of size {val.size}")
            elif hasattr(val, 'item'):
                return float(val.item())
            else:
                return float(val)

        if input1_arr.size == 1 and input2_arr.size == 1:
            return CP.PropsSI(prop, input1_name, to_scalar(input1_arr[0]),
                            input2_name, to_scalar(input2_arr[0]), self.coolprop_name)
        elif input1_arr.size == 1:
            # input1 is scalar, input2 is array
            scalar1 = to_scalar(input1_arr[0])
            # Flatten input2 to ensure proper iteration
            input2_flat = input2_arr.flatten()
            result = np.array([CP.PropsSI(prop, input1_name, scalar1,
                                        input2_name, to_scalar(val2), self.coolprop_name)
                            for val2 in input2_flat])
            # Restore original shape
            if original_shape is not None:
                result = result.reshape(original_shape)
            return result
        elif input2_arr.size == 1:
            # input2 is scalar, input1 is array
            scalar2 = to_scalar(input2_arr[0])
            # Flatten input1 to ensure proper iteration
            input1_flat = input1_arr.flatten()
            result = np.array([CP.PropsSI(prop, input1_name, to_scalar(val1),
                                        input2_name, scalar2, self.coolprop_name)
                            for val1 in input1_flat])
            # Restore original shape
            if original_shape is not None:
                result = result.reshape(original_shape)
            return result
        else:
            # Both are arrays - element-wise operation
            # Flatten both arrays to ensure proper iteration
            input1_flat = input1_arr.flatten()
            input2_flat = input2_arr.flatten()
            result = np.array([CP.PropsSI(prop, input1_name, to_scalar(val1),
                                        input2_name, to_scalar(val2), self.coolprop_name)
                            for val1, val2 in zip(input1_flat, input2_flat)])
            # Restore original shape
            if original_shape is not None:
                result = result.reshape(original_shape)
            return result

    def saturation_pressure(self, T):
        """Temperature in Celsius, returns pressure in bar"""
        T_arr = np.atleast_1d(T)
        T_K = T_arr + 273.15

        # Handle nan values
        if np.any(np.isnan(T_K)):
            # Create output array with same shape, filled with nan
            p_Pa = np.full_like(T_K, np.nan)
            # Only compute for valid (non-nan) values
            valid_mask = ~np.isnan(T_K)
            if np.any(valid_mask):
                p_Pa[valid_mask] = self._vectorize_coolprop('P', 'T', T_K[valid_mask], 'Q', 0)
        else:
            p_Pa = self._vectorize_coolprop('P', 'T', T_K, 'Q', 0)

        return p_Pa / 1e5  # Convert Pa to bar

    def saturation_temperature(self, p):
        """Pressure in bar, returns temperature in Celsius"""
        p_arr = np.atleast_1d(p)
        p_Pa = p_arr * 1e5

        # Handle nan values
        if np.any(np.isnan(p_Pa)):
            # Create output array with same shape, filled with nan
            T_K = np.full_like(p_Pa, np.nan)
            # Only compute for valid (non-nan) values
            valid_mask = ~np.isnan(p_Pa)
            if np.any(valid_mask):
                T_K[valid_mask] = self._vectorize_coolprop('T', 'P', p_Pa[valid_mask], 'Q', 0)
        else:
            T_K = self._vectorize_coolprop('T', 'P', p_Pa, 'Q', 0)

        return T_K - 273.15

    def enthalpy_liquid(self, T=None, p=None):
        """Returns enthalpy in kJ/kg"""
        if T is not None:
            T_arr = np.atleast_1d(T)
            T_K = T_arr + 273.15
            # Handle nan values
            if np.any(np.isnan(T_K)):
                h_J = np.full_like(T_K, np.nan)
                valid_mask = ~np.isnan(T_K)
                if np.any(valid_mask):
                    h_J[valid_mask] = self._vectorize_coolprop('H', 'T', T_K[valid_mask], 'Q', 0)
            else:
                h_J = self._vectorize_coolprop('H', 'T', T_K, 'Q', 0)
        elif p is not None:
            p_arr = np.atleast_1d(p)
            p_Pa = p_arr * 1e5
            # Handle nan values
            if np.any(np.isnan(p_Pa)):
                h_J = np.full_like(p_Pa, np.nan)
                valid_mask = ~np.isnan(p_Pa)
                if np.any(valid_mask):
                    h_J[valid_mask] = self._vectorize_coolprop('H', 'P', p_Pa[valid_mask], 'Q', 0)
            else:
                h_J = self._vectorize_coolprop('H', 'P', p_Pa, 'Q', 0)
        else:
            raise ValueError("Must provide either T or p")
        return h_J / 1000  # Convert J/kg to kJ/kg

    def enthalpy_vapor(self, T=None, p=None):
        """Returns enthalpy in kJ/kg"""
        if T is not None:
            T_arr = np.atleast_1d(T)
            T_K = T_arr + 273.15
            # Handle nan values
            if np.any(np.isnan(T_K)):
                h_J = np.full_like(T_K, np.nan)
                valid_mask = ~np.isnan(T_K)
                if np.any(valid_mask):
                    h_J[valid_mask] = self._vectorize_coolprop('H', 'T', T_K[valid_mask], 'Q', 1)
            else:
                h_J = self._vectorize_coolprop('H', 'T', T_K, 'Q', 1)
        elif p is not None:
            p_arr = np.atleast_1d(p)
            p_Pa = p_arr * 1e5
            # Handle nan values
            if np.any(np.isnan(p_Pa)):
                h_J = np.full_like(p_Pa, np.nan)
                valid_mask = ~np.isnan(p_Pa)
                if np.any(valid_mask):
                    h_J[valid_mask] = self._vectorize_coolprop('H', 'P', p_Pa[valid_mask], 'Q', 1)
            else:
                h_J = self._vectorize_coolprop('H', 'P', p_Pa, 'Q', 1)
        else:
            raise ValueError("Must provide either T or p")
        return h_J / 1000  # Convert J/kg to kJ/kg

    def entropy_liquid(self, T=None, p=None):
        """Returns entropy in kJ/kgK"""
        if T is not None:
            T_arr = np.atleast_1d(T)
            T_K = T_arr + 273.15
            # Handle nan values
            if np.any(np.isnan(T_K)):
                s_J = np.full_like(T_K, np.nan)
                valid_mask = ~np.isnan(T_K)
                if np.any(valid_mask):
                    s_J[valid_mask] = self._vectorize_coolprop('S', 'T', T_K[valid_mask], 'Q', 0)
            else:
                s_J = self._vectorize_coolprop('S', 'T', T_K, 'Q', 0)
        elif p is not None:
            p_arr = np.atleast_1d(p)
            p_Pa = p_arr * 1e5
            # Handle nan values
            if np.any(np.isnan(p_Pa)):
                s_J = np.full_like(p_Pa, np.nan)
                valid_mask = ~np.isnan(p_Pa)
                if np.any(valid_mask):
                    s_J[valid_mask] = self._vectorize_coolprop('S', 'P', p_Pa[valid_mask], 'Q', 0)
            else:
                s_J = self._vectorize_coolprop('S', 'P', p_Pa, 'Q', 0)
        else:
            raise ValueError("Must provide either T or p")
        return s_J / 1000  # Convert J/kgK to kJ/kgK

    def entropy_vapor(self, T=None, p=None):
        """Returns entropy in kJ/kgK"""
        if T is not None:
            T_arr = np.atleast_1d(T)
            T_K = T_arr + 273.15
            # Handle nan values
            if np.any(np.isnan(T_K)):
                s_J = np.full_like(T_K, np.nan)
                valid_mask = ~np.isnan(T_K)
                if np.any(valid_mask):
                    s_J[valid_mask] = self._vectorize_coolprop('S', 'T', T_K[valid_mask], 'Q', 1)
            else:
                s_J = self._vectorize_coolprop('S', 'T', T_K, 'Q', 1)
        elif p is not None:
            p_arr = np.atleast_1d(p)
            p_Pa = p_arr * 1e5
            # Handle nan values
            if np.any(np.isnan(p_Pa)):
                s_J = np.full_like(p_Pa, np.nan)
                valid_mask = ~np.isnan(p_Pa)
                if np.any(valid_mask):
                    s_J[valid_mask] = self._vectorize_coolprop('S', 'P', p_Pa[valid_mask], 'Q', 1)
            else:
                s_J = self._vectorize_coolprop('S', 'P', p_Pa, 'Q', 1)
        else:
            raise ValueError("Must provide either T or p")
        return s_J / 1000  # Convert J/kgK to kJ/kgK

    def density_liquid(self, T=None, p=None):
        """Returns density in kg/m3"""
        if T is not None:
            T_arr = np.atleast_1d(T)
            T_K = T_arr + 273.15
            # Handle nan values
            if np.any(np.isnan(T_K)):
                rho = np.full_like(T_K, np.nan)
                valid_mask = ~np.isnan(T_K)
                if np.any(valid_mask):
                    rho[valid_mask] = self._vectorize_coolprop('D', 'T', T_K[valid_mask], 'Q', 0)
            else:
                rho = self._vectorize_coolprop('D', 'T', T_K, 'Q', 0)
        elif p is not None:
            p_arr = np.atleast_1d(p)
            p_Pa = p_arr * 1e5
            # Handle nan values
            if np.any(np.isnan(p_Pa)):
                rho = np.full_like(p_Pa, np.nan)
                valid_mask = ~np.isnan(p_Pa)
                if np.any(valid_mask):
                    rho[valid_mask] = self._vectorize_coolprop('D', 'P', p_Pa[valid_mask], 'Q', 0)
            else:
                rho = self._vectorize_coolprop('D', 'P', p_Pa, 'Q', 0)
        else:
            raise ValueError("Must provide either T or p")
        return rho


class PolynomialAmmonia(WorkingFluid):
    """
    Ammonia using polynomial correlations (original OTEX implementation)
    Fast but less accurate than CoolProp
    Valid range: approximately 0-40°C
    """

    def __init__(self):
        super().__init__('Ammonia_Polynomial')
        self.molecular_weight = 17.031  # kg/kmol
        self.critical_temp = 405.4      # K
        self.critical_pressure = 113.3e5  # Pa

    def saturation_pressure(self, T):
        """
        Polynomial correlation for NH3 saturation pressure
        T in Celsius, returns p in bar
        """
        T = np.atleast_1d(T)
        p = 0.00002196*T**3 + 0.00193103*T**2 + 0.1695763*T + 4.25739601
        return p

    def saturation_temperature(self, p):
        """
        Inverse of saturation pressure (numerical solution)
        p in bar, returns T in Celsius
        """
        from scipy.optimize import fsolve

        p = np.atleast_1d(p)
        T_sat = np.zeros_like(p)

        for i, p_val in enumerate(p):
            # Initial guess: linear approximation
            T_guess = (p_val - 4.25739601) / 0.1695763
            # Solve for exact value
            T_sat[i] = fsolve(lambda T: self.saturation_pressure(T) - p_val, T_guess)[0]

        return T_sat

    def enthalpy_vapor(self, T=None, p=None):
        """
        Enthalpy of saturated vapor
        Returns h in kJ/kg
        """
        if p is not None:
            p_val = np.atleast_1d(p)
        elif T is not None:
            p_val = self.saturation_pressure(T)
        else:
            raise ValueError("Must provide either T or p")

        h_g = 28.276*np.log(p_val) + 1418.1
        return h_g

    def entropy_vapor(self, T=None, p=None):
        """
        Entropy of saturated vapor
        Returns s in kJ/kgK
        """
        if p is not None:
            p_val = np.atleast_1d(p)
        elif T is not None:
            p_val = self.saturation_pressure(T)
        else:
            raise ValueError("Must provide either T or p")

        s_g = -0.352*np.log(p_val) + 6.1284
        return s_g

    def enthalpy_liquid(self, T=None, p=None):
        """
        Enthalpy of saturated liquid
        Returns h in kJ/kg
        """
        if p is not None:
            p_val = np.atleast_1d(p)
        elif T is not None:
            p_val = self.saturation_pressure(T)
        else:
            raise ValueError("Must provide either T or p")

        h_f = -0.0235*p_val**4 + 0.9083*p_val**3 - 12.93*p_val**2 + 97.316*p_val - 39.559
        return h_f

    def entropy_liquid(self, T=None, p=None):
        """
        Entropy of saturated liquid
        Returns s in kJ/kgK
        """
        if p is not None:
            p_val = np.atleast_1d(p)
        elif T is not None:
            p_val = self.saturation_pressure(T)
        else:
            raise ValueError("Must provide either T or p")

        s_f = 0.3947*np.log(p_val) + 0.4644
        return s_f

    def density_liquid(self, T=None, p=None):
        """
        Density of saturated liquid NH3
        Returns rho in kg/m3
        """
        # Simple correlation for NH3 liquid density
        # Based on temperature (pressure has small effect on liquid density)
        if T is not None:
            T_val = np.atleast_1d(T)
        elif p is not None:
            T_val = self.saturation_temperature(p)
        else:
            raise ValueError("Must provide either T or p")

        # Linear approximation: rho decreases with temperature
        # At 0°C: ~640 kg/m3, at 40°C: ~600 kg/m3
        rho = 640 - 1.0 * T_val
        return rho


def get_working_fluid(fluid_type='ammonia', use_coolprop=True):
    """
    Factory function to get a working fluid instance

    Args:
        fluid_type: Type of fluid ('ammonia', 'r134a', 'r245fa', 'propane', 'isobutane')
        use_coolprop: If True, use CoolProp (if available), else use polynomial

    Returns:
        WorkingFluid instance
    """

    fluid_type = fluid_type.lower()

    # If CoolProp requested and available
    if use_coolprop and COOLPROP_AVAILABLE:
        fluid_map = {
            'ammonia': ('Ammonia', 'NH3'),
            'r134a': ('R134a', 'R134a'),
            'r245fa': ('R245fa', 'R245fa'),
            'propane': ('Propane', 'Propane'),
            'isobutane': ('Isobutane', 'IsoButane'),
        }

        if fluid_type in fluid_map:
            name, cp_name = fluid_map[fluid_type]
            return CoolPropFluid(name, cp_name)
        else:
            raise ValueError(f"Unknown fluid type: {fluid_type}")

    # Fallback to polynomial correlations
    else:
        if fluid_type == 'ammonia':
            return PolynomialAmmonia()
        else:
            raise ValueError(f"Polynomial correlations only available for ammonia. "
                           f"Install CoolProp to use {fluid_type}")


if __name__ == "__main__":
    # Test the working fluids module
    print("Testing Working Fluids Module\n")
    print("="*60)

    # Test polynomial ammonia (always available)
    print("\n1. Testing Polynomial Ammonia:")
    nh3_poly = PolynomialAmmonia()
    T_test = np.array([5, 10, 15, 20, 25])
    p_sat = nh3_poly.saturation_pressure(T_test)
    print(f"Temperature [°C]: {T_test}")
    print(f"Sat. Pressure [bar]: {p_sat}")
    print(f"Vapor Enthalpy [kJ/kg]: {nh3_poly.enthalpy_vapor(T=T_test)}")
    print(f"Liquid Density [kg/m3]: {nh3_poly.density_liquid(T=T_test)}")

    # Test CoolProp if available
    if COOLPROP_AVAILABLE:
        print("\n2. Testing CoolProp Ammonia:")
        nh3_cp = CoolPropFluid('Ammonia', 'NH3')
        p_sat_cp = nh3_cp.saturation_pressure(T_test)
        print(f"Temperature [°C]: {T_test}")
        print(f"Sat. Pressure [bar]: {p_sat_cp}")
        print(f"Vapor Enthalpy [kJ/kg]: {nh3_cp.enthalpy_vapor(T=T_test)}")
        print(f"Liquid Density [kg/m3]: {nh3_cp.density_liquid(T=T_test)}")

        print("\n3. Comparison (Polynomial vs CoolProp):")
        print(f"Pressure difference: {np.abs(p_sat - p_sat_cp).max():.4f} bar (max)")

        print("\n4. Testing other fluids with CoolProp:")
        for fluid in ['r134a', 'propane']:
            f = get_working_fluid(fluid, use_coolprop=True)
            print(f"\n{f.name}:")
            print(f"  Sat. Pressure at 20°C: {f.saturation_pressure(20)[0]:.3f} bar")
            print(f"  Vapor Enthalpy at 20°C: {f.enthalpy_vapor(T=20)[0]:.2f} kJ/kg")
    else:
        print("\nCoolProp not available. Install with: pip install CoolProp")

    print("\n" + "="*60)
    print("Testing complete!")
