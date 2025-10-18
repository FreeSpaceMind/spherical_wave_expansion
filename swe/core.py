"""
Spherical Wave Expansion (SWE) Module

This module provides functionality for working with spherical wave expansions
of electromagnetic fields, including far-field pattern reconstruction and
coefficient extraction from measured data.

Key Features:
- Read/write TICRA .sph files
- Calculate far-field patterns from Q coefficients
- Calculate near-field E and H components at arbitrary points
- Calculate equivalent surface currents on specified surfaces
- Extract Q coefficients from measurements:
  * Far-field patterns
  * Spherical near-field measurements
  * Planar near-field measurements  
  * Cylindrical near-field measurements

Physical Conventions:
- Time dependence: exp(+iωt)
- Frequency: Hz
- Wavenumber k: rad/m
- Electric field E: V/m
- Magnetic field H: A/m
- Impedance η₀ = 376.73 Ω (free space)
- Outgoing spherical waves: h_n^(3)(kr) = j_n(kr) - i·y_n(kr)

Coordinate System:
- r: radial distance (m)
- θ: polar angle from +z axis (radians, 0 to π)
- φ: azimuthal angle from +x axis (radians, 0 to 2π)

Near-Field Measurement Geometries:
- Spherical: measurements on sphere of radius r
- Planar: measurements on z=constant plane (rectangular scan)
- Cylindrical: measurements on cylinder of radius ρ

References:
- J.E. Hansen, "Spherical Near-Field Antenna Measurements" (1988)
- TICRA documentation for .sph file format
- IEEE Std 1720-2012: Recommended Practice for Near-Field Antenna Measurements
"""

import math
import numpy as np
np.math = math 
from scipy.special import lpmv, spherical_jn, spherical_yn
from scipy.optimize import lsq_linear
from typing import Dict, Tuple, Optional, Union
import warnings
from multiprocessing import Pool
import os


# ==============================================================================
# Helper Functions for Normalized Associated Legendre Functions
# ==============================================================================

def cartesian_to_spherical(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert Cartesian coordinates to spherical coordinates.
    
    Args:
        x, y, z: Cartesian coordinates in meters
        
    Returns:
        r: Radial distance in meters
        theta: Polar angle in radians (0 to π)
        phi: Azimuthal angle in radians (0 to 2π)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / np.maximum(r, 1e-10), -1, 1))
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2*np.pi, phi)  # Ensure [0, 2π]
    return r, theta, phi


def spherical_to_cartesian(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        r: Radial distance in meters
        theta: Polar angle in radians
        phi: Azimuthal angle in radians
        
    Returns:
        x, y, z: Cartesian coordinates in meters
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def spherical_to_cartesian_field(E_r: np.ndarray, E_theta: np.ndarray, E_phi: np.ndarray,
                                 theta: np.ndarray, phi: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert field components from spherical to Cartesian basis.
    
    Args:
        E_r, E_theta, E_phi: Field components in spherical basis
        theta: Polar angle in radians
        phi: Azimuthal angle in radians
        
    Returns:
        E_x, E_y, E_z: Field components in Cartesian basis
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    E_x = (sin_theta * cos_phi * E_r + 
           cos_theta * cos_phi * E_theta - 
           sin_phi * E_phi)
    E_y = (sin_theta * sin_phi * E_r + 
           cos_theta * sin_phi * E_theta + 
           cos_phi * E_phi)
    E_z = cos_theta * E_r - sin_theta * E_theta
    
    return E_x, E_y, E_z

class LegendreCoefficientCache:
    """Cache for factorial ratios and normalization constants."""
    
    def __init__(self):
        self._factorial_ratio_cache = {}
        self._norm_factor_cache = {}
    
    def factorial_ratio(self, n: int, m: int) -> float:
        """
        Compute (n-m)! / (n+m)! efficiently without computing large factorials.
        
        For m >= 0: (n-m)! / (n+m)! = 1 / [(n-m+1)(n-m+2)...(n+m)]
        """
        key = (n, m)
        if key in self._factorial_ratio_cache:
            return self._factorial_ratio_cache[key]
        
        if m == 0:
            result = 1.0
        elif m > 0:
            result = 1.0
            for i in range(n - m + 1, n + m + 1):
                result /= i
        else:  # m < 0
            result = 1.0
            for i in range(n + m + 1, n - m + 1):
                result *= i
        
        self._factorial_ratio_cache[key] = result
        return result
    
    def normalization_factor(self, n: int, m: int) -> float:
        """
        Compute Hansen normalization factor: sqrt((2n+1)/2 * (n-m)!/(n+m)!)
        """
        key = (n, abs(m))
        if key in self._norm_factor_cache:
            return self._norm_factor_cache[key]
        
        abs_m = abs(m)
        fac_ratio = self.factorial_ratio(n, abs_m)
        result = np.sqrt((2 * n + 1) / 2 * fac_ratio)
        
        self._norm_factor_cache[key] = result
        return result


# Global cache instance
_legendre_cache = LegendreCoefficientCache()

import numpy as np
from typing import Tuple, Dict
import math


# ==============================================================================
# Optimized Associated Legendre Functions Using Recurrence Relations
# ==============================================================================

class LegendreCoefficientCache:
    """Cache for factorial ratios and normalization constants."""
    
    def __init__(self):
        self._factorial_ratio_cache = {}
        self._norm_factor_cache = {}
    
    def factorial_ratio(self, n: int, m: int) -> float:
        """
        Compute (n-m)! / (n+m)! efficiently without computing large factorials.
        
        For m >= 0: (n-m)! / (n+m)! = 1 / [(n-m+1)(n-m+2)...(n+m)]
        """
        key = (n, m)
        if key in self._factorial_ratio_cache:
            return self._factorial_ratio_cache[key]
        
        if m == 0:
            result = 1.0
        elif m > 0:
            result = 1.0
            for i in range(n - m + 1, n + m + 1):
                result /= i
        else:  # m < 0
            result = 1.0
            for i in range(n + m + 1, n - m + 1):
                result *= i
        
        self._factorial_ratio_cache[key] = result
        return result
    
    def normalization_factor(self, n: int, m: int) -> float:
        """
        Compute Hansen normalization factor: sqrt((2n+1)/2 * (n-m)!/(n+m)!)
        """
        key = (n, abs(m))
        if key in self._norm_factor_cache:
            return self._norm_factor_cache[key]
        
        abs_m = abs(m)
        fac_ratio = self.factorial_ratio(n, abs_m)
        result = np.sqrt((2 * n + 1) / 2 * fac_ratio)
        
        self._norm_factor_cache[key] = result
        return result


# Global cache instance
_legendre_cache = LegendreCoefficientCache()


def compute_legendre_recurrence(n_max: int, m: int, cos_theta: np.ndarray) -> np.ndarray:
    """
    Compute associated Legendre functions P_n^m(cos θ) for all n from m to n_max
    using recurrence relations. Much faster than calling lpmv repeatedly.
    
    Args:
        n_max: Maximum degree
        m: Order (can be negative)
        cos_theta: cos(θ) values, shape (N,)
    
    Returns:
        Array of shape (n_max - m + 1, N) containing P_m^m, P_{m+1}^m, ..., P_{n_max}^m
    """
    abs_m = abs(m)
    cos_theta = np.atleast_1d(cos_theta)
    N = len(cos_theta)
    
    # Avoid singularities
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    # Number of n values we need
    n_count = n_max - abs_m + 1
    P = np.zeros((n_count, N))
    
    if abs_m > n_max:
        return P
    
    # Starting value: P_m^m using the formula
    # P_m^m = (-1)^m * (2m-1)!! * sin^m(θ)
    if abs_m == 0:
        P[0, :] = 1.0
    else:
        # Double factorial: (2m-1)!! = 1*3*5*...*(2m-1)
        double_fac = 1.0
        for i in range(1, 2*abs_m, 2):
            double_fac *= i
        P[0, :] = double_fac * (sin_theta ** abs_m)
        if abs_m % 2 == 1:
            P[0, :] *= -1
    
    if n_count == 1:
        return P
    
    # Next value: P_{m+1}^m using specific recurrence
    # P_{m+1}^m = (2m+1) * cos(θ) * P_m^m
    if abs_m + 1 <= n_max:
        P[1, :] = (2 * abs_m + 1) * cos_theta * P[0, :]
    
    # General recurrence for n >= m+2:
    # (n-m) P_n^m = (2n-1) cos(θ) P_{n-1}^m - (n+m-1) P_{n-2}^m
    for i in range(2, n_count):
        n = abs_m + i
        coeff1 = (2 * n - 1) / (n - abs_m)
        coeff2 = (n + abs_m - 1) / (n - abs_m)
        P[i, :] = coeff1 * cos_theta * P[i-1, :] - coeff2 * P[i-2, :]
    
    return P


def compute_legendre_derivative_recurrence(n_max: int, m: int, cos_theta: np.ndarray,
                                           P: np.ndarray = None) -> np.ndarray:
    """
    Compute derivatives dP_n^m/dθ using recurrence relations.
    
    The derivative can be computed from:
    sin(θ) dP_n^m/dθ = n*cos(θ)*P_n^m - (n+m)*P_{n-1}^m
    
    Or for better numerical stability:
    dP_n^m/dθ = (n*cos(θ)*P_n^m - (n+m)*P_{n-1}^m) / sin(θ)
    
    Args:
        n_max: Maximum degree
        m: Order
        cos_theta: cos(θ) values
        P: Pre-computed P values (optional, will compute if not provided)
    
    Returns:
        Array of shape (n_max - |m| + 1, N) containing derivatives
    """
    abs_m = abs(m)
    cos_theta = np.atleast_1d(cos_theta)
    N = len(cos_theta)
    
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    # Compute P if not provided
    if P is None:
        P = compute_legendre_recurrence(n_max, m, cos_theta)
    
    n_count = n_max - abs_m + 1
    dP = np.zeros((n_count, N))
    
    # For n = m, special case:
    # dP_m^m/dθ = m * cos(θ)/sin(θ) * P_m^m
    # But we need P_{m-1}^m which is 0, so:
    # sin(θ) dP_m^m/dθ = m*cos(θ)*P_m^m - 0
    if n_count > 0:
        # Avoid division by zero at poles
        safe_sin = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
        dP[0, :] = abs_m * cos_theta * P[0, :] / safe_sin
    
    # For n > m, use the recurrence with P_{n-1}^m
    for i in range(1, n_count):
        n = abs_m + i
        safe_sin = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
        dP[i, :] = (n * cos_theta * P[i, :] - (n + abs_m) * P[i-1, :]) / safe_sin
    
    return dP


def normalized_associated_legendre(n: int, m: int, 
                                             theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized version using recurrence relations and caching.
    Drop-in replacement for normalized_associated_legendre.
    
    Args:
        n: Degree
        m: Order (can be negative)
        theta: Polar angle(s) in radians
    
    Returns:
        P_norm: Normalized associated Legendre function
        dP_norm: Derivative with respect to theta
    """
    theta = np.atleast_1d(theta)
    cos_theta = np.cos(theta)
    
    abs_m = abs(m)
    
    # Compute P_n^m for this n and all lower n values using recurrence
    # We need P_{n-1}^m for the derivative, so compute up to n
    P_all = compute_legendre_recurrence(n, abs(m), cos_theta)
    
    # Extract P_n^m (it's the last one in the array)
    P_n_m = P_all[-1, :]
    
    # Compute derivative
    dP_all = compute_legendre_derivative_recurrence(n, abs(m), cos_theta, P_all)
    dP_n_m_dtheta = dP_all[-1, :]
    
    # Apply Hansen normalization
    norm_factor = _legendre_cache.normalization_factor(n, m)
    P_norm = norm_factor * P_n_m
    dP_norm = norm_factor * dP_n_m_dtheta
    
    return P_norm, dP_norm


def compute_all_modes_legendre(n_max: int, m_max: int, 
                               theta: np.ndarray) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    """
    Compute normalized Legendre functions and derivatives for ALL modes at once.
    This is much more efficient when you need many (n,m) pairs.
    
    Args:
        n_max: Maximum degree
        m_max: Maximum order
        theta: Polar angle(s) in radians
    
    Returns:
        Dictionary mapping (n, m) -> (P_norm, dP_norm)
    """
    theta = np.atleast_1d(theta)
    
    # CRITICAL: Apply pole avoidance BEFORE computing Legendre functions
    theta_safe = np.copy(theta)
    epsilon = 1e-3
    at_north_pole = theta < epsilon
    at_south_pole = theta > (np.pi - epsilon)
    
    theta_safe = np.where(at_north_pole, epsilon, theta_safe)
    theta_safe = np.where(at_south_pole, np.pi - epsilon, theta_safe)
    
    cos_theta = np.cos(theta_safe)
    
    results = {}
    
    # For each order m, compute all degrees n >= |m| at once
    for m in range(-m_max, m_max + 1):
        abs_m = abs(m)
        
        if abs_m > n_max:
            continue
        
        # Compute all P_n^m for n from abs_m to n_max
        P_all = compute_legendre_recurrence(n_max, m, cos_theta)
        dP_all = compute_legendre_derivative_recurrence(n_max, m, cos_theta, P_all)
        
        # Store normalized results for each n
        for i, n in enumerate(range(abs_m, n_max + 1)):
            if n < 1:  # Skip n=0 for spherical wave expansion
                continue
            
            norm_factor = _legendre_cache.normalization_factor(n, m)
            P_norm = norm_factor * P_all[i, :]
            dP_norm = norm_factor * dP_all[i, :]
            
            results[(n, m)] = (P_norm, dP_norm)
    
    return results

def far_field_pattern_functions(n: int, m: int, 
                                         theta: np.ndarray, 
                                         phi: np.ndarray,
                                         legendre_cache: Dict = None):
    """
    Optimized version that can use pre-computed Legendre functions.
    
    IMPORTANT: Always applies pole avoidance (epsilon=1e-3) consistently
    with how the cache was computed.
    
    Args:
        n: Degree
        m: Order
        theta: Polar angles  
        phi: Azimuthal angles
        legendre_cache: Optional pre-computed dict from compute_all_modes_legendre
    
    Returns:
        (K1_theta, K1_phi), (K2_theta, K2_phi)
    """
    abs_m = abs(m)
    
    # Always apply pole avoidance for consistency
    theta_safe = np.copy(theta)
    epsilon = 1e-3
    at_north_pole = theta < epsilon
    at_south_pole = theta > (np.pi - epsilon)
    
    theta_safe = np.where(at_north_pole, epsilon, theta_safe)
    theta_safe = np.where(at_south_pole, np.pi - epsilon, theta_safe)
    
    # Get Legendre functions (from cache or compute)
    if legendre_cache is not None and (n, m) in legendre_cache:
        P_norm, dP_norm_dtheta = legendre_cache[(n, m)]
    else:
        P_norm, dP_norm_dtheta = normalized_associated_legendre(n, m, theta_safe)
    
    prefactor = np.sqrt(2 / (n * (n + 1)))
    
    if m == 0:
        sign_factor = 1.0
    else:
        sign_factor = (m / abs(m)) ** m
    
    phase = np.exp(-1j * m * phi)
    i_factor_1 = (1j) ** (n)
    i_factor_2 = (1j) ** (n + 1)
    
    sin_theta = np.sin(theta_safe)
    mP_over_sin = m * P_norm / sin_theta
    
    K1_theta = prefactor * sign_factor * phase * i_factor_1 * (1j * mP_over_sin)
    K1_phi = prefactor * sign_factor * phase * i_factor_1 * (dP_norm_dtheta)
    
    K2_theta = prefactor * sign_factor * phase * i_factor_2 * (dP_norm_dtheta)
    K2_phi = prefactor * sign_factor * phase * i_factor_2 * (-1j * mP_over_sin)
    
    return (K1_theta, K1_phi), (K2_theta, K2_phi)

def near_field_pattern_functions(n: int, m: int, r: np.ndarray, 
                                 theta: np.ndarray, phi: np.ndarray, 
                                 k: float, legendre_cache: Dict = None):
    """
    Calculate near-field pattern functions using TICRA convention.
    Uses Hankel function of second kind h_n^(2) = j_n - i*y_n
    """
    abs_m = abs(m)
    
    # Apply pole avoidance
    theta_safe = np.copy(theta)
    epsilon = 1e-3
    theta_safe = np.where(theta < epsilon, epsilon, theta_safe)
    theta_safe = np.where(theta > (np.pi - epsilon), np.pi - epsilon, theta_safe)
    
    # Get Legendre functions
    if legendre_cache is not None and (n, m) in legendre_cache:
        P_norm, dP_norm = legendre_cache[(n, m)]
    else:
        P_norm, dP_norm = normalized_associated_legendre(n, m, theta_safe)
    
    # Radial functions - Hankel function of second kind
    kr = k * r
    j_n = spherical_jn(n, kr)
    y_n = spherical_yn(n, kr)
    h_n = j_n - 1j * y_n  # h_n^(2)
    
    # Derivative of kr*h_n
    if n == 0:
        j_n_m1 = 0.0
        y_n_m1 = -np.cos(kr) / kr
    else:
        j_n_m1 = spherical_jn(n-1, kr)
        y_n_m1 = spherical_yn(n-1, kr)
    
    h_n_m1 = j_n_m1 - 1j * y_n_m1
    dkrh_n = kr * h_n_m1 - n * h_n  # d/d(kr){kr*h_n^(2)}
    
    # Common factors from equations (4.214-4.215)
    prefactor = 1 / np.sqrt(2 * np.pi) * 1 / np.sqrt(n * (n + 1))
    
    # TICRA sign factor: (m/|m|)^m (positive, not negative)
    if m == 0:
        sign_factor = 1.0
    else:
        sign_factor = (m / abs(m)) ** m
    
    # TICRA phase convention: exp(-jmφ)
    phase = np.exp(-1j * m * phi)
    
    sin_theta = np.sin(theta_safe)
    jmP_over_sin = 1j * m * P_norm / sin_theta
    
    coef = prefactor * sign_factor * phase
    
    # Mode 1 (TE): m'_nm components from eq (4.216)
    # Note: TICRA uses negative signs
    F1_E_r = 0.0
    F1_E_theta = -coef * h_n * jmP_over_sin
    F1_E_phi = -coef * h_n * dP_norm
    
    # Mode 2 (TM): n'_nm components from eq (4.217)
    F2_E_r = coef * (n * (n + 1) / kr) * h_n * P_norm
    F2_E_theta = coef * (dkrh_n / kr) * dP_norm
    F2_E_phi = -coef * (dkrh_n / kr) * jmP_over_sin  # Note negative sign
    
    # H fields - swap per reciprocity
    F1_H_r = F2_E_r
    F1_H_theta = F2_E_theta
    F1_H_phi = F2_E_phi
    
    F2_H_r = F1_E_r
    F2_H_theta = F1_E_theta
    F2_H_phi = F1_E_phi
    
    return (F1_E_r, F1_E_theta, F1_E_phi), (F2_E_r, F2_E_theta, F2_E_phi), \
           (F1_H_r, F1_H_theta, F1_H_phi), (F2_H_r, F2_H_theta, F2_H_phi)

def compute_mode_coefficients_batch(args):
    """Compute Q coefficients for a batch of modes - for parallel processing"""
    modes_batch, THETA, PHI, E_THETA, E_PHI, sin_theta, theta_unique, phi_unique, norm_factor, legendre_data = args
    
    results = []
    
    for n, m in modes_batch:
        # Unpack Legendre data for this mode
        P_norm, dP_norm = legendre_data[(n, m)]
        
        # Broadcast to 2D grid
        P_norm_2d = P_norm[:, np.newaxis]
        dP_norm_2d = dP_norm[:, np.newaxis]
        
        # TICRA pattern functions
        prefactor = np.sqrt(2 / (n * (n + 1)))
        sign_factor = (m / abs(m)) ** m if m != 0 else 1.0
        phase = np.exp(-1j * m * PHI)
        i_factor_1 = (1j) ** n
        i_factor_2 = (1j) ** (n + 1)
        
        sin_theta_safe = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
        mP_over_sin = 1j * m * P_norm_2d / sin_theta_safe
        
        # Pattern functions
        K1_theta = prefactor * sign_factor * phase * i_factor_1 * mP_over_sin
        K1_phi = prefactor * sign_factor * phase * i_factor_1 * dP_norm_2d
        
        K2_theta = prefactor * sign_factor * phase * i_factor_2 * (-dP_norm_2d)
        K2_phi = prefactor * sign_factor * phase * i_factor_2 * (-mP_over_sin)
        
        # Inner product: ∫∫ (E · F*) sin(θ) dθ dφ
        integrand_1 = (E_THETA * np.conj(K1_theta) + E_PHI * np.conj(K1_phi)) * sin_theta
        Q1 = np.trapz(np.trapz(integrand_1, phi_unique, axis=1), theta_unique, axis=0)
        Q1 *= norm_factor
        
        integrand_2 = (E_THETA * np.conj(K2_theta) + E_PHI * np.conj(K2_phi)) * sin_theta
        Q2 = np.trapz(np.trapz(integrand_2, phi_unique, axis=1), theta_unique, axis=0)
        Q2 *= norm_factor
        
        # Calculate mode power
        mode_power = (abs(Q1)**2 + abs(Q2)**2) / 2.0
        
        results.append(((n, m), Q1, Q2, mode_power))
    
    return results
# ==============================================================================
# Main SWE Class
# ==============================================================================

class SphericalWaveExpansion:
    """
    Spherical Wave Expansion representation of electromagnetic fields.
    
    Attributes:
        Q1_coeffs: Dictionary of Q₁ coefficients with keys (n, m)
        Q2_coeffs: Dictionary of Q₂ coefficients with keys (n, m)
        frequency: Frequency in Hz
        k: Wavenumber in rad/m
        NMAX: Maximum degree
        MMAX: Maximum order
    """
    
    def __init__(self, 
                 Q1_coeffs: Optional[Dict[Tuple[int, int], complex]] = None,
                 Q2_coeffs: Optional[Dict[Tuple[int, int], complex]] = None,
                 frequency: Optional[float] = None,
                 NMAX: Optional[int] = None,
                 MMAX: Optional[int] = None):
        """
        Initialize SWE object.
        
        Args:
            Q1_coeffs: Q₁ coefficients dictionary
            Q2_coeffs: Q₂ coefficients dictionary
            frequency: Frequency in Hz
            NMAX: Maximum degree (auto-detected if None)
            MMAX: Maximum order (auto-detected if None)
        """
        self.Q1_coeffs = Q1_coeffs if Q1_coeffs is not None else {}
        self.Q2_coeffs = Q2_coeffs if Q2_coeffs is not None else {}
        self._frequency = frequency
        
        # Auto-detect NMAX and MMAX if not provided
        if NMAX is None or MMAX is None:
            all_keys = list(self.Q1_coeffs.keys()) + list(self.Q2_coeffs.keys())
            if all_keys:
                self.NMAX = max(n for n, m in all_keys)
                self.MMAX = max(abs(m) for n, m in all_keys)
            else:
                self.NMAX = NMAX if NMAX is not None else 0
                self.MMAX = MMAX if MMAX is not None else 0
        else:
            self.NMAX = NMAX
            self.MMAX = MMAX
    
    @property
    def frequency(self) -> Optional[float]:
        """Frequency in Hz."""
        return self._frequency
    
    @frequency.setter
    def frequency(self, freq: float):
        """Set frequency in Hz."""
        self._frequency = freq
    
    @property
    def k(self) -> Optional[float]:
        """Wavenumber in rad/m."""
        if self._frequency is None:
            return None
        return 2 * np.pi * self._frequency / 299792458.0
    
    @property
    def wavelength(self) -> Optional[float]:
        """Wavelength in meters."""
        if self._frequency is None:
            return None
        return 299792458.0 / self._frequency
    
    def far_field(self, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimized far_field calculation using batch Legendre computation.
        
        Replace the existing far_field method with this one.
        
        Args:
            theta: Polar angle(s) in radians
            phi: Azimuthal angle(s) in radians
            
        Returns:
            E_theta: Theta component of electric field
            E_phi: Phi component of electric field
        """
        if self.k is None:
            raise ValueError("Frequency must be set before computing far field")
        
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)
        
        if theta.shape != phi.shape:
            theta, phi = np.broadcast_arrays(theta, phi)
        
        E_theta = np.zeros_like(theta, dtype=complex)
        E_phi = np.zeros_like(phi, dtype=complex)
        
        all_modes = set(self.Q1_coeffs.keys()) | set(self.Q2_coeffs.keys())
        
        if not all_modes:
            return E_theta, E_phi
        
        # PRE-COMPUTE ALL LEGENDRE FUNCTIONS AT ONCE
        # This is the key optimization - compute once, use many times
        legendre_cache = compute_all_modes_legendre(self.NMAX, self.MMAX, theta)
        
        # Now loop through modes using the pre-computed values
        for (n, m) in all_modes:
            # Use optimized version with cache
            (K1_theta, K1_phi), (K2_theta, K2_phi) = \
                far_field_pattern_functions(n, m, theta, phi, legendre_cache)
            
            if (n, m) in self.Q1_coeffs:
                Q1 = self.Q1_coeffs[(n, m)]
                E_theta += Q1 * K1_theta
                E_phi += Q1 * K1_phi
            
            if (n, m) in self.Q2_coeffs:
                Q2 = self.Q2_coeffs[(n, m)]
                E_theta += Q2 * K2_theta
                E_phi += Q2 * K2_phi
                
        return E_theta, E_phi
    
    @classmethod
    def from_far_field(cls,
                            theta: np.ndarray,
                            phi: np.ndarray,
                            E_theta: np.ndarray,
                            E_phi: np.ndarray,
                            frequency: float,
                            r0: float = None,
                            NMAX_initial: int = 100,  # Start larger
                            MMAX_initial: int = 50,
                            power_threshold: float = 0.995,
                            high_mode_power_threshold: float = 0.005,
                            azimuthal_power_threshold: float = 0.00005,
                            use_multiprocessing: bool = True,
                            n_workers: int = None):
        """
        Adaptively calculate Q coefficients from far-field patterns using orthogonality integral.
        
        Uses Hansen's reciprocity theorem with adaptive mode selection and parallel computation.
        
        Args:
            theta: Polar angles (radians) - must be on regular grid
            phi: Azimuthal angles (radians) - must be on regular grid
            E_theta: Theta component of electric field
            E_phi: Phi component of electric field
            frequency: Frequency in Hz
            r0: Radius of minimum sphere enclosing sources (meters)
            NMAX_initial: Starting NMAX (default: 50)
            MMAX_initial: Starting MMAX (if None, grows with NMAX)
            power_threshold: Retain modes with this fraction of power (0.995 = 99.5%)
            high_mode_power_threshold: Max power in top 10% n-modes (0.005 = 0.5%)
            azimuthal_power_threshold: Min power per |m| to include (0.00005 = 0.005%)
            use_multiprocessing: Enable parallel computation
            n_workers: Number of parallel workers (None = auto-detect)
        
        Returns:
            SphericalWaveExpansion object with adaptively determined modes
        """
        
        k = 2 * np.pi * frequency / 299792458.0
        
        # Reshape to 2D grid if needed
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)
        E_theta = np.atleast_1d(E_theta)
        E_phi = np.atleast_1d(E_phi)
        
        # Check if data is on regular grid
        theta_unique = np.unique(theta)
        phi_unique = np.unique(phi)
        
        if len(theta_unique) * len(phi_unique) != len(theta):
            raise ValueError("Data must be on regular (theta, phi) grid for integration")
        
        # Reshape to 2D grid
        n_theta = len(theta_unique)
        n_phi = len(phi_unique)
        
        THETA = theta.reshape(n_theta, n_phi)
        PHI = phi.reshape(n_theta, n_phi)
        E_THETA = E_theta.reshape(n_theta, n_phi)
        E_PHI = E_phi.reshape(n_theta, n_phi)
        
        sin_theta = np.sin(THETA)
        
        # Override NMAX_initial from r0 if provided
        if r0 is not None:
            kr0 = k * r0
            NMAX_estimated = int(np.ceil(kr0 + max(10, 3.6 * (kr0 ** (1/3)))))
            NMAX_initial = max(NMAX_initial, NMAX_estimated)
            print(f"Estimated NMAX from r0={r0}m: kr0={kr0:.2f}, NMAX={NMAX_estimated}")
            print(f"Using NMAX_initial={NMAX_initial}")
        else:
            print(f"Using NMAX_initial={NMAX_initial}")
        
        # MMAX_initial = None means let it grow with NMAX
        use_adaptive_mmax = (MMAX_initial is None)
        if MMAX_initial is None:
            MMAX_initial = NMAX_initial
        
        # Normalization factor from Hansen
        norm_factor = 1.0 / (4.0 * np.pi)
        
        # Auto-detect number of workers
        if n_workers is None:
            n_workers = min(os.cpu_count(), 8)  # Cap at 8 to avoid overhead
        
        NMAX = NMAX_initial
        max_iterations = 10
        
        for iteration in range(max_iterations):
            # Let MMAX grow with NMAX if not explicitly specified
            if use_adaptive_mmax:
                MMAX_current = NMAX
            else:
                MMAX_current = min(MMAX_initial, NMAX)
            
            print(f"\nIteration {iteration + 1}: Computing coefficients for NMAX={NMAX}, MMAX={MMAX_current}")
            
            # Pre-compute Legendre functions for all modes
            print(f"  Computing Legendre functions...")
            legendre_cache = compute_all_modes_legendre(NMAX, MMAX_current, THETA[:, 0])
            
            # Build mode list
            modes = []
            for n in range(1, NMAX + 1):
                for m in range(-min(n, MMAX_current), min(n, MMAX_current) + 1):
                    modes.append((n, m))
            
            total_modes = len(modes)
            print(f"  Extracting {total_modes} mode coefficients via integration...")
            
            # Compute coefficients (parallel or serial)
            if use_multiprocessing and total_modes > 100:
                # Split modes into batches for parallel processing
                batch_size = max(10, total_modes // (n_workers * 4))
                mode_batches = [modes[i:i + batch_size] for i in range(0, len(modes), batch_size)]
                
                print(f"    Using {n_workers} workers, {len(mode_batches)} batches...")
                
                # Prepare arguments for each batch
                args_list = []
                for batch in mode_batches:
                    # Create a subset of legendre data for this batch
                    legendre_subset = {(n, m): legendre_cache[(n, m)] for (n, m) in batch}
                    args_list.append((batch, THETA, PHI, E_THETA, E_PHI, sin_theta, 
                                    theta_unique, phi_unique, norm_factor, legendre_subset))
                
                # Compute in parallel
                with Pool(n_workers) as pool:
                    batch_results = pool.map(compute_mode_coefficients_batch, args_list)
                
                # Flatten results
                Q1_coeffs = {}
                Q2_coeffs = {}
                mode_powers = []
                for batch_result in batch_results:
                    for (n, m), Q1, Q2, mode_power in batch_result:
                        Q1_coeffs[(n, m)] = Q1
                        Q2_coeffs[(n, m)] = Q2
                        mode_powers.append(((n, m), mode_power))
            
            else:
                # Serial computation
                Q1_coeffs = {}
                Q2_coeffs = {}
                mode_powers = []
                
                for mode_idx, (n, m) in enumerate(modes):
                    P_norm, dP_norm = legendre_cache[(n, m)]
                    
                    P_norm_2d = P_norm[:, np.newaxis]
                    dP_norm_2d = dP_norm[:, np.newaxis]
                    
                    prefactor = np.sqrt(2 / (n * (n + 1)))
                    sign_factor = (m / abs(m)) ** m if m != 0 else 1.0
                    phase = np.exp(-1j * m * PHI)
                    i_factor_1 = (1j) ** n
                    i_factor_2 = (1j) ** (n + 1)
                    
                    sin_theta_safe = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
                    mP_over_sin = 1j * m * P_norm_2d / sin_theta_safe
                    
                    K1_theta = prefactor * sign_factor * phase * i_factor_1 * mP_over_sin
                    K1_phi = prefactor * sign_factor * phase * i_factor_1 * dP_norm_2d
                    
                    K2_theta = prefactor * sign_factor * phase * i_factor_2 * (-dP_norm_2d)
                    K2_phi = prefactor * sign_factor * phase * i_factor_2 * (-mP_over_sin)
                    
                    integrand_1 = (E_THETA * np.conj(K1_theta) + E_PHI * np.conj(K1_phi)) * sin_theta
                    Q1 = np.trapz(np.trapz(integrand_1, phi_unique, axis=1), theta_unique, axis=0)
                    Q1 *= norm_factor
                    
                    integrand_2 = (E_THETA * np.conj(K2_theta) + E_PHI * np.conj(K2_phi)) * sin_theta
                    Q2 = np.trapz(np.trapz(integrand_2, phi_unique, axis=1), theta_unique, axis=0)
                    Q2 *= norm_factor
                    
                    Q1_coeffs[(n, m)] = Q1
                    Q2_coeffs[(n, m)] = Q2
                    
                    mode_power = (abs(Q1)**2 + abs(Q2)**2) / 2.0
                    mode_powers.append(((n, m), mode_power))
            
            print(f"    Completed {len(Q1_coeffs)} modes")
            
            # Calculate total power
            mode_powers_array = np.array([p for _, p in mode_powers])
            total_power = np.sum(mode_powers_array)
            
            print(f"  Total power: {total_power:.6e}")
            
            # Check power in top 10% of n values
            n_values = np.array([n for (n, m), _ in mode_powers])
            n_cutoff = max(1, int(np.ceil(0.1 * NMAX)))
            high_n_mask = n_values > (NMAX - n_cutoff)
            high_mode_power = np.sum(mode_powers_array[high_n_mask])
            
            high_mode_fraction = high_mode_power / total_power if total_power > 0 else 0
            print(f"  Power in top {n_cutoff} n-modes: {high_mode_fraction*100:.2f}%")
            
            # Check convergence
            if high_mode_fraction < high_mode_power_threshold:
                print(f"  ✓ Convergence achieved: high modes contain {high_mode_fraction*100:.2f}% < {high_mode_power_threshold*100:.0f}%")
                
                # Calculate power per |m| for azimuthal truncation
                power_per_m = np.zeros(MMAX_current + 1)
                for (n, m), mode_power in mode_powers:
                    power_per_m[abs(m)] += mode_power
                
                # Determine MMAX based on azimuthal power distribution
                # Keep all m where power is above threshold
                MMAX_truncated = 0
                for m_abs in range(MMAX_current, -1, -1):
                    if power_per_m[m_abs] / total_power >= azimuthal_power_threshold:
                        MMAX_truncated = m_abs
                        break  # Found the highest m with significant power

                # Alternative: print diagnostic info
                print(f"  Power per |m|:")
                for m_abs in range(min(10, MMAX_current + 1)):
                    print(f"    |m|={m_abs}: {power_per_m[m_abs]/total_power*100:.3f}%")
                if MMAX_current > 10:
                    print(f"    ... (showing first 10)")
                    print(f"    |m|={MMAX_current}: {power_per_m[MMAX_current]/total_power*100:.3f}%")

                print(f"  Azimuthal truncation: MMAX {MMAX_current} → {MMAX_truncated} (threshold: {azimuthal_power_threshold*100:.2f}%)")
                
                # Power-based mode truncation
                mode_powers_sorted = sorted(mode_powers, key=lambda x: x[1], reverse=True)
                
                cumulative_power = 0.0
                Q1_final = {}
                Q2_final = {}
                NMAX_truncated = 0
                
                for (n, m), mode_power in mode_powers_sorted:
                    if abs(m) <= MMAX_truncated:
                        Q1_final[(n, m)] = Q1_coeffs[(n, m)]
                        Q2_final[(n, m)] = Q2_coeffs[(n, m)]
                        cumulative_power += mode_power
                        NMAX_truncated = max(NMAX_truncated, n)
                        
                        if cumulative_power / total_power >= power_threshold:
                            break
                
                retained_fraction = cumulative_power / total_power
                print(f"\n  Power-based truncation: {len(Q1_final)} modes retained")
                print(f"  NMAX: {NMAX} → {NMAX_truncated}, MMAX: {MMAX_current} → {MMAX_truncated}")
                print(f"  Retained power: {retained_fraction*100:.2f}%")
                
                # Verify final power distribution
                n_values_final = np.array([n for (n, m) in Q1_final.keys()])
                mode_powers_final = np.array([(abs(Q1_final[(n, m)])**2 + abs(Q2_final[(n, m)])**2) / 2.0 
                                            for (n, m) in Q1_final.keys()])
                total_power_final = np.sum(mode_powers_final)
                
                n_cutoff_final = max(1, int(np.ceil(0.1 * NMAX_truncated)))
                high_n_mask_final = n_values_final > (NMAX_truncated - n_cutoff_final)
                high_mode_power_final = np.sum(mode_powers_final[high_n_mask_final])
                high_mode_fraction_final = high_mode_power_final / total_power_final if total_power_final > 0 else 0
                
                print(f"  Final solution: {len(Q1_final)} modes, total power = {total_power_final:.6e}")
                print(f"  Final high mode check: {high_mode_fraction_final*100:.2f}% in top {n_cutoff_final} n-modes")
                
                if high_mode_fraction_final >= high_mode_power_threshold:
                    print(f"  ⚠ Warning: High mode power criterion not met after truncation ({high_mode_fraction_final*100:.2f}% >= {high_mode_power_threshold*100:.0f}%).")
                
                print(f"\n✓ Adaptive coefficient extraction complete: NMAX={NMAX_truncated}, MMAX={MMAX_truncated}")
                return cls(Q1_final, Q2_final, frequency, NMAX_truncated, MMAX_truncated)
            
            else:
                # Need more modes
                NMAX += 50
                print(f"  → Increasing NMAX to {NMAX}")
        
        # Max iterations reached
        print(f"\n⚠ Warning: Max iterations ({max_iterations}) reached without full convergence.")
        print(f"  Returning solution with NMAX={NMAX}, MMAX={MMAX_current}")
        return cls(Q1_coeffs, Q2_coeffs, frequency, NMAX, MMAX_current)

    @classmethod
    def from_sph_file(cls, filename: str) -> 'SphericalWaveExpansion':
        """
        Create SWE object from TICRA .sph file.
        
        Args:
            filename: Path to .sph file
            
        Returns:
            SphericalWaveExpansion object
        """
        data = read_ticra_sph(filename)
        
        # Convert frequency from GHz to Hz
        frequency = data['frequency'] * 1e9 if data['frequency'] is not None else None
        
        return cls(
            Q1_coeffs=data['Q1_coeffs'],
            Q2_coeffs=data['Q2_coeffs'],
            frequency=frequency,
            NMAX=data['NMAX'],
            MMAX=data['MMAX']
        )
    
    def to_sph_file(self, filename: str, 
                   NTHE: int = 181, NPHI: int = 361,
                   description: str = "Generated by SWE module"):
        """
        Write SWE object to TICRA .sph file.
        
        Args:
            filename: Output file path
            NTHE: Number of theta samples (for header)
            NPHI: Number of phi samples (for header)
            description: Description text
        """
        write_ticra_sph(
            filename,
            self.Q1_coeffs,
            self.Q2_coeffs,
            self.frequency / 1e9 if self.frequency is not None else 1.0,
            NTHE, NPHI,
            self.NMAX, self.MMAX,
            description
        )
    
    def near_field(self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> \
            Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], 
                Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calculate near-field E and H components.
        
        Args:
            r: Radial distance(s) in meters
            theta: Polar angle(s) in radians
            phi: Azimuthal angle(s) in radians
            
        Returns:
            E: Tuple of (E_r, E_theta, E_phi) in V/m
            H: Tuple of (H_r, H_theta, H_phi) in A/m
        """
        if self.k is None:
            raise ValueError("Frequency must be set before computing near field")
        
        r = np.atleast_1d(r)
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)
        
        if not (r.shape == theta.shape == phi.shape):
            r, theta, phi = np.broadcast_arrays(r, theta, phi)
        
        # Initialize output arrays
        E_r = np.zeros_like(r, dtype=complex)
        E_theta = np.zeros_like(theta, dtype=complex)
        E_phi = np.zeros_like(phi, dtype=complex)
        H_r = np.zeros_like(r, dtype=complex)
        H_theta = np.zeros_like(theta, dtype=complex)
        H_phi = np.zeros_like(phi, dtype=complex)
        
        all_modes = set(self.Q1_coeffs.keys()) | set(self.Q2_coeffs.keys())
        
        if not all_modes:
            return (E_r, E_theta, E_phi), (H_r, H_theta, H_phi)
        
        # Pre-compute Legendre functions
        legendre_cache = compute_all_modes_legendre(self.NMAX, self.MMAX, theta)

        # Scaling for 4pi power
        E_prefactor = np.sqrt(4*np.pi)
        H_prefactor = np.sqrt(4*np.pi)
        
        # Loop through modes
        for (n, m) in all_modes:
            # Get near-field pattern functions for this mode
            F1_E, F2_E, F1_H, F2_H = near_field_pattern_functions(
                n, m, r, theta, phi, self.k, legendre_cache
            )
            
            # Q1 contribution (TE to r modes)
            if (n, m) in self.Q1_coeffs:
                Q1 = self.Q1_coeffs[(n, m)]
                E_r += E_prefactor * Q1 * F1_E[0]
                E_theta += E_prefactor * Q1 * F1_E[1]
                E_phi += E_prefactor * Q1 * F1_E[2]
                
                H_r += H_prefactor * Q1 * F1_H[0]
                H_theta += H_prefactor * Q1 * F1_H[1]
                H_phi += H_prefactor * Q1 * F1_H[2]
            
            # Q2 contribution (TM to r modes)
            if (n, m) in self.Q2_coeffs:
                Q2 = self.Q2_coeffs[(n, m)]
                E_r += E_prefactor * Q2 * F2_E[0]
                E_theta += E_prefactor * Q2 * F2_E[1]
                E_phi += E_prefactor * Q2 * F2_E[2]
                
                H_r += H_prefactor * Q2 * F2_H[0]
                H_theta += H_prefactor * Q2 * F2_H[1]
                H_phi += H_prefactor * Q2 * F2_H[2]
        
        return (E_r, E_theta, E_phi), (H_r, H_theta, H_phi)
        
    def near_field_cartesian(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> \
            Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
                  Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calculate near-field E and H in Cartesian coordinates.
        
        Args:
            x, y, z: Cartesian coordinates in meters
            
        Returns:
            E: Tuple of (E_x, E_y, E_z) components in V/m
            H: Tuple of (H_x, H_y, H_z) components in A/m
        """
        # Convert to spherical coordinates
        r, theta, phi = cartesian_to_spherical(x, y, z)
        
        # Get fields in spherical basis
        (E_r, E_theta, E_phi), (H_r, H_theta, H_phi) = self.near_field(r, theta, phi)
        
        # Convert to Cartesian basis
        E_x, E_y, E_z = spherical_to_cartesian_field(E_r, E_theta, E_phi, theta, phi)
        H_x, H_y, H_z = spherical_to_cartesian_field(H_r, H_theta, H_phi, theta, phi)
        
        return (E_x, E_y, E_z), (H_x, H_y, H_z)
    
    def currents_on_surface(self, rr: np.ndarray, unr: np.ndarray, dSr: np.ndarray,
                                swe_origin: np.ndarray = None,
                                swe_rotation: Optional[Tuple[float, float, float]] = None,
                                chunk_size: Optional[int] = None,
                                n_threads: Optional[int] = None) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Calculate equivalent surface currents on an arbitrary reflector surface from SWE source.
        
        Uses the reciprocity-based surface current formulation from Hansen Appendix A1.
        Surface currents: J = n × H, M = -n × E
        
        Args:
            rr: Nr x 3 array of reflector surface points (Cartesian, meters)
            unr: Nr x 3 array of surface normal vectors (outward)
            dSr: Nr array of surface element areas (m²)
            swe_origin: 3-element array, SWE coordinate origin in reflector frame (meters)
                        Default is [0, 0, 0]
            swe_rotation: (alpha, beta, gamma) Euler angles (radians, ZYZ convention) 
                        rotating SWE frame to reflector frame. Default is no rotation.
            chunk_size: Points per chunk for parallel processing (None for auto)
            n_threads: Number of worker threads (None for auto)
            
        Returns:
            Jrr: Nr x 3 array of equivalent electric currents (A)
            Mrr: Nr x 3 array of equivalent magnetic currents (V)
        """
        from concurrent.futures import ThreadPoolExecutor
        import os
        
        Nr = len(rr)
        
        # Default origin at coordinate system origin
        if swe_origin is None:
            swe_origin = np.array([0., 0., 0.])
        
        # Transform reflector points to SWE coordinate system
        rr_swe = rr - swe_origin[np.newaxis, :]
        
        if swe_rotation is not None:
            # Apply inverse rotation (reflector -> SWE frame)
            rr_swe = self._apply_inverse_rotation(rr_swe, swe_rotation)
            unr_swe = self._apply_inverse_rotation(unr, swe_rotation)
        else:
            unr_swe = unr.copy()
        
        # Auto-optimize parameters
        if chunk_size is None:
            chunk_size = min(1000, max(100, Nr // 8))
        if n_threads is None:
            n_threads = min(8, os.cpu_count() or 1)
        
        # Create chunks
        chunks = [(i, min(i + chunk_size, Nr)) for i in range(0, Nr, chunk_size)]
        
        # Worker function
        def _compute_fields_chunk(start: int, end: int):
            r_chunk = rr_swe[start:end]
            x, y, z = r_chunk[:, 0], r_chunk[:, 1], r_chunk[:, 2]
            (Ex, Ey, Ez), (Hx, Hy, Hz) = self.near_field_cartesian(x, y, z)
            return np.stack([Ex, Ey, Ez], axis=1), np.stack([Hx, Hy, Hz], axis=1)
        
        # Parallel field calculation
        E_total = np.zeros((Nr, 3), dtype=np.complex128)
        H_total = np.zeros((Nr, 3), dtype=np.complex128)
        
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(_compute_fields_chunk, start, end) for start, end in chunks]
            for i, future in enumerate(futures):
                start, end = chunks[i]
                E_chunk, H_chunk = future.result()
                E_total[start:end] = E_chunk
                H_total[start:end] = H_chunk
        
        # Transform E and H back to reflector frame if rotated
        if swe_rotation is not None:
            E_total = self._apply_rotation(E_total, swe_rotation)
            H_total = self._apply_rotation(H_total, swe_rotation)
        
        # Calculate surface currents: J = n × H, M = -n × E
        Jr = np.cross(unr, H_total)
        Mr = -np.cross(unr, E_total)
        
        # Apply surface element areas and sign convention (matches GetCurrents)
        Jrr = -Jr * dSr[:, np.newaxis]
        Mrr = Mr * dSr[:, np.newaxis]
        
        return Jrr, Mrr

    def _apply_rotation(self, vectors: np.ndarray, angles: Tuple[float, float, float]) -> np.ndarray:
        """
        Apply ZYZ Euler rotation to vectors.
        
        Args:
            vectors: N x 3 array of vectors
            angles: (alpha, beta, gamma) Euler angles in radians
            
        Returns:
            Rotated N x 3 array
        """
        alpha, beta, gamma = angles
        
        # ZYZ Euler rotation matrices
        Rz_alpha = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                            [np.sin(alpha), np.cos(alpha), 0],
                            [0, 0, 1]])
        Ry_beta = np.array([[np.cos(beta), 0, np.sin(beta)],
                            [0, 1, 0],
                            [-np.sin(beta), 0, np.cos(beta)]])
        Rz_gamma = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                            [np.sin(gamma), np.cos(gamma), 0],
                            [0, 0, 1]])
        
        # Combined rotation: R = Rz(γ) Ry(β) Rz(α)
        R = Rz_gamma @ Ry_beta @ Rz_alpha
        
        return vectors @ R.T

    def _apply_inverse_rotation(self, vectors: np.ndarray, angles: Tuple[float, float, float]) -> np.ndarray:
        """
        Apply inverse ZYZ Euler rotation.
        
        Args:
            vectors: N x 3 array of vectors
            angles: (alpha, beta, gamma) Euler angles in radians
            
        Returns:
            Inverse rotated N x 3 array
        """
        alpha, beta, gamma = angles
        # Inverse rotation: apply in reverse order with negated angles
        return self._apply_rotation(vectors, (-gamma, -beta, -alpha))


# ==============================================================================
# File I/O Functions
# ==============================================================================

def read_ticra_sph(filename: str) -> Dict:
    """
    Read TICRA .sph file containing spherical wave expansion coefficients.
    
    Args:
        filename: Path to the .sph file
        
    Returns:
        Dictionary containing all data from .sph file
    """

    # NORMALIZATION FACTOR (from Ticra)
    # ticra has a comment about 1/sqrt(8pi) normalization, but that doesn't apply in this formulation
    normalization_factor = 1


    with open(filename, 'r') as f:
        lines = f.readlines()
    
    line_idx = 0
    
    # Record 1: PRGTAG
    prgtag = lines[line_idx].strip()
    line_idx += 1
    
    frequency = None
    if 'Freq [GHz]:' in prgtag:
        freq_str = prgtag.split('Freq [GHz]:')[1].strip().split()[0]
        frequency = float(freq_str)
    
    # Record 2: IDSTRG
    idstrg = lines[line_idx].strip()
    line_idx += 1
    
    # Record 3: NTHE, NPHI, NMAX, MMAX
    control_data = lines[line_idx].strip().split()
    NTHE = int(control_data[0])
    NPHI = int(control_data[1])
    NMAX = int(control_data[2])
    MMAX = int(control_data[3])
    line_idx += 1
    
    # Record 4: Rotation angles
    rotation_line = lines[line_idx].strip()
    line_idx += 1
    if 'Rotation angles' in rotation_line:
        angles_str = rotation_line.split('=')[1].strip('()')
        rotation_angles = tuple(float(x.strip()) for x in angles_str.split(','))
    else:
        rotation_angles = (0.0, 0.0, 0.0)
    
    # Records 5-8: Dummy data
    line_idx += 4
    
    # Read coefficients
    Q1_coeffs = {}
    Q2_coeffs = {}
    power = {}
    
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        if not line:
            line_idx += 1
            continue
            
        parts = line.split()
        if len(parts) >= 2:
            try:
                m_index = int(parts[0])
                powerm = float(parts[1])
                power[abs(m_index)] = powerm
                line_idx += 1
                
                abs_m = abs(m_index)
                n_start = max(1, abs_m)
                
                if m_index == 0:
                    for n in range(n_start, NMAX + 1):
                        if line_idx >= len(lines):
                            break
                        coeff_line = lines[line_idx].strip()
                        line_idx += 1
                        
                        coeff_parts = coeff_line.split()
                        if len(coeff_parts) >= 4:
                            Q1_coeffs[(n, 0)] = normalization_factor * (float(coeff_parts[0]) + 1j * float(coeff_parts[1]))
                            Q2_coeffs[(n, 0)] = normalization_factor * (float(coeff_parts[2]) + 1j * float(coeff_parts[3]))
                        else:
                            line_idx -= 1
                            break
                else:
                    for n in range(n_start, NMAX + 1):
                        # -m coefficients
                        if line_idx >= len(lines):
                            break
                        coeff_line = lines[line_idx].strip()
                        line_idx += 1
                        
                        coeff_parts = coeff_line.split()
                        if len(coeff_parts) >= 4:
                            Q1_coeffs[(n, -abs_m)] = normalization_factor * (float(coeff_parts[0]) + 1j * float(coeff_parts[1]))
                            Q2_coeffs[(n, -abs_m)] = normalization_factor * (float(coeff_parts[2]) + 1j * float(coeff_parts[3]))
                        else:
                            line_idx -= 1
                            break
                        
                        # +m coefficients
                        if line_idx >= len(lines):
                            break
                        coeff_line = lines[line_idx].strip()
                        line_idx += 1
                        
                        coeff_parts = coeff_line.split()
                        if len(coeff_parts) >= 4:
                            Q1_coeffs[(n, abs_m)] = normalization_factor * (float(coeff_parts[0]) + 1j * float(coeff_parts[1]))
                            Q2_coeffs[(n, abs_m)] = normalization_factor * (float(coeff_parts[2]) + 1j * float(coeff_parts[3]))
                        else:
                            line_idx -= 1
                            break
                            
            except (ValueError, IndexError):
                line_idx += 1
        else:
            line_idx += 1
    
    # conjugate coefficients to match Q_smn definition
    for key in Q1_coeffs:
        Q1_coeffs[key] = np.conj(Q1_coeffs[key])
    for key in Q2_coeffs:
        Q2_coeffs[key] = np.conj(Q2_coeffs[key])

    return {
        'frequency': frequency,
        'NTHE': NTHE,
        'NPHI': NPHI,
        'NMAX': NMAX,
        'MMAX': MMAX,
        'rotation_angles': rotation_angles,
        'Q1_coeffs': Q1_coeffs,
        'Q2_coeffs': Q2_coeffs,
        'power': power
    }


def write_ticra_sph(filename: str,
                    Q1_coeffs: Dict[Tuple[int, int], complex],
                    Q2_coeffs: Dict[Tuple[int, int], complex],
                    frequency_GHz: float,
                    NTHE: int, NPHI: int,
                    NMAX: int, MMAX: int,
                    description: str = "Generated by SWE module"):
    """
    Write spherical wave coefficients to TICRA .sph file format.
    
    Input coefficients should be in TICRA working convention.
    They will be conjugated before writing to match the .sph file format.
    """
    
    with open(filename, 'w') as f:
        # Record 1: PRGTAG
        f.write(f"TICRA-SWE Freq [GHz]: {frequency_GHz:.6f}\n")
        
        # Record 2: IDSTRG
        f.write(f"{description}\n")
        
        # Record 3: NTHE, NPHI, NMAX, MMAX
        f.write(f"{NTHE:5d}{NPHI:5d}{NMAX:5d}{MMAX:5d}\n")
        
        # Record 4: Rotation angles
        f.write("Rotation angles = (  0.00000,  0.00000,  0.00000)\n")
        
        # Records 5-8: Dummy data
        f.write("Dummy text string\n")
        f.write("  0.00000000000000E+00  0.00000000000000E+00  0.00000000000000E+00  0.00000000000000E+00  0.00000000000000E+00\n")
        f.write("  0.00000000000000E+00  0.00000000000000E+00  0.00000000000000E+00  0.00000000000000E+00  0.00000000000000E+00\n")
        f.write("Dummy text string\n")
        f.write("Dummy text string\n")
        
        # Write coefficients for each |m|
        for m_val in range(0, MMAX + 1):
            # Calculate power for this m
            power = 0.0
            for n in range(max(1, m_val), NMAX + 1):
                if m_val == 0:
                    Q1 = Q1_coeffs.get((n, 0), 0.0)
                    Q2 = Q2_coeffs.get((n, 0), 0.0)
                    power += abs(Q1)**2 + abs(Q2)**2
                else:
                    Q1_pos = Q1_coeffs.get((n, m_val), 0.0)
                    Q2_pos = Q2_coeffs.get((n, m_val), 0.0)
                    Q1_neg = Q1_coeffs.get((n, -m_val), 0.0)
                    Q2_neg = Q2_coeffs.get((n, -m_val), 0.0)
                    power += abs(Q1_pos)**2 + abs(Q2_pos)**2 + abs(Q1_neg)**2 + abs(Q2_neg)**2
            
            power = power / 2.0
            
            # Write m header
            f.write(f"{m_val:5d}  {power:23.16E}\n")
            
            # Write coefficients for each n
            for n in range(max(1, m_val), NMAX + 1):
                if m_val == 0:
                    # Conjugate before writing (reverse read operation)
                    Q1 = np.conj(Q1_coeffs.get((n, 0), 0.0))
                    Q2 = np.conj(Q2_coeffs.get((n, 0), 0.0))
                    
                    f.write(f"  {Q1.real:23.16E} {Q1.imag:23.16E} "
                           f"{Q2.real:23.16E} {Q2.imag:23.16E}\n")
                else:
                    # -m line (conjugate before writing)
                    Q1_neg = np.conj(Q1_coeffs.get((n, -m_val), 0.0))
                    Q2_neg = np.conj(Q2_coeffs.get((n, -m_val), 0.0))
                    
                    f.write(f"  {Q1_neg.real:23.16E} {Q1_neg.imag:23.16E} "
                           f"{Q2_neg.real:23.16E} {Q2_neg.imag:23.16E}\n")
                    
                    # +m line (conjugate before writing)
                    Q1_pos = np.conj(Q1_coeffs.get((n, m_val), 0.0))
                    Q2_pos = np.conj(Q2_coeffs.get((n, m_val), 0.0))
                    
                    f.write(f"  {Q1_pos.real:23.16E} {Q1_pos.imag:23.16E} "
                           f"{Q2_pos.real:23.16E} {Q2_pos.imag:23.16E}\n")