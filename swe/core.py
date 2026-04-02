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
- Time dependence: exp(jωt) (matches Ticra)
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

import logging
import math
import numpy as np
from scipy.special import lpmv, spherical_jn, spherical_yn
from scipy.optimize import lsq_linear
from typing import Dict, List, Tuple, Optional, Union
import warnings
from multiprocessing import Pool
import os

# Optional Numba acceleration for performance-critical functions
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Configure module-level logger
logger = logging.getLogger(__name__)


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


def compute_pole_limits(n: int, m: int, pole: str = 'north') -> Tuple[float, float]:
    """
    Compute analytical limits of normalized Legendre terms at poles (theta=0 or pi).

    At the poles, sin(theta)=0 causes division issues. This function computes
    the proper limits using L'Hopital's rule and Legendre function identities.

    Mathematical basis:
    - P_n^1(cos theta) = -sin(theta) * dP_n/d(cos theta)
    - Therefore: P_n^1/sin(theta) = -dP_n/d(cos theta)|_{x=1} = -n(n+1)/2
    - For |m|>=2: m*P_n^m/sin(theta) -> 0 as theta->0

    At south pole (theta=pi), the sign factors differ for the two terms:
    - mP_over_sin: sign factor is (-1)^(n+1)
    - dP/dtheta: sign factor is (-1)^n

    Args:
        n: Degree (n >= 1)
        m: Order
        pole: 'north' (theta=0) or 'south' (theta=pi)

    Returns:
        (mP_over_sin_limit, dP_limit): Normalized limits at the pole
    """
    norm_factor = _legendre_cache.normalization_factor(n, m)
    abs_m = abs(m)

    # Sign factors for south pole (derived from numerical analysis)
    if pole == 'north':
        mP_sign = 1.0
        dP_sign = 1.0
    else:
        # At south pole, mP_over_sin and dP have different sign factors
        mP_sign = (-1.0) ** (n + 1)
        dP_sign = (-1.0) ** n

    if abs_m == 0:
        # m=0: m*P_n^m/sin(theta) = 0
        # dP_n^0/dtheta|_{pole} = 0
        mP_over_sin_limit = 0.0
        dP_limit = 0.0
    elif abs_m == 1:
        # For m=+/-1:
        # lim P_n^1/sin(theta) = -n(n+1)/2 (unnormalized)
        # dP_n^1/dtheta|_{theta=0} = -n(n+1)/2 (unnormalized)
        unnorm_limit = -n * (n + 1) / 2.0

        # m * P_n^m / sin(theta) limit
        mP_over_sin_limit = m * unnorm_limit * norm_factor * mP_sign

        # dP/dtheta limit at pole
        dP_limit = unnorm_limit * norm_factor * dP_sign
    else:
        # |m| >= 2: both limits are zero
        mP_over_sin_limit = 0.0
        dP_limit = 0.0

    return mP_over_sin_limit, dP_limit


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

    # Apply pole avoidance for numerical stability in Legendre computation
    # Using very small epsilon since analytical limits are computed in pattern functions
    theta_safe = np.copy(theta)
    epsilon = 1e-6
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

    Uses consistent epsilon-based pole avoidance matching the Legendre cache
    computation. This ensures smooth pattern values near poles by avoiding
    numerical instabilities without introducing discontinuities.

    Args:
        n: Degree
        m: Order
        theta: Polar angles
        phi: Azimuthal angles
        legendre_cache: Optional pre-computed dict from compute_all_modes_legendre

    Returns:
        (K1_theta, K1_phi), (K2_theta, K2_phi)
    """
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)

    # Use same epsilon as compute_all_modes_legendre for consistency
    # This ensures cache values and sin(theta) computation are aligned
    epsilon = 1e-6
    at_north_pole = theta < epsilon
    at_south_pole = theta > (np.pi - epsilon)

    # Apply pole avoidance - must match cache computation exactly
    theta_safe = np.copy(theta)
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
        sign_factor = (-m / abs(m)) ** m

    # Ticra sign convention: negative phase progression
    phase = np.exp(-1j * m * phi)
    i_factor_1 = (1j) ** (n + 1)
    i_factor_2 = (1j) ** (n)

    sin_theta = np.sin(theta_safe)
    mP_over_sin = m * P_norm / sin_theta

    # signs are tricky here, had to compare near field form of Ticra and Hansen to
    # derive far field forms for Ticra
    K1_theta = prefactor * sign_factor * phase * i_factor_1 * (-1j * mP_over_sin)
    K1_phi = prefactor * sign_factor * phase * i_factor_1 * (-dP_norm_dtheta)
    
    K2_theta = prefactor * sign_factor * phase * i_factor_2 * (dP_norm_dtheta)
    K2_phi = prefactor * sign_factor * phase * i_factor_2 * (-1j * mP_over_sin)
    
    return (K1_theta, K1_phi), (K2_theta, K2_phi)


def _precompute_bessel_scipy(nmax: int, kr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Scipy-based Bessel pre-computation (fallback when Numba not available)."""
    N = len(kr)
    j_all = np.zeros((nmax + 1, N))
    y_all = np.zeros((nmax + 1, N))

    for n in range(nmax + 1):
        j_all[n] = spherical_jn(n, kr)
        y_all[n] = spherical_yn(n, kr)

    return j_all, y_all


# Define Numba-accelerated version if available
if HAS_NUMBA:
    @jit(nopython=True, parallel=True, cache=True)
    def _precompute_bessel_numba(nmax, kr):
        """
        Numba-accelerated spherical Bessel computation using recurrence relations.

        Uses upward recurrence which is stable for y_n (growing) and j_n when n < kr.
        For n > kr, j_n may lose some precision, but this is acceptable for most
        antenna applications where kr is typically larger than nmax.
        """
        N = len(kr)
        j_all = np.zeros((nmax + 1, N))
        y_all = np.zeros((nmax + 1, N))

        # Parallel loop over spatial points
        for i in prange(N):
            x = kr[i]
            if x < 1e-30:
                x = 1e-30  # Avoid division by zero

            sin_x = np.sin(x)
            cos_x = np.cos(x)

            # Initial values (analytic formulas)
            j_all[0, i] = sin_x / x                          # j_0
            y_all[0, i] = -cos_x / x                         # y_0

            if nmax >= 1:
                j_all[1, i] = sin_x / (x * x) - cos_x / x    # j_1
                y_all[1, i] = -cos_x / (x * x) - sin_x / x   # y_1

            # Upward recurrence: f_{n+1} = (2n+1)/x * f_n - f_{n-1}
            for n in range(1, nmax):
                factor = (2 * n + 1) / x
                j_all[n + 1, i] = factor * j_all[n, i] - j_all[n - 1, i]
                y_all[n + 1, i] = factor * y_all[n, i] - y_all[n - 1, i]

        return j_all, y_all


def precompute_spherical_bessel(nmax: int, kr: np.ndarray,
                                 use_numba: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute all spherical Bessel functions j_n and y_n for n=0 to nmax.

    This provides a significant speedup when computing near-field patterns,
    since Bessel functions depend only on (n, kr), not on the azimuthal order m.
    Pre-computing eliminates redundant calls when looping over modes.

    Args:
        nmax: Maximum order n
        kr: Array of k*r values, shape (N,)
        use_numba: If True and Numba is available, use JIT-compiled version
                   with recurrence relations for additional speedup

    Returns:
        j_all: Array of shape (nmax+1, N) containing j_0, j_1, ..., j_nmax
        y_all: Array of shape (nmax+1, N) containing y_0, y_1, ..., y_nmax
    """
    kr = np.atleast_1d(kr).ravel()

    if use_numba and HAS_NUMBA:
        return _precompute_bessel_numba(nmax, kr)
    else:
        return _precompute_bessel_scipy(nmax, kr)


def near_field_pattern_functions(n: int, m: int, r: np.ndarray,
                                 theta: np.ndarray, phi: np.ndarray,
                                 k: float, legendre_cache: Dict = None,
                                 bessel_cache: Tuple[np.ndarray, np.ndarray] = None):
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

    if bessel_cache is not None:
        # Use pre-computed Bessel functions (much faster for many modes)
        j_all, y_all = bessel_cache
        j_n = j_all[n]
        y_n = y_all[n]
        if n == 0:
            j_n_m1 = 0.0
            y_n_m1 = -np.cos(kr) / kr
        else:
            j_n_m1 = j_all[n - 1]
            y_n_m1 = y_all[n - 1]
    else:
        # Fallback to scipy calls (backward compatibility)
        j_n = spherical_jn(n, kr)
        y_n = spherical_yn(n, kr)
        if n == 0:
            j_n_m1 = 0.0
            y_n_m1 = -np.cos(kr) / kr
        else:
            j_n_m1 = spherical_jn(n - 1, kr)
            y_n_m1 = spherical_yn(n - 1, kr)

    h_n = j_n - 1j * y_n  # h_n^(2)
    h_n_m1 = j_n_m1 - 1j * y_n_m1
    dkrh_n = kr * h_n_m1 - n * h_n  # d/d(kr){kr*h_n^(2)}
    
    # Common factors from equations (4.214-4.215)
    prefactor = 1 / np.sqrt(2 * np.pi) * 1 / np.sqrt(n * (n + 1))
    
    if m == 0:
        sign_factor = 1.0
    else:
        sign_factor = (-m / abs(m)) ** m
    
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
    # Note: Pattern functions are normalized such that H = E_swap (impedance absorbed in normalization)
    F1_H_r = F2_E_r
    F1_H_theta = F2_E_theta
    F1_H_phi = F2_E_phi

    F2_H_r = F1_E_r
    F2_H_theta = F1_E_theta
    F2_H_phi = F1_E_phi
    
    return (F1_E_r, F1_E_theta, F1_E_phi), (F2_E_r, F2_E_theta, F2_E_phi), \
           (F1_H_r, F1_H_theta, F1_H_phi), (F2_H_r, F2_H_theta, F2_H_phi)

def compute_mode_coefficients_batch(args):
    """Compute Q coefficients for a batch of modes - with FFT optimization for phi integration."""
    modes_batch, THETA, PHI, E_THETA, E_PHI, sin_theta, theta_unique, phi_unique, norm_factor, legendre_data = args
    
    # Check if phi is uniformly spaced
    dphi = np.diff(phi_unique)
    use_fft = len(phi_unique) > 4 and np.allclose(dphi, dphi[0], rtol=1e-6)
    
    if not use_fft:
        # Fall back to original trapz method (for non-uniform phi grids)
        return compute_mode_coefficients_batch_trapz(args)
    
    # FFT-optimized method for uniform phi grids
    results = []
    N_phi = len(phi_unique)
    
    for n, m in modes_batch:
        # Unpack Legendre data
        P_norm, dP_norm = legendre_data[(n, m)]
        P_norm_2d = P_norm[:, np.newaxis]
        dP_norm_2d = dP_norm[:, np.newaxis]
        
        # TICRA pattern functions WITHOUT exp(-imφ) phase term
        prefactor = np.sqrt(2 / (n * (n + 1)))
        sign_factor = (m / abs(m)) ** m if m != 0 else 1.0
        i_factor_1 = (1j) ** n
        i_factor_2 = (1j) ** (n + 1)
        
        # Use same epsilon as compute_all_modes_legendre for consistency
        sin_theta_safe = np.where(np.abs(sin_theta) < 1e-6, 1e-6, sin_theta)
        mP_over_sin = 1j * m * P_norm_2d / sin_theta_safe
        
        # Pattern functions without exp(-imφ)
        K1_theta_base = prefactor * sign_factor * i_factor_1 * mP_over_sin
        K1_phi_base = prefactor * sign_factor * i_factor_1 * dP_norm_2d
        K2_theta_base = prefactor * sign_factor * i_factor_2 * dP_norm_2d
        K2_phi_base = prefactor * sign_factor * i_factor_2 * (-mP_over_sin)
        
        # Integrand without exp(-imφ) in K (but will have exp(+imφ) from conjugate)
        # We'll compute: ∫∫ E * conj(K_base) * exp(+imφ) * sin(θ) dθ dφ
        integrand_1_base = (E_THETA * np.conj(K1_theta_base) + 
                            E_PHI * np.conj(K1_phi_base)) * sin_theta
        integrand_2_base = (E_THETA * np.conj(K2_theta_base) + 
                            E_PHI * np.conj(K2_phi_base)) * sin_theta
        
        # Use FFT for phi integration
        # We need: ∫ integrand_base * exp(+imφ) dφ
        # Multiply by exp(+imφ) then take FFT to extract DC component
        phi_grid = phi_unique[np.newaxis, :]  # shape (1, N_phi)
        exp_plus_imphi = np.exp(1j * m * phi_grid)
        
        # Multiply integrand by exp(+imφ)
        integrand_1_with_phase = integrand_1_base * exp_plus_imphi
        integrand_2_with_phase = integrand_2_base * exp_plus_imphi
        
        # FFT along phi axis (axis=1), extract DC component (k=0)
        # FFT gives sum, so divide by N_phi and multiply by 2π to get integral
        phi_integral_1 = np.fft.fft(integrand_1_with_phase, axis=1)[:, 0] * (2 * np.pi / N_phi)
        phi_integral_2 = np.fft.fft(integrand_2_with_phase, axis=1)[:, 0] * (2 * np.pi / N_phi)
        
        # Now integrate over theta
        Q1 = np.trapz(phi_integral_1, theta_unique, axis=0) * norm_factor
        Q2 = np.trapz(phi_integral_2, theta_unique, axis=0) * norm_factor
        
        mode_power = (abs(Q1)**2 + abs(Q2)**2) / 2.0
        results.append(((n, m), Q1, Q2, mode_power))
    
    return results


def compute_mode_coefficients_batch_trapz(args):
    """Original trapz method - fallback for non-uniform grids."""
    modes_batch, THETA, PHI, E_THETA, E_PHI, sin_theta, theta_unique, phi_unique, norm_factor, legendre_data = args
    
    results = []
    
    for n, m in modes_batch:
        P_norm, dP_norm = legendre_data[(n, m)]
        P_norm_2d = P_norm[:, np.newaxis]
        dP_norm_2d = dP_norm[:, np.newaxis]
        
        prefactor = np.sqrt(2 / (n * (n + 1)))
        sign_factor = (m / abs(m)) ** m if m != 0 else 1.0
        phase = np.exp(-1j * m * PHI)
        i_factor_1 = (1j) ** n
        i_factor_2 = (1j) ** (n + 1)
        
        # Use same epsilon as compute_all_modes_legendre for consistency
        sin_theta_safe = np.where(np.abs(sin_theta) < 1e-6, 1e-6, sin_theta)
        mP_over_sin = 1j * m * P_norm_2d / sin_theta_safe
        
        K1_theta = prefactor * sign_factor * phase * i_factor_1 * mP_over_sin
        K1_phi = prefactor * sign_factor * phase * i_factor_1 * dP_norm_2d
        K2_theta = prefactor * sign_factor * phase * i_factor_2 * (dP_norm_2d)
        K2_phi = prefactor * sign_factor * phase * i_factor_2 * (-mP_over_sin)
        
        integrand_1 = (E_THETA * np.conj(K1_theta) + E_PHI * np.conj(K1_phi)) * sin_theta
        Q1 = np.trapz(np.trapz(integrand_1, phi_unique, axis=1), theta_unique, axis=0)
        Q1 *= norm_factor
        
        integrand_2 = (E_THETA * np.conj(K2_theta) + E_PHI * np.conj(K2_phi)) * sin_theta
        Q2 = np.trapz(np.trapz(integrand_2, phi_unique, axis=1), theta_unique, axis=0)
        Q2 *= norm_factor
        
        mode_power = (abs(Q1)**2 + abs(Q2)**2) / 2.0
        results.append(((n, m), Q1, Q2, mode_power))
    
    return results
# ==============================================================================
# Main SWE Class
# ==============================================================================

class SphericalWaveExpansion:
    """
    Multi-frequency Spherical Wave Expansion representation of electromagnetic fields.

    Stores Q₁ and Q₂ coefficients for one or more frequencies.  All compute
    methods (far_field, near_field, …) require an explicit ``frequency``
    argument so that the caller selects which frequency block to evaluate.

    Typical usage::

        swe = SphericalWaveExpansion.from_sph_file("antenna.sph")
        for f in swe.frequencies:
            Et, Ep = swe.far_field(theta, phi, frequency=f)

    Attributes (per-frequency, accessed via methods):
        frequencies : sorted list of loaded frequencies in Hz
        Q1_coeffs(f): Dict[(n, m) -> complex] for frequency f
        Q2_coeffs(f): Dict[(n, m) -> complex] for frequency f
        NMAX(f)     : maximum degree for frequency f
        MMAX(f)     : maximum order  for frequency f
        k(f)        : wavenumber (rad/m) for frequency f
        wavelength(f): wavelength (m) for frequency f
        total_power(f): Σ(|Q1|²+|Q2|²) for frequency f
    """

    _C = 299792458.0  # speed of light (m/s)

    def __init__(self,
                 Q1_coeffs: Optional[Dict[float, Dict[Tuple[int, int], complex]]] = None,
                 Q2_coeffs: Optional[Dict[float, Dict[Tuple[int, int], complex]]] = None,
                 NMAX: Optional[Dict[float, int]] = None,
                 MMAX: Optional[Dict[float, int]] = None):
        """
        Initialize a multi-frequency SWE object.

        Args:
            Q1_coeffs: ``{freq_Hz: {(n, m): complex}}`` mapping for Q₁ modes.
            Q2_coeffs: ``{freq_Hz: {(n, m): complex}}`` mapping for Q₂ modes.
            NMAX: ``{freq_Hz: int}`` maximum degree per frequency.
                  Auto-detected from coefficient keys when omitted.
            MMAX: ``{freq_Hz: int}`` maximum order per frequency.
                  Auto-detected from coefficient keys when omitted.
        """
        self._Q1: Dict[float, Dict[Tuple[int, int], complex]] = (
            Q1_coeffs if Q1_coeffs is not None else {}
        )
        self._Q2: Dict[float, Dict[Tuple[int, int], complex]] = (
            Q2_coeffs if Q2_coeffs is not None else {}
        )

        # Ensure every frequency present in Q1 also exists in Q2 and vice versa
        all_freqs = set(self._Q1.keys()) | set(self._Q2.keys())
        for f in all_freqs:
            self._Q1.setdefault(f, {})
            self._Q2.setdefault(f, {})

        # Build per-frequency NMAX / MMAX
        self._nmax: Dict[float, int] = {}
        self._mmax: Dict[float, int] = {}
        for f in all_freqs:
            all_keys = list(self._Q1[f].keys()) + list(self._Q2[f].keys())
            if NMAX is not None and f in NMAX:
                self._nmax[f] = NMAX[f]
            elif all_keys:
                self._nmax[f] = max(n for n, m in all_keys)
            else:
                self._nmax[f] = 0

            if MMAX is not None and f in MMAX:
                self._mmax[f] = MMAX[f]
            elif all_keys:
                self._mmax[f] = max(abs(m) for n, m in all_keys)
            else:
                self._mmax[f] = 0

        if all_freqs:
            for f in sorted(all_freqs):
                logger.debug(
                    f"SphericalWaveExpansion: {f/1e9:.4f} GHz — "
                    f"NMAX={self._nmax[f]}, MMAX={self._mmax[f]}, "
                    f"{len(self._Q1[f])} Q1 modes, {len(self._Q2[f])} Q2 modes"
                )

    # ------------------------------------------------------------------
    # Per-frequency accessors
    # ------------------------------------------------------------------

    @property
    def frequencies(self) -> List[float]:
        """Sorted list of loaded frequencies in Hz."""
        return sorted(self._Q1.keys())

    def _validate_freq(self, freq: float) -> None:
        """Raise KeyError with a helpful message if *freq* is not loaded."""
        if freq not in self._Q1:
            avail = [f"{f/1e9:.4f}" for f in sorted(self._Q1.keys())]
            raise KeyError(
                f"Frequency {freq/1e9:.4f} GHz not found. "
                f"Available: [{', '.join(avail)}] GHz"
            )

    def Q1_coeffs(self, freq: float) -> Dict[Tuple[int, int], complex]:
        """Return Q₁ coefficient dict for *freq* (Hz)."""
        self._validate_freq(freq)
        return self._Q1[freq]

    def Q2_coeffs(self, freq: float) -> Dict[Tuple[int, int], complex]:
        """Return Q₂ coefficient dict for *freq* (Hz)."""
        self._validate_freq(freq)
        return self._Q2[freq]

    def NMAX(self, freq: float) -> int:
        """Maximum degree for *freq* (Hz)."""
        self._validate_freq(freq)
        return self._nmax[freq]

    def MMAX(self, freq: float) -> int:
        """Maximum order for *freq* (Hz)."""
        self._validate_freq(freq)
        return self._mmax[freq]

    def k(self, freq: float) -> float:
        """Wavenumber in rad/m for *freq* (Hz)."""
        return 2 * np.pi * freq / self._C

    def wavelength(self, freq: float) -> float:
        """Wavelength in metres for *freq* (Hz)."""
        return self._C / freq

    def total_power(self, freq: float) -> float:
        """Total power Σ(|Q1|²+|Q2|²) for *freq* (Hz).

        When coefficients are normalized (normalize_coefficients()),
        total_power == 1.0 and far_field/near_field with normalize=True
        give directivity-referenced fields.
        """
        self._validate_freq(freq)
        power = sum(abs(Q)**2 for Q in self._Q1[freq].values())
        power += sum(abs(Q)**2 for Q in self._Q2[freq].values())
        return power

    # ------------------------------------------------------------------
    # Coefficient normalization
    # ------------------------------------------------------------------

    def normalize_coefficients(self, freq: Optional[float] = None) -> None:
        """Normalize Q coefficients so that total_power == 1.

        After normalization, far_field(normalize=True) and
        near_field(normalize=True) produce directivity-referenced fields
        regardless of the original coefficient convention.

        Args:
            freq: Frequency (Hz) to normalize.  When *None* (default), all
                  stored frequencies are normalized.
        """
        freqs = [freq] if freq is not None else list(self._Q1.keys())
        for f in freqs:
            tp = self.total_power(f)
            if tp <= 0:
                logger.warning(f"Cannot normalize {f/1e9:.4f} GHz: total_power is zero")
                continue
            norm = np.sqrt(tp)
            logger.debug(f"Normalizing {f/1e9:.4f} GHz: total_power={tp:.6f}, norm={norm:.6f}")
            for key in self._Q1[f]:
                self._Q1[f][key] /= norm
            for key in self._Q2[f]:
                self._Q2[f][key] /= norm

    # ------------------------------------------------------------------
    # Multi-frequency merge helper
    # ------------------------------------------------------------------

    def add_frequency(self, other: 'SphericalWaveExpansion') -> None:
        """Merge all frequency blocks from *other* into this object.

        Args:
            other: Another SphericalWaveExpansion whose frequencies will be
                   added to this one.

        Raises:
            ValueError: If any frequency in *other* already exists here.
        """
        conflicts = set(other._Q1.keys()) & set(self._Q1.keys())
        if conflicts:
            cstr = ', '.join(f"{f/1e9:.4f} GHz" for f in sorted(conflicts))
            raise ValueError(f"Frequencies already present: {cstr}")
        for f in other.frequencies:
            self._Q1[f] = dict(other._Q1[f])
            self._Q2[f] = dict(other._Q2[f])
            self._nmax[f] = other._nmax[f]
            self._mmax[f] = other._mmax[f]
        logger.debug(f"add_frequency: added {len(other.frequencies)} block(s)")

    def far_field(self, theta: np.ndarray, phi: np.ndarray,
                  frequency: float,
                  power_threshold: float = 0.999,
                  normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute far-field pattern from SWE coefficients.

        By default, normalizes the output so that |E_theta|² + |E_phi|²
        equals directivity, matching the convention used by TICRA .cut/.ffd files.
        Set normalize=False to get the raw (unnormalized) field sum.

        Args:
            theta: Polar angle(s) in radians
            phi: Azimuthal angle(s) in radians
            frequency: Frequency in Hz (must be a loaded frequency)
            power_threshold: Only include modes containing this fraction of total power.
                           Default 0.999 (99.9%) filters out noise in high-n modes
                           that can cause boresight artifacts. Set to 1.0 to include all modes.
            normalize: If True (default), divide by sqrt(total_power) so that
                      |E|² = directivity. If False, return raw Σ Q·K.

        Returns:
            E_theta: Theta component of electric field
            E_phi: Phi component of electric field
        """
        self._validate_freq(frequency)

        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)

        if theta.shape != phi.shape:
            theta, phi = np.broadcast_arrays(theta, phi)

        q1 = self._Q1[frequency]
        q2 = self._Q2[frequency]

        # Compute mode powers and filter by cumulative power threshold
        all_modes = set(q1.keys()) | set(q2.keys())

        if not all_modes:
            logger.debug("No modes present, returning zero field")
            E_theta = np.zeros_like(theta, dtype=complex)
            E_phi = np.zeros_like(phi, dtype=complex)
            return E_theta, E_phi

        nmax = self._nmax[frequency]
        mmax = self._mmax[frequency]

        # Calculate power per mode and sort by n
        mode_powers = []
        for (n, m) in all_modes:
            Q1 = q1.get((n, m), 0)
            Q2 = q2.get((n, m), 0)
            power = abs(Q1)**2 + abs(Q2)**2
            mode_powers.append(((n, m), power))

        total_power = sum(p for _, p in mode_powers)

        # Find effective NMAX based on power threshold
        if power_threshold < 1.0 and total_power > 0:
            n_power = {}
            for (n, m), p in mode_powers:
                n_power[n] = n_power.get(n, 0) + p

            cumulative = 0
            effective_nmax = nmax
            for n in sorted(n_power.keys()):
                cumulative += n_power[n]
                if cumulative >= power_threshold * total_power:
                    effective_nmax = n
                    break

            filtered_modes = [(nm, p) for nm, p in mode_powers if nm[0] <= effective_nmax]
            if len(filtered_modes) < len(mode_powers):
                logger.debug(f"Power filtering: using NMAX={effective_nmax} "
                             f"({len(filtered_modes)}/{len(mode_powers)} modes, "
                             f"{100*power_threshold:.2f}% power)")
        else:
            filtered_modes = mode_powers
            effective_nmax = nmax

        logger.debug(f"Computing far field at {len(theta.flatten())} points, "
                     f"effective NMAX={effective_nmax}")

        E_theta = np.zeros_like(theta, dtype=complex)
        E_phi = np.zeros_like(phi, dtype=complex)

        effective_mmax = min(mmax, effective_nmax)
        legendre_cache = compute_all_modes_legendre(effective_nmax, effective_mmax, theta)

        for (n, m), _ in filtered_modes:
            if (n, m) not in legendre_cache:
                continue

            (K1_theta, K1_phi), (K2_theta, K2_phi) = \
                far_field_pattern_functions(n, m, theta, phi, legendre_cache)

            if (n, m) in q1:
                E_theta += q1[(n, m)] * K1_theta
                E_phi += q1[(n, m)] * K1_phi

            if (n, m) in q2:
                E_theta += q2[(n, m)] * K2_theta
                E_phi += q2[(n, m)] * K2_phi

        # Normalize so |E|² = directivity (matching TICRA .cut/.ffd convention)
        if normalize and total_power > 0:
            norm = np.sqrt(total_power)
            E_theta /= norm
            E_phi /= norm

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
                            power_threshold: float = 0.999,
                            high_mode_power_threshold: float = 0.00001,
                            azimuthal_power_threshold: float = 0.00001,
                            use_multiprocessing: bool = True,
                            n_workers: int = None,
                            normalize: bool = True):
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
            power_threshold: Retain modes with this fraction of power (0.999 = 99.9%)
            high_mode_power_threshold: Max power in top 10% n-modes (0.001 = 0.51)
            azimuthal_power_threshold: Min power per |m| to include (0.00005 = 0.005%)
            use_multiprocessing: Enable parallel computation
            n_workers: Number of parallel workers (None = auto-detect)
            normalize: If True (default), normalize coefficients so total_power=1.
                      If False, preserve original scaling for absolute field comparison.

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
            logger.debug(f"Estimated NMAX from r0={r0}m: kr0={kr0:.2f}, NMAX={NMAX_estimated}")
            NMAX_initial = max(NMAX_initial, NMAX_estimated)

        logger.info(f"Starting SWE coefficient extraction: NMAX_initial={NMAX_initial}, frequency={frequency/1e9:.4f} GHz")
        
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

            logger.info(f"Iteration {iteration + 1}: Computing coefficients for NMAX={NMAX}, MMAX={MMAX_current}")

            # Pre-compute Legendre functions for all modes
            logger.debug("Computing Legendre functions...")
            legendre_cache = compute_all_modes_legendre(NMAX, MMAX_current, THETA[:, 0])
            
            # Build mode list
            modes = []
            for n in range(1, NMAX + 1):
                for m in range(-min(n, MMAX_current), min(n, MMAX_current) + 1):
                    modes.append((n, m))
            
            total_modes = len(modes)
            logger.debug(f"Extracting {total_modes} mode coefficients via integration...")

            # Compute coefficients (parallel or serial)
            if use_multiprocessing and total_modes > 100:
                # Split modes into batches for parallel processing
                batch_size = max(10, total_modes // (n_workers * 4))
                mode_batches = [modes[i:i + batch_size] for i in range(0, len(modes), batch_size)]
                logger.debug(f"Using {n_workers} workers, {len(mode_batches)} batches...")
                
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
                logger.debug(f"Completed {len(Q1_coeffs)} modes")
            
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
                    
                    K2_theta = prefactor * sign_factor * phase * i_factor_2 * (dP_norm_2d)
                    K2_phi = prefactor * sign_factor * phase * i_factor_2 * (-mP_over_sin)
                    
                    integrand_1 = (E_THETA * np.conj(K1_theta) + E_PHI * np.conj(K1_phi)) * sin_theta
                    Q1 = np.trapezoid(np.trapezoid(integrand_1, phi_unique, axis=1), theta_unique, axis=0)
                    Q1 *= norm_factor
                    
                    integrand_2 = (E_THETA * np.conj(K2_theta) + E_PHI * np.conj(K2_phi)) * sin_theta
                    Q2 = np.trapezoid(np.trapezoid(integrand_2, phi_unique, axis=1), theta_unique, axis=0)
                    Q2 *= norm_factor
                    
                    Q1_coeffs[(n, m)] = Q1
                    Q2_coeffs[(n, m)] = Q2
                    
                    mode_power = (abs(Q1)**2 + abs(Q2)**2) / 2.0
                    mode_powers.append(((n, m), mode_power))
            
            # Calculate total power
            mode_powers_array = np.array([p for _, p in mode_powers])
            total_power = np.sum(mode_powers_array)
            
            # Check power in top 10% of n values
            n_values = np.array([n for (n, m), _ in mode_powers])
            n_cutoff = max(1, int(np.ceil(0.1 * NMAX)))
            high_n_mask = n_values > (NMAX - n_cutoff)
            high_mode_power = np.sum(mode_powers_array[high_n_mask])
            
            high_mode_fraction = high_mode_power / total_power if total_power > 0 else 0
            logger.debug(f"Power in top {n_cutoff} n-modes: {high_mode_fraction*100:.2f}%")

            # Check convergence
            if high_mode_fraction < high_mode_power_threshold:
                logger.info(f"Convergence achieved: high modes contain {high_mode_fraction*100:.2f}% < {high_mode_power_threshold*100:.0f}%")
                # Calculate power per |m| for azimuthal truncation
                power_per_m = np.zeros(MMAX_current + 1)
                for (n, m), mode_power in mode_powers:
                    power_per_m[abs(m)] += mode_power
                
                # Calculate power per |m| for azimuthal truncation
                power_per_m = np.zeros(MMAX_current + 1)
                for (n, m), mode_power in mode_powers:
                    power_per_m[abs(m)] += mode_power

                # Find MMAX where tail is small AND cumulative power is sufficient
                MMAX_truncated = 0
                for m_test in range(0, MMAX_current + 1):
                    # Cumulative power up to m_test
                    cumulative_m = sum(power_per_m[m] for m in range(0, m_test + 1))
                    cumulative_m_fraction = cumulative_m / total_power if total_power > 0 else 0
                    
                    # Check power in top modes near m_test
                    m_cutoff = max(1, int(np.ceil(0.1 * m_test))) if m_test > 0 else 0
                    high_m = range(max(0, m_test - m_cutoff + 1), m_test + 1)
                    high_m_power = sum(power_per_m[m] for m in high_m)
                    high_m_fraction = high_m_power / total_power if total_power > 0 else 0
                    
                    # Accept if BOTH: tail is small AND we have enough power
                    if (high_m_fraction < azimuthal_power_threshold and
                        cumulative_m_fraction >= power_threshold):
                        MMAX_truncated = m_test
                        logger.debug(f"Azimuthal truncation: MMAX {MMAX_current} -> {MMAX_truncated}, tail power: {high_m_fraction*100:.3f}%")
                        break
                
                # Find highest n where tail modes have negligible power
                power_by_n = {n: 0 for n in range(1, NMAX+1)}
                for (n, m), pwr in mode_powers:
                    power_by_n[n] += pwr

                NMAX_truncated = 1
                for n_test in range(1, NMAX+1):
                    # Check power in top 10% of modes at this truncation level
                    n_cutoff = max(1, int(np.ceil(0.1 * n_test)))
                    high_modes = range(max(1, n_test - n_cutoff + 1), n_test + 1)
                    high_power = sum(power_by_n.get(n, 0) for n in high_modes)
                    high_power_fraction = high_power / total_power if total_power > 0 else 0
                    
                    # Also check cumulative power up to n_test
                    cumulative = sum(power_by_n.get(n, 0) for n in range(1, n_test+1))
                    cumulative_fraction = cumulative / total_power if total_power > 0 else 0
                    
                    # Accept if both: (1) tail is small AND (2) we have enough total power
                    if (high_power_fraction < high_mode_power_threshold and 
                        cumulative_fraction >= power_threshold):
                        NMAX_truncated = n_test
                        break

                # Keep ALL modes up to NMAX_truncated and MMAX_truncated
                Q1_final = {(n,m): Q1_coeffs[(n,m)] for (n,m) in Q1_coeffs.keys() 
                            if n <= NMAX_truncated and abs(m) <= MMAX_truncated}
                Q2_final = {(n,m): Q2_coeffs[(n,m)] for (n,m) in Q2_coeffs.keys() 
                            if n <= NMAX_truncated and abs(m) <= MMAX_truncated}

                retained_power = sum(power_by_n.get(n, 0) for n in range(1, NMAX_truncated+1))
                retained_fraction = retained_power / total_power
                
                # Verify final power distribution
                n_values_final = np.array([n for (n, m) in Q1_final.keys()])
                mode_powers_final = np.array([(abs(Q1_final[(n, m)])**2 + abs(Q2_final[(n, m)])**2) / 2.0 
                                            for (n, m) in Q1_final.keys()])
                total_power_final = np.sum(mode_powers_final)
                
                n_cutoff_final = max(1, int(np.ceil(0.1 * NMAX_truncated)))
                high_n_mask_final = n_values_final > (NMAX_truncated - n_cutoff_final)
                high_mode_power_final = np.sum(mode_powers_final[high_n_mask_final])
                high_mode_fraction_final = high_mode_power_final / total_power_final if total_power_final > 0 else 0
                
                # Check if final grid is adequately sampled
                theta_samples = len(theta_unique)
                phi_samples = len(phi_unique)
                theta_required = 2 * NMAX_truncated + 1
                phi_required = 2 * MMAX_truncated + 1

                if theta_samples < theta_required or phi_samples < phi_required:
                    logger.warning(
                        f"Input pattern is undersampled for NMAX={NMAX_truncated}, MMAX={MMAX_truncated}. "
                        f"Need at least {theta_required}x{phi_required} grid, but have {theta_samples}x{phi_samples}. "
                        f"Results may be inaccurate due to aliasing."
                    )
                    warnings.warn(
                        f"Input pattern is undersampled for NMAX={NMAX_truncated}, MMAX={MMAX_truncated}. "
                        f"Need at least {theta_required}×{phi_required} grid, but have {theta_samples}×{phi_samples}. "
                        f"Results may be inaccurate due to aliasing.",
                        UserWarning
                    )

                logger.info(
                    f"SWE extraction complete: NMAX={NMAX_truncated}, MMAX={MMAX_truncated}, "
                    f"retained power={retained_fraction*100:.2f}%, {len(Q1_final)} modes"
                )
                swe = cls(
                    {frequency: Q1_final},
                    {frequency: Q2_final},
                    NMAX={frequency: NMAX_truncated},
                    MMAX={frequency: MMAX_truncated},
                )
                if normalize:
                    swe.normalize_coefficients()
                return swe

            else:
                # Need more modes
                logger.debug(f"High mode power fraction {high_mode_fraction*100:.2f}% exceeds threshold, increasing NMAX from {NMAX} to {NMAX + 50}")
                NMAX += 50

        # Max iterations reached
        logger.warning(
            f"Maximum iterations ({max_iterations}) reached in adaptive mode calculation. "
            f"Final NMAX={NMAX}, MMAX={MMAX_current}. Results may be inaccurate."
        )
        warnings.warn(
            "Maximum iterations reached in adaptive mode calculation. "
            "Results may be inaccurate.",
            UserWarning
        )
        swe = cls(
            {frequency: Q1_coeffs},
            {frequency: Q2_coeffs},
            NMAX={frequency: NMAX},
            MMAX={frequency: MMAX_current},
        )
        if normalize:
            swe.normalize_coefficients()
        return swe

    @classmethod
    def from_sph_file(cls, filename: str, normalize: bool = True) -> 'SphericalWaveExpansion':
        """
        Create a multi-frequency SWE object from a TICRA .sph file.

        All frequency blocks present in the file are loaded.

        Note:
            When exporting .sph files from TICRA/GRASP, enable power normalization
            in the export settings so that the coefficients are normalized to unit
            radiated power. Unnormalized TICRA exports can produce incorrect
            near-field magnitudes in PO analysis.

        Args:
            filename: Path to .sph file
            normalize: If True (default), normalize coefficients so total_power=1
                      per frequency (directivity-referenced fields). If False,
                      preserve absolute scaling for comparison with GRASP data.

        Returns:
            SphericalWaveExpansion object with all frequencies loaded.
        """
        logger.info(f"Creating SphericalWaveExpansion from file: {filename}")
        blocks = read_ticra_sph(filename)

        Q1_all: Dict[float, Dict[Tuple[int, int], complex]] = {}
        Q2_all: Dict[float, Dict[Tuple[int, int], complex]] = {}
        nmax_all: Dict[float, int] = {}
        mmax_all: Dict[float, int] = {}

        for block in blocks:
            freq_hz = block['frequency'] * 1e9 if block['frequency'] is not None else 0.0
            Q1_all[freq_hz] = block['Q1_coeffs']
            Q2_all[freq_hz] = block['Q2_coeffs']
            nmax_all[freq_hz] = block['NMAX']
            mmax_all[freq_hz] = block['MMAX']

        swe = cls(Q1_coeffs=Q1_all, Q2_coeffs=Q2_all,
                  NMAX=nmax_all, MMAX=mmax_all)

        if normalize:
            swe.normalize_coefficients()

        return swe

    def to_sph_file(self, filename: str,
                    NTHE: int = 181, NPHI: int = 361,
                    description: str = "Generated by SWE module"):
        """
        Write all frequency blocks in this SWE object to a TICRA .sph file.

        Args:
            filename: Output file path
            NTHE: Number of theta samples written in every block header
            NPHI: Number of phi samples written in every block header
            description: Description text written in every block header
        """
        logger.info(f"Writing SphericalWaveExpansion to file: {filename}")
        freq_data_list = []
        for freq in self.frequencies:
            freq_data_list.append({
                'Q1_coeffs': self._Q1[freq],
                'Q2_coeffs': self._Q2[freq],
                'frequency_GHz': freq / 1e9,
                'NMAX': self._nmax[freq],
                'MMAX': self._mmax[freq],
                'NTHE': NTHE,
                'NPHI': NPHI,
            })
        write_ticra_sph(filename, freq_data_list, description)
    
    def near_field(self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray,
                   frequency: float,
                   normalize: bool = True) -> \
            Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
                  Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calculate near-field E and H components.

        By default, normalizes by sqrt(total_power) so that the fields are
        consistent with the directivity-normalized far field.

        Scaling note: the prefactor used here is sqrt(4π) for E and
        j*sqrt(4π)/Z0 for H. This differs from Hansen's textbook expression
        of k*sqrt(Z0) because the .sph file I/O absorbs a 1/sqrt(8π) factor
        into the stored Q coefficients, cancelling the k*sqrt(Z0) term and
        leaving sqrt(4π) as the effective prefactor. The far-field K-function
        normalisation uses the same convention.

        Args:
            r: Radial distance(s) in meters
            theta: Polar angle(s) in radians
            phi: Azimuthal angle(s) in radians
            frequency: Frequency in Hz (must be a loaded frequency)
            normalize: If True (default), divide by sqrt(total_power) for
                      consistency with directivity-normalized far field.

        Returns:
            E: Tuple of (E_r, E_theta, E_phi) in V/m
            H: Tuple of (H_r, H_theta, H_phi) in A/m
        """
        self._validate_freq(frequency)

        r = np.atleast_1d(r)
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)

        if not (r.shape == theta.shape == phi.shape):
            r, theta, phi = np.broadcast_arrays(r, theta, phi)

        nmax = self._nmax[frequency]
        mmax = self._mmax[frequency]
        q1 = self._Q1[frequency]
        q2 = self._Q2[frequency]

        logger.debug(f"Computing near field at {len(r.flatten())} points, "
                     f"NMAX={nmax}, MMAX={mmax}")

        E_r = np.zeros_like(r, dtype=complex)
        E_theta = np.zeros_like(theta, dtype=complex)
        E_phi = np.zeros_like(phi, dtype=complex)
        H_r = np.zeros_like(r, dtype=complex)
        H_theta = np.zeros_like(theta, dtype=complex)
        H_phi = np.zeros_like(phi, dtype=complex)

        all_modes = set(q1.keys()) | set(q2.keys())

        if not all_modes:
            logger.debug("No modes present, returning zero fields")
            return (E_r, E_theta, E_phi), (H_r, H_theta, H_phi)

        legendre_cache = compute_all_modes_legendre(nmax, mmax, theta)

        # Pre-compute ALL spherical Bessel functions for n=0 to NMAX
        # (Bessel functions depend on (n, kr) only, not on m)
        k = self.k(frequency)
        kr = k * r.ravel()
        bessel_cache = precompute_spherical_bessel(nmax, kr)

        # Scaling prefactors for near-field E and H.
        # The Q coefficients from the .sph file are read with normalization_factor=1/sqrt(8π),
        # which absorbs the k*sqrt(Z0) factor present in Hansen's formulation.
        # The effective prefactor consistent with the far-field K-function scaling is sqrt(4π).
        Z0 = 376.730313668
        E_prefactor = np.sqrt(4 * np.pi)
        H_prefactor = 1j * np.sqrt(4 * np.pi) / Z0

        for (n, m) in all_modes:
            F1_E, F2_E, F1_H, F2_H = near_field_pattern_functions(
                n, m, r, theta, phi, k, legendre_cache, bessel_cache
            )

            if (n, m) in q1:
                Q1 = q1[(n, m)]
                E_r += E_prefactor * Q1 * F1_E[0]
                E_theta += E_prefactor * Q1 * F1_E[1]
                E_phi += E_prefactor * Q1 * F1_E[2]
                H_r += H_prefactor * Q1 * F1_H[0]
                H_theta += H_prefactor * Q1 * F1_H[1]
                H_phi += H_prefactor * Q1 * F1_H[2]

            if (n, m) in q2:
                Q2 = q2[(n, m)]
                E_r += E_prefactor * Q2 * F2_E[0]
                E_theta += E_prefactor * Q2 * F2_E[1]
                E_phi += E_prefactor * Q2 * F2_E[2]
                H_r += H_prefactor * Q2 * F2_H[0]
                H_theta += H_prefactor * Q2 * F2_H[1]
                H_phi += H_prefactor * Q2 * F2_H[2]

        if normalize:
            tp = self.total_power(frequency)
            if tp > 0:
                norm = np.sqrt(tp)
                E_r /= norm
                E_theta /= norm
                E_phi /= norm
                H_r /= norm
                H_theta /= norm
                H_phi /= norm

        return (E_r, E_theta, E_phi), (H_r, H_theta, H_phi)

    def near_field_cartesian(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                             frequency: float,
                             normalize: bool = True) -> \
            Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
                  Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calculate near-field E and H in Cartesian coordinates.

        Args:
            x, y, z: Cartesian coordinates in meters
            frequency: Frequency in Hz (must be a loaded frequency)
            normalize: If True (default), divide by sqrt(total_power) for
                      consistency with directivity-normalized far field.

        Returns:
            E: Tuple of (E_x, E_y, E_z) components in V/m
            H: Tuple of (H_x, H_y, H_z) components in A/m
        """
        r, theta, phi = cartesian_to_spherical(x, y, z)
        (E_r, E_theta, E_phi), (H_r, H_theta, H_phi) = self.near_field(
            r, theta, phi, frequency=frequency, normalize=normalize
        )
        E_x, E_y, E_z = spherical_to_cartesian_field(E_r, E_theta, E_phi, theta, phi)
        H_x, H_y, H_z = spherical_to_cartesian_field(H_r, H_theta, H_phi, theta, phi)
        return (E_x, E_y, E_z), (H_x, H_y, H_z)
    
    def currents_on_surface(self, rr: np.ndarray, unr: np.ndarray, dSr: np.ndarray,
                            frequency: float,
                            swe_origin: np.ndarray = None,
                            swe_rotation: Optional[Tuple[float, float, float]] = None) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Calculate equivalent surface currents on an arbitrary reflector surface.

        Uses the Surface Equivalence Theorem: J = n × H, M = -n × E.

        Args:
            rr: Nr x 3 array of reflector surface points (Cartesian, meters)
            unr: Nr x 3 array of surface normal vectors (outward)
            dSr: Nr array of surface element areas (m²)
            frequency: Frequency in Hz (must be a loaded frequency)
            swe_origin: 3-element array, SWE coordinate origin in reflector frame.
                        Default is [0, 0, 0].
            swe_rotation: (alpha, beta, gamma) Euler angles (radians, ZYZ convention)
                        rotating SWE frame to reflector frame. Default is no rotation.

        Returns:
            Jrr: Nr x 3 array of equivalent electric currents (A)
            Mrr: Nr x 3 array of equivalent magnetic currents (V)
        """
        self._validate_freq(frequency)
        Nr = len(rr)
        logger.debug(f"Computing surface currents on {Nr} surface points")

        if swe_origin is None:
            swe_origin = np.array([0., 0., 0.])

        rr_swe = rr - swe_origin[np.newaxis, :]

        if swe_rotation is not None:
            rr_swe = self._apply_inverse_rotation(rr_swe, swe_rotation)
            unr_swe = self._apply_inverse_rotation(unr, swe_rotation)
        else:
            unr_swe = unr.copy()

        x, y, z = rr_swe[:, 0], rr_swe[:, 1], rr_swe[:, 2]
        (Ex, Ey, Ez), (Hx, Hy, Hz) = self.near_field_cartesian(
            x, y, z, frequency=frequency
        )

        E_total = np.column_stack([Ex, Ey, Ez])
        H_total = np.column_stack([Hx, Hy, Hz])

        if swe_rotation is not None:
            E_total = self._apply_rotation(E_total, swe_rotation)
            H_total = self._apply_rotation(H_total, swe_rotation)

        Jrr = np.cross(unr_swe, H_total, axis=-1) * dSr[:, np.newaxis]
        Mrr = -np.cross(unr_swe, E_total, axis=-1) * dSr[:, np.newaxis]

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

def read_ticra_sph(filename: str) -> List[Dict]:
    """
    Read TICRA .sph file containing spherical wave expansion coefficients.

    Reads all frequency blocks present in the file.

    Args:
        filename: Path to the .sph file

    Returns:
        List of dicts, one per frequency block, each containing:
          'frequency' (GHz), 'NMAX', 'MMAX', 'NTHE', 'NPHI',
          'rotation_angles', 'Q1_coeffs', 'Q2_coeffs', 'power'

    Raises:
        ValueError: If no frequency blocks are found in the file.
    """
    logger.info(f"Reading TICRA .sph file: {filename}")

    # NORMALIZATION FACTOR (from Ticra)
    normalization_factor = 1/np.sqrt(8*np.pi)

    with open(filename, 'r') as f:
        lines = f.readlines()

    results = []
    line_idx = 0

    while line_idx < len(lines):
        # Scan forward to the next PRGTAG line
        while line_idx < len(lines) and 'Freq [GHz]:' not in lines[line_idx]:
            line_idx += 1
        if line_idx >= len(lines):
            break

        # Record 1: PRGTAG
        prgtag = lines[line_idx].strip()
        line_idx += 1

        frequency = None
        if 'Freq [GHz]:' in prgtag:
            freq_str = prgtag.split('Freq [GHz]:')[1].strip().split()[0]
            frequency = float(freq_str)

        # Record 2: IDSTRG
        line_idx += 1  # skip description line

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
            angles_str = rotation_line.split('=')[1].strip().strip('()')
            rotation_angles = tuple(float(x.strip()) for x in angles_str.split(','))
        else:
            rotation_angles = (0.0, 0.0, 0.0)

        # Records 5-8: Dummy data (skip 4 lines)
        line_idx += 4

        # Read coefficients for this frequency block
        Q1_coeffs: Dict[Tuple[int, int], complex] = {}
        Q2_coeffs: Dict[Tuple[int, int], complex] = {}
        power: Dict[int, float] = {}
        m_blocks_read = 0

        while line_idx < len(lines) and m_blocks_read <= MMAX:
            line = lines[line_idx].strip()
            if not line:
                line_idx += 1
                continue

            parts = line.split()
            if len(parts) >= 2:
                try:
                    m_index = int(parts[0])
                    powerm = float(parts[1])
                except (ValueError, IndexError):
                    line_idx += 1
                    continue

                if abs(m_index) > MMAX:
                    break

                power[abs(m_index)] = powerm
                line_idx += 1
                m_blocks_read += 1

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
                            Q1_coeffs[(n, 0)] = complex(float(coeff_parts[0]), float(coeff_parts[1]))
                            Q2_coeffs[(n, 0)] = complex(float(coeff_parts[2]), float(coeff_parts[3]))
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
                            Q1_coeffs[(n, -abs_m)] = complex(float(coeff_parts[0]), float(coeff_parts[1]))
                            Q2_coeffs[(n, -abs_m)] = complex(float(coeff_parts[2]), float(coeff_parts[3]))
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
                            Q1_coeffs[(n, abs_m)] = complex(float(coeff_parts[0]), float(coeff_parts[1]))
                            Q2_coeffs[(n, abs_m)] = complex(float(coeff_parts[2]), float(coeff_parts[3]))
                        else:
                            line_idx -= 1
                            break
            else:
                line_idx += 1

        # Convert from TICRA file convention to internal convention:
        # Q_internal = -conj(Q_file) * normalization_factor
        for key in Q1_coeffs:
            Q1_coeffs[key] = -np.conj(Q1_coeffs[key]) * normalization_factor
        for key in Q2_coeffs:
            Q2_coeffs[key] = -np.conj(Q2_coeffs[key]) * normalization_factor

        logger.info(f"Loaded block: {frequency} GHz, NMAX={NMAX}, MMAX={MMAX}, "
                    f"{len(Q1_coeffs)} Q1 modes, {len(Q2_coeffs)} Q2 modes")

        results.append({
            'frequency': frequency,
            'NTHE': NTHE,
            'NPHI': NPHI,
            'NMAX': NMAX,
            'MMAX': MMAX,
            'rotation_angles': rotation_angles,
            'Q1_coeffs': Q1_coeffs,
            'Q2_coeffs': Q2_coeffs,
            'power': power,
        })

    if not results:
        raise ValueError(f"No frequency blocks found in {filename}")

    logger.info(f"Read {len(results)} frequency block(s) from {filename}")
    return results


def write_ticra_sph(filename: str,
                    freq_data_list: List[Dict],
                    description: str = "Generated by SWE module"):
    """
    Write spherical wave coefficients to TICRA .sph file format.

    Args:
        filename: Output file path
        freq_data_list: List of per-frequency dicts, each containing:
            'Q1_coeffs'    : Dict[(n, m) -> complex]  (internal convention)
            'Q2_coeffs'    : Dict[(n, m) -> complex]
            'frequency_GHz': float
            'NMAX'         : int
            'MMAX'         : int
            'NTHE'         : int  (optional, default 181)
            'NPHI'         : int  (optional, default 361)
        description: Description text written in every block header

    Internal coefficients are converted back to file convention on write:
        Q_file = -conj(Q_internal) * sqrt(8*pi)
    """
    logger.info(f"Writing TICRA .sph file: {filename} ({len(freq_data_list)} frequency block(s))")

    # Inverse of the 1/sqrt(8*pi) applied on read
    normalization_factor = np.sqrt(8 * np.pi)

    with open(filename, 'w') as f:
        for block in freq_data_list:
            Q1_coeffs = block['Q1_coeffs']
            Q2_coeffs = block['Q2_coeffs']
            frequency_GHz = block['frequency_GHz']
            NMAX = block['NMAX']
            MMAX = block['MMAX']
            NTHE = block.get('NTHE', 181)
            NPHI = block.get('NPHI', 361)

            logger.debug(f"Writing block: {frequency_GHz:.6f} GHz, NMAX={NMAX}, MMAX={MMAX}, "
                         f"{len(Q1_coeffs)} Q1 modes, {len(Q2_coeffs)} Q2 modes")

            # Record 1: PRGTAG
            f.write(f"TICRA-SWE Freq [GHz]: {frequency_GHz:.6f}\n")
            # Record 2: IDSTRG
            f.write(f"{description}\n")
            # Record 3: NTHE, NPHI, NMAX, MMAX
            f.write(f"{NTHE:5d}{NPHI:5d}{NMAX:5d}{MMAX:5d}\n")
            # Record 4: Rotation angles
            f.write("Rotation angles = (  0.00000,  0.00000,  0.00000)\n")
            # Records 5-8: Dummy data (4 lines matching TICRA format)
            f.write("  0.00000000000000E+00  0.00000000000000E+00  0.00000000000000E+00  0.00000000000000E+00  0.00000000000000E+00\n")
            f.write("  0.00000000000000E+00  0.00000000000000E+00  0.00000000000000E+00  0.00000000000000E+00  0.00000000000000E+00\n")
            f.write("SWEP_DUMMY_FILE_NAME\n")
            f.write("SWEP_DUMMY_FILE_NAME\n")

            # Write coefficients for each |m|
            for m_val in range(0, MMAX + 1):
                # Calculate power for this m (in internal units)
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

                f.write(f"{m_val:5d}  {power:23.16E}\n")

                for n in range(max(1, m_val), NMAX + 1):
                    if m_val == 0:
                        # internal -> file: Q_file = -conj(Q_internal) * sqrt(8pi)
                        Q1 = -np.conj(Q1_coeffs.get((n, 0), 0.0)) * normalization_factor
                        Q2 = -np.conj(Q2_coeffs.get((n, 0), 0.0)) * normalization_factor
                        f.write(f"  {Q1.real:23.16E} {Q1.imag:23.16E} "
                                f"{Q2.real:23.16E} {Q2.imag:23.16E}\n")
                    else:
                        # -m line
                        Q1_neg = -np.conj(Q1_coeffs.get((n, -m_val), 0.0)) * normalization_factor
                        Q2_neg = -np.conj(Q2_coeffs.get((n, -m_val), 0.0)) * normalization_factor
                        f.write(f"  {Q1_neg.real:23.16E} {Q1_neg.imag:23.16E} "
                                f"{Q2_neg.real:23.16E} {Q2_neg.imag:23.16E}\n")
                        # +m line
                        Q1_pos = -np.conj(Q1_coeffs.get((n, m_val), 0.0)) * normalization_factor
                        Q2_pos = -np.conj(Q2_coeffs.get((n, m_val), 0.0)) * normalization_factor
                        f.write(f"  {Q1_pos.real:23.16E} {Q1_pos.imag:23.16E} "
                                f"{Q2_pos.real:23.16E} {Q2_pos.imag:23.16E}\n")