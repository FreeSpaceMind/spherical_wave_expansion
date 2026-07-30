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
- Extract Q coefficients from far-field patterns (from_far_field)

Physical Conventions:
- Time dependence: exp(jωt) (matches Ticra)
- Frequency: Hz
- Wavenumber k: rad/m
- Electric field E: V/m
- Magnetic field H: A/m
- Impedance η₀ = 376.73 Ω (free space)
- Outgoing spherical waves: h_n^(2)(kr) = j_n(kr) - i·y_n(kr)

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
from typing import Dict, Tuple, Optional, Union, Iterable, List
import warnings
from multiprocessing import Pool

# NumPy 1.x/2.x compatibility: np.trapezoid appeared in 2.0 and np.trapz
# is removed there. Downstream hosts (UmbraAntennaDesigner) pin 1.26, so
# resolve whichever this NumPy provides once, at import.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz
import os

# Optional Numba acceleration for performance-critical functions
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Configure module-level logger
logger = logging.getLogger(__name__)


def _match_frequency_key(frequency: float,
                         available: Iterable[float],
                         rtol: float = 1e-6) -> float:
    """Return the available frequency matching ``frequency`` within tolerance."""
    available_list = [float(freq) for freq in available]
    if not available_list:
        raise ValueError("No SWE frequencies are available")

    target = float(frequency)
    matches = [
        freq for freq in available_list
        if np.isclose(freq, target, rtol=rtol, atol=0.0)
    ]
    if matches:
        return min(matches, key=lambda freq: abs(freq - target))

    available_text = ", ".join(f"{freq / 1e9:.9g} GHz" for freq in available_list)
    raise ValueError(
        f"Frequency {target / 1e9:.9g} GHz is not available. "
        f"Available frequencies: {available_text}"
    )


class _FrequencyIndexedCoefficients(dict):
    """Dict-compatible coefficient view that can also be called by frequency."""

    def __init__(self,
                 by_frequency: Dict[float, Dict[Tuple[int, int], complex]],
                 active_frequency: Optional[float]):
        self._by_frequency = by_frequency
        self._active_frequency = active_frequency
        active = self._active_dict()
        super().__init__(active)

    def _active_key(self) -> Optional[float]:
        if not self._by_frequency:
            return None
        if self._active_frequency is None:
            return next(iter(self._by_frequency))
        return _match_frequency_key(self._active_frequency, self._by_frequency.keys())

    def _active_dict(self) -> Dict[Tuple[int, int], complex]:
        key = self._active_key()
        if key is None:
            return {}
        return self._by_frequency[key]

    def set_active_frequency(self, frequency: Optional[float]) -> None:
        self._active_frequency = frequency
        super().clear()
        super().update(self._active_dict())

    def __call__(self, frequency: Optional[float] = None) -> Dict[Tuple[int, int], complex]:
        if frequency is None:
            return self._active_dict()
        key = _match_frequency_key(float(frequency), self._by_frequency.keys())
        return self._by_frequency[key]

    def __setitem__(self, key, value):
        active = self._active_dict()
        active[key] = value
        super().__setitem__(key, value)

    def __delitem__(self, key):
        active = self._active_dict()
        if key in active:
            del active[key]
        super().__delitem__(key)

    def clear(self):
        self._active_dict().clear()
        super().clear()

    def update(self, *args, **kwargs):
        active = self._active_dict()
        active.update(*args, **kwargs)
        super().clear()
        super().update(active)


class _FrequencyIndexedInt(int):
    """Integer that keeps old int behavior and supports calls by frequency."""

    def __new__(cls,
                value: int,
                by_frequency: Dict[float, int],
                active_frequency: Optional[float]):
        obj = int.__new__(cls, int(value))
        obj._by_frequency = by_frequency
        obj._active_frequency = active_frequency
        return obj

    def __call__(self, frequency: Optional[float] = None) -> int:
        if frequency is None or not self._by_frequency:
            return int(self)
        key = _match_frequency_key(float(frequency), self._by_frequency.keys())
        return int(self._by_frequency[key])


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
    Compute NORMALIZED associated Legendre functions P̄_n^m(cos θ) for all n from
    |m| to n_max using a fully normalized recurrence.

    Normalization (Hansen convention):
        P̄_n^m = sqrt((2n+1)/2 * (n-|m|)! / (n+|m|)!) * P_n^m

    The recurrence operates entirely on normalized values, preventing float64
    overflow at any degree/order (including abs_m > ~143 where the un-normalized
    P_m^m seed would otherwise exceed 1.8e308).

    Seed P̄_m^m is computed in log-space:
        log|P̄_m^m| = 0.5*(log(2m+1) - log(2) - lgamma(2m+1))
                    + lgamma(m+0.5) + m*log(2) - 0.5*log(π)
                    + m*log(sin θ)

    Normalized recurrence (n > m+1):
        P̄_n^m = A_nm * cos(θ) * P̄_{n-1}^m - B_nm * P̄_{n-2}^m
        A_nm = sqrt((4n²-1) / (n²-m²))
        B_nm = sqrt((2n+1)(n-1-m)(n-1+m) / ((2n-3)(n²-m²)))

    Returns:
        Array of shape (n_max - |m| + 1, N) containing P̄_{|m|}^m, ..., P̄_{n_max}^m
        (already Hansen-normalized; callers must NOT multiply by norm_factor again)
    """
    abs_m = abs(m)
    cos_theta = np.atleast_1d(cos_theta)
    N = len(cos_theta)

    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sin_theta = np.sqrt(1 - cos_theta**2)

    n_count = n_max - abs_m + 1
    P = np.zeros((n_count, N))

    if abs_m > n_max:
        return P

    # -----------------------------------------------------------------
    # Normalized seed P̄_m^m in log-space to prevent float64 overflow.
    # log C = 0.5*(log(2m+1) - log(2) - lgamma(2m+1))
    #        + lgamma(m + 0.5) + m*log(2) - 0.5*log(π)
    # P̄_m^m = (-1)^m * C * sin^m(θ)   (C > 0 by construction)
    # -----------------------------------------------------------------
    if abs_m == 0:
        P[0, :] = math.sqrt(0.5)   # P̄_0^0 = sqrt(1/2)
    else:
        log_C = (0.5 * (math.log(2 * abs_m + 1) - math.log(2) - math.lgamma(2 * abs_m + 1))
                 + math.lgamma(abs_m + 0.5) + abs_m * math.log(2) - 0.5 * math.log(math.pi))
        with np.errstate(divide='ignore'):
            log_sin = np.where(sin_theta > 0.0, np.log(sin_theta), -np.inf)
        log_seed = log_C + abs_m * log_sin
        # Clamp to avoid both overflow (>709) and extreme underflow — values
        # beyond these bounds round to +inf or 0 in float64 anyway.
        P[0, :] = np.where(np.isfinite(log_seed),
                           np.exp(np.clip(log_seed, -708.0, 708.0)), 0.0)
        if abs_m % 2 == 1:
            P[0, :] *= -1

    if n_count == 1:
        return P

    # -----------------------------------------------------------------
    # First step: P̄_{m+1}^m = sqrt(2m+3) * cos(θ) * P̄_m^m
    # (derived from A_{m+1,m} = sqrt((4(m+1)²-1)/((m+1)²-m²)) = sqrt(2m+3))
    # -----------------------------------------------------------------
    P[1, :] = math.sqrt(2 * abs_m + 3) * cos_theta * P[0, :]

    # -----------------------------------------------------------------
    # General normalized recurrence for n >= abs_m + 2:
    #   P̄_n^m = A_nm * cos(θ) * P̄_{n-1}^m - B_nm * P̄_{n-2}^m
    # -----------------------------------------------------------------
    for i in range(2, n_count):
        n = abs_m + i
        n2 = n * n
        m2 = abs_m * abs_m
        n2m2 = n2 - m2                          # n² - m²
        A = math.sqrt((4 * n2 - 1) / n2m2)
        B = math.sqrt((2 * n + 1) * (n - 1 - abs_m) * (n - 1 + abs_m)
                      / ((2 * n - 3) * n2m2))
        P[i, :] = A * cos_theta * P[i - 1, :] - B * P[i - 2, :]

    return P


def compute_legendre_derivative_recurrence(n_max: int, m: int, cos_theta: np.ndarray,
                                           P: np.ndarray = None) -> np.ndarray:
    """
    Compute derivatives dP̄_n^m/dθ for the NORMALIZED Legendre functions returned
    by compute_legendre_recurrence.

    Uses the normalized derivative formula:
        sin(θ) dP̄_n^m/dθ = n*cos(θ)*P̄_n^m - sqrt((n²-m²)(2n+1)/(2n-1)) * P̄_{n-1}^m

    For n = |m| (seed): P̄_{m-1}^m ≡ 0, so:
        dP̄_m^m/dθ = |m| * cos(θ)/sin(θ) * P̄_m^m

    Args:
        n_max: Maximum degree
        m: Order
        cos_theta: cos(θ) values
        P: Pre-computed normalized P̄ values from compute_legendre_recurrence
           (computed if not provided)

    Returns:
        Array of shape (n_max - |m| + 1, N) containing dP̄_n^m/dθ
    """
    abs_m = abs(m)
    cos_theta = np.atleast_1d(cos_theta)
    N = len(cos_theta)

    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sin_theta = np.sqrt(1 - cos_theta**2)

    if P is None:
        P = compute_legendre_recurrence(n_max, m, cos_theta)

    n_count = n_max - abs_m + 1
    dP = np.zeros((n_count, N))

    safe_sin = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)

    # Seed: dP̄_m^m/dθ = abs_m * cos(θ)/sin(θ) * P̄_m^m
    if n_count > 0:
        dP[0, :] = abs_m * cos_theta * P[0, :] / safe_sin

    # General: sin(θ)*dP̄_n^m/dθ = n*cos(θ)*P̄_n^m - D_nm * P̄_{n-1}^m
    # where D_nm = sqrt((n²-m²)(2n+1)/(2n-1))
    for i in range(1, n_count):
        n = abs_m + i
        D = math.sqrt((n * n - abs_m * abs_m) * (2 * n + 1) / (2 * n - 1))
        dP[i, :] = (n * cos_theta * P[i, :] - D * P[i - 1, :]) / safe_sin

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
    
    # compute_legendre_recurrence returns already-normalized P̄_n^m values.
    # We need up to n so the derivative recurrence has P̄_{n-1}^m available.
    P_all = compute_legendre_recurrence(n, abs(m), cos_theta)
    dP_all = compute_legendre_derivative_recurrence(n, abs(m), cos_theta, P_all)

    return P_all[-1, :], dP_all[-1, :]


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
        
        # compute_legendre_recurrence now returns already-normalized P̄_n^m values;
        # no additional norm_factor multiplication is needed.
        for i, n in enumerate(range(abs_m, n_max + 1)):
            if n < 1:  # Skip n=0 for spherical wave expansion
                continue
            results[(n, m)] = (P_all[i, :], dP_all[i, :])
    
    return results

def far_field_pattern_functions(n: int, m: int,
                                         theta: np.ndarray,
                                         phi: np.ndarray,
                                         legendre_cache: Dict = None,
                                         phase_cache: Dict = None):
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
        phase_cache: Optional dict mapping m -> exp(-1j*m*phi) (avoids recomputing
                     the same phase for every n at a fixed m value)

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
    if phase_cache is not None and m in phase_cache:
        phase = phase_cache[m]
    else:
        phase = np.exp(-1j * m * phi)
    i_factor_1 = (1j) ** (n + 1)
    i_factor_2 = (1j) ** (n)

    sin_theta = np.sin(theta_safe)
    mP_over_sin = m * P_norm / sin_theta

    K1_theta = prefactor * sign_factor * phase * i_factor_1 * (-1j * mP_over_sin)
    K1_phi = prefactor * sign_factor * phase * i_factor_1 * (-dP_norm_dtheta)
    
    K2_theta = prefactor * sign_factor * phase * i_factor_2 * (dP_norm_dtheta)
    K2_phi = prefactor * sign_factor * phase * i_factor_2 * (-1j * mP_over_sin)
    
    return (K1_theta, K1_phi), (K2_theta, K2_phi)


def _precompute_bessel_numpy(nmax: int, kr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numpy upward recurrence for spherical Bessel functions (no Numba required).

    Uses the same recurrence as the Numba version:
        f_{n+1}(x) = (2n+1)/x * f_n(x) - f_{n-1}(x)

    This is stable for y_n (Neumann, growing) and for j_n when n < kr.
    For n >> kr the j_n recurrence loses relative precision, but this is
    acceptable in practice because those modes carry negligible power.

    Each recurrence step is a single vectorised numpy operation over all
    kr values, replacing the 2*(nmax+1) individual scipy calls.
    """
    kr = np.where(kr < 1e-30, 1e-30, kr)  # guard against division by zero

    j_all = np.empty((nmax + 1, len(kr)))
    y_all = np.empty((nmax + 1, len(kr)))

    # Analytic initial values
    j_all[0] = np.sin(kr) / kr
    y_all[0] = -np.cos(kr) / kr

    if nmax >= 1:
        j_all[1] = np.sin(kr) / kr**2 - np.cos(kr) / kr
        y_all[1] = -np.cos(kr) / kr**2 - np.sin(kr) / kr

    # Upward recurrence: f_{n+1} = (2n+1)/x * f_n - f_{n-1}
    for n in range(1, nmax):
        factor = (2 * n + 1) / kr
        j_all[n + 1] = factor * j_all[n] - j_all[n - 1]
        y_all[n + 1] = factor * y_all[n] - y_all[n - 1]

    return j_all, y_all


def _precompute_bessel_scipy(nmax: int, kr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Scipy-based Bessel pre-computation (kept as reference/fallback)."""
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
        return _precompute_bessel_numpy(nmax, kr)


if HAS_NUMBA:
    @jit(nopython=True, parallel=True)
    def _near_field_accumulate_numba(n_arr, m_arr, q1_arr, q2_arr,
                                     prefactor_arr, sign_arr,
                                     P_arr, dP_arr, h_all, dkrh_all,
                                     kr, sin_theta, phi,
                                     E_prefactor, H_prefactor):
        n_points = phi.shape[0]
        n_modes = n_arr.shape[0]
        E_r = np.zeros(n_points, dtype=np.complex128)
        E_theta = np.zeros(n_points, dtype=np.complex128)
        E_phi = np.zeros(n_points, dtype=np.complex128)
        H_r = np.zeros(n_points, dtype=np.complex128)
        H_theta = np.zeros(n_points, dtype=np.complex128)
        H_phi = np.zeros(n_points, dtype=np.complex128)

        for i in prange(n_points):
            er = 0.0 + 0.0j
            et = 0.0 + 0.0j
            ep = 0.0 + 0.0j
            hr = 0.0 + 0.0j
            ht = 0.0 + 0.0j
            hp = 0.0 + 0.0j
            kr_i = kr[i]
            if abs(kr_i) < 1e-300:
                kr_i = 1e-300
            sin_i = sin_theta[i]
            if abs(sin_i) < 1e-12:
                sin_i = 1e-12

            for mode_idx in range(n_modes):
                n = n_arr[mode_idx]
                m = m_arr[mode_idx]
                q1 = q1_arr[mode_idx]
                q2 = q2_arr[mode_idx]
                phase_angle = -m * phi[i]
                phase = math.cos(phase_angle) + 1j * math.sin(phase_angle)
                coef = prefactor_arr[mode_idx] * sign_arr[mode_idx] * phase

                P = P_arr[mode_idx, i]
                dP = dP_arr[mode_idx, i]
                h_n = h_all[n, i]
                dkrh_n = dkrh_all[n, i]
                jmP_over_sin = 1j * m * P / sin_i

                f1e_theta = -coef * h_n * jmP_over_sin
                f1e_phi = -coef * h_n * dP
                f2e_r = coef * (n * (n + 1) / kr_i) * h_n * P
                f2e_theta = coef * (dkrh_n / kr_i) * dP
                f2e_phi = -coef * (dkrh_n / kr_i) * jmP_over_sin

                er += E_prefactor * q2 * f2e_r
                et += E_prefactor * (q1 * f1e_theta + q2 * f2e_theta)
                ep += E_prefactor * (q1 * f1e_phi + q2 * f2e_phi)

                hr += H_prefactor * q1 * f2e_r
                ht += H_prefactor * (q1 * f2e_theta + q2 * f1e_theta)
                hp += H_prefactor * (q1 * f2e_phi + q2 * f1e_phi)

            E_r[i] = er
            E_theta[i] = et
            E_phi[i] = ep
            H_r[i] = hr
            H_theta[i] = ht
            H_phi[i] = hp

        return E_r, E_theta, E_phi, H_r, H_theta, H_phi


def near_field_pattern_functions(n: int, m: int, r: np.ndarray,
                                 theta: np.ndarray, phi: np.ndarray,
                                 k: float, legendre_cache: Dict = None,
                                 bessel_cache: Tuple[np.ndarray, np.ndarray] = None,
                                 phase_cache: Dict = None):
    """
    Calculate near-field pattern functions using TICRA convention.
    Uses Hankel function of second kind h_n^(2) = j_n - i*y_n
    """
    abs_m = abs(m)
    
    # Apply pole avoidance — epsilon must match compute_all_modes_legendre (1e-6)
    # so that P_norm from cache and sin_theta are evaluated at the same clamped θ.
    # A larger epsilon here would make mP_over_sin = jm·P_norm/sin_theta wrong
    # at the exact pole (θ=0) by a factor of epsilon_here/epsilon_cache.
    theta_safe = np.copy(theta)
    epsilon = 1e-6
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
        # No pre-computed cache: call scipy directly
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
    if phase_cache is not None and m in phase_cache:
        phase = phase_cache[m]
    else:
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


def _cc_theta_weights(theta_unique: np.ndarray) -> np.ndarray:
    """Effective theta quadrature weights for Clenshaw-Curtis integration.

    The existing integration code computes:

        Q ∝ sum_j w_j * phi_integral_j

    where phi_integral_j already contains the sin(theta_j) Jacobian factor.
    The weights returned here satisfy:

        sum_j w_j * [g(theta_j) * sin(theta_j)]
            ≈ integral_0^pi g(theta) sin(theta) dtheta

    i.e.,  w_j = w_CC_j / sin(theta_j)   (with w_j=0 at poles where sin=0).

    For a uniform grid theta_j = j*pi/(N-1) (CGL nodes in x=cos(theta)), w_CC
    is computed via the Waldvogel/Trefethen FFT algorithm.  For non-uniform
    grids, plain trapezoidal step-sizes are returned instead.
    """
    N = len(theta_unique)
    if N < 2:
        return np.ones(N)

    n = N - 1  # number of intervals

    # Check that this is a uniform grid
    expected = np.linspace(theta_unique[0], theta_unique[-1], N)
    if not np.allclose(theta_unique, expected, rtol=1e-6):
        # Non-uniform grid: return trapezoidal step sizes
        dt = np.zeros(N)
        dt[0] = (theta_unique[1] - theta_unique[0]) / 2
        dt[-1] = (theta_unique[-1] - theta_unique[-2]) / 2
        dt[1:-1] = (theta_unique[2:] - theta_unique[:-2]) / 2
        return dt

    # Waldvogel algorithm for CC weights on CGL nodes (Trefethen clencurt.m).
    # w_CC satisfies: sum_j w_CC[j] * f(x_j) ≈ integral_{-1}^{1} f(x) dx
    # which equals integral_0^pi f(cos θ) sin(θ) dθ  (x = cos θ).
    # numpy ifft normalises by len(c_mirror)=2n; multiply by 2 to restore [-1,1] scale.
    c = np.zeros(n + 1)
    c[0::2] = 2.0 / (1.0 - np.arange(0, n + 1, 2, dtype=float) ** 2)
    c_mirror = np.r_[c, c[-2:0:-1]]      # length 2n = (n+1)+(n-1)
    w_cc = np.real(np.fft.ifft(c_mirror))[:n + 1] * 2  # factor 2: ifft 1/(2n) vs needed 1/n
    w_cc[0] /= 2.0   # endpoint nodes appear twice in mirror
    w_cc[-1] /= 2.0

    # Convert to effective weights for integrands that already carry sin(theta).
    # w_eff[j] = w_CC[j] / sin(theta_j), except at poles (sin=0) where the
    # integrand vanishes anyway so w_eff=0.
    sin_t = np.sin(theta_unique)
    w_eff = np.zeros(N)
    ok = sin_t > 1e-10
    w_eff[ok] = w_cc[ok] / sin_t[ok]
    return w_eff


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
    w_theta = _cc_theta_weights(theta_unique)

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
        
        # Now integrate over theta using Clenshaw-Curtis quadrature
        Q1 = np.dot(w_theta, phi_integral_1) * norm_factor
        Q2 = np.dot(w_theta, phi_integral_2) * norm_factor
        
        mode_power = (abs(Q1)**2 + abs(Q2)**2) / 2.0
        results.append(((n, m), Q1, Q2, mode_power))
    
    return results


def compute_mode_coefficients_batch_trapz(args):
    """Original trapz method - fallback for non-uniform grids."""
    modes_batch, THETA, PHI, E_THETA, E_PHI, sin_theta, theta_unique, phi_unique, norm_factor, legendre_data = args

    results = []
    w_theta = _cc_theta_weights(theta_unique)

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
        Q1 = np.dot(w_theta, np.trapz(integrand_1, phi_unique, axis=1)) * norm_factor

        integrand_2 = (E_THETA * np.conj(K2_theta) + E_PHI * np.conj(K2_phi)) * sin_theta
        Q2 = np.dot(w_theta, np.trapz(integrand_2, phi_unique, axis=1)) * norm_factor
        
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
                 Q1_coeffs: Optional[Dict[Union[float, Tuple[int, int]], Union[Dict[Tuple[int, int], complex], complex]]] = None,
                 Q2_coeffs: Optional[Dict[Union[float, Tuple[int, int]], Union[Dict[Tuple[int, int], complex], complex]]] = None,
                 frequency: Optional[float] = None,
                 NMAX: Optional[Union[Dict[float, int], int]] = None,
                 MMAX: Optional[Union[Dict[float, int], int]] = None):
        """
        Initialize a multi-frequency SWE object.

        Args:
            Q1_coeffs: Either ``{freq_Hz: {(n, m): complex}}`` or a legacy
                single-frequency ``{(n, m): complex}`` mapping.
            Q2_coeffs: Same shape as ``Q1_coeffs`` for Q₂ modes.
            frequency: Frequency in Hz for legacy single-frequency inputs.
            NMAX: Maximum degree, as an int or ``{freq_Hz: int}``.
            MMAX: Maximum order, as an int or ``{freq_Hz: int}``.
        """
        q1_raw = dict(Q1_coeffs) if Q1_coeffs is not None else {}
        q2_raw = dict(Q2_coeffs) if Q2_coeffs is not None else {}
        self._frequency_order: List[float] = []
        self._Q1_by_frequency: Dict[float, Dict[Tuple[int, int], complex]] = {}
        self._Q2_by_frequency: Dict[float, Dict[Tuple[int, int], complex]] = {}
        self._NMAX_by_frequency: Dict[float, int] = {}
        self._MMAX_by_frequency: Dict[float, int] = {}
        self._frequency = float(frequency) if frequency is not None else None

        def _is_frequency_indexed(mapping: Dict) -> bool:
            return bool(mapping) and all(isinstance(value, dict) for value in mapping.values())

        def _to_mode_dict(mapping: Dict) -> Dict[Tuple[int, int], complex]:
            return {
                (int(n), int(m)): complex(value)
                for (n, m), value in mapping.items()
            }

        q1_is_indexed = _is_frequency_indexed(q1_raw)
        q2_is_indexed = _is_frequency_indexed(q2_raw)
        nmax_is_indexed = isinstance(NMAX, dict)
        mmax_is_indexed = isinstance(MMAX, dict)
        indexed_input = q1_is_indexed or q2_is_indexed or nmax_is_indexed or mmax_is_indexed

        self._unindexed_Q1_coeffs: Dict[Tuple[int, int], complex] = {}
        self._unindexed_Q2_coeffs: Dict[Tuple[int, int], complex] = {}
        self._unindexed_NMAX = 0
        self._unindexed_MMAX = 0

        if indexed_input:
            freq_set = set()
            if q1_is_indexed:
                freq_set.update(float(freq) for freq in q1_raw.keys())
            elif q1_raw and self._frequency is not None:
                freq_set.add(self._frequency)
            if q2_is_indexed:
                freq_set.update(float(freq) for freq in q2_raw.keys())
            elif q2_raw and self._frequency is not None:
                freq_set.add(self._frequency)
            if nmax_is_indexed:
                freq_set.update(float(freq) for freq in NMAX.keys())
            if mmax_is_indexed:
                freq_set.update(float(freq) for freq in MMAX.keys())
            if self._frequency is not None:
                freq_set.add(self._frequency)

            self._frequency_order = sorted(freq_set)
            if self._frequency is None and self._frequency_order:
                self._frequency = self._frequency_order[0]

            for freq in self._frequency_order:
                q1_modes = q1_raw.get(freq, q1_raw.get(float(freq), {})) if q1_is_indexed else (
                    q1_raw if self._frequency == freq else {}
                )
                q2_modes = q2_raw.get(freq, q2_raw.get(float(freq), {})) if q2_is_indexed else (
                    q2_raw if self._frequency == freq else {}
                )
                q1_modes = _to_mode_dict(q1_modes)
                q2_modes = _to_mode_dict(q2_modes)
                all_keys = list(q1_modes.keys()) + list(q2_modes.keys())

                if nmax_is_indexed and freq in NMAX:
                    nmax_value = int(NMAX[freq])
                elif nmax_is_indexed and float(freq) in NMAX:
                    nmax_value = int(NMAX[float(freq)])
                elif isinstance(NMAX, int):
                    nmax_value = int(NMAX)
                elif all_keys:
                    nmax_value = max(n for n, _ in all_keys)
                else:
                    nmax_value = 0

                if mmax_is_indexed and freq in MMAX:
                    mmax_value = int(MMAX[freq])
                elif mmax_is_indexed and float(freq) in MMAX:
                    mmax_value = int(MMAX[float(freq)])
                elif isinstance(MMAX, int):
                    mmax_value = int(MMAX)
                elif all_keys:
                    mmax_value = max(abs(m) for _, m in all_keys)
                else:
                    mmax_value = 0

                self._Q1_by_frequency[freq] = q1_modes
                self._Q2_by_frequency[freq] = q2_modes
                self._NMAX_by_frequency[freq] = nmax_value
                self._MMAX_by_frequency[freq] = mmax_value
        else:
            q1_modes = _to_mode_dict(q1_raw)
            q2_modes = _to_mode_dict(q2_raw)
            all_keys = list(q1_modes.keys()) + list(q2_modes.keys())
            nmax_value = int(NMAX) if isinstance(NMAX, int) else (
                max((n for n, _ in all_keys), default=0)
            )
            mmax_value = int(MMAX) if isinstance(MMAX, int) else (
                max((abs(m) for _, m in all_keys), default=0)
            )

            self._unindexed_Q1_coeffs = q1_modes
            self._unindexed_Q2_coeffs = q2_modes
            self._unindexed_NMAX = nmax_value
            self._unindexed_MMAX = mmax_value

            if self._frequency is not None:
                self._frequency_order = [self._frequency]
                self._Q1_by_frequency[self._frequency] = q1_modes
                self._Q2_by_frequency[self._frequency] = q2_modes
                self._NMAX_by_frequency[self._frequency] = nmax_value
                self._MMAX_by_frequency[self._frequency] = mmax_value

        self._sync_active_views()

        active_q1 = self.Q1_coeffs() if callable(self.Q1_coeffs) else self.Q1_coeffs
        active_q2 = self.Q2_coeffs() if callable(self.Q2_coeffs) else self.Q2_coeffs
        n_modes = len(active_q1) + len(active_q2)
        if n_modes > 0:
            logger.debug(
                f"SphericalWaveExpansion initialized: NMAX={int(self.NMAX)}, "
                f"MMAX={int(self.MMAX)}, {len(active_q1)} Q1 modes, {len(active_q2)} Q2 modes"
            )

    @classmethod
    def from_frequency_data(cls, frequency_data: Iterable[Dict]) -> 'SphericalWaveExpansion':
        """Create a possibly multi-frequency SWE object from parsed blocks."""
        obj = cls()
        for block in frequency_data:
            freq = float(block['frequency'])
            obj._frequency_order.append(freq)
            obj._Q1_by_frequency[freq] = dict(block.get('Q1_coeffs', {}))
            obj._Q2_by_frequency[freq] = dict(block.get('Q2_coeffs', {}))
            obj._NMAX_by_frequency[freq] = int(block.get('NMAX', 0) or 0)
            obj._MMAX_by_frequency[freq] = int(block.get('MMAX', 0) or 0)

        if obj._frequency_order:
            obj._frequency = obj._frequency_order[0]
        obj._sync_active_views()
        return obj

    def _sync_active_views(self) -> None:
        if self._frequency_order and self._frequency is None:
            self._frequency = self._frequency_order[0]
        active = self._active_frequency_key(required=False)
        active_q1 = self._Q1_by_frequency.get(active, {})
        active_q2 = self._Q2_by_frequency.get(active, {})
        active_nmax = self._NMAX_by_frequency.get(active, 0)
        active_mmax = self._MMAX_by_frequency.get(active, 0)
        if active is None:
            self.Q1_coeffs = self._unindexed_Q1_coeffs
            self.Q2_coeffs = self._unindexed_Q2_coeffs
            self.NMAX = self._unindexed_NMAX
            self.MMAX = self._unindexed_MMAX
            return

        self.Q1_coeffs = _FrequencyIndexedCoefficients(self._Q1_by_frequency, active)
        self.Q2_coeffs = _FrequencyIndexedCoefficients(self._Q2_by_frequency, active)
        self.NMAX = _FrequencyIndexedInt(active_nmax, self._NMAX_by_frequency, active)
        self.MMAX = _FrequencyIndexedInt(active_mmax, self._MMAX_by_frequency, active)

    def _active_frequency_key(self, frequency: Optional[float] = None,
                              required: bool = True) -> Optional[float]:
        if not self._frequency_order:
            if required:
                raise ValueError("No SWE frequencies are available")
            return None
        target = self._frequency if frequency is None else float(frequency)
        if target is None:
            return self._frequency_order[0]
        return _match_frequency_key(target, self._frequency_order)

    @property
    def frequencies(self) -> List[float]:
        """Available frequencies in Hz, in ascending order."""
        return sorted(self._frequency_order)
    
    @property
    def frequency(self) -> Optional[float]:
        """Frequency in Hz."""
        return self._frequency
    
    @frequency.setter
    def frequency(self, freq: float):
        """Set frequency in Hz."""
        self._frequency = float(freq) if freq is not None else None
        self._sync_active_views()

    def _resolve_frequency(self, frequency: Optional[float] = None) -> Optional[float]:
        """Return the active frequency key, or a legacy unindexed frequency."""
        if self._frequency_order:
            return self._active_frequency_key(frequency)
        return self._frequency if frequency is None else float(frequency)

    def _coefficients_for_frequency(
        self, frequency: Optional[float] = None
    ) -> Tuple[Optional[float], Dict[Tuple[int, int], complex],
               Dict[Tuple[int, int], complex], int, int]:
        """Return frequency, coefficient dicts, NMAX, and MMAX for computation."""
        if self._frequency_order:
            freq_key = self._active_frequency_key(frequency)
            return (
                freq_key,
                self._Q1_by_frequency[freq_key],
                self._Q2_by_frequency[freq_key],
                self._NMAX_by_frequency[freq_key],
                self._MMAX_by_frequency[freq_key],
            )

        freq_key = self._frequency if frequency is None else float(frequency)
        return (
            freq_key,
            self._unindexed_Q1_coeffs,
            self._unindexed_Q2_coeffs,
            self._unindexed_NMAX,
            self._unindexed_MMAX,
        )

    def _validate_freq(self, freq: Optional[float]) -> None:
        """Raise a helpful error if *freq* is not loaded."""
        if self._frequency_order:
            self._active_frequency_key(freq)
            return
        if freq is None and self._frequency is None:
            raise ValueError("Frequency must be set before computing fields")
    
    def k(self, freq: Optional[float] = None) -> float:
        """Wavenumber in rad/m."""
        freq_key = self._resolve_frequency(freq)
        if freq_key is None:
            raise ValueError("Frequency must be set before computing wavenumber")
        return 2 * np.pi * float(freq_key) / self._C
    
    def wavelength(self, freq: Optional[float] = None) -> float:
        """Wavelength in meters."""
        freq_key = self._resolve_frequency(freq)
        if freq_key is None:
            raise ValueError("Frequency must be set before computing wavelength")
        return self._C / float(freq_key)

    def total_power(self, freq: Optional[float] = None) -> float:
        """Total power in the SWE coefficients: Σ(|Q1|² + |Q2|²).

        Used for directivity normalization. When coefficients are normalized
        (via normalize_coefficients()), total_power = 1.0 and the far field
        with normalize=True gives |E|² = directivity.
        """
        _, q1_coeffs, q2_coeffs, _, _ = self._coefficients_for_frequency(freq)
        return self._total_power_for(q1_coeffs, q2_coeffs)

    @staticmethod
    def _total_power_for(q1_coeffs: Dict[Tuple[int, int], complex],
                         q2_coeffs: Dict[Tuple[int, int], complex]) -> float:
        power = 0.0
        for Q in q1_coeffs.values():
            power += abs(Q) ** 2
        for Q in q2_coeffs.values():
            power += abs(Q) ** 2
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
        if self._frequency_order:
            for freq in self._frequency_order:
                q1 = self._Q1_by_frequency[freq]
                q2 = self._Q2_by_frequency[freq]
                tp = self._total_power_for(q1, q2)
                if tp <= 0:
                    logger.warning(
                        "Cannot normalize %.9g GHz: total_power is zero",
                        freq / 1e9,
                    )
                    continue
                norm = np.sqrt(tp)
                logger.debug(
                    f"Normalizing coefficients at {freq/1e9:.6g} GHz: "
                    f"total_power={tp:.6f}, dividing by {norm:.6f}"
                )
                for key in list(q1.keys()):
                    q1[key] /= norm
                for key in list(q2.keys()):
                    q2[key] /= norm
            self._sync_active_views()
            return

        tp = self.total_power()
        if tp <= 0:
            logger.warning("Cannot normalize: total_power is zero")
            return
        norm = np.sqrt(tp)
        logger.debug(f"Normalizing coefficients: total_power={tp:.6f}, dividing by {norm:.6f}")
        for key in self.Q1_coeffs:
            self.Q1_coeffs[key] /= norm
        for key in self.Q2_coeffs:
            self.Q2_coeffs[key] /= norm

    @staticmethod
    def _select_active_modes(q1_coeffs: Dict[Tuple[int, int], complex],
                             q2_coeffs: Dict[Tuple[int, int], complex],
                             nmax: int,
                             mmax: int,
                             power_threshold: float = 0.999,
                             azimuthal_power_threshold: float = 1e-5,
                             per_mode_power_threshold: float = 1e-8) -> \
            Tuple[List[Tuple[int, int]], int, int, float]:
        """Select active modes using n-shell and per-|m| power filtering."""
        all_modes = set(q1_coeffs.keys()) | set(q2_coeffs.keys())
        if not all_modes:
            return [], 0, 0, 0.0

        mode_powers = []
        for mode in all_modes:
            q1 = q1_coeffs.get(mode, 0.0)
            q2 = q2_coeffs.get(mode, 0.0)
            mode_powers.append((mode, abs(q1) ** 2 + abs(q2) ** 2))

        total_power = sum(power for _, power in mode_powers)
        if total_power <= 0.0:
            active = sorted(all_modes)
            return active, nmax, min(mmax, nmax), total_power

        effective_nmax = nmax
        if power_threshold < 1.0:
            n_power: Dict[int, float] = {}
            for (n, _m), power in mode_powers:
                n_power[n] = n_power.get(n, 0.0) + power

            cumulative = 0.0
            for n in sorted(n_power):
                cumulative += n_power[n]
                if cumulative >= power_threshold * total_power:
                    effective_nmax = n
                    break

        m_power: Dict[int, float] = {}
        for (n, m), power in mode_powers:
            if n <= effective_nmax:
                m_power[abs(m)] = m_power.get(abs(m), 0.0) + power

        active_abs_m = {
            abs_m for abs_m, power in m_power.items()
            if power >= azimuthal_power_threshold * total_power
        }
        if not active_abs_m and m_power:
            active_abs_m = {max(m_power, key=m_power.get)}

        cutoff = per_mode_power_threshold * total_power
        active_modes = [
            mode for mode, power in mode_powers
            if mode[0] <= effective_nmax
            and abs(mode[1]) in active_abs_m
            and power > cutoff
        ]
        active_modes.sort()

        effective_mmax = max((abs(m) for _, m in active_modes), default=0)
        effective_mmax = min(effective_mmax, mmax, effective_nmax)

        if len(active_modes) < len(all_modes):
            logger.debug(
                "Power filtering: NMAX %s -> %s, MMAX %s -> %s, "
                "%s/%s modes kept",
                nmax, effective_nmax, mmax, effective_mmax,
                len(active_modes), len(all_modes),
            )

        return active_modes, effective_nmax, effective_mmax, total_power

    def far_field(self, theta: np.ndarray, phi: np.ndarray,
                  frequency: Optional[float] = None,
                  power_threshold: float = 0.999,
                  azimuthal_power_threshold: float = 1e-5,
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
        freq_key, q1_coeffs, q2_coeffs, nmax, mmax = self._coefficients_for_frequency(frequency)
        if freq_key is None:
            raise ValueError("Frequency must be set before computing far field")

        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)

        if theta.shape != phi.shape:
            theta, phi = np.broadcast_arrays(theta, phi)

        active_modes, effective_nmax, effective_mmax, total_power = self._select_active_modes(
            q1_coeffs,
            q2_coeffs,
            nmax,
            mmax,
            power_threshold=power_threshold,
            azimuthal_power_threshold=azimuthal_power_threshold,
        )

        if not active_modes:
            logger.debug("No modes present, returning zero field")
            E_theta = np.zeros_like(theta, dtype=complex)
            E_phi = np.zeros_like(phi, dtype=complex)
            return E_theta, E_phi

        logger.debug(f"Computing far field at {len(theta.flatten())} points, "
                     f"effective NMAX={effective_nmax}")

        E_theta = np.zeros_like(theta, dtype=complex)
        E_phi = np.zeros_like(phi, dtype=complex)

        legendre_cache = compute_all_modes_legendre(effective_nmax, effective_mmax, theta)

        # Precompute exp(-1j*m*phi) for each unique m in filtered_modes.
        # Many n values share the same m, so this avoids redundant complex exponentials.
        unique_m = {m for _, m in active_modes}
        phase_cache = {m_val: np.exp(-1j * m_val * phi) for m_val in unique_m}

        for n, m in active_modes:
            if (n, m) not in legendre_cache:
                continue

            (K1_theta, K1_phi), (K2_theta, K2_phi) = \
                far_field_pattern_functions(n, m, theta, phi, legendre_cache, phase_cache)

            if (n, m) in q1_coeffs:
                Q1 = q1_coeffs[(n, m)]
                E_theta += Q1 * K1_theta
                E_phi += Q1 * K1_phi

            if (n, m) in q2_coeffs:
                Q2 = q2_coeffs[(n, m)]
                E_theta += Q2 * K2_theta
                E_phi += Q2 * K2_phi

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
                            NMAX_initial: int = 100,
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
            NMAX_initial: Starting NMAX (default: 100)
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

            # ------------------------------------------------------------------
            # Azimuthal pre-screening: compute m=0 and m=±1 modes first.
            # If they already account for ≥ power_threshold of extracted power,
            # cap MMAX_current=1 and skip the remaining |m| values entirely.
            # This avoids computing thousands of empty modes for rotationally
            # symmetric or near-symmetric antennas.
            # ------------------------------------------------------------------
            if use_adaptive_mmax and MMAX_current > 1:
                prescreened_mmax = min(1, MMAX_current)
                leg_pre = compute_all_modes_legendre(NMAX, prescreened_mmax, THETA[:, 0])
                modes_pre = [(n, m) for n in range(1, NMAX + 1)
                             for m in range(-min(n, prescreened_mmax), min(n, prescreened_mmax) + 1)]
                args_pre = [(modes_pre, THETA, PHI, E_THETA, E_PHI, sin_theta,
                             theta_unique, phi_unique, norm_factor,
                             {nm: leg_pre[nm] for nm in modes_pre if nm in leg_pre})]
                pre_results = compute_mode_coefficients_batch(args_pre[0])
                pre_total = sum(pw for _, _, _, pw in pre_results)
                all_m01_power = pre_total

                # Estimate full total power by sampling a few higher |m| modes
                # (m=2,3) to see if they carry significant energy
                if MMAX_current >= 2:
                    leg_m2 = compute_all_modes_legendre(min(NMAX, 10), 2, THETA[:, 0])
                    modes_m2 = [(n, m) for n in range(1, min(NMAX, 10) + 1)
                                for m in [-2, 2] if abs(m) <= n]
                    modes_m2 = [nm for nm in modes_m2 if nm in leg_m2]
                    if modes_m2:
                        args_m2 = [(modes_m2, THETA, PHI, E_THETA, E_PHI, sin_theta,
                                    theta_unique, phi_unique, norm_factor,
                                    {nm: leg_m2[nm] for nm in modes_m2})]
                        m2_results = compute_mode_coefficients_batch(args_m2[0])
                        m2_power = sum(pw for _, _, _, pw in m2_results)
                    else:
                        m2_power = 0.0
                else:
                    m2_power = 0.0

                # If |m|≥2 modes contain < 0.1% of the |m|≤1 power, cap MMAX
                if all_m01_power > 0 and m2_power / all_m01_power < 0.001:
                    MMAX_current = 1
                    logger.info(f"Azimuthal pre-screening: |m|≥2 power is {m2_power/all_m01_power*100:.4f}% "
                                f"of |m|≤1 power — capping MMAX={MMAX_current}")

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
                w_theta = _cc_theta_weights(theta_unique)

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
                    Q1 = np.dot(w_theta, _trapezoid(integrand_1, phi_unique, axis=1)) * norm_factor

                    integrand_2 = (E_THETA * np.conj(K2_theta) + E_PHI * np.conj(K2_phi)) * sin_theta
                    Q2 = np.dot(w_theta, _trapezoid(integrand_2, phi_unique, axis=1)) * norm_factor
                    
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

                # Find MMAX where tail power is negligible AND cumulative power is sufficient
                MMAX_truncated = MMAX_current  # default: keep everything
                for m_test in range(0, MMAX_current + 1):
                    # Cumulative power up to |m| = m_test
                    cumulative_m = sum(power_per_m[m] for m in range(0, m_test + 1))
                    cumulative_m_fraction = cumulative_m / total_power if total_power > 0 else 0

                    # Tail power: power at |m| > m_test (modes we'd discard)
                    tail_power = sum(power_per_m[m] for m in range(m_test + 1, MMAX_current + 1))
                    tail_fraction = tail_power / total_power if total_power > 0 else 0

                    # Accept if: enough cumulative power AND discarded tail is small
                    if (cumulative_m_fraction >= power_threshold and
                            tail_fraction < azimuthal_power_threshold):
                        MMAX_truncated = m_test
                        logger.debug(f"Azimuthal truncation: MMAX {MMAX_current} -> {MMAX_truncated}, "
                                     f"tail power: {tail_fraction*100:.4f}%")
                        break
                
                # Find highest n where tail modes have negligible power
                power_by_n = {n: 0 for n in range(1, NMAX+1)}
                for (n, m), pwr in mode_powers:
                    power_by_n[n] += pwr

                # Find smallest NMAX where local top modes are at noise level AND cumulative is sufficient
                NMAX_truncated = NMAX  # default: keep everything if no noise floor found
                for n_test in range(1, NMAX + 1):
                    # Check power in top few modes near n_test (noise-floor detection)
                    n_cutoff = max(1, int(np.ceil(0.1 * n_test)))
                    high_modes = range(max(1, n_test - n_cutoff + 1), n_test + 1)
                    high_power = sum(power_by_n.get(n, 0) for n in high_modes)
                    high_power_fraction = high_power / total_power if total_power > 0 else 0

                    # Cumulative power up to n_test
                    cumulative = sum(power_by_n.get(n, 0) for n in range(1, n_test + 1))
                    cumulative_fraction = cumulative / total_power if total_power > 0 else 0

                    # Accept if: top modes are at noise level AND enough cumulative power
                    if (high_power_fraction < high_mode_power_threshold and
                            cumulative_fraction >= power_threshold):
                        NMAX_truncated = n_test
                        logger.debug(f"N truncation: NMAX {NMAX} -> {NMAX_truncated}, "
                                     f"local power: {high_power_fraction*100:.4f}%")
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
    def from_sph_file(cls, filename: str,
                      frequencies: Optional[Iterable[float]] = None,
                      normalize: bool = True) -> 'SphericalWaveExpansion':
        """
        Create a multi-frequency SWE object from a TICRA .sph file.

        All frequency blocks present in the file are loaded.

        Args:
            filename: Path to .sph file
            frequencies: Optional frequencies in Hz to retain. If None, all
                blocks in the file are loaded.

        Returns:
            SphericalWaveExpansion object with all frequencies loaded.
        """
        logger.info(f"Creating SphericalWaveExpansion from file: {filename}")
        blocks = read_ticra_sph_blocks(filename)

        if frequencies is not None:
            requested = np.asarray(list(np.atleast_1d(frequencies)), dtype=float).reshape(-1)
            available = [block['frequency'] * 1e9 for block in blocks]
            selected_blocks = []
            for requested_freq in requested:
                match = _match_frequency_key(float(requested_freq), available)
                selected_blocks.append(
                    blocks[available.index(match)]
                )
            blocks = selected_blocks

        frequency_data = []
        for data in blocks:
            frequency = data['frequency'] * 1e9 if data['frequency'] is not None else None
            if frequency is None:
                continue
            frequency_data.append({
                'frequency': frequency,
                'Q1_coeffs': data['Q1_coeffs'],
                'Q2_coeffs': data['Q2_coeffs'],
                'NMAX': data['NMAX'],
                'MMAX': data['MMAX'],
            })

        if not frequency_data:
            raise ValueError(f"No frequency blocks found in {filename}")

        swe = cls.from_frequency_data(frequency_data)

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
        frequencies = self.frequencies
        if len(frequencies) == 0:
            frequencies = np.asarray([self.frequency], dtype=float)

        for idx, freq in enumerate(frequencies):
            if freq is None or not np.isfinite(freq):
                continue
            q1 = self.Q1_coeffs(freq) if callable(self.Q1_coeffs) else self.Q1_coeffs
            q2 = self.Q2_coeffs(freq) if callable(self.Q2_coeffs) else self.Q2_coeffs
            nmax = self.NMAX(freq) if callable(self.NMAX) else self.NMAX
            mmax = self.MMAX(freq) if callable(self.MMAX) else self.MMAX
            write_ticra_sph(
                filename,
                q1,
                q2,
                float(freq) / 1e9,
                NTHE, NPHI,
                int(nmax), int(mmax),
                description,
                file_mode='w' if idx == 0 else 'a',
            )

    @staticmethod
    def _near_field_prefactors(k: float) -> Tuple[complex, complex]:
        """Return E/H scaling prefactors for the near-field pattern functions."""
        Z0 = 376.730313668
        return k * np.sqrt(Z0), 1j * k / np.sqrt(Z0)

    @staticmethod
    def _evaluate_near_field_general(
        r: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        k: float,
        active_modes: List[Tuple[int, int]],
        q1_coeffs: Dict[Tuple[int, int], complex],
        q2_coeffs: Dict[Tuple[int, int], complex],
        effective_nmax: int,
        effective_mmax: int,
        use_numba: bool = True,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Pure NumPy reference path for arbitrary point clouds."""
        E_r = np.zeros_like(r, dtype=complex)
        E_theta = np.zeros_like(theta, dtype=complex)
        E_phi = np.zeros_like(phi, dtype=complex)
        H_r = np.zeros_like(r, dtype=complex)
        H_theta = np.zeros_like(theta, dtype=complex)
        H_phi = np.zeros_like(phi, dtype=complex)

        if not active_modes:
            return (E_r, E_theta, E_phi), (H_r, H_theta, H_phi)

        legendre_cache = compute_all_modes_legendre(effective_nmax, effective_mmax, theta)
        bessel_cache = precompute_spherical_bessel(effective_nmax, k * r.ravel(), use_numba=use_numba)
        phase_cache = {m: np.exp(-1j * m * phi) for _, m in active_modes}
        E_prefactor, H_prefactor = SphericalWaveExpansion._near_field_prefactors(k)

        for n, m in active_modes:
            F1_E, F2_E, F1_H, F2_H = near_field_pattern_functions(
                n, m, r, theta, phi, k, legendre_cache, bessel_cache, phase_cache
            )

            if (n, m) in q1_coeffs:
                q1 = q1_coeffs[(n, m)]
                E_r += E_prefactor * q1 * F1_E[0]
                E_theta += E_prefactor * q1 * F1_E[1]
                E_phi += E_prefactor * q1 * F1_E[2]
                H_r += H_prefactor * q1 * F1_H[0]
                H_theta += H_prefactor * q1 * F1_H[1]
                H_phi += H_prefactor * q1 * F1_H[2]

            if (n, m) in q2_coeffs:
                q2 = q2_coeffs[(n, m)]
                E_r += E_prefactor * q2 * F2_E[0]
                E_theta += E_prefactor * q2 * F2_E[1]
                E_phi += E_prefactor * q2 * F2_E[2]
                H_r += H_prefactor * q2 * F2_H[0]
                H_theta += H_prefactor * q2 * F2_H[1]
                H_phi += H_prefactor * q2 * F2_H[2]

        return (E_r, E_theta, E_phi), (H_r, H_theta, H_phi)

    @staticmethod
    def _evaluate_near_field_numba(
        r: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        k: float,
        active_modes: List[Tuple[int, int]],
        q1_coeffs: Dict[Tuple[int, int], complex],
        q2_coeffs: Dict[Tuple[int, int], complex],
        effective_nmax: int,
        effective_mmax: int,
        chunk_size: int = 4096,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Chunked Numba accumulation path for arbitrary point clouds."""
        if not HAS_NUMBA:
            return SphericalWaveExpansion._evaluate_near_field_general(
                r, theta, phi, k, active_modes, q1_coeffs, q2_coeffs,
                effective_nmax, effective_mmax, use_numba=False,
            )

        n_points = len(r)
        E_r = np.zeros(n_points, dtype=complex)
        E_theta = np.zeros(n_points, dtype=complex)
        E_phi = np.zeros(n_points, dtype=complex)
        H_r = np.zeros(n_points, dtype=complex)
        H_theta = np.zeros(n_points, dtype=complex)
        H_phi = np.zeros(n_points, dtype=complex)

        n_arr = np.array([n for n, _ in active_modes], dtype=np.int64)
        m_arr = np.array([m for _, m in active_modes], dtype=np.int64)
        q1_arr = np.array([q1_coeffs.get(mode, 0.0 + 0.0j) for mode in active_modes],
                          dtype=np.complex128)
        q2_arr = np.array([q2_coeffs.get(mode, 0.0 + 0.0j) for mode in active_modes],
                          dtype=np.complex128)
        prefactor_arr = np.array(
            [1.0 / np.sqrt(2.0 * np.pi) / np.sqrt(n * (n + 1)) for n, _ in active_modes],
            dtype=np.float64,
        )
        sign_arr = np.array(
            [1.0 if m == 0 else float((-m / abs(m)) ** m) for _, m in active_modes],
            dtype=np.float64,
        )
        E_prefactor, H_prefactor = SphericalWaveExpansion._near_field_prefactors(k)

        for start in range(0, n_points, chunk_size):
            stop = min(start + chunk_size, n_points)
            r_chunk = r[start:stop]
            theta_chunk = theta[start:stop]
            phi_chunk = phi[start:stop]
            kr = k * r_chunk.ravel()

            legendre_cache = compute_all_modes_legendre(
                effective_nmax, effective_mmax, theta_chunk
            )
            P_arr = np.empty((len(active_modes), stop - start), dtype=np.float64)
            dP_arr = np.empty_like(P_arr)
            for mode_idx, mode in enumerate(active_modes):
                P_arr[mode_idx], dP_arr[mode_idx] = legendre_cache[mode]

            j_all, y_all = precompute_spherical_bessel(effective_nmax, kr, use_numba=True)
            h_all = j_all - 1j * y_all
            dkrh_all = np.empty_like(h_all)
            safe_kr = np.where(np.abs(kr) < 1e-300, 1e-300, kr)
            dkrh_all[0] = 1j * np.cos(safe_kr)
            for n in range(1, effective_nmax + 1):
                dkrh_all[n] = safe_kr * h_all[n - 1] - n * h_all[n]

            theta_safe = np.copy(theta_chunk)
            epsilon = 1e-6
            theta_safe = np.where(theta_safe < epsilon, epsilon, theta_safe)
            theta_safe = np.where(theta_safe > (np.pi - epsilon), np.pi - epsilon, theta_safe)
            sin_theta = np.sin(theta_safe)

            er, et, ep, hr, ht, hp = _near_field_accumulate_numba(
                n_arr, m_arr, q1_arr, q2_arr,
                prefactor_arr, sign_arr,
                P_arr, dP_arr, h_all, dkrh_all,
                safe_kr, sin_theta, phi_chunk,
                E_prefactor, H_prefactor,
            )
            E_r[start:stop] = er
            E_theta[start:stop] = et
            E_phi[start:stop] = ep
            H_r[start:stop] = hr
            H_theta[start:stop] = ht
            H_phi[start:stop] = hp

        return (E_r, E_theta, E_phi), (H_r, H_theta, H_phi)

    @staticmethod
    def _detect_rings(
        r: np.ndarray,
        theta: np.ndarray,
        rtol: float = 1e-9,
        theta_atol: float = 1e-10,
        max_ring_fraction: float = 0.25,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Group points by equal (r, theta), returning None when not beneficial."""
        r_flat = np.asarray(r, dtype=float).ravel()
        theta_flat = np.asarray(theta, dtype=float).ravel()
        n_points = len(r_flat)
        if n_points == 0:
            return None

        r_scale = max(float(np.max(np.abs(r_flat))), 1.0)
        r_step = max(rtol * r_scale, np.finfo(float).eps)
        theta_step = max(theta_atol, np.finfo(float).eps)
        quantized = np.column_stack([
            np.round(r_flat / r_step).astype(np.int64),
            np.round(theta_flat / theta_step).astype(np.int64),
        ])

        _unique, first_index, point_to_ring = np.unique(
            quantized, axis=0, return_index=True, return_inverse=True
        )
        n_rings = len(first_index)
        if n_rings == n_points or n_rings / n_points > max_ring_fraction:
            return None
        if np.any(np.bincount(point_to_ring) == 1):
            return None

        ring_r = r_flat[first_index]
        ring_theta = theta_flat[first_index]
        if np.max(np.abs(r_flat - ring_r[point_to_ring])) > r_step:
            return None
        if np.max(np.abs(theta_flat - ring_theta[point_to_ring])) > theta_step:
            return None

        return ring_r, ring_theta, point_to_ring

    @staticmethod
    def _evaluate_near_field_bor(
        ring_r: np.ndarray,
        ring_theta: np.ndarray,
        phi: np.ndarray,
        point_to_ring: np.ndarray,
        k: float,
        active_modes: List[Tuple[int, int]],
        q1_coeffs: Dict[Tuple[int, int], complex],
        q2_coeffs: Dict[Tuple[int, int], complex],
        effective_nmax: int,
        effective_mmax: int,
        use_numba: bool = True,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """BOR fast path: evaluate radial/theta terms once per ring."""
        n_points = len(phi)
        E_r = np.zeros(n_points, dtype=complex)
        E_theta = np.zeros(n_points, dtype=complex)
        E_phi = np.zeros(n_points, dtype=complex)
        H_r = np.zeros(n_points, dtype=complex)
        H_theta = np.zeros(n_points, dtype=complex)
        H_phi = np.zeros(n_points, dtype=complex)

        if not active_modes:
            return (E_r, E_theta, E_phi), (H_r, H_theta, H_phi)

        ring_phi = np.zeros_like(ring_r)
        legendre_cache = compute_all_modes_legendre(effective_nmax, effective_mmax, ring_theta)
        bessel_cache = precompute_spherical_bessel(
            effective_nmax, k * ring_r.ravel(), use_numba=use_numba
        )
        ring_phase_cache = {m: np.ones_like(ring_r, dtype=complex) for _, m in active_modes}
        point_phase_cache = {m: np.exp(-1j * m * phi) for _, m in active_modes}
        E_prefactor, H_prefactor = SphericalWaveExpansion._near_field_prefactors(k)

        def expand(base, phase):
            if np.isscalar(base):
                return base
            return base[point_to_ring] * phase

        for n, m in active_modes:
            F1_E, F2_E, F1_H, F2_H = near_field_pattern_functions(
                n, m, ring_r, ring_theta, ring_phi, k,
                legendre_cache, bessel_cache, ring_phase_cache
            )
            phase = point_phase_cache[m]

            if (n, m) in q1_coeffs:
                q1 = q1_coeffs[(n, m)]
                E_r += E_prefactor * q1 * expand(F1_E[0], phase)
                E_theta += E_prefactor * q1 * expand(F1_E[1], phase)
                E_phi += E_prefactor * q1 * expand(F1_E[2], phase)
                H_r += H_prefactor * q1 * expand(F1_H[0], phase)
                H_theta += H_prefactor * q1 * expand(F1_H[1], phase)
                H_phi += H_prefactor * q1 * expand(F1_H[2], phase)

            if (n, m) in q2_coeffs:
                q2 = q2_coeffs[(n, m)]
                E_r += E_prefactor * q2 * expand(F2_E[0], phase)
                E_theta += E_prefactor * q2 * expand(F2_E[1], phase)
                E_phi += E_prefactor * q2 * expand(F2_E[2], phase)
                H_r += H_prefactor * q2 * expand(F2_H[0], phase)
                H_theta += H_prefactor * q2 * expand(F2_H[1], phase)
                H_phi += H_prefactor * q2 * expand(F2_H[2], phase)

        return (E_r, E_theta, E_phi), (H_r, H_theta, H_phi)
    
    def near_field(self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray,
                   frequency: Optional[float] = None,
                   power_threshold: float = 0.999,
                   azimuthal_power_threshold: float = 1e-5,
                   normalize: bool = True,
                   use_numba: bool = True,
                   force_general: bool = False) -> \
            Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
                  Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calculate near-field E and H components.

        Automatic BOR ring detection is used when the points share repeated
        (r, theta) rings in the SWE frame. Set ``force_general=True`` to use
        the pure NumPy arbitrary-point reference path.
        """
        if isinstance(frequency, (bool, np.bool_)):
            normalize = bool(frequency)
            frequency = None

        freq_key, q1_coeffs, q2_coeffs, nmax, mmax = self._coefficients_for_frequency(frequency)
        if freq_key is None:
            raise ValueError("Frequency must be set before computing near field")
        k = 2 * np.pi * float(freq_key) / self._C

        r = np.atleast_1d(r)
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)
        if not (r.shape == theta.shape == phi.shape):
            r, theta, phi = np.broadcast_arrays(r, theta, phi)

        output_shape = r.shape
        r_flat = np.asarray(r, dtype=float).ravel()
        theta_flat = np.asarray(theta, dtype=float).ravel()
        phi_flat = np.asarray(phi, dtype=float).ravel()

        logger.debug(
            "Computing near field at %s points, NMAX=%s, MMAX=%s",
            len(r_flat), nmax, mmax,
        )

        active_modes, effective_nmax, effective_mmax, total_power = self._select_active_modes(
            q1_coeffs,
            q2_coeffs,
            nmax,
            mmax,
            power_threshold=power_threshold,
            azimuthal_power_threshold=azimuthal_power_threshold,
        )

        if not active_modes:
            zero = np.zeros(output_shape, dtype=complex)
            return (zero.copy(), zero.copy(), zero.copy()), (zero.copy(), zero.copy(), zero.copy())

        ring_data = None if force_general else self._detect_rings(r_flat, theta_flat)
        if ring_data is not None:
            ring_r, ring_theta, point_to_ring = ring_data
            logger.debug(
                "Near-field BOR path: %s points compressed to %s rings",
                len(r_flat), len(ring_r),
            )
            E, H = self._evaluate_near_field_bor(
                ring_r,
                ring_theta,
                phi_flat,
                point_to_ring,
                k,
                active_modes,
                q1_coeffs,
                q2_coeffs,
                effective_nmax,
                effective_mmax,
                use_numba=use_numba,
            )
        else:
            logger.debug("Near-field general path: %s points", len(r_flat))
            if use_numba and HAS_NUMBA:
                E, H = self._evaluate_near_field_numba(
                    r_flat,
                    theta_flat,
                    phi_flat,
                    k,
                    active_modes,
                    q1_coeffs,
                    q2_coeffs,
                    effective_nmax,
                    effective_mmax,
                )
            else:
                E, H = self._evaluate_near_field_general(
                    r_flat,
                    theta_flat,
                    phi_flat,
                    k,
                    active_modes,
                    q1_coeffs,
                    q2_coeffs,
                    effective_nmax,
                    effective_mmax,
                    use_numba=use_numba,
                )

        if normalize and total_power > 0:
            norm = np.sqrt(total_power)
            E = tuple(component / norm for component in E)
            H = tuple(component / norm for component in H)

        E = tuple(component.reshape(output_shape) for component in E)
        H = tuple(component.reshape(output_shape) for component in H)
        return E, H
    def near_field_cartesian(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                             frequency: Optional[float] = None,
                             power_threshold: float = 0.999,
                             azimuthal_power_threshold: float = 1e-5,
                             normalize: bool = True,
                             use_numba: bool = True,
                             force_general: bool = False) -> \
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
        if isinstance(frequency, (bool, np.bool_)):
            normalize = bool(frequency)
            frequency = None

        # Convert to spherical coordinates
        r, theta, phi = cartesian_to_spherical(x, y, z)

        # Get fields in spherical basis
        (E_r, E_theta, E_phi), (H_r, H_theta, H_phi) = self.near_field(
            r, theta, phi,
            frequency=frequency,
            power_threshold=power_threshold,
            azimuthal_power_threshold=azimuthal_power_threshold,
            normalize=normalize,
            use_numba=use_numba,
            force_general=force_general,
        )
        
        # Convert to Cartesian basis
        E_x, E_y, E_z = spherical_to_cartesian_field(E_r, E_theta, E_phi, theta, phi)
        H_x, H_y, H_z = spherical_to_cartesian_field(H_r, H_theta, H_phi, theta, phi)
        return (E_x, E_y, E_z), (H_x, H_y, H_z)
    
    def currents_on_surface(self, rr: np.ndarray, unr: np.ndarray, dSr: np.ndarray,
                                swe_origin: np.ndarray = None,
                                swe_rotation: Optional[Tuple[float, float, float]] = None,
                                frequency: Optional[float] = None,
                                power_threshold: float = 0.999,
                                azimuthal_power_threshold: float = 1e-5,
                                use_numba: bool = True,
                                force_general: bool = False) -> \
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
            x, y, z,
            frequency=frequency,
            power_threshold=power_threshold,
            azimuthal_power_threshold=azimuthal_power_threshold,
            use_numba=use_numba,
            force_general=force_general,
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

def _parse_ticra_sph_block(lines: List[str]) -> Dict:
    """Parse one TICRA .sph block."""
    normalization_factor = 1 / np.sqrt(8 * np.pi)
    line_idx = 0

    prgtag = lines[line_idx].strip()
    line_idx += 1

    frequency = None
    if 'Freq [GHz]:' in prgtag:
        freq_str = prgtag.split('Freq [GHz]:')[1].strip().split()[0]
        frequency = float(freq_str)

    idstrg = lines[line_idx].strip()
    line_idx += 1

    control_data = lines[line_idx].strip().split()
    NTHE = int(control_data[0])
    NPHI = int(control_data[1])
    NMAX = int(control_data[2])
    MMAX = int(control_data[3])
    line_idx += 1

    rotation_line = lines[line_idx].strip()
    line_idx += 1
    if 'Rotation angles' in rotation_line:
        angles_str = rotation_line.split('=')[1].strip().strip('()')
        rotation_angles = tuple(float(x.strip()) for x in angles_str.split(','))
    else:
        rotation_angles = (0.0, 0.0, 0.0)

    line_idx += 4

    Q1_coeffs = {}
    Q2_coeffs = {}
    power = {}

    while line_idx < len(lines):
        line = lines[line_idx].strip()
        if not line:
            line_idx += 1
            continue

        parts = line.split()
        if len(parts) < 2:
            line_idx += 1
            continue

        try:
            m_index = int(parts[0])
            powerm = float(parts[1])
        except (ValueError, IndexError):
            line_idx += 1
            continue

        power[abs(m_index)] = powerm
        line_idx += 1

        abs_m = abs(m_index)
        n_start = max(1, abs_m)

        if m_index == 0:
            for n in range(n_start, NMAX + 1):
                if line_idx >= len(lines):
                    break
                coeff_parts = lines[line_idx].strip().split()
                line_idx += 1
                if len(coeff_parts) < 4:
                    line_idx -= 1
                    break
                Q1_coeffs[(n, 0)] = normalization_factor * (
                    float(coeff_parts[0]) + 1j * float(coeff_parts[1])
                )
                Q2_coeffs[(n, 0)] = normalization_factor * (
                    float(coeff_parts[2]) + 1j * float(coeff_parts[3])
                )
        else:
            for n in range(n_start, NMAX + 1):
                if line_idx >= len(lines):
                    break
                coeff_parts = lines[line_idx].strip().split()
                line_idx += 1
                if len(coeff_parts) < 4:
                    line_idx -= 1
                    break
                Q1_coeffs[(n, -abs_m)] = normalization_factor * (
                    float(coeff_parts[0]) + 1j * float(coeff_parts[1])
                )
                Q2_coeffs[(n, -abs_m)] = normalization_factor * (
                    float(coeff_parts[2]) + 1j * float(coeff_parts[3])
                )

                if line_idx >= len(lines):
                    break
                coeff_parts = lines[line_idx].strip().split()
                line_idx += 1
                if len(coeff_parts) < 4:
                    line_idx -= 1
                    break
                Q1_coeffs[(n, abs_m)] = normalization_factor * (
                    float(coeff_parts[0]) + 1j * float(coeff_parts[1])
                )
                Q2_coeffs[(n, abs_m)] = normalization_factor * (
                    float(coeff_parts[2]) + 1j * float(coeff_parts[3])
                )

    for key in Q1_coeffs:
        Q1_coeffs[key] = np.conj(Q1_coeffs[key])
    for key in Q2_coeffs:
        Q2_coeffs[key] = np.conj(Q2_coeffs[key])

    logger.info(
        f"Loaded SWE: NMAX={NMAX}, MMAX={MMAX}, frequency={frequency} GHz, "
        f"{len(Q1_coeffs)} Q1 modes, {len(Q2_coeffs)} Q2 modes"
    )
    logger.debug(f"Rotation angles: {rotation_angles}")

    return {
        'frequency': frequency,
        'idstrg': idstrg,
        'NTHE': NTHE,
        'NPHI': NPHI,
        'NMAX': NMAX,
        'MMAX': MMAX,
        'rotation_angles': rotation_angles,
        'Q1_coeffs': Q1_coeffs,
        'Q2_coeffs': Q2_coeffs,
        'power': power,
    }


def read_ticra_sph_blocks(filename: str) -> List[Dict]:
    """Read all concatenated TICRA .sph frequency blocks."""
    logger.info(f"Reading TICRA .sph file: {filename}")
    with open(filename, 'r') as f:
        lines = f.readlines()

    starts = [
        idx for idx, line in enumerate(lines)
        if 'Freq [GHz]:' in line
    ]
    if not starts:
        raise ValueError(f"No TICRA .sph frequency blocks found in {filename}")

    blocks = []
    for block_idx, start in enumerate(starts):
        end = starts[block_idx + 1] if block_idx + 1 < len(starts) else len(lines)
        blocks.append(_parse_ticra_sph_block(lines[start:end]))
    return blocks


def read_ticra_sph(filename: str) -> List[Dict]:
    """
    Read TICRA .sph file containing spherical wave expansion coefficients.

    Returns all frequency blocks present in the file.
    """
    return read_ticra_sph_blocks(filename)


def write_ticra_sph(filename: str,
                    Q1_coeffs: Dict[Tuple[int, int], complex],
                    Q2_coeffs: Dict[Tuple[int, int], complex],
                    frequency_GHz: float,
                    NTHE: int, NPHI: int,
                    NMAX: int, MMAX: int,
                    description: str = "Generated by SWE module",
                    file_mode: str = 'w'):
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
        Q_file = -conj(Q_internal)
    """
    logger.info(
        f"Writing TICRA .sph file: {filename}, "
        f"NMAX={NMAX}, MMAX={MMAX}, frequency={frequency_GHz} GHz"
    )
    normalization_factor = np.sqrt(8 * np.pi)

    with open(filename, file_mode) as f:
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
                    # Conjugate and scale before writing (reverse read operation)
                    Q1 = np.conj(Q1_coeffs.get((n, 0), 0.0))*normalization_factor
                    Q2 = np.conj(Q2_coeffs.get((n, 0), 0.0))*normalization_factor
                    
                    f.write(f"  {Q1.real:23.16E} {Q1.imag:23.16E} "
                           f"{Q2.real:23.16E} {Q2.imag:23.16E}\n")
                else:
                    # -m line (conjugate before writing)
                    Q1_neg = np.conj(Q1_coeffs.get((n, -m_val), 0.0))*normalization_factor
                    Q2_neg = np.conj(Q2_coeffs.get((n, -m_val), 0.0))*normalization_factor
                    
                    f.write(f"  {Q1_neg.real:23.16E} {Q1_neg.imag:23.16E} "
                           f"{Q2_neg.real:23.16E} {Q2_neg.imag:23.16E}\n")
                    
                    # +m line (conjugate before writing)
                    Q1_pos = np.conj(Q1_coeffs.get((n, m_val), 0.0))*normalization_factor
                    Q2_pos = np.conj(Q2_coeffs.get((n, m_val), 0.0))*normalization_factor
                    
                    f.write(f"  {Q1_pos.real:23.16E} {Q1_pos.imag:23.16E} "
                           f"{Q2_pos.real:23.16E} {Q2_pos.imag:23.16E}\n")
