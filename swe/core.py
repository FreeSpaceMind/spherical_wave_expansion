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

def hansen_associated_legendre(n: int, m: int, cos_theta: np.ndarray) -> np.ndarray:
    """
    Compute unnormalized associated Legendre function per Hansen's convention.
    
    Hansen (following Stratton) defines:
        P_n^m(cos θ) = (sin θ)^m × d^m P_n(cos θ) / d(cos θ)^m
    
    This does NOT include the Condon-Shortley phase factor (-1)^m.
    
    Args:
        n: Degree (n >= 0)
        m: Order (0 <= m <= n, use absolute value)
        cos_theta: Cosine of polar angle
        
    Returns:
        P_n^m(cos θ) in Hansen's convention
    """
    abs_m = abs(m)
    
    # scipy's lpmv includes (-1)^m phase, so we remove it
    P_scipy = lpmv(abs_m, n, cos_theta)
    P_hansen = (-1)**abs_m * P_scipy
    
    return P_hansen

def normalized_associated_legendre(n: int, m: int, theta: np.ndarray):
    """
    Compute normalized associated Legendre function per Hansen equation A1.25.
    
    P̄_n^m(cos θ) = sqrt((2n+1)/2 × (n-m)!/(n+m)!) × P_n^m(cos θ)
    
    Also computes the derivative dP̄_n^m/dθ.
    
    Args:
        n: Degree (n >= 1)
        m: Order (-n <= m <= n)
        theta: Polar angle(s) in radians
        
    Returns:
        P_norm: Normalized associated Legendre function
        dP_norm: Derivative with respect to theta
    """
    abs_m = abs(m)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Safe sin_theta to avoid division by zero at poles
    sin_theta_safe = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
    
    # Compute unnormalized associated Legendre function (Hansen convention)
    P_unnorm = hansen_associated_legendre(n, abs_m, cos_theta)
    
    # Normalization factor from equation A1.25
    norm_factor = np.sqrt((2*n + 1) / 2 * 
                         np.math.factorial(n - abs_m) / 
                         np.math.factorial(n + abs_m))
    
    P_norm = norm_factor * P_unnorm
    
    # Compute derivative using Hansen's equation A1.34b
    # For normalized functions, the coefficients change due to normalization factor ratios
    if abs_m == 0:
        # For m=0: dP̄_n^0/dθ = -√[n(n+1)] × P̄_n^1
        P_bar_n_1 = hansen_associated_legendre(n, 1, cos_theta)
        norm_factor_n_1 = np.sqrt((2*n + 1) / 2 * 
                                 np.math.factorial(n - 1) / 
                                 np.math.factorial(n + 1))
        P_bar_n_1_norm = norm_factor_n_1 * P_bar_n_1
        
        coeff = np.sqrt(n * (n + 1))
        dP_norm = -coeff * P_bar_n_1_norm
    else:
        # For m > 0: dP̄_n^m/dθ = (1/2){√[(n-m+1)(n+m)] P̄_n^(m-1) - √[(n+m+1)(n-m)] P̄_n^(m+1)}
        
        # First term: √[(n-m+1)(n+m)] × P̄_n^(m-1)
        P_n_m_minus = hansen_associated_legendre(n, abs_m - 1, cos_theta)
        norm_factor_m_minus = np.sqrt((2*n + 1) / 2 * 
                                     np.math.factorial(n - (abs_m - 1)) / 
                                     np.math.factorial(n + (abs_m - 1)))
        P_bar_n_m_minus = norm_factor_m_minus * P_n_m_minus
        
        coeff1 = np.sqrt((n - abs_m + 1) * (n + abs_m))
        term1 = coeff1 * P_bar_n_m_minus
        
        # Second term: √[(n+m+1)(n-m)] × P̄_n^(m+1)
        if abs_m + 1 <= n:
            P_n_m_plus = hansen_associated_legendre(n, abs_m + 1, cos_theta)
            norm_factor_m_plus = np.sqrt((2*n + 1) / 2 * 
                                        np.math.factorial(n - (abs_m + 1)) / 
                                        np.math.factorial(n + (abs_m + 1)))
            P_bar_n_m_plus = norm_factor_m_plus * P_n_m_plus
            
            coeff2 = np.sqrt((n + abs_m + 1) * (n - abs_m))
            term2 = coeff2 * P_bar_n_m_plus
        else:
            # P̄_n^(n+1) = 0
            term2 = 0.0
        
        dP_norm = 0.5 * (term1 - term2)
    
    return P_norm, dP_norm

def far_field_pattern_functions(n: int, m: int, theta: np.ndarray, phi: np.ndarray):
    abs_m = abs(m)
    
    # Avoid exact evaluation at poles
    theta_safe = np.copy(theta)
    epsilon = 1e-3
    at_north_pole = theta < epsilon
    at_south_pole = theta > (np.pi - epsilon)
    
    theta_safe = np.where(at_north_pole, epsilon, theta_safe)
    theta_safe = np.where(at_south_pole, np.pi - epsilon, theta_safe)
    
    # Now compute with safe theta values
    P_norm, dP_norm_dtheta = normalized_associated_legendre(n, m, theta_safe)
    
    prefactor = np.sqrt(2 / (n * (n + 1)))
    
    if m == 0:
        sign_factor = 1.0
    else:
        sign_factor = (-m / abs(m)) ** m
    
    phase = np.exp(1j * m * phi)
    i_factor_1 = (-1j) ** (n + 1)
    i_factor_2 = (-1j) ** n
    
    # Now sin_theta_safe is never zero
    sin_theta = np.sin(theta_safe)
    mP_over_sin = m * P_norm / sin_theta
    
    K1_theta = prefactor * sign_factor * phase * i_factor_1 * (1j * mP_over_sin)
    K1_phi = prefactor * sign_factor * phase * i_factor_1 * (-dP_norm_dtheta)
    
    K2_theta = prefactor * sign_factor * phase * i_factor_2 * dP_norm_dtheta
    K2_phi = prefactor * sign_factor * phase * i_factor_2 * (1j * mP_over_sin)
    
    return (K1_theta, K1_phi), (K2_theta, K2_phi)


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
        Calculate far-field E_theta and E_phi components.
        
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
        
        for (n, m) in all_modes:
            (K1_theta, K1_phi), (K2_theta, K2_phi) = \
                far_field_pattern_functions(n, m, theta, phi)
            
            if (n, m) in self.Q1_coeffs:
                Q1 = self.Q1_coeffs[(n, m)]
                E_theta += Q1 * K1_theta
                E_phi += Q1 * K1_phi
            
            if (n, m) in self.Q2_coeffs:
                Q2 = self.Q2_coeffs[(n, m)]
                E_theta += Q2 * K2_theta
                E_phi += Q2 * K2_phi
        
        # Apply overall factor k * sqrt(η₀) from TICRA equation (4)
        # E_SI(r,θ,φ) = k√ζ Σ Q_smn F_smn(r,θ,φ)
        # where ζ = η₀ = 376.73 Ω
        Z0 = 376.730313668  # Free space impedance in ohms
        E_theta *= self.k / np.sqrt(Z0)
        E_phi *= self.k / np.sqrt(Z0)
                
        return E_theta, E_phi
    
    @classmethod
    def from_spherical_near_field(cls,
                                  r: Union[float, np.ndarray],
                                  theta: np.ndarray,
                                  phi: np.ndarray,
                                  E_theta: np.ndarray,
                                  E_phi: np.ndarray,
                                  frequency: float,
                                  NMAX: int,
                                  MMAX: Optional[int] = None,
                                  use_H: bool = False,
                                  H_theta: Optional[np.ndarray] = None,
                                  H_phi: Optional[np.ndarray] = None) -> 'SphericalWaveExpansion':
        """
        Create SWE object from near-field measurements on a spherical surface.
        
        This is the most natural measurement geometry for SWE. Measurements should
        be tangential field components (E_theta, E_phi or H_theta, H_phi) on a 
        sphere of radius r.
        
        Args:
            r: Radius of measurement sphere in meters (scalar or array)
            theta: Polar angles in radians (1D array of length N)
            phi: Azimuthal angles in radians (1D array of length N)
            E_theta: Measured E_theta component (1D array of length N)
            E_phi: Measured E_phi component (1D array of length N)
            frequency: Frequency in Hz
            NMAX: Maximum degree for expansion
            MMAX: Maximum order (default: NMAX)
            use_H: If True, also include H field measurements in fit
            H_theta: Measured H_theta component (optional)
            H_phi: Measured H_phi component (optional)
            
        Returns:
            SphericalWaveExpansion object with fitted coefficients
        """
        if MMAX is None:
            MMAX = NMAX
        
        # Convert to 1D arrays
        r = np.atleast_1d(r)
        if r.size == 1:
            r = np.full(len(theta), r.item())
        
        theta = np.atleast_1d(theta).flatten()
        phi = np.atleast_1d(phi).flatten()
        E_theta = np.atleast_1d(E_theta).flatten()
        E_phi = np.atleast_1d(E_phi).flatten()
        
        N_points = len(theta)
        if not (len(r) == len(phi) == len(E_theta) == len(E_phi) == N_points):
            raise ValueError("All input arrays must have the same length")
        
        k = 2 * np.pi * frequency / 299792458.0
        
        # Build mode list
        modes = []
        for n in range(1, NMAX + 1):
            for m in range(-min(n, MMAX), min(n, MMAX) + 1):
                modes.append((n, m))
        
        N_modes = len(modes)
        
        # Build design matrix for near-field (includes radial functions)
        # Similar to far_field but with full h_n functions
        
        # Determine number of equations
        if use_H and H_theta is not None and H_phi is not None:
            H_theta = np.atleast_1d(H_theta).flatten()
            H_phi = np.atleast_1d(H_phi).flatten()
            use_H_field = True
        else:
            use_H_field = False
        
        # Each mode contributes 4 unknowns: Re(Q1), Im(Q1), Re(Q2), Im(Q2)
        N_coeffs = 4 * N_modes
        
        # Build lists to accumulate matrix rows
        A_rows = []
        b_list = []
        
        kr = k * r
        
        for mode_idx, (n, m) in enumerate(modes):
            # Get angular functions
            P_norm, dP_norm = normalized_associated_legendre(n, m, theta)
            
            # Radial functions
            j_n = spherical_jn(n, kr)
            y_n = spherical_yn(n, kr)
            h_n = j_n - 1j * y_n
            
            # Derivatives
            if n == 0:
                dh_n = -j_n + 1j * y_n
            else:
                j_n_minus = spherical_jn(n-1, kr)
                y_n_minus = spherical_yn(n-1, kr)
                h_n_minus = j_n_minus - 1j * y_n_minus
                dh_n = h_n_minus - (n + 1) / kr * h_n
            
            exp_imphi = np.exp(1j * m * phi)
            
            prefactor = np.sqrt(2 / (n * (n + 1)))
            if m == 0:
                sign_factor = 1.0
            else:
                sign_factor = (-m / abs(m)) ** m
            
            sin_theta = np.sin(theta)
            sin_theta_safe = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
            
            # Column indices for this mode's coefficients
            col_Q1_re = 4 * mode_idx
            col_Q1_im = 4 * mode_idx + 1
            col_Q2_re = 4 * mode_idx + 2
            col_Q2_im = 4 * mode_idx + 3
            
            # Q1 contribution to E field
            coef = prefactor * sign_factor * exp_imphi * k
            E_theta_Q1 = coef * (1j * m / sin_theta_safe) * P_norm * h_n / kr
            E_phi_Q1 = -coef * dP_norm * h_n / kr
            
            # Q2 contribution to E field
            E_theta_Q2 = coef * dP_norm * (k * dh_n - h_n / r) / k
            E_phi_Q2 = coef * (1j * m / sin_theta_safe) * P_norm * (k * dh_n - h_n / r) / k
            
            if mode_idx == 0:
                # Initialize A matrix on first iteration
                if use_H_field:
                    A = np.zeros((8 * N_points, N_coeffs), dtype=float)
                else:
                    A = np.zeros((4 * N_points, N_coeffs), dtype=float)
            
            # Fill A matrix for E_theta real part
            A[0*N_points:1*N_points, col_Q1_re] = np.real(E_theta_Q1)
            A[0*N_points:1*N_points, col_Q1_im] = -np.imag(E_theta_Q1)
            A[0*N_points:1*N_points, col_Q2_re] = np.real(E_theta_Q2)
            A[0*N_points:1*N_points, col_Q2_im] = -np.imag(E_theta_Q2)
            
            # Fill A matrix for E_theta imaginary part
            A[1*N_points:2*N_points, col_Q1_re] = np.imag(E_theta_Q1)
            A[1*N_points:2*N_points, col_Q1_im] = np.real(E_theta_Q1)
            A[1*N_points:2*N_points, col_Q2_re] = np.imag(E_theta_Q2)
            A[1*N_points:2*N_points, col_Q2_im] = np.real(E_theta_Q2)
            
            # Fill A matrix for E_phi real part
            A[2*N_points:3*N_points, col_Q1_re] = np.real(E_phi_Q1)
            A[2*N_points:3*N_points, col_Q1_im] = -np.imag(E_phi_Q1)
            A[2*N_points:3*N_points, col_Q2_re] = np.real(E_phi_Q2)
            A[2*N_points:3*N_points, col_Q2_im] = -np.imag(E_phi_Q2)
            
            # Fill A matrix for E_phi imaginary part
            A[3*N_points:4*N_points, col_Q1_re] = np.imag(E_phi_Q1)
            A[3*N_points:4*N_points, col_Q1_im] = np.real(E_phi_Q1)
            A[3*N_points:4*N_points, col_Q2_re] = np.imag(E_phi_Q2)
            A[3*N_points:4*N_points, col_Q2_im] = np.real(E_phi_Q2)
            
            # If using H field as well
            if use_H_field:
                Z0 = 376.730313668
                factor_H = k / (1j * k * Z0)
                
                # Q1 contribution to H field
                H_theta_Q1 = factor_H * coef * dP_norm * (k * dh_n - h_n / r) / k / k
                H_phi_Q1 = factor_H * coef * (1j * m / sin_theta_safe) * P_norm * (k * dh_n - h_n / r) / k / k
                
                # Q2 contribution to H field
                H_theta_Q2 = -factor_H * coef * (1j * m / sin_theta_safe) * P_norm * h_n / kr / k
                H_phi_Q2 = factor_H * coef * dP_norm * h_n / kr / k
                
                # Fill A matrix for H_theta real part
                A[4*N_points:5*N_points, col_Q1_re] = np.real(H_theta_Q1)
                A[4*N_points:5*N_points, col_Q1_im] = -np.imag(H_theta_Q1)
                A[4*N_points:5*N_points, col_Q2_re] = np.real(H_theta_Q2)
                A[4*N_points:5*N_points, col_Q2_im] = -np.imag(H_theta_Q2)
                
                # Fill A matrix for H_theta imaginary part
                A[5*N_points:6*N_points, col_Q1_re] = np.imag(H_theta_Q1)
                A[5*N_points:6*N_points, col_Q1_im] = np.real(H_theta_Q1)
                A[5*N_points:6*N_points, col_Q2_re] = np.imag(H_theta_Q2)
                A[5*N_points:6*N_points, col_Q2_im] = np.real(H_theta_Q2)
                
                # Fill A matrix for H_phi real part
                A[6*N_points:7*N_points, col_Q1_re] = np.real(H_phi_Q1)
                A[6*N_points:7*N_points, col_Q1_im] = -np.imag(H_phi_Q1)
                A[6*N_points:7*N_points, col_Q2_re] = np.real(H_phi_Q2)
                A[6*N_points:7*N_points, col_Q2_im] = -np.imag(H_phi_Q2)
                
                # Fill A matrix for H_phi imaginary part
                A[7*N_points:8*N_points, col_Q1_re] = np.imag(H_phi_Q1)
                A[7*N_points:8*N_points, col_Q1_im] = np.real(H_phi_Q1)
                A[7*N_points:8*N_points, col_Q2_re] = np.imag(H_phi_Q2)
                A[7*N_points:8*N_points, col_Q2_im] = np.real(H_phi_Q2)
        
        # Build b vector (measurements)
        b = np.concatenate([
            np.real(E_theta), np.imag(E_theta),
            np.real(E_phi), np.imag(E_phi)
        ])
        
        if use_H_field:
            b = np.concatenate([
                b,
                np.real(H_theta), np.imag(H_theta),
                np.real(H_phi), np.imag(H_phi)
            ])
        
        # Solve least squares
        result = lsq_linear(A, b, verbose=0)
        coeffs = result.x
        
        # Extract Q coefficients
        Q1_coeffs = {}
        Q2_coeffs = {}
        
        for mode_idx, (n, m) in enumerate(modes):
            Q1_re = coeffs[4 * mode_idx]
            Q1_im = coeffs[4 * mode_idx + 1]
            Q2_re = coeffs[4 * mode_idx + 2]
            Q2_im = coeffs[4 * mode_idx + 3]
            
            Q1_coeffs[(n, m)] = Q1_re + 1j * Q1_im
            Q2_coeffs[(n, m)] = Q2_re + 1j * Q2_im
        
        return cls(Q1_coeffs, Q2_coeffs, frequency, NMAX, MMAX)
    
    @classmethod
    def from_planar_near_field(cls,
                               x: np.ndarray,
                               y: np.ndarray,
                               z_scan: float,
                               E_x: np.ndarray,
                               E_y: np.ndarray,
                               frequency: float,
                               NMAX: int,
                               MMAX: Optional[int] = None,
                               origin_offset: Tuple[float, float, float] = (0., 0., 0.)) -> 'SphericalWaveExpansion':
        """
        Create SWE object from planar near-field measurements.
        
        Planar near-field scanners measure tangential E fields on a plane.
        This method transforms the planar measurements to spherical wave coefficients.
        
        Args:
            x: X coordinates of measurement points in meters (1D array)
            y: Y coordinates of measurement points in meters (1D array)
            z_scan: Z coordinate of scan plane in meters (scalar)
            E_x: Measured E_x component (1D array)
            E_y: Measured E_y component (1D array)
            frequency: Frequency in Hz
            NMAX: Maximum degree for expansion
            MMAX: Maximum order (default: NMAX)
            origin_offset: (x0, y0, z0) offset of spherical coordinate origin from planar scan
            
        Returns:
            SphericalWaveExpansion object with fitted coefficients
        """
        if MMAX is None:
            MMAX = NMAX
        
        # Convert planar measurements to spherical coordinates relative to origin
        x_rel = x - origin_offset[0]
        y_rel = y - origin_offset[1]
        z_rel = z_scan - origin_offset[2]
        
        # For each measurement point, compute (r, theta, phi)
        r, theta, phi = cartesian_to_spherical(x_rel, y_rel, np.full_like(x_rel, z_rel))
        
        # Convert E field from Cartesian to spherical components
        # We need E_theta and E_phi at each measurement point
        # First, get unit vectors
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        # E_z = 0 on the measurement plane (tangential components only)
        E_z = np.zeros_like(E_x)
        
        # Transform to spherical components
        # theta_hat = cos(theta)cos(phi) x_hat + cos(theta)sin(phi) y_hat - sin(theta) z_hat
        # phi_hat = -sin(phi) x_hat + cos(phi) y_hat
        E_theta = (cos_theta * cos_phi * E_x + cos_theta * sin_phi * E_y - sin_theta * E_z)
        E_phi = (-sin_phi * E_x + cos_phi * E_y)
        
        # Now use spherical near-field fitting
        return cls.from_spherical_near_field(
            r, theta, phi, E_theta, E_phi,
            frequency, NMAX, MMAX
        )
    
    @classmethod
    def from_cylindrical_near_field(cls,
                                    rho: np.ndarray,
                                    phi: np.ndarray,
                                    z: np.ndarray,
                                    E_rho: np.ndarray,
                                    E_z: np.ndarray,
                                    frequency: float,
                                    NMAX: int,
                                    MMAX: Optional[int] = None,
                                    origin_offset: Tuple[float, float, float] = (0., 0., 0.)) -> 'SphericalWaveExpansion':
        """
        Create SWE object from cylindrical near-field measurements.
        
        Cylindrical near-field scanners measure tangential E fields on a cylinder.
        
        Args:
            rho: Radial coordinate in cylindrical system (meters, 1D array)
            phi: Azimuthal angle in cylindrical system (radians, 1D array)
            z: Vertical coordinate (meters, 1D array)
            E_rho: Measured E_rho component (tangential, 1D array)
            E_z: Measured E_z component (tangential, 1D array)
            frequency: Frequency in Hz
            NMAX: Maximum degree for expansion
            MMAX: Maximum order (default: NMAX)
            origin_offset: (x0, y0, z0) offset of spherical origin from cylindrical origin
            
        Returns:
            SphericalWaveExpansion object with fitted coefficients
        """
        if MMAX is None:
            MMAX = NMAX
        
        # Convert cylindrical to Cartesian
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        
        # Apply origin offset
        x_rel = x - origin_offset[0]
        y_rel = y - origin_offset[1]
        z_rel = z - origin_offset[2]
        
        # Convert to spherical coordinates
        r, theta, phi_sph = cartesian_to_spherical(x_rel, y_rel, z_rel)
        
        # Convert E field from cylindrical to spherical components
        # Need to transform E_rho and E_z to E_theta and E_phi
        # Cylindrical: (rho_hat, phi_hat, z_hat)
        # Spherical: (r_hat, theta_hat, phi_hat)
        
        # Note: phi_cyl = phi_sph (azimuthal angles are the same)
        # E_phi_cyl = E_phi_sph
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # rho_hat = sin(theta) r_hat + cos(theta) theta_hat
        # z_hat = cos(theta) r_hat - sin(theta) theta_hat
        # phi_hat is the same in both systems
        
        # E_rho = E_rho_cyl * rho_hat = E_rho_cyl * (sin(theta) r_hat + cos(theta) theta_hat)
        # E_z = E_z_cyl * z_hat = E_z_cyl * (cos(theta) r_hat - sin(theta) theta_hat)
        # E_phi_cyl = 0 (tangential measurement on cylinder)
        
        # In spherical components:
        # E_r = E_rho * sin(theta) + E_z * cos(theta)  (but we don't need this)
        E_theta = E_rho * cos_theta - E_z * sin_theta
        E_phi = np.zeros_like(E_rho)  # E_phi_cyl was zero
        
        # Now use spherical near-field fitting
        return cls.from_spherical_near_field(
            r, theta, phi_sph, E_theta, E_phi,
            frequency, NMAX, MMAX
        )
    
    @classmethod
    def from_far_field(cls, 
                      theta: np.ndarray, 
                      phi: np.ndarray,
                      E_theta: np.ndarray, 
                      E_phi: np.ndarray,
                      frequency: float,
                      NMAX: int,
                      MMAX: Optional[int] = None) -> 'SphericalWaveExpansion':
        """
        Create SWE object from far-field measurements using least squares.
        
        Args:
            theta: Polar angles in radians (1D array of length N)
            phi: Azimuthal angles in radians (1D array of length N)
            E_theta: Measured theta component (1D array of length N)
            E_phi: Measured phi component (1D array of length N)
            frequency: Frequency in Hz
            NMAX: Maximum degree for expansion
            MMAX: Maximum order (default: NMAX)
            
        Returns:
            SphericalWaveExpansion object with fitted coefficients
        """
        if MMAX is None:
            MMAX = NMAX
        
        theta = np.atleast_1d(theta).flatten()
        phi = np.atleast_1d(phi).flatten()
        E_theta = np.atleast_1d(E_theta).flatten()
        E_phi = np.atleast_1d(E_phi).flatten()
        
        N_points = len(theta)
        if not (len(phi) == len(E_theta) == len(E_phi) == N_points):
            raise ValueError("All input arrays must have the same length")
        
        k = 2 * np.pi * frequency / 299792458.0
        
        # Build list of modes
        modes = []
        for n in range(1, NMAX + 1):
            for m in range(-min(n, MMAX), min(n, MMAX) + 1):
                modes.append((n, m))
        
        N_modes = len(modes)
        N_coeffs = 2 * N_modes  # Q1 and Q2 for each mode
        
        # Build design matrix: each row corresponds to a measurement point
        # Columns correspond to [Re(Q1), Im(Q1), Re(Q2), Im(Q2)] for each mode
        A_theta = np.zeros((N_points, N_coeffs), dtype=float)
        A_phi = np.zeros((N_points, N_coeffs), dtype=float)
        
        for mode_idx, (n, m) in enumerate(modes):
            (K1_theta, K1_phi), (K2_theta, K2_phi) = \
                far_field_pattern_functions(n, m, theta, phi)
            
            # Q1 contribution (real and imaginary parts)
            col_Q1_re = 2 * mode_idx
            col_Q1_im = 2 * mode_idx + 1
            
            A_theta[:, col_Q1_re] = k * np.real(K1_theta)
            A_theta[:, col_Q1_im] = k * -np.imag(K1_theta)  # Note the minus sign
            A_phi[:, col_Q1_re] = k * np.real(K1_phi)
            A_phi[:, col_Q1_im] = k * -np.imag(K1_phi)
            
            # Q2 contribution
            col_Q2_re = 2 * mode_idx + N_modes * 2
            col_Q2_im = 2 * mode_idx + N_modes * 2 + 1
            
            # Actually, let me reorganize: alternate Q1, Q2 for each mode
            # Columns: [Re(Q1_mode0), Im(Q1_mode0), Re(Q2_mode0), Im(Q2_mode0), ...]
            col_Q1_re = 4 * mode_idx
            col_Q1_im = 4 * mode_idx + 1
            col_Q2_re = 4 * mode_idx + 2
            col_Q2_im = 4 * mode_idx + 3
            
        # Reinitialize with correct size
        N_coeffs = 4 * N_modes
        A_theta = np.zeros((N_points, N_coeffs), dtype=float)
        A_phi = np.zeros((N_points, N_coeffs), dtype=float)
        
        for mode_idx, (n, m) in enumerate(modes):
            (K1_theta, K1_phi), (K2_theta, K2_phi) = \
                far_field_pattern_functions(n, m, theta, phi)
            
            col_Q1_re = 4 * mode_idx
            col_Q1_im = 4 * mode_idx + 1
            col_Q2_re = 4 * mode_idx + 2
            col_Q2_im = 4 * mode_idx + 3
            
            A_theta[:, col_Q1_re] = k * np.real(K1_theta)
            A_theta[:, col_Q1_im] = k * -np.imag(K1_theta)
            A_theta[:, col_Q2_re] = k * np.real(K2_theta)
            A_theta[:, col_Q2_im] = k * -np.imag(K2_theta)
            
            A_phi[:, col_Q1_re] = k * np.real(K1_phi)
            A_phi[:, col_Q1_im] = k * -np.imag(K1_phi)
            A_phi[:, col_Q2_re] = k * np.real(K2_phi)
            A_phi[:, col_Q2_im] = k * -np.imag(K2_phi)
        
        # Stack theta and phi equations
        A = np.vstack([A_theta, A_phi])
        b = np.concatenate([np.real(E_theta), np.imag(E_theta),
                           np.real(E_phi), np.imag(E_phi)])
        
        # Solve least squares problem
        result = lsq_linear(A, b)
        coeffs = result.x
        
        # Extract Q coefficients
        Q1_coeffs = {}
        Q2_coeffs = {}
        
        for mode_idx, (n, m) in enumerate(modes):
            Q1_re = coeffs[4 * mode_idx]
            Q1_im = coeffs[4 * mode_idx + 1]
            Q2_re = coeffs[4 * mode_idx + 2]
            Q2_im = coeffs[4 * mode_idx + 3]
            
            Q1_coeffs[(n, m)] = Q1_re + 1j * Q1_im
            Q2_coeffs[(n, m)] = Q2_re + 1j * Q2_im
        
        return cls(Q1_coeffs, Q2_coeffs, frequency, NMAX, MMAX)
    
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
        Calculate near-field E and H components at arbitrary points.
        
        Uses full spherical wave expansion with Hankel functions.
        
        Args:
            r: Radial distance(s) in meters
            theta: Polar angle(s) in radians
            phi: Azimuthal angle(s) in radians
            
        Returns:
            E: Tuple of (E_r, E_theta, E_phi) components
            H: Tuple of (H_r, H_theta, H_phi) components
        """
        if self.k is None:
            raise ValueError("Frequency must be set before computing near field")
        
        r = np.atleast_1d(r)
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)
        
        if not (r.shape == theta.shape == phi.shape):
            r, theta, phi = np.broadcast_arrays(r, theta, phi)
        
        # Initialize field components
        E_r = np.zeros_like(r, dtype=complex)
        E_theta = np.zeros_like(theta, dtype=complex)
        E_phi = np.zeros_like(phi, dtype=complex)
        H_r = np.zeros_like(r, dtype=complex)
        H_theta = np.zeros_like(theta, dtype=complex)
        H_phi = np.zeros_like(phi, dtype=complex)
        
        kr = self.k * r
        
        all_modes = set(self.Q1_coeffs.keys()) | set(self.Q2_coeffs.keys())
        
        for (n, m) in all_modes:
            # Get pattern functions (theta/phi dependence)
            P_norm, dP_norm = normalized_associated_legendre(n, m, theta)
            
            # Radial functions
            # h_n^(3) = j_n - i*y_n (outgoing wave)
            j_n = spherical_jn(n, kr)
            y_n = spherical_yn(n, kr)
            h_n = j_n - 1j * y_n
            
            # Derivatives
            if n == 0:
                dh_n = -j_n + 1j * y_n  # d/d(kr)[h_0] = -h_1
            else:
                j_n_minus = spherical_jn(n-1, kr)
                y_n_minus = spherical_yn(n-1, kr)
                h_n_minus = j_n_minus - 1j * y_n_minus
                # d/d(kr)[h_n] = h_{n-1} - (n+1)/kr * h_n
                dh_n = h_n_minus - (n + 1) / kr * h_n
            
            # Azimuthal phase
            exp_imphi = np.exp(1j * m * phi)
            
            # Common factors
            prefactor = np.sqrt(2 / (n * (n + 1)))
            if m == 0:
                sign_factor = 1.0
            else:
                sign_factor = (-m / abs(m)) ** m
                        
            # Safe sin(theta)
            sin_theta = np.sin(theta)
            sin_theta_safe = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
            
            # Q1 contribution (TE to r, TM from r in Hansen notation)
            if (n, m) in self.Q1_coeffs:
                Q1 = self.Q1_coeffs[(n, m)]
                coef = prefactor * sign_factor * exp_imphi * Q1
                
                # E field from Q1 (proportional to M_mn in Hansen)
                # M has no radial component
                E_r += 0.0
                E_theta += coef * (1j * m / sin_theta_safe) * P_norm * h_n / kr
                E_phi += -coef * dP_norm * h_n / kr
                
                # H field from Q1 (curl of E / (i*omega*mu))
                # This is proportional to curl(M) / (ik*Z0)
                # Z0 = sqrt(mu/eps) = 377 ohms
                Z0 = 376.730313668
                factor_H = 1 / (1j * self.k * Z0)
                
                H_r += factor_H * coef * n * (n + 1) * P_norm * h_n / kr**2
                H_theta += factor_H * coef * dP_norm * (self.k * dh_n - h_n / r) / self.k
                H_phi += factor_H * coef * (1j * m / sin_theta_safe) * P_norm * (self.k * dh_n - h_n / r) / self.k
            
            # Q2 contribution (TM to r, TE from r)
            if (n, m) in self.Q2_coeffs:
                Q2 = self.Q2_coeffs[(n, m)]
                coef = prefactor * sign_factor * exp_imphi * Q2
                
                # E field from Q2 (proportional to N_mn)
                # N has all three components
                E_r += coef * n * (n + 1) * P_norm * h_n / kr**2
                E_theta += coef * dP_norm * (self.k * dh_n - h_n / r) / self.k
                E_phi += coef * (1j * m / sin_theta_safe) * P_norm * (self.k * dh_n - h_n / r) / self.k
                
                # H field from Q2 (curl of E / (i*omega*mu))
                Z0 = 376.730313668
                factor_H = 1 / (1j * self.k * Z0)
                
                H_r += 0.0
                H_theta += -factor_H * coef * (1j * m / sin_theta_safe) * P_norm * h_n / kr
                H_phi += factor_H * coef * dP_norm * h_n / kr
        
        # Apply overall normalization
        # The field should have units of V/m for E and A/m for H
        E_r *= self.k
        E_theta *= self.k
        E_phi *= self.k
        H_r *= self.k
        H_theta *= self.k
        H_phi *= self.k
        
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
    
    def surface_currents(self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray,
                        surface_normal: str = 'outward') -> \
            Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
                  Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calculate equivalent surface currents on a surface.
        
        The surface is defined by the (r, theta, phi) points. Surface currents are:
        J_s = n × H (electric surface current, A/m)
        M_s = -n × E (magnetic surface current, V/m)
        
        Args:
            r: Radial distance(s) to surface points in meters
            theta: Polar angle(s) in radians
            phi: Azimuthal angle(s) in radians
            surface_normal: Direction of normal ('outward' or 'inward')
            
        Returns:
            J: Tuple of (J_r, J_theta, J_phi) electric surface current
            M: Tuple of (M_r, M_theta, M_phi) magnetic surface current
        """
        # Get E and H fields at the surface
        (E_r, E_theta, E_phi), (H_r, H_theta, H_phi) = self.near_field(r, theta, phi)
        
        # For a spherical surface at radius r, the outward normal is n = r_hat
        # J_s = n × H = r_hat × H
        # M_s = -n × E = -r_hat × E
        
        if surface_normal == 'outward':
            sign = 1.0
        elif surface_normal == 'inward':
            sign = -1.0
        else:
            raise ValueError("surface_normal must be 'outward' or 'inward'")
        
        # J = r_hat × H
        # In spherical coordinates: r_hat × (H_r, H_θ, H_φ) = (0, H_φ, -H_θ)
        J_r = np.zeros_like(H_r)
        J_theta = sign * H_phi
        J_phi = -sign * H_theta
        
        # M = -r_hat × E
        M_r = np.zeros_like(E_r)
        M_theta = -sign * E_phi
        M_phi = sign * E_theta
        
        return (J_r, J_theta, J_phi), (M_r, M_theta, M_phi)
    
    def __repr__(self) -> str:
        freq_str = f"{self.frequency/1e9:.3f} GHz" if self.frequency else "unset"
        return (f"SphericalWaveExpansion(NMAX={self.NMAX}, MMAX={self.MMAX}, "
                f"frequency={freq_str}, modes={len(self.Q1_coeffs)})")


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

    # Fixed normalization factor from Ticra documentation
    NORMALIZATION_FACTOR = np.sqrt(8 * np.pi)

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
                            Q1_coeffs[(n, 0)] = float(coeff_parts[0]) + 1j * float(coeff_parts[1])
                            Q2_coeffs[(n, 0)] = float(coeff_parts[2]) + 1j * float(coeff_parts[3])
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
                            Q1_coeffs[(n, -abs_m)] = float(coeff_parts[0]) + 1j * float(coeff_parts[1])
                            Q2_coeffs[(n, -abs_m)] = float(coeff_parts[2]) + 1j * float(coeff_parts[3])
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
                            Q1_coeffs[(n, abs_m)] = float(coeff_parts[0]) + 1j * float(coeff_parts[1])
                            Q2_coeffs[(n, abs_m)] = float(coeff_parts[2]) + 1j * float(coeff_parts[3])
                        else:
                            line_idx -= 1
                            break
                            
            except (ValueError, IndexError):
                line_idx += 1
        else:
            line_idx += 1
    
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
    
    Applies inverse normalization to convert from Q_smn coefficients
    to Q'_smn file format according to TICRA spec equation (6):
    Q'_smn = (1/√8π) * Q*_smn
    
    Args:
        filename: Output file path
        Q1_coeffs: Q₁ coefficients dictionary
        Q2_coeffs: Q₂ coefficients dictionary
        frequency_GHz: Frequency in GHz
        NTHE: Number of theta samples
        NPHI: Number of phi samples
        NMAX: Maximum degree
        MMAX: Maximum order
        description: Description text
    """
    from datetime import datetime
    
    # Normalization factor from TICRA spec
    # Q'_smn = (1/√8π) * Q*_smn
    NORMALIZATION_FACTOR = np.sqrt(8 * np.pi)
    
    with open(filename, 'w') as f:
        # Record 1: PRGTAG
        timestamp = datetime.now().strftime("%Y/%m/%d at %H:%M:%S")
        f.write(f"File created by SWE module on {timestamp}, {description}, "
                f"Freq [GHz]: {frequency_GHz:15.9f}\n")
        
        # Record 2: IDSTRG
        f.write("SWE\n")
        
        # Record 3: Control data
        f.write(f"{NTHE:6d} {NPHI:6d} {NMAX:6d} {MMAX:6d}\n")
        
        # Record 4: Rotation angles
        f.write(f"Rotation angles (Theta, Phi, Chi)=({0.0:.5f}, {0.0:.5f}, {0.0:.5f})\n")
        
        # Records 5-6: Dummy data (5 real numbers each)
        f.write(f"  {'0.0000':>14s}  {'180.00':>14s}  {'0.0000':>14s}  {'359.99':>14s}  {'0.00000':>14s}\n")
        f.write(f"  {'0.0000':>14s}  {'180.00':>14s}  {'0.0000':>14s}  {'359.99':>14s}  {'0.00000':>14s}\n")
        
        # Records 7-8: Dummy file names
        f.write("SWEP_DUMMY_FILE_NAME\n")
        f.write("SWEP_DUMMY_FILE_NAME\n")
        
        # Write coefficients for each |m|
        for m_val in range(MMAX + 1):
            # Calculate power for this mode (sum of |Q'|^2 over all n)
            # Note: Power is calculated from Q' coefficients in the file
            power = 0.0
            for n in range(max(1, m_val), NMAX + 1):
                for m_sign in ([-m_val, m_val] if m_val > 0 else [0]):
                    if (n, m_sign) in Q1_coeffs:
                        # Convert to Q' for power calculation
                        Q1_prime = np.conj(Q1_coeffs[(n, m_sign)]) / NORMALIZATION_FACTOR
                        power += abs(Q1_prime)**2
                    if (n, m_sign) in Q2_coeffs:
                        Q2_prime = np.conj(Q2_coeffs[(n, m_sign)]) / NORMALIZATION_FACTOR
                        power += abs(Q2_prime)**2
            
            # Write mode header
            f.write(f"{m_val:6d}  {power:24.16E}\n")
            
            # Write coefficients
            for n in range(max(1, m_val), NMAX + 1):
                if m_val == 0:
                    # m=0 case: one line per n
                    # Apply inverse normalization: Q' = conj(Q) / sqrt(8π)
                    Q1 = Q1_coeffs.get((n, 0), 0.0 + 0.0j)
                    Q2 = Q2_coeffs.get((n, 0), 0.0 + 0.0j)
                    
                    f.write(f"  {Q1.real:23.16E} {Q1.imag:23.16E} "
                           f"{Q2.real:23.16E} {Q2.imag:23.16E}\n")
                else:
                    # -m line
                    Q1_neg = Q1_coeffs.get((n, -m_val), 0.0 + 0.0j)
                    Q2_neg = Q2_coeffs.get((n, -m_val), 0.0 + 0.0j)
                    
                    f.write(f"  {Q1_neg.real:23.16E} {Q1_neg.imag:23.16E} "
                           f"{Q2_neg.real:23.16E} {Q2_neg.imag:23.16E}\n")
                    
                    # +m line
                    Q1_pos = Q1_coeffs.get((n, m_val), 0.0 + 0.0j)
                    Q2_pos = Q2_coeffs.get((n, m_val), 0.0 + 0.0j)
                    
                    f.write(f"  {Q1_pos.real:23.16E} {Q1_pos.imag:23.16E} "
                           f"{Q2_pos.real:23.16E} {Q2_pos.imag:23.16E}\n")


# ==============================================================================
# Example Usage
# ==============================================================================

def verify_against_hansen_table():
    """
    Verify implementation against values from Hansen's table (page 324).
    """
    import numpy as np
    
    theta = np.pi / 3  # 60 degrees for easy checking
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    print("Verification against Hansen table (Appendix A1):")
    print(f"θ = {np.degrees(theta):.1f}°, cos(θ) = {cos_theta:.4f}, sin(θ) = {sin_theta:.4f}\n")
    
    # Test cases from Hansen tables
    # Page 322: Function values P̄^|m|_n(cos θ)  
    # Page 324: Derivatives dP̄^|m|_n/dθ
    test_cases = [
        # (n, m, expected_P, expected_dP_dtheta, description)
        (1, 0, np.sqrt(6)/2 * cos_theta, -np.sqrt(6)/2 * sin_theta, 
         "P̄_1^0 = √6/2 cos θ, dP̄_1^0/dθ = -√6/2 sin θ"),
        (1, 1, np.sqrt(3)/2 * sin_theta, np.sqrt(3)/2 * cos_theta,
         "P̄_1^1 = √3/2 sin θ, dP̄_1^1/dθ = √3/2 cos θ"),
        (2, 0, np.sqrt(10)/8 * (3*np.cos(2*theta) + 1), -3*np.sqrt(10)/4 * np.sin(2*theta),
         "P̄_2^0 = √10/8 (3cos 2θ + 1), dP̄_2^0/dθ = -3√10/4 sin 2θ"),
        (2, 1, np.sqrt(15)/4 * np.sin(2*theta), np.sqrt(15)/2 * np.cos(2*theta),
         "P̄_2^1 = √15/4 sin 2θ, dP̄_2^1/dθ = √15/2 cos 2θ"),
    ]
    
    print("Expected values at θ = 60° (from Hansen Appendix A1):")
    print()
    for n, m, exp_P, exp_dP, desc in test_cases:
        P_norm, dP_norm = normalized_associated_legendre(n, m, theta)
        
        print(f"{desc}")
        print(f"  P̄_{n}^{m}: computed={P_norm:.6f}, expected={exp_P:.6f}, " 
              f"error={abs(P_norm - exp_P):.2e}")
        print(f"  dP̄_{n}^{m}/dθ: computed={dP_norm:.6f}, expected={exp_dP:.6f}, "
              f"error={abs(dP_norm - exp_dP):.2e}")
        print()

if __name__ == "__main__":
    verify_against_hansen_table()