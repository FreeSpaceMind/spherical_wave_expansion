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

import numpy as np
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


def normalized_associated_legendre(n: int, m: int, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized associated Legendre function and its derivative.
    
    Based on equation A1.25 from Hansen's documentation:
    P̄ₙᵐ(cos θ) = sqrt((2n+1)/2 * (n-m)!/(n+m)!) * Pₙᵐ(cos θ)
    
    Args:
        n: Degree (n >= 1)
        m: Order (-n <= m <= n)
        theta: Polar angle(s) in radians
        
    Returns:
        P_norm: Normalized associated Legendre function values
        dP_norm: Derivative with respect to theta
    """
    abs_m = abs(m)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Compute unnormalized associated Legendre function
    P_unnorm = lpmv(abs_m, n, cos_theta)
    
    # Normalization factor
    norm_factor = np.sqrt((2*n + 1) / 2 * 
                         np.math.factorial(n - abs_m) / 
                         np.math.factorial(n + abs_m))
    
    P_norm = norm_factor * P_unnorm
    
    # Compute derivative using recurrence relations
    if abs_m == 0:
        if n == 1:
            dP_norm = -norm_factor * np.sqrt((n*(n+1))/2) * sin_theta
        else:
            P_n_minus_1 = lpmv(0, n-1, cos_theta) * np.sqrt((2*(n-1) + 1) / 2)
            dP_norm = -(1/(2*n + 1)) * ((n-abs_m+1)*(n+abs_m) * P_n_minus_1 - 
                                         (n+1) * cos_theta * P_norm) / sin_theta
    else:
        if n > abs_m:
            P_n_minus_1 = lpmv(abs_m, n-1, cos_theta) * np.sqrt((2*(n-1) + 1) / 2 * 
                              np.math.factorial(n-1-abs_m) / 
                              np.math.factorial(n-1+abs_m))
            dP_norm = 0.5 * ((n - abs_m + 1) * (n + abs_m) * P_n_minus_1 - 
                            (n + 1) * cos_theta * P_norm) / sin_theta
        else:
            dP_norm = -abs_m * cos_theta * P_norm / sin_theta
    
    return P_norm, dP_norm


def far_field_pattern_functions(n: int, m: int, theta: np.ndarray, phi: np.ndarray) -> \
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Compute far-field pattern functions K̄₁ₘₙ and K̄₂ₘₙ.
    
    Based on equations A1.59 and A1.60 from Hansen.
    
    Args:
        n: Degree (n >= 1)
        m: Order (-n <= m <= n)
        theta: Polar angle(s) in radians
        phi: Azimuthal angle(s) in radians
        
    Returns:
        K1: Tuple of (K1_theta, K1_phi) components
        K2: Tuple of (K2_theta, K2_phi) components
    """
    abs_m = abs(m)
    
    # Get normalized Legendre function and derivative
    P_norm, dP_norm_dtheta = normalized_associated_legendre(n, m, theta)
    
    # Common factors
    prefactor = np.sqrt(2 / (n * (n + 1)))
    
    # Sign factor: (-m/|m|)^m
    if m == 0:
        sign_factor = 1.0
    else:
        sign_factor = (-m / abs_m) ** abs_m
    
    # Azimuthal phase
    phase = np.exp(1j * m * phi)
    
    # Powers of -i
    i_factor_1 = (-1j) ** (n + 1)
    i_factor_2 = (-1j) ** n
    
    # Handle division by sin(theta) carefully
    sin_theta = np.sin(theta)
    sin_theta_safe = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
    
    # K̄₁ₘₙ components
    K1_theta = prefactor * sign_factor * phase * i_factor_1 * \
               (1j * m * P_norm / sin_theta_safe)
    K1_phi = prefactor * sign_factor * phase * i_factor_1 * \
             (-dP_norm_dtheta)
    
    # K̄₂ₘₙ components
    K2_theta = prefactor * sign_factor * phase * i_factor_2 * \
               dP_norm_dtheta
    K2_phi = prefactor * sign_factor * phase * i_factor_2 * \
             (1j * m * P_norm / sin_theta_safe)
    
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
        
        # Apply overall factor k (assuming η=1)
        E_theta *= self.k
        E_phi *= self.k
        
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
            sign_factor = 1.0 if m == 0 else (-m / abs(m)) ** abs(m)
            
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
                sign_factor = (-m / abs(m)) ** abs(m)
            
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
            # Calculate power for this mode (sum of |Q|^2 over all n)
            power = 0.0
            for n in range(max(1, m_val), NMAX + 1):
                for m_sign in ([-m_val, m_val] if m_val > 0 else [0]):
                    if (n, m_sign) in Q1_coeffs:
                        power += abs(Q1_coeffs[(n, m_sign)])**2
                    if (n, m_sign) in Q2_coeffs:
                        power += abs(Q2_coeffs[(n, m_sign)])**2
            
            # Write mode header
            f.write(f"{m_val:6d}  {power:24.16E}\n")
            
            # Write coefficients
            for n in range(max(1, m_val), NMAX + 1):
                if m_val == 0:
                    # m=0 case: one line per n
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

if __name__ == "__main__":
    
    print("=== Example 1: Create SWE from coefficients ===")
    # Create simple dipole-like pattern
    Q1_coeffs = {
        (1, -1): 0.5 + 0.1j,
        (1, 0): 1.0,
        (1, 1): 0.5 - 0.1j,
    }
    Q2_coeffs = {
        (1, -1): 0.1j,
        (1, 0): 0.0,
        (1, 1): -0.1j,
    }
    
    swe = SphericalWaveExpansion(Q1_coeffs, Q2_coeffs, frequency=10e9)
    print(swe)
    
    # Calculate far field
    theta = np.linspace(0, np.pi, 91)
    phi = np.zeros_like(theta)
    E_theta, E_phi = swe.far_field(theta, phi)
    print(f"Field calculated at {len(theta)} points")
    print(f"Max |E_theta|: {np.max(np.abs(E_theta)):.3e}")
    
    print("\n=== Example 2: Read from .sph file ===")
    # swe_from_file = SphericalWaveExpansion.from_sph_file("output_truncated.sph")
    # print(swe_from_file)
    
    print("\n=== Example 3: Fit SWE from far-field data ===")
    # Generate synthetic far-field data
    theta_meas = np.random.uniform(0, np.pi, 100)
    phi_meas = np.random.uniform(0, 2*np.pi, 100)
    E_theta_meas, E_phi_meas = swe.far_field(theta_meas, phi_meas)
    
    # Fit new SWE
    swe_fitted = SphericalWaveExpansion.from_far_field(
        theta_meas, phi_meas, E_theta_meas, E_phi_meas,
        frequency=10e9, NMAX=1
    )
    print(swe_fitted)
    print("Original Q1(1,0):", Q1_coeffs[(1, 0)])
    print("Fitted Q1(1,0):  ", swe_fitted.Q1_coeffs[(1, 0)])
    
    print("\n=== Example 4: Near-field calculation ===")
    # Calculate E and H fields at near-field points
    r_near = np.array([0.1, 0.2, 0.5])  # meters
    theta_near = np.array([np.pi/4, np.pi/2, 3*np.pi/4])
    phi_near = np.array([0.0, np.pi/2, np.pi])
    
    (E_r, E_theta_nf, E_phi_nf), (H_r, H_theta_nf, H_phi_nf) = swe.near_field(
        r_near, theta_near, phi_near
    )
    print(f"Near-field calculated at {len(r_near)} points")
    print(f"E_r at r={r_near[0]:.2f}m: {E_r[0]:.3e} V/m")
    print(f"H_theta at r={r_near[1]:.2f}m: {H_theta_nf[1]:.3e} A/m")
    
    print("\n=== Example 5: Surface currents ===")
    # Calculate equivalent surface currents on a sphere
    r_surface = 0.3  # meters
    theta_surf = np.linspace(0, np.pi, 37)
    phi_surf = np.linspace(0, 2*np.pi, 73)
    theta_grid, phi_grid = np.meshgrid(theta_surf, phi_surf, indexing='ij')
    r_grid = np.full_like(theta_grid, r_surface)
    
    (J_r, J_theta, J_phi), (M_r, M_theta, M_phi) = swe.surface_currents(
        r_grid, theta_grid, phi_grid, surface_normal='outward'
    )
    print(f"Surface currents calculated on {theta_grid.size} surface points")
    print(f"Max |J_theta|: {np.max(np.abs(J_theta)):.3e} A/m")
    print(f"Max |M_phi|: {np.max(np.abs(M_phi)):.3e} V/m")
    print(f"J_r should be zero: max|J_r| = {np.max(np.abs(J_r)):.3e}")
    
    print("\n=== Example 6: Write to file ===")
    # swe.to_sph_file("output_example.sph", description="Example SWE output")
    
    print("\n=== Example 7: Cartesian coordinates ===")
    # Calculate fields at Cartesian grid points
    x_pts = np.array([0.1, 0.0, 0.1])
    y_pts = np.array([0.0, 0.1, 0.1])
    z_pts = np.array([0.2, 0.2, 0.0])
    
    (E_x, E_y, E_z), (H_x, H_y, H_z) = swe.near_field_cartesian(x_pts, y_pts, z_pts)
    print(f"Fields in Cartesian basis at {len(x_pts)} points")
    print(f"E at (x,y,z) = ({x_pts[0]:.1f}, {y_pts[0]:.1f}, {z_pts[0]:.1f}):")
    print(f"  E_x = {E_x[0]:.3e} V/m")
    print(f"  E_y = {E_y[0]:.3e} V/m")
    print(f"  E_z = {E_z[0]:.3e} V/m")
    
    print("\n=== Example 8: Create SWE from spherical near-field measurements ===")
    # Generate synthetic near-field data on a sphere
    r_meas = 0.5  # meters
    theta_nf_meas = np.random.uniform(0, np.pi, 150)
    phi_nf_meas = np.random.uniform(0, 2*np.pi, 150)
    
    # Get "measured" fields from original SWE
    (_, E_theta_nf_meas, E_phi_nf_meas), (_, H_theta_nf_meas, H_phi_nf_meas) = swe.near_field(
        np.full_like(theta_nf_meas, r_meas), theta_nf_meas, phi_nf_meas
    )
    
    # Fit new SWE from near-field measurements
    swe_from_nf = SphericalWaveExpansion.from_spherical_near_field(
        r_meas, theta_nf_meas, phi_nf_meas,
        E_theta_nf_meas, E_phi_nf_meas,
        frequency=10e9, NMAX=1
    )
    print(f"SWE fitted from spherical near-field: {swe_from_nf}")
    print(f"Comparison of Q1(1,0):")
    print(f"  Original: {Q1_coeffs[(1, 0)]:.6f}")
    print(f"  Fitted:   {swe_from_nf.Q1_coeffs[(1, 0)]:.6f}")
    
    print("\n=== Example 9: Create SWE from planar near-field measurements ===")
    # Simulate planar scan at z = 0.3m
    x_scan = np.linspace(-0.3, 0.3, 31)
    y_scan = np.linspace(-0.3, 0.3, 31)
    X_scan, Y_scan = np.meshgrid(x_scan, y_scan)
    x_flat = X_scan.flatten()
    y_flat = Y_scan.flatten()
    z_scan_plane = 0.3
    
    # Get "measured" fields in Cartesian coordinates
    (E_x_meas, E_y_meas, E_z_meas), _ = swe.near_field_cartesian(
        x_flat, y_flat, np.full_like(x_flat, z_scan_plane)
    )
    
    # Fit SWE from planar measurements (using tangential components)
    swe_from_planar = SphericalWaveExpansion.from_planar_near_field(
        x_flat, y_flat, z_scan_plane,
        E_x_meas, E_y_meas,
        frequency=10e9, NMAX=1,
        origin_offset=(0., 0., 0.)
    )
    print(f"SWE fitted from planar near-field: {swe_from_planar}")
    print(f"Comparison of Q1(1,0):")
    print(f"  Original: {Q1_coeffs[(1, 0)]:.6f}")
    print(f"  Fitted:   {swe_from_planar.Q1_coeffs[(1, 0)]:.6f}")
    
    print("\n=== Example 10: Create SWE from cylindrical near-field measurements ===")
    # Simulate cylindrical scan at rho = 0.4m
    phi_cyl_scan = np.linspace(0, 2*np.pi, 72)
    z_cyl_scan = np.linspace(-0.3, 0.3, 31)
    PHI_cyl, Z_cyl = np.meshgrid(phi_cyl_scan, z_cyl_scan)
    phi_flat = PHI_cyl.flatten()
    z_flat = Z_cyl.flatten()
    rho_scan = 0.4
    
    # Convert to Cartesian to get fields
    x_cyl = rho_scan * np.cos(phi_flat)
    y_cyl = rho_scan * np.sin(phi_flat)
    (E_x_cyl, E_y_cyl, E_z_cyl), _ = swe.near_field_cartesian(x_cyl, y_cyl, z_flat)
    
    # Convert to cylindrical components
    E_rho_meas = E_x_cyl * np.cos(phi_flat) + E_y_cyl * np.sin(phi_flat)
    # E_phi_meas = -E_x_cyl * np.sin(phi_flat) + E_y_cyl * np.cos(phi_flat)  # Not used
    E_z_meas = E_z_cyl
    
    # Fit SWE from cylindrical measurements
    swe_from_cyl = SphericalWaveExpansion.from_cylindrical_near_field(
        np.full_like(phi_flat, rho_scan), phi_flat, z_flat,
        E_rho_meas, E_z_meas,
        frequency=10e9, NMAX=1,
        origin_offset=(0., 0., 0.)
    )
    print(f"SWE fitted from cylindrical near-field: {swe_from_cyl}")
    print(f"Comparison of Q1(1,0):")
    print(f"  Original: {Q1_coeffs[(1, 0)]:.6f}")
    print(f"  Fitted:   {swe_from_cyl.Q1_coeffs[(1, 0)]:.6f}")