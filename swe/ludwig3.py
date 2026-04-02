"""
Ludwig-3 Polarization Utilities

Conversion functions between spherical (E_theta, E_phi) and Ludwig-3 (Eco, Ecx)
polarization components, plus helpers for GRASP .cut file multi-frequency and
negative-theta handling.

Ludwig-3 definition (GRASP convention):
    Eco = E_theta * cos(phi) - E_phi * sin(phi)
    Ecx = E_theta * sin(phi) + E_phi * cos(phi)

References:
    - A.C. Ludwig, "The definition of cross polarization," IEEE Trans. AP, 1973
    - TICRA GRASP Technical Description
"""

import numpy as np
from typing import Dict, List, Tuple


def spherical_to_ludwig3(E_theta: np.ndarray, E_phi: np.ndarray,
                         phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert spherical field components to Ludwig-3 polarization.

    Args:
        E_theta: Theta component of electric field (complex)
        E_phi: Phi component of electric field (complex)
        phi: Azimuthal angle in radians

    Returns:
        Eco: Co-polar component (Ludwig-3)
        Ecx: Cross-polar component (Ludwig-3)
    """
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    Eco = E_theta * cos_phi - E_phi * sin_phi
    Ecx = E_theta * sin_phi + E_phi * cos_phi
    return Eco, Ecx


def ludwig3_to_spherical(Eco: np.ndarray, Ecx: np.ndarray,
                         phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Ludwig-3 polarization to spherical field components.

    Args:
        Eco: Co-polar component (Ludwig-3, complex)
        Ecx: Cross-polar component (Ludwig-3, complex)
        phi: Azimuthal angle in radians

    Returns:
        E_theta: Theta component of electric field
        E_phi: Phi component of electric field
    """
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    E_theta = Eco * cos_phi + Ecx * sin_phi
    E_phi = -Eco * sin_phi + Ecx * cos_phi
    return E_theta, E_phi


def remap_negative_theta(theta_deg: np.ndarray,
                         phi_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remap negative theta values from GRASP .cut files.

    In GRASP .cut files, theta ranges from -180 to +180 degrees. Negative
    theta is equivalent to positive theta with phi shifted by 180 degrees.

    Args:
        theta_deg: Theta values in degrees (may include negatives)
        phi_deg: Constant phi value for this cut (degrees)

    Returns:
        theta_rad: Remapped theta in radians (all >= 0)
        phi_rad: Remapped phi in radians (shifted by pi where theta was negative)
    """
    theta_deg = np.asarray(theta_deg, dtype=float)
    negative_mask = theta_deg < 0

    theta_remapped = np.abs(theta_deg)
    phi_remapped = np.full_like(theta_deg, phi_deg)
    phi_remapped[negative_mask] += 180.0

    return np.deg2rad(theta_remapped), np.deg2rad(phi_remapped)


def extract_cut_frequency_set(cut_data: Dict, freq_index: int,
                              n_phi: int) -> Dict:
    """
    Extract cuts for a single frequency from a multi-frequency .cut file.

    GRASP .cut files with multiple frequencies store n_phi cuts per frequency,
    with frequencies in sequential blocks.

    Args:
        cut_data: Dictionary returned by read_grasp_cut()
        freq_index: 0-based frequency index (0 = first frequency)
        n_phi: Number of phi cuts per frequency

    Returns:
        Dictionary with 'cuts' key containing only the selected frequency's cuts
    """
    start = freq_index * n_phi
    end = start + n_phi
    return {'cuts': cut_data['cuts'][start:end]}
