"""
Spherical Wave Expansion Package

A Python package for spherical wave expansion analysis of electromagnetic fields.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes and functions
from .core import (
    SphericalWaveExpansion,
    read_ticra_sph,
    write_ticra_sph,
    normalized_associated_legendre,
    far_field_pattern_functions,
    cartesian_to_spherical,
    spherical_to_cartesian,
    spherical_to_cartesian_field,
)

__all__ = [
    "SphericalWaveExpansion",
    "read_ticra_sph",
    "write_ticra_sph",
    "normalized_associated_legendre",
    "far_field_pattern_functions",
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    "spherical_to_cartesian_field",
]