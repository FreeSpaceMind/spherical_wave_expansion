"""
Spherical Wave Expansion Package

A Python package for spherical wave expansion analysis of electromagnetic fields.
"""

__version__ = "0.1.0"
__author__ = "Justin Long"
__email__ = "justinwlong1@gmail.com"

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

from .ticra_io import (
    read_grasp_cut,
    write_grasp_cut,
    cut_to_fields,
    read_grasp_grd,
    write_grasp_grd,
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
    "read_grasp_cut",
    "write_grasp_cut",
    "cut_to_fields",
    "read_grasp_grd",
    "write_grasp_grd",
]