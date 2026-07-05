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
    read_ticra_sph_blocks,
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

from .ludwig3 import (
    spherical_to_ludwig3,
    ludwig3_to_spherical,
    remap_negative_theta,
    extract_cut_frequency_set,
)

__all__ = [
    "SphericalWaveExpansion",
    "read_ticra_sph",
    "read_ticra_sph_blocks",
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
<<<<<<< HEAD
]
=======
    "spherical_to_ludwig3",
    "ludwig3_to_spherical",
    "remap_negative_theta",
    "extract_cut_frequency_set",
]
>>>>>>> 1f18d7cb49b3732a462bd44c50d28f77172b4a52
