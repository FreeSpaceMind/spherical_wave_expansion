"""
Shared fixtures and utilities for SWE test suite.

Test data files (.sph, .cut, .grd) should be placed in tests/.
Tests that require these files are skipped if the files are not present.
"""

import os
import numpy as np
import pytest

# Path to test data directory (files live directly in tests/)
TEST_DATA_DIR = os.path.dirname(__file__)


def get_test_file(filename):
    """Return full path to a test data file, or None if it doesn't exist."""
    path = os.path.join(TEST_DATA_DIR, filename)
    return path if os.path.exists(path) else None


# Discover available test data files
# Discover available test data files
SPH_FILE = get_test_file("example.sph")
CUT_FILE = get_test_file("example.cut")
GRD_FILE = get_test_file("example.grd")

requires_sph = pytest.mark.skipif(SPH_FILE is None, reason="example.sph not found in tests/")
requires_cut = pytest.mark.skipif(CUT_FILE is None, reason="example.cut not found in tests/")
requires_grd = pytest.mark.skipif(GRD_FILE is None, reason="example.grd not found in tests/")

# Shared constants for multi-frequency test data
Z_DISTANCE = 0.25          # meters, z-distance for .grd planar scan
N_PHI_PER_FREQ = 37        # phi cuts per frequency in .cut file
FREQ_INDEX_8GHZ = 0        # index of 8 GHz in multi-freq .cut/.grd


# ==============================================================================
# Comparison utilities
# ==============================================================================

def compute_comparison_metrics(computed, reference, label=""):
    """
    Compute and print comparison metrics between computed and reference fields.

    Args:
        computed: complex array of computed field values
        reference: complex array of reference field values
        label: descriptive label for printout

    Returns:
        dict with keys: rms_error, max_error, rms_relative_error,
                        scaling_ratio, phase_offset_deg
    """
    computed = np.asarray(computed).ravel()
    reference = np.asarray(reference).ravel()

    # Absolute errors
    diff = computed - reference
    rms_error = np.sqrt(np.mean(np.abs(diff) ** 2))
    max_error = np.max(np.abs(diff))

    # Relative error (normalized to reference)
    ref_mag = np.abs(reference)
    mask = ref_mag > np.max(ref_mag) * 1e-3  # only compare significant values
    if np.any(mask):
        rms_relative_error = np.sqrt(np.mean(np.abs(diff[mask] / reference[mask]) ** 2))
    else:
        rms_relative_error = np.inf

    # Scaling ratio: best-fit complex scale factor (least-squares)
    # computed ≈ scale * reference
    if np.any(mask):
        scale = np.sum(computed[mask] * np.conj(reference[mask])) / np.sum(np.abs(reference[mask]) ** 2)
        scaling_ratio = np.abs(scale)
        phase_offset_deg = np.angle(scale, deg=True)
    else:
        scaling_ratio = np.nan
        phase_offset_deg = np.nan

    # Normalized RMS error after removing best-fit scale
    if np.any(mask) and not np.isnan(scaling_ratio):
        scaled_diff = computed - scale * reference
        normalized_rms = np.sqrt(np.mean(np.abs(scaled_diff[mask]) ** 2)) / np.sqrt(
            np.mean(np.abs(reference[mask]) ** 2))
    else:
        normalized_rms = np.inf

    metrics = {
        'rms_error': rms_error,
        'max_error': max_error,
        'rms_relative_error': rms_relative_error,
        'scaling_ratio': scaling_ratio,
        'phase_offset_deg': phase_offset_deg,
        'normalized_rms_after_scaling': normalized_rms,
    }

    if label:
        print(f"\n--- {label} ---")
        print(f"  RMS error:              {rms_error:.6e}")
        print(f"  Max error:              {max_error:.6e}")
        print(f"  RMS relative error:     {rms_relative_error:.6e}")
        print(f"  Best-fit scale ratio:   {scaling_ratio:.6f}")
        print(f"  Best-fit phase offset:  {phase_offset_deg:.4f} deg")
        print(f"  Normalized RMS (post-scale): {normalized_rms:.6e}")

    return metrics
