"""
Test: Load .sph -> compute near field on planar grid -> compare with Ticra .grd reference.

Validates absolute near-field levels (V/m) between the SWE package and Ticra GRASP,
using Ludwig-3 (Eco/Ecx) polarization at z=0.25m planar scan.
"""

import numpy as np
import pytest

from swe import SphericalWaveExpansion, cartesian_to_spherical
from swe.ticra_io import read_grasp_grd
from swe.ludwig3 import spherical_to_ludwig3
from tests.conftest import (
    SPH_FILE, GRD_FILE, requires_sph, requires_grd,
    compute_comparison_metrics, Z_DISTANCE, FREQ_INDEX_8GHZ
)


@requires_sph
@requires_grd
class TestSphToNearField:

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load reference data with absolute scaling preserved."""
        swe_full = SphericalWaveExpansion.from_sph_file(SPH_FILE, normalize=False)
        self.grd_data = read_grasp_grd(GRD_FILE)
        self.field = self.grd_data['fields'][FREQ_INDEX_8GHZ]

        # Use the first frequency (8 GHz) — matches FREQ_INDEX_8GHZ in the .grd
        self.freq = swe_full.frequencies[FREQ_INDEX_8GHZ]

        # Truncate modes for near-field safety: n must be < kr_min to avoid
        # overflow in spherical Bessel functions. kr_min corresponds to the
        # closest grid point (smallest r).
        r_min = Z_DISTANCE  # on-axis point has smallest r
        kr_min = swe_full.k(self.freq) * r_min
        nmax_safe = int(kr_min) - 5  # conservative margin
        nmax_safe = max(nmax_safe, swe_full.MMAX(self.freq))

        q1_all = swe_full.Q1_coeffs(self.freq)
        q2_all = swe_full.Q2_coeffs(self.freq)
        Q1_trunc = {(n, m): v for (n, m), v in q1_all.items() if n <= nmax_safe}
        Q2_trunc = {(n, m): v for (n, m), v in q2_all.items() if n <= nmax_safe}

        self.swe = SphericalWaveExpansion(
            Q1_coeffs={self.freq: Q1_trunc},
            Q2_coeffs={self.freq: Q2_trunc},
            NMAX={self.freq: nmax_safe},
            MMAX={self.freq: swe_full.MMAX(self.freq)},
        )
        print(f"\n  Truncated NMAX: {swe_full.NMAX(self.freq)} -> {nmax_safe} "
              f"(kr_min={kr_min:.1f})")

    def _get_grid(self):
        """Build the planar grid at z=Z_DISTANCE from .grd extents."""
        x = np.linspace(self.field['grid_min_x'], self.field['grid_max_x'],
                        self.field['nx'])
        y = np.linspace(self.field['grid_min_y'], self.field['grid_max_y'],
                        self.field['ny'])
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, Z_DISTANCE)
        return X, Y, Z

    def test_grd_file_loads(self):
        """Verify .grd file loads with valid structure."""
        assert self.grd_data['nset'] > 0, "Should have at least one field set"
        assert len(self.grd_data['fields']) > 0, "Should have field data"

        print(f"\nLoaded .grd: nset={self.grd_data['nset']}, "
              f"icomp={self.grd_data['icomp']}, ncomp={self.grd_data['ncomp']}, "
              f"igrid={self.grd_data['igrid']}")
        print(f"  Grid: [{self.field['grid_min_x']:.4f}, {self.field['grid_min_y']:.4f}] -> "
              f"[{self.field['grid_max_x']:.4f}, {self.field['grid_max_y']:.4f}]")
        print(f"  Size: {self.field['nx']} x {self.field['ny']}")

    def test_near_field_absolute_vs_grd(self):
        """
        Compare absolute near-field with .grd reference at z=0.25m.

        Computes near field in spherical coordinates, converts to Ludwig-3,
        and compares with .grd data (icomp=3: Eco, Ecx).
        """
        X, Y, Z = self._get_grid()

        r, theta, phi = cartesian_to_spherical(X.ravel(), Y.ravel(), Z.ravel())

        print(f"\n  Computing near field at {len(r)} points, z={Z_DISTANCE}m...")
        print(f"  SWE: NMAX={self.swe.NMAX(self.freq)}, MMAX={self.swe.MMAX(self.freq)}, "
              f"freq={self.freq/1e9:.4f} GHz")

        (E_r, E_theta, E_phi), _ = self.swe.near_field(
            r, theta, phi, frequency=self.freq, normalize=False
        )

        Eco, Ecx = spherical_to_ludwig3(E_theta, E_phi, phi)
        Eco = Eco.reshape(X.shape)
        Ecx = Ecx.reshape(X.shape)

        # Reference data from .grd (icomp=3: component 0=Eco, component 1=Ecx)
        ref_Eco = self.field['data'][:, :, 0]
        ref_Ecx = self.field['data'][:, :, 1]

        metrics_co = compute_comparison_metrics(
            Eco, ref_Eco, label="Near-field Eco: SWE vs .grd (absolute)"
        )
        metrics_cx = compute_comparison_metrics(
            Ecx, ref_Ecx, label="Near-field Ecx: SWE vs .grd (absolute)"
        )

        print(f"\n  Eco scaling ratio: {metrics_co['scaling_ratio']:.6f} (expect ~1.0)")
        print(f"  Ecx scaling ratio: {metrics_cx['scaling_ratio']:.6f} (expect ~1.0)")
        print(f"  Eco normalized RMS: {metrics_co['normalized_rms_after_scaling']:.6e}")
        print(f"  Ecx normalized RMS: {metrics_cx['normalized_rms_after_scaling']:.6e}")

    def test_near_field_not_zero(self):
        """Sanity check: near field should not be identically zero."""
        (E_r, E_theta, E_phi), _ = self.swe.near_field(
            np.array([1.0]), np.array([np.pi / 4]), np.array([0.0]),
            frequency=self.freq, normalize=False
        )
        total = np.abs(E_r) + np.abs(E_theta) + np.abs(E_phi)
        assert total[0] > 0, "Near field is identically zero at test point"
