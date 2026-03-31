"""
Test: Load .sph -> compute near field on planar grid -> compare with Ticra .grd reference.

This is the key test for identifying the known near-field scaling issue
between the SWE package and Ticra GRASP.
"""

import numpy as np
import pytest

from swe import SphericalWaveExpansion, cartesian_to_spherical
from swe.ticra_io import read_grasp_grd
from tests.conftest import SPH_FILE, GRD_FILE, requires_sph, requires_grd, compute_comparison_metrics


@requires_sph
@requires_grd
class TestSphToNearField:

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load the reference data."""
        self.swe = SphericalWaveExpansion.from_sph_file(SPH_FILE)
        self.grd_data = read_grasp_grd(GRD_FILE)

    def _get_grid_coordinates(self, field_idx=0):
        """
        Extract the planar grid coordinates from .grd data.

        The .grd file stores a planar near-field scan. The grid coordinates
        depend on igrid type. For a planar scan, the coordinates are typically
        in meters (x, y) at a fixed z distance.

        Returns:
            x, y: 2D arrays of grid coordinates (meters)
        """
        field = self.grd_data['fields'][field_idx]
        x = np.linspace(field['grid_min_x'], field['grid_max_x'], field['nx'])
        y = np.linspace(field['grid_min_y'], field['grid_max_y'], field['ny'])
        X, Y = np.meshgrid(x, y)
        return X, Y

    def test_grd_file_loads(self):
        """Verify .grd file loads with valid structure."""
        assert self.grd_data['nset'] > 0, "Should have at least one field set"
        assert len(self.grd_data['fields']) > 0, "Should have field data"

        field = self.grd_data['fields'][0]
        assert field['nx'] > 0 and field['ny'] > 0, "Grid should have points"

        print(f"\nLoaded .grd: nset={self.grd_data['nset']}, "
              f"icomp={self.grd_data['icomp']}, ncomp={self.grd_data['ncomp']}, "
              f"igrid={self.grd_data['igrid']}")
        print(f"  Grid: [{field['grid_min_x']:.4f}, {field['grid_min_y']:.4f}] -> "
              f"[{field['grid_max_x']:.4f}, {field['grid_max_y']:.4f}]")
        print(f"  Size: {field['nx']} x {field['ny']}")

    def test_near_field_vs_grd(self):
        """
        Compare near-field computation with .grd reference.

        NOTE: The z-distance of the planar scan must be known. The .grd file
        header may contain this info, or it must be specified. This test
        will attempt to detect it or use a reasonable default.
        """
        field = self.grd_data['fields'][0]
        X, Y = self._get_grid_coordinates(0)

        # The z-distance must be determined from the .grd header or known geometry.
        # Parse header for scan distance if available.
        z_distance = None
        for line in self.grd_data.get('header', []):
            # Try to find z-distance in header text
            lower = line.lower()
            if 'z_distance' in lower or 'scan_distance' in lower or 'z =' in lower:
                parts = line.split()
                for i, p in enumerate(parts):
                    try:
                        z_distance = float(p)
                        break
                    except ValueError:
                        continue

        if z_distance is None:
            # If igrid suggests a far-field grid (uv, theta-phi), handle differently
            if self.grd_data['igrid'] in [1, 7]:
                print("\n  Grid appears to be a far-field grid (igrid={}).".format(
                    self.grd_data['igrid']))
                print("  Comparing as far-field pattern instead of planar near-field.")
                self._compare_as_farfield_grid(field, X, Y)
                return

            print("\n  WARNING: z-distance not found in .grd header.")
            print("  Set z_distance in the test or .grd header to enable near-field comparison.")
            print("  Skipping quantitative comparison.")
            pytest.skip("z-distance for planar scan not determined")

        Z = np.full_like(X, z_distance)

        # Compute near field at the grid points
        (E_x, E_y, E_z), (H_x, H_y, H_z) = self.swe.near_field_cartesian(
            X.ravel(), Y.ravel(), Z.ravel()
        )

        E_x = E_x.reshape(X.shape)
        E_y = E_y.reshape(X.shape)

        # Reference data
        ref_E1 = field['data'][:, :, 0]
        ref_E2 = field['data'][:, :, 1]

        # Compare (assuming icomp=1: E_theta/E_phi or Ex/Ey depending on context)
        metrics_1 = compute_comparison_metrics(
            E_x, ref_E1, label="Near-field component 1 (Ex vs ref)"
        )
        metrics_2 = compute_comparison_metrics(
            E_y, ref_E2, label="Near-field component 2 (Ey vs ref)"
        )

        print(f"\n  If scaling_ratio != 1.0, there is a near-field normalization mismatch.")
        print(f"  Component 1 scale: {metrics_1['scaling_ratio']:.6f}")
        print(f"  Component 2 scale: {metrics_2['scaling_ratio']:.6f}")

    def _compare_as_farfield_grid(self, field, X, Y):
        """
        Compare SWE far-field with a .grd file that contains far-field data
        on a uv or theta-phi grid.
        """
        igrid = self.grd_data['igrid']

        if igrid == 1:
            # uv grid: u = sin(theta)*cos(phi), v = sin(theta)*sin(phi)
            u, v = X, Y
            r2 = u ** 2 + v ** 2
            # Only valid for r2 < 1 (forward hemisphere)
            valid = r2 < 1.0
            theta = np.arcsin(np.sqrt(np.clip(r2, 0, 1)))
            phi = np.arctan2(v, u)
            phi = np.where(phi < 0, phi + 2 * np.pi, phi)
        elif igrid == 7:
            # theta-phi grid: X=phi (rad), Y=theta (rad)
            phi = X
            theta = Y
            valid = np.ones_like(theta, dtype=bool)
        else:
            print(f"  Unsupported igrid={igrid} for far-field comparison")
            return

        theta_flat = theta[valid].ravel()
        phi_flat = phi[valid].ravel()

        E_theta, E_phi = self.swe.far_field(theta_flat, phi_flat)

        ref_E1 = field['data'][:, :, 0][valid].ravel()
        ref_E2 = field['data'][:, :, 1][valid].ravel()

        compute_comparison_metrics(E_theta, ref_E1, label="Far-field grid E1 (E_theta)")
        compute_comparison_metrics(E_phi, ref_E2, label="Far-field grid E2 (E_phi)")

    def test_near_field_not_zero(self):
        """Sanity check: near field should not be identically zero."""
        # Compute at a single test point
        r, theta, phi = 1.0, np.pi / 4, 0.0
        if self.swe.frequency is not None:
            (E_r, E_theta, E_phi), _ = self.swe.near_field(
                np.array([r]), np.array([theta]), np.array([phi])
            )
            total = np.abs(E_r) + np.abs(E_theta) + np.abs(E_phi)
            assert total[0] > 0, "Near field is identically zero at test point"
