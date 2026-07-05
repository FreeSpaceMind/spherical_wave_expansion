"""
Test: Load .sph -> compute near field on planar grid -> compare with Ticra .grd reference.

Validates absolute near-field levels (V/m) between the SWE package and Ticra GRASP,
using Ludwig-3 (Eco/Ecx) polarization at z=0.25m planar scan.
Loops over all frequencies present in the .sph/.grd files.
"""

import numpy as np
import pytest

from swe import SphericalWaveExpansion, cartesian_to_spherical
from swe.ticra_io import read_grasp_grd
from swe.ludwig3 import spherical_to_ludwig3
from tests.conftest import (
    SPH_FILE, GRD_FILE, requires_sph, requires_grd,
    compute_comparison_metrics, Z_DISTANCE,
)


@requires_sph
@requires_grd
class TestSphToNearField:

    @pytest.fixture(autouse=True)
    def setup(self):
        """
        Load reference data for all frequencies and build per-frequency
        mode-truncated SWE objects.

        Modes with n >= kr_min are dropped to prevent Bessel function overflow;
        the safe limit is computed per-frequency since k differs.
        """
        swe_full = SphericalWaveExpansion.from_sph_file(SPH_FILE, normalize=False)
        grd_data = read_grasp_grd(GRD_FILE)

        n_freqs = len(swe_full.frequencies)
        assert len(grd_data['fields']) >= n_freqs, (
            f".grd has {len(grd_data['fields'])} field sets but .sph has {n_freqs} freqs"
        )

        # For each frequency build a (truncated_swe, grd_field) pair
        self.freq_field_pairs = []
        for i, freq in enumerate(swe_full.frequencies):
            field = grd_data['fields'][i]

            r_min = Z_DISTANCE
            kr_min = swe_full.k(freq) * r_min
            nmax_safe = max(int(kr_min) - 5, swe_full.MMAX(freq))

            q1_trunc = {k: v for k, v in swe_full.Q1_coeffs(freq).items()
                        if k[0] <= nmax_safe}
            q2_trunc = {k: v for k, v in swe_full.Q2_coeffs(freq).items()
                        if k[0] <= nmax_safe}

            swe_trunc = SphericalWaveExpansion(
                Q1_coeffs={freq: q1_trunc},
                Q2_coeffs={freq: q2_trunc},
                NMAX={freq: nmax_safe},
                MMAX={freq: swe_full.MMAX(freq)},
            )
            self.freq_field_pairs.append((freq, swe_trunc, field))
            print(f"  {freq/1e9:.4f} GHz — NMAX {swe_full.NMAX(freq)} -> {nmax_safe} "
                  f"(kr_min={kr_min:.1f})")

        # Keep grd metadata for the structural test
        self.grd_data = grd_data

    def _build_grid(self, field):
        """Build the planar grid at Z_DISTANCE from .grd extents."""
        x = np.linspace(field['grid_min_x'], field['grid_max_x'], field['nx'])
        y = np.linspace(field['grid_min_y'], field['grid_max_y'], field['ny'])
        X, Y = np.meshgrid(x, y)
        return X, Y, np.full_like(X, Z_DISTANCE)

    def test_grd_file_loads(self):
        """Verify .grd file structure."""
        assert self.grd_data['nset'] > 0
        assert len(self.grd_data['fields']) > 0
        print(f"\nLoaded .grd: nset={self.grd_data['nset']}, "
              f"icomp={self.grd_data['icomp']}, ncomp={self.grd_data['ncomp']}, "
              f"igrid={self.grd_data['igrid']}")
        field = self.grd_data['fields'][0]
        print(f"  Grid (set 0): [{field['grid_min_x']:.4f}, {field['grid_min_y']:.4f}] -> "
              f"[{field['grid_max_x']:.4f}, {field['grid_max_y']:.4f}], "
              f"size={field['nx']}x{field['ny']}")

    def test_near_field_absolute_vs_grd(self):
        """
        Compare absolute near-field against .grd at z=0.25m for all frequencies.

        Converts computed E_theta/E_phi to Ludwig-3 (Eco, Ecx) and compares
        with .grd data (icomp=3: component 0=Eco, 1=Ecx).
        """
        for freq, swe, field in self.freq_field_pairs:
            X, Y, Z = self._build_grid(field)
            r, theta, phi = cartesian_to_spherical(X.ravel(), Y.ravel(), Z.ravel())

            (_, E_theta, E_phi), _ = swe.near_field(
                r, theta, phi, frequency=freq, normalize=False
            )

            Eco = spherical_to_ludwig3(E_theta, E_phi, phi)[0].reshape(X.shape)
            Ecx = spherical_to_ludwig3(E_theta, E_phi, phi)[1].reshape(X.shape)

            met_co = compute_comparison_metrics(
                Eco, field['data'][:, :, 0],
                label=f"Near-field Eco {freq/1e9:.4f} GHz: SWE vs .grd"
            )
            met_cx = compute_comparison_metrics(
                Ecx, field['data'][:, :, 1],
                label=f"Near-field Ecx {freq/1e9:.4f} GHz: SWE vs .grd"
            )
            print(f"  {freq/1e9:.4f} GHz — "
                  f"Eco scale={met_co['scaling_ratio']:.4f}, "
                  f"Ecx scale={met_cx['scaling_ratio']:.4f}, "
                  f"Eco nRMS={met_co['normalized_rms_after_scaling']:.2e}")

    def test_near_field_not_zero(self):
        """Sanity check: near field is non-zero at a test point for every frequency."""
        for freq, swe, _ in self.freq_field_pairs:
            (E_r, E_theta, E_phi), _ = swe.near_field(
                np.array([1.0]), np.array([np.pi / 4]), np.array([0.0]),
                frequency=freq, normalize=False
            )
            total = np.abs(E_r) + np.abs(E_theta) + np.abs(E_phi)
            assert total[0] > 0, f"Near field is zero at {freq/1e9:.4f} GHz"
