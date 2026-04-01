"""
Test: Load .sph -> compute far field -> compare with Ticra .cut reference.

This is the key test for identifying far-field scaling issues between
the SWE package and Ticra GRASP.
"""

import numpy as np
import pytest

from swe import SphericalWaveExpansion
from swe.ticra_io import read_grasp_cut, cut_to_fields
from tests.conftest import SPH_FILE, CUT_FILE, requires_sph, requires_cut, compute_comparison_metrics


@requires_sph
@requires_cut
class TestSphToFarField:

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load the reference data."""
        self.swe = SphericalWaveExpansion.from_sph_file(SPH_FILE)
        self.cut_data = read_grasp_cut(CUT_FILE)
        self.theta_ref, self.phi_ref, self.E_theta_ref, self.E_phi_ref = \
            cut_to_fields(self.cut_data)

    def test_far_field_shape(self):
        """Verify far_field returns arrays with correct shapes."""
        E_theta, E_phi = self.swe.far_field(self.theta_ref, self.phi_ref)
        assert E_theta.shape == self.theta_ref.shape
        assert E_phi.shape == self.phi_ref.shape

    def test_far_field_vs_cut_e_theta(self):
        """Compare E_theta from SWE far_field with Ticra .cut reference."""
        E_theta, E_phi = self.swe.far_field(self.theta_ref, self.phi_ref)

        metrics = compute_comparison_metrics(
            E_theta, self.E_theta_ref, label="E_theta: SWE vs .cut"
        )

        # Report but don't hard-fail on scaling (this test helps diagnose the issue)
        print(f"\n  If scaling_ratio != 1.0, there is a normalization mismatch.")
        print(f"  Scaling ratio: {metrics['scaling_ratio']:.6f}")
        print(f"  Expected: 1.000000")

    def test_far_field_vs_cut_e_phi(self):
        """Compare E_phi from SWE far_field with Ticra .cut reference."""
        E_theta, E_phi = self.swe.far_field(self.theta_ref, self.phi_ref)

        metrics = compute_comparison_metrics(
            E_phi, self.E_phi_ref, label="E_phi: SWE vs .cut"
        )

        print(f"\n  If scaling_ratio != 1.0, there is a normalization mismatch.")
        print(f"  Scaling ratio: {metrics['scaling_ratio']:.6f}")
        print(f"  Expected: 1.000000")

    def test_far_field_per_cut(self):
        """Compare field for each individual phi cut separately."""
        print(f"\n{'='*70}")
        print(f"Per-cut comparison (SWE far_field vs Ticra .cut)")
        print(f"{'='*70}")

        for i, cut in enumerate(self.cut_data['cuts']):
            phi_deg = cut['constant']
            angles_deg = cut['v_ini'] + np.arange(cut['v_num']) * cut['v_inc']
            theta_rad = np.deg2rad(angles_deg)
            phi_rad = np.full_like(theta_rad, np.deg2rad(phi_deg))

            E_theta_comp, E_phi_comp = self.swe.far_field(theta_rad, phi_rad)

            ref_E_theta = cut['data'][:, 0]
            ref_E_phi = cut['data'][:, 1]

            met_th = compute_comparison_metrics(
                E_theta_comp, ref_E_theta,
                label=f"Cut {i} (phi={phi_deg}deg) E_theta"
            )
            met_ph = compute_comparison_metrics(
                E_phi_comp, ref_E_phi,
                label=f"Cut {i} (phi={phi_deg}deg) E_phi"
            )

    def test_far_field_pattern_symmetry(self):
        """Sanity check: far-field pattern should be smooth and physical."""
        theta = np.linspace(0, np.pi, 181)
        phi = np.zeros_like(theta)

        E_theta, E_phi = self.swe.far_field(theta, phi)

        # Pattern should not be identically zero
        assert np.max(np.abs(E_theta)) > 0 or np.max(np.abs(E_phi)) > 0, \
            "Far field is identically zero"

        # Pattern should not have NaN
        assert not np.any(np.isnan(E_theta)), "E_theta contains NaN"
        assert not np.any(np.isnan(E_phi)), "E_phi contains NaN"
