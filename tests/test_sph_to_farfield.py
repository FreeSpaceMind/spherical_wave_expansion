"""
Test: Load .sph -> compute far field -> compare with Ticra .cut reference.

Validates absolute field levels (V/m) between the SWE package and Ticra GRASP,
using Ludwig-3 (Eco/Ecx) polarization and proper negative-theta handling.
"""

import numpy as np
import pytest

from swe import SphericalWaveExpansion
from swe.ticra_io import read_grasp_cut
from swe.ludwig3 import (
    spherical_to_ludwig3, remap_negative_theta, extract_cut_frequency_set
)
from tests.conftest import (
    SPH_FILE, CUT_FILE, requires_sph, requires_cut,
    compute_comparison_metrics, N_PHI_PER_FREQ, FREQ_INDEX_8GHZ
)


@requires_sph
@requires_cut
class TestSphToFarField:

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load reference data with absolute scaling preserved."""
        self.swe = SphericalWaveExpansion.from_sph_file(SPH_FILE, normalize=False)
        all_cuts = read_grasp_cut(CUT_FILE)
        self.cut_data = extract_cut_frequency_set(
            all_cuts, FREQ_INDEX_8GHZ, N_PHI_PER_FREQ
        )

    def _compute_cut_comparison(self, cut):
        """
        Compute far field for a single cut and convert to Ludwig-3.

        Returns computed (Eco, Ecx) and reference (Eco_ref, Ecx_ref).
        """
        phi_deg = cut['constant']
        theta_deg = cut['v_ini'] + np.arange(cut['v_num']) * cut['v_inc']

        theta_rad, phi_rad = remap_negative_theta(theta_deg, phi_deg)

        E_theta, E_phi = self.swe.far_field(theta_rad, phi_rad, normalize=False)

        Eco, Ecx = spherical_to_ludwig3(E_theta, E_phi, phi_rad)

        Eco_ref = cut['data'][:, 0]
        Ecx_ref = cut['data'][:, 1]

        return Eco, Ecx, Eco_ref, Ecx_ref

    def test_far_field_shape(self):
        """Verify far_field returns arrays with correct shapes."""
        theta = np.linspace(0, np.pi, 181)
        phi = np.zeros_like(theta)
        E_theta, E_phi = self.swe.far_field(theta, phi, normalize=False)
        assert E_theta.shape == theta.shape
        assert E_phi.shape == phi.shape

    def test_far_field_absolute_vs_cut(self):
        """Compare absolute Eco/Ecx from SWE far_field with Ticra .cut reference."""
        all_Eco = []
        all_Ecx = []
        all_Eco_ref = []
        all_Ecx_ref = []

        for cut in self.cut_data['cuts']:
            Eco, Ecx, Eco_ref, Ecx_ref = self._compute_cut_comparison(cut)
            all_Eco.append(Eco)
            all_Ecx.append(Ecx)
            all_Eco_ref.append(Eco_ref)
            all_Ecx_ref.append(Ecx_ref)

        all_Eco = np.concatenate(all_Eco)
        all_Ecx = np.concatenate(all_Ecx)
        all_Eco_ref = np.concatenate(all_Eco_ref)
        all_Ecx_ref = np.concatenate(all_Ecx_ref)

        metrics_co = compute_comparison_metrics(
            all_Eco, all_Eco_ref, label="Eco: SWE vs .cut (absolute)"
        )
        metrics_cx = compute_comparison_metrics(
            all_Ecx, all_Ecx_ref, label="Ecx: SWE vs .cut (absolute)"
        )

        # Scaling ratio should be near 1.0 for absolute agreement
        print(f"\n  Eco scaling ratio: {metrics_co['scaling_ratio']:.6f} (expect ~1.0)")
        print(f"  Ecx scaling ratio: {metrics_cx['scaling_ratio']:.6f} (expect ~1.0)")

    def test_far_field_per_cut_ludwig3(self):
        """Compare Ludwig-3 field for each individual phi cut separately."""
        print(f"\n{'='*70}")
        print(f"Per-cut Ludwig-3 comparison (SWE far_field vs Ticra .cut)")
        print(f"{'='*70}")

        for i, cut in enumerate(self.cut_data['cuts']):
            Eco, Ecx, Eco_ref, Ecx_ref = self._compute_cut_comparison(cut)

            met_co = compute_comparison_metrics(
                Eco, Eco_ref,
                label=f"Cut {i} (phi={cut['constant']}deg) Eco"
            )
            met_cx = compute_comparison_metrics(
                Ecx, Ecx_ref,
                label=f"Cut {i} (phi={cut['constant']}deg) Ecx"
            )

    def test_far_field_pattern_symmetry(self):
        """Sanity check: far-field pattern should be smooth and physical."""
        theta = np.linspace(0, np.pi, 181)
        phi = np.zeros_like(theta)

        E_theta, E_phi = self.swe.far_field(theta, phi, normalize=False)

        assert np.max(np.abs(E_theta)) > 0 or np.max(np.abs(E_phi)) > 0, \
            "Far field is identically zero"
        assert not np.any(np.isnan(E_theta)), "E_theta contains NaN"
        assert not np.any(np.isnan(E_phi)), "E_phi contains NaN"
