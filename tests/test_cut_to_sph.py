"""
Test: Load .cut -> extract Q coefficients via from_far_field -> compare with .sph reference.

Validates the inverse problem: given a far-field pattern, can we recover
the spherical wave expansion coefficients that match the original .sph file?
"""

import numpy as np
import pytest

from swe import SphericalWaveExpansion
from swe.ticra_io import read_grasp_cut, cut_to_fields
from tests.conftest import (
    SPH_FILE, CUT_FILE, requires_sph, requires_cut, compute_comparison_metrics
)


@requires_sph
@requires_cut
class TestCutToSph:

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load the reference data."""
        self.swe_ref = SphericalWaveExpansion.from_sph_file(SPH_FILE)
        self.cut_data = read_grasp_cut(CUT_FILE)

    def _build_full_sphere_grid(self):
        """
        Build a full-sphere (theta, phi) grid from the .cut file data.

        The .cut file typically only has cuts at a few phi values.
        For from_far_field to work well, we need data on a regular 2D grid.
        This function checks if the .cut data covers enough of the sphere.

        Returns:
            theta, phi, E_theta, E_phi: arrays on a regular grid, or None if
            the .cut data is insufficient.
        """
        cuts = self.cut_data['cuts']
        if not cuts:
            return None, None, None, None

        # Check if we have enough phi cuts for a full 2D grid
        phi_values = sorted(set(c['constant'] for c in cuts))
        n_phi = len(phi_values)

        # Use the first cut to determine theta sampling
        first_cut = cuts[0]
        theta_deg = first_cut['v_ini'] + np.arange(first_cut['v_num']) * first_cut['v_inc']
        n_theta = len(theta_deg)

        print(f"\n  Cut file has {n_phi} phi values, {n_theta} theta points per cut")
        print(f"  Phi values: {phi_values}")
        print(f"  Theta range: [{theta_deg[0]:.1f}, {theta_deg[-1]:.1f}] deg")

        if n_phi < 3:
            print("  WARNING: Too few phi cuts for reliable SWE extraction.")
            print("  Need at least a few phi values for from_far_field.")

        # Build 2D grid: theta varies along rows, phi along columns
        theta_rad = np.deg2rad(theta_deg)
        phi_rad = np.deg2rad(np.array(phi_values))

        THETA, PHI = np.meshgrid(theta_rad, phi_rad, indexing='ij')

        E_THETA = np.zeros((n_theta, n_phi), dtype=complex)
        E_PHI = np.zeros((n_theta, n_phi), dtype=complex)

        # Map cuts to grid columns
        phi_to_col = {phi: j for j, phi in enumerate(phi_values)}
        for cut in cuts:
            j = phi_to_col.get(cut['constant'])
            if j is None:
                continue
            # Verify theta sampling matches
            cut_theta = cut['v_ini'] + np.arange(cut['v_num']) * cut['v_inc']
            if len(cut_theta) != n_theta or not np.allclose(cut_theta, theta_deg, atol=0.01):
                print(f"  WARNING: Cut at phi={cut['constant']} has different theta sampling")
                continue
            E_THETA[:, j] = cut['data'][:, 0]
            E_PHI[:, j] = cut['data'][:, 1]

        return THETA.ravel(), PHI.ravel(), E_THETA.ravel(), E_PHI.ravel()

    def test_cut_to_swe_extraction(self):
        """Extract SWE coefficients from .cut data and compare with .sph reference."""
        theta, phi, E_theta, E_phi = self._build_full_sphere_grid()

        if theta is None:
            pytest.skip("Could not build grid from .cut data")

        if self.swe_ref.frequency is None:
            pytest.skip("Reference .sph has no frequency info")

        print(f"\n  Extracting SWE from {len(theta)} field points...")
        print(f"  Reference: NMAX={self.swe_ref.NMAX}, MMAX={self.swe_ref.MMAX}")

        # Extract coefficients
        swe_extracted = SphericalWaveExpansion.from_far_field(
            theta, phi, E_theta, E_phi,
            frequency=self.swe_ref.frequency,
            NMAX_initial=self.swe_ref.NMAX,
            MMAX_initial=self.swe_ref.MMAX,
            use_multiprocessing=False,
        )

        print(f"  Extracted: NMAX={swe_extracted.NMAX}, MMAX={swe_extracted.MMAX}")
        print(f"  Q1 modes: {len(swe_extracted.Q1_coeffs)}")
        print(f"  Q2 modes: {len(swe_extracted.Q2_coeffs)}")

        # Compare Q coefficients
        self._compare_coefficients(swe_extracted, self.swe_ref)

    def _compare_coefficients(self, swe_comp, swe_ref):
        """Compare Q coefficients between computed and reference SWE."""
        # Gather all Q1 values for common modes
        common_keys_q1 = set(swe_comp.Q1_coeffs.keys()) & set(swe_ref.Q1_coeffs.keys())
        common_keys_q2 = set(swe_comp.Q2_coeffs.keys()) & set(swe_ref.Q2_coeffs.keys())

        print(f"\n  Common Q1 modes: {len(common_keys_q1)}")
        print(f"  Common Q2 modes: {len(common_keys_q2)}")

        if common_keys_q1:
            comp_q1 = np.array([swe_comp.Q1_coeffs[k] for k in sorted(common_keys_q1)])
            ref_q1 = np.array([swe_ref.Q1_coeffs[k] for k in sorted(common_keys_q1)])
            compute_comparison_metrics(comp_q1, ref_q1, label="Q1 coefficients")

        if common_keys_q2:
            comp_q2 = np.array([swe_comp.Q2_coeffs[k] for k in sorted(common_keys_q2)])
            ref_q2 = np.array([swe_ref.Q2_coeffs[k] for k in sorted(common_keys_q2)])
            compute_comparison_metrics(comp_q2, ref_q2, label="Q2 coefficients")

    def test_cut_roundtrip_via_swe(self):
        """
        Load .cut -> extract SWE -> recompute far field -> compare with original .cut.

        Even if the Q coefficients don't perfectly match the reference .sph,
        the reconstructed far field from those Q's should match the input .cut.
        """
        theta, phi, E_theta_ref, E_phi_ref = self._build_full_sphere_grid()

        if theta is None:
            pytest.skip("Could not build grid from .cut data")

        if self.swe_ref.frequency is None:
            pytest.skip("Reference .sph has no frequency info")

        # Extract SWE
        swe_extracted = SphericalWaveExpansion.from_far_field(
            theta, phi, E_theta_ref, E_phi_ref,
            frequency=self.swe_ref.frequency,
            NMAX_initial=self.swe_ref.NMAX,
            MMAX_initial=self.swe_ref.MMAX,
            use_multiprocessing=False,
        )

        # Recompute far field
        E_theta_recomp, E_phi_recomp = swe_extracted.far_field(theta, phi)

        metrics_th = compute_comparison_metrics(
            E_theta_recomp, E_theta_ref,
            label="Far-field roundtrip E_theta (.cut -> SWE -> far_field)"
        )
        metrics_ph = compute_comparison_metrics(
            E_phi_recomp, E_phi_ref,
            label="Far-field roundtrip E_phi (.cut -> SWE -> far_field)"
        )

        # The roundtrip should be quite good if enough modes are used
        print(f"\n  E_theta normalized RMS: {metrics_th['normalized_rms_after_scaling']:.6e}")
        print(f"  E_phi normalized RMS:   {metrics_ph['normalized_rms_after_scaling']:.6e}")
