"""
Test: Load .cut -> extract Q coefficients via from_far_field -> compare with .cut and .grd.

Validates the inverse problem: given a far-field pattern in Ludwig-3, can we
recover SWE coefficients that reproduce both the far field (.cut) and near field (.grd)?
"""

import numpy as np
import pytest

from swe import SphericalWaveExpansion, cartesian_to_spherical
from swe.ticra_io import read_grasp_cut, read_grasp_grd
from swe.ludwig3 import (
    spherical_to_ludwig3, ludwig3_to_spherical,
    remap_negative_theta, extract_cut_frequency_set
)
from tests.conftest import (
    SPH_FILE, CUT_FILE, GRD_FILE,
    requires_sph, requires_cut, requires_grd,
    compute_comparison_metrics,
    N_PHI_PER_FREQ, FREQ_INDEX_8GHZ, Z_DISTANCE
)


def _build_full_sphere_grid(cuts):
    """
    Build a full-sphere (theta, phi) grid from .cut data with negative-theta remapping.

    Each cut has theta from -180 to +180 deg at a fixed phi. Negative theta maps to
    theta'=|theta|, phi'=phi+180. With phi from 0-180 deg, this covers the full sphere.

    Returns:
        theta, phi, E_theta, E_phi as 1D arrays on a regular grid,
        sorted by (phi, theta) for from_far_field compatibility.
    """
    # Collect remapped data from all cuts
    grid_points = {}  # (theta_round, phi_round) -> (E_theta, E_phi)

    for cut in cuts:
        phi_deg = cut['constant']
        theta_deg_arr = cut['v_ini'] + np.arange(cut['v_num']) * cut['v_inc']
        theta_rad, phi_rad = remap_negative_theta(theta_deg_arr, phi_deg)

        Eco = cut['data'][:, 0]
        Ecx = cut['data'][:, 1]
        E_theta, E_phi = ludwig3_to_spherical(Eco, Ecx, phi_rad)

        for k in range(len(theta_rad)):
            # Round to avoid floating point key issues
            th_key = round(np.rad2deg(theta_rad[k]), 4)
            ph_key = round(np.rad2deg(phi_rad[k]) % 360, 4)
            grid_points[(th_key, ph_key)] = (E_theta[k], E_phi[k])

    # Sort into regular grid
    all_theta_deg = sorted(set(k[0] for k in grid_points.keys()))
    all_phi_deg = sorted(set(k[1] for k in grid_points.keys()))

    n_theta = len(all_theta_deg)
    n_phi = len(all_phi_deg)

    print(f"\n  Grid: {n_theta} theta x {n_phi} phi = {n_theta * n_phi} points")
    print(f"  Theta: [{all_theta_deg[0]:.1f}, {all_theta_deg[-1]:.1f}] deg")
    print(f"  Phi: [{all_phi_deg[0]:.1f}, {all_phi_deg[-1]:.1f}] deg")

    theta_grid = np.deg2rad(np.array(all_theta_deg))
    phi_grid = np.deg2rad(np.array(all_phi_deg))
    THETA, PHI = np.meshgrid(theta_grid, phi_grid, indexing='ij')

    E_THETA = np.zeros((n_theta, n_phi), dtype=complex)
    E_PHI = np.zeros((n_theta, n_phi), dtype=complex)

    for i, th in enumerate(all_theta_deg):
        for j, ph in enumerate(all_phi_deg):
            key = (round(th, 4), round(ph, 4))
            if key in grid_points:
                E_THETA[i, j], E_PHI[i, j] = grid_points[key]

    return THETA.ravel(), PHI.ravel(), E_THETA.ravel(), E_PHI.ravel()


@requires_sph
@requires_cut
class TestCutToSph:

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load reference data."""
        self.swe_ref = SphericalWaveExpansion.from_sph_file(SPH_FILE, normalize=False)
        all_cuts = read_grasp_cut(CUT_FILE)
        self.cut_data = extract_cut_frequency_set(
            all_cuts, FREQ_INDEX_8GHZ, N_PHI_PER_FREQ
        )

    def test_cut_to_swe_extraction(self):
        """Extract SWE coefficients from .cut data and compare with .sph reference."""
        theta, phi, E_theta, E_phi = _build_full_sphere_grid(self.cut_data['cuts'])

        print(f"\n  Extracting SWE from {len(theta)} field points...")
        print(f"  Reference: NMAX={self.swe_ref.NMAX}, MMAX={self.swe_ref.MMAX}")

        swe_extracted = SphericalWaveExpansion.from_far_field(
            theta, phi, E_theta, E_phi,
            frequency=self.swe_ref.frequency,
            NMAX_initial=self.swe_ref.NMAX,
            MMAX_initial=self.swe_ref.MMAX,
            use_multiprocessing=False,
            normalize=False,
        )

        print(f"  Extracted: NMAX={swe_extracted.NMAX}, MMAX={swe_extracted.MMAX}")
        print(f"  Q1 modes: {len(swe_extracted.Q1_coeffs)}")
        print(f"  Q2 modes: {len(swe_extracted.Q2_coeffs)}")

        # Compare Q coefficients
        common_q1 = set(swe_extracted.Q1_coeffs.keys()) & set(self.swe_ref.Q1_coeffs.keys())
        common_q2 = set(swe_extracted.Q2_coeffs.keys()) & set(self.swe_ref.Q2_coeffs.keys())
        print(f"  Common Q1 modes: {len(common_q1)}")
        print(f"  Common Q2 modes: {len(common_q2)}")

        if common_q1:
            comp_q1 = np.array([swe_extracted.Q1_coeffs[k] for k in sorted(common_q1)])
            ref_q1 = np.array([self.swe_ref.Q1_coeffs[k] for k in sorted(common_q1)])
            compute_comparison_metrics(comp_q1, ref_q1, label="Q1 coefficients")

        if common_q2:
            comp_q2 = np.array([swe_extracted.Q2_coeffs[k] for k in sorted(common_q2)])
            ref_q2 = np.array([self.swe_ref.Q2_coeffs[k] for k in sorted(common_q2)])
            compute_comparison_metrics(comp_q2, ref_q2, label="Q2 coefficients")

    def test_cut_roundtrip_farfield(self):
        """
        Load .cut -> extract SWE -> recompute far field -> compare with original .cut.

        The reconstructed far field from extracted Q's should match the input .cut
        in Ludwig-3 (Eco/Ecx) at absolute levels.
        """
        theta, phi, E_theta, E_phi = _build_full_sphere_grid(self.cut_data['cuts'])

        swe_extracted = SphericalWaveExpansion.from_far_field(
            theta, phi, E_theta, E_phi,
            frequency=self.swe_ref.frequency,
            NMAX_initial=self.swe_ref.NMAX,
            MMAX_initial=self.swe_ref.MMAX,
            use_multiprocessing=False,
            normalize=False,
        )

        # Compare per-cut in Ludwig-3
        print(f"\n{'='*70}")
        print(f"Roundtrip: .cut -> SWE -> far_field -> Ludwig-3 vs original .cut")
        print(f"{'='*70}")

        for i, cut in enumerate(self.cut_data['cuts'][:5]):  # first 5 cuts for speed
            phi_deg = cut['constant']
            theta_deg = cut['v_ini'] + np.arange(cut['v_num']) * cut['v_inc']
            theta_rad, phi_rad = remap_negative_theta(theta_deg, phi_deg)

            E_th, E_ph = swe_extracted.far_field(theta_rad, phi_rad, normalize=False)
            Eco, Ecx = spherical_to_ludwig3(E_th, E_ph, phi_rad)

            Eco_ref = cut['data'][:, 0]
            Ecx_ref = cut['data'][:, 1]

            compute_comparison_metrics(
                Eco, Eco_ref,
                label=f"Roundtrip cut {i} (phi={phi_deg}deg) Eco"
            )


@requires_sph
@requires_cut
@requires_grd
class TestCutSweReproducesGrd:
    """Cross-validation: SWE extracted from .cut should reproduce .grd near field."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load data and extract SWE from .cut."""
        self.swe_ref = SphericalWaveExpansion.from_sph_file(SPH_FILE, normalize=False)
        all_cuts = read_grasp_cut(CUT_FILE)
        cut_data = extract_cut_frequency_set(
            all_cuts, FREQ_INDEX_8GHZ, N_PHI_PER_FREQ
        )
        self.grd_data = read_grasp_grd(GRD_FILE)

        theta, phi, E_theta, E_phi = _build_full_sphere_grid(cut_data['cuts'])
        self.swe_extracted = SphericalWaveExpansion.from_far_field(
            theta, phi, E_theta, E_phi,
            frequency=self.swe_ref.frequency,
            NMAX_initial=self.swe_ref.NMAX,
            MMAX_initial=self.swe_ref.MMAX,
            use_multiprocessing=False,
            normalize=False,
        )

    def test_extracted_swe_reproduces_grd(self):
        """Compute near field from extracted SWE, compare with .grd at z=0.25m."""
        field = self.grd_data['fields'][FREQ_INDEX_8GHZ]
        x = np.linspace(field['grid_min_x'], field['grid_max_x'], field['nx'])
        y = np.linspace(field['grid_min_y'], field['grid_max_y'], field['ny'])
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, Z_DISTANCE)

        r, theta, phi = cartesian_to_spherical(X.ravel(), Y.ravel(), Z.ravel())

        print(f"\n  Computing near field at {len(r)} points, z={Z_DISTANCE}m...")
        (E_r, E_theta, E_phi), _ = self.swe_extracted.near_field(
            r, theta, phi, normalize=False
        )

        Eco, Ecx = spherical_to_ludwig3(E_theta, E_phi, phi)
        Eco = Eco.reshape(X.shape)
        Ecx = Ecx.reshape(X.shape)

        ref_Eco = field['data'][:, :, 0]
        ref_Ecx = field['data'][:, :, 1]

        metrics_co = compute_comparison_metrics(
            Eco, ref_Eco,
            label="Cross-val near-field Eco: extracted SWE vs .grd"
        )
        metrics_cx = compute_comparison_metrics(
            Ecx, ref_Ecx,
            label="Cross-val near-field Ecx: extracted SWE vs .grd"
        )

        print(f"\n  Eco scaling ratio: {metrics_co['scaling_ratio']:.6f}")
        print(f"  Ecx scaling ratio: {metrics_cx['scaling_ratio']:.6f}")
