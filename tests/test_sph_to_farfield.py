"""
Test: Load .sph -> compute far field -> compare with Ticra .cut reference.

Validates absolute field levels (V/m) between the SWE package and Ticra GRASP,
using Ludwig-3 (Eco/Ecx) polarization and proper negative-theta handling.
Loops over all frequencies present in the .sph/.cut files.
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
    compute_comparison_metrics, N_PHI_PER_FREQ,
)


@requires_sph
@requires_cut
class TestSphToFarField:

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load reference data for all frequencies."""
        self.swe = SphericalWaveExpansion.from_sph_file(SPH_FILE, normalize=False)
        all_cuts = read_grasp_cut(CUT_FILE)

        # Build a list of (frequency_Hz, cut_data) aligned by sequential index
        n_freqs = len(self.swe.frequencies)
        n_cut_blocks = len(all_cuts['cuts']) // N_PHI_PER_FREQ
        assert n_cut_blocks >= n_freqs, (
            f".cut file has {n_cut_blocks} frequency blocks but .sph has {n_freqs}"
        )

        self.freq_cut_pairs = [
            (self.swe.frequencies[i],
             extract_cut_frequency_set(all_cuts, i, N_PHI_PER_FREQ))
            for i in range(n_freqs)
        ]
        print(f"\n  Loaded {n_freqs} frequencies: "
              f"{[f'{f/1e9:.4f} GHz' for f in self.swe.frequencies]}")

    def _compute_cut_comparison(self, freq, cut):
        """Compute far field for one cut and convert to Ludwig-3."""
        theta_deg = cut['v_ini'] + np.arange(cut['v_num']) * cut['v_inc']
        theta_rad, phi_rad = remap_negative_theta(theta_deg, cut['constant'])

        E_theta, E_phi = self.swe.far_field(
            theta_rad, phi_rad, frequency=freq, normalize=False
        )
        Eco, Ecx = spherical_to_ludwig3(E_theta, E_phi, phi_rad)
        return Eco, Ecx, cut['data'][:, 0], cut['data'][:, 1]

    def test_far_field_shape(self):
        """Verify far_field returns arrays with correct shape for every frequency."""
        theta = np.linspace(0, np.pi, 181)
        phi = np.zeros_like(theta)
        for freq, _ in self.freq_cut_pairs:
            E_theta, E_phi = self.swe.far_field(theta, phi, frequency=freq, normalize=False)
            assert E_theta.shape == theta.shape, f"Shape mismatch at {freq/1e9:.4f} GHz"
            assert E_phi.shape == phi.shape,     f"Shape mismatch at {freq/1e9:.4f} GHz"

    def test_far_field_absolute_vs_cut(self):
        """Compare absolute Eco/Ecx against Ticra .cut for all frequencies."""
        for freq, cut_data in self.freq_cut_pairs:
            all_Eco, all_Ecx, all_Eco_ref, all_Ecx_ref = [], [], [], []
            for cut in cut_data['cuts']:
                Eco, Ecx, Eco_ref, Ecx_ref = self._compute_cut_comparison(freq, cut)
                all_Eco.append(Eco);     all_Ecx.append(Ecx)
                all_Eco_ref.append(Eco_ref); all_Ecx_ref.append(Ecx_ref)

            met_co = compute_comparison_metrics(
                np.concatenate(all_Eco), np.concatenate(all_Eco_ref),
                label=f"Eco {freq/1e9:.4f} GHz: SWE vs .cut"
            )
            met_cx = compute_comparison_metrics(
                np.concatenate(all_Ecx), np.concatenate(all_Ecx_ref),
                label=f"Ecx {freq/1e9:.4f} GHz: SWE vs .cut"
            )
            print(f"  {freq/1e9:.4f} GHz — "
                  f"Eco scale={met_co['scaling_ratio']:.4f}, "
                  f"Ecx scale={met_cx['scaling_ratio']:.4f}")

    def test_far_field_per_cut_ludwig3(self):
        """Compare Ludwig-3 per phi-cut for all frequencies."""
        for freq, cut_data in self.freq_cut_pairs:
            print(f"\n{'='*60}")
            print(f"Per-cut comparison at {freq/1e9:.4f} GHz")
            print(f"{'='*60}")
            for i, cut in enumerate(cut_data['cuts']):
                Eco, Ecx, Eco_ref, Ecx_ref = self._compute_cut_comparison(freq, cut)
                compute_comparison_metrics(
                    Eco, Eco_ref,
                    label=f"{freq/1e9:.4f} GHz cut {i} (phi={cut['constant']}deg) Eco"
                )
                compute_comparison_metrics(
                    Ecx, Ecx_ref,
                    label=f"{freq/1e9:.4f} GHz cut {i} (phi={cut['constant']}deg) Ecx"
                )

    def test_far_field_pattern_symmetry(self):
        """Sanity check: far-field is non-zero and NaN-free for every frequency."""
        theta = np.linspace(0, np.pi, 181)
        phi = np.zeros_like(theta)
        for freq, _ in self.freq_cut_pairs:
            E_theta, E_phi = self.swe.far_field(theta, phi, frequency=freq, normalize=False)
            assert np.max(np.abs(E_theta)) > 0 or np.max(np.abs(E_phi)) > 0, \
                f"Far field is zero at {freq/1e9:.4f} GHz"
            assert not np.any(np.isnan(E_theta)), f"NaN in E_theta at {freq/1e9:.4f} GHz"
            assert not np.any(np.isnan(E_phi)),   f"NaN in E_phi at {freq/1e9:.4f} GHz"
