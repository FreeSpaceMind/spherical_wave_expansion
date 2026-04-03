"""
Validation script: compare SWE far-field and near-field against TICRA reference data.
Prints scaling ratios and nRMS for all 9 frequencies.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from swe import SphericalWaveExpansion, cartesian_to_spherical
from swe.ticra_io import read_grasp_cut, read_grasp_grd
from swe.ludwig3 import spherical_to_ludwig3, remap_negative_theta, extract_cut_frequency_set

SPH_FILE = "tests/example.sph"
CUT_FILE = "tests/example.cut"
GRD_FILE  = "tests/example.grd"
Z_DISTANCE = 0.25
N_PHI_PER_FREQ = 37


def scale_and_nrms(computed, reference):
    """Return best-fit scale ratio and normalized RMS after scaling."""
    c = np.asarray(computed).ravel()
    r = np.asarray(reference).ravel()
    mask = np.abs(r) > np.max(np.abs(r)) * 1e-3
    if not np.any(mask):
        return np.nan, np.nan
    scale = np.sum(c[mask] * np.conj(r[mask])) / np.sum(np.abs(r[mask])**2)
    nrms = (np.sqrt(np.mean(np.abs(c[mask] - scale * r[mask])**2))
            / np.sqrt(np.mean(np.abs(r[mask])**2)))
    return np.abs(scale), float(nrms)


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading .sph (normalize=False for absolute comparison)...")
swe_ff = SphericalWaveExpansion.from_sph_file(SPH_FILE, normalize=False)
freqs  = swe_ff.frequencies
print(f"  {len(freqs)} frequencies: {[f'{f/1e9:.4f}' for f in freqs]} GHz\n")

all_cuts = read_grasp_cut(CUT_FILE)
grd_data  = read_grasp_grd(GRD_FILE)

# ── Far-field comparison ───────────────────────────────────────────────────────
print("=" * 72)
print(f"{'FAR-FIELD vs TICRA .cut':^72}")
print("=" * 72)
print(f"{'Freq (GHz)':<12} {'Eco scale':>10} {'Eco nRMS':>10} {'Ecx scale':>10} {'Ecx nRMS':>10}")
print("-" * 72)

ff_results = []
for i, freq in enumerate(freqs):
    cut_set = extract_cut_frequency_set(all_cuts, i, N_PHI_PER_FREQ)
    all_Eco, all_Ecx, all_Eco_ref, all_Ecx_ref = [], [], [], []

    for cut in cut_set['cuts']:
        theta_deg = cut['v_ini'] + np.arange(cut['v_num']) * cut['v_inc']
        theta_rad, phi_rad = remap_negative_theta(theta_deg, cut['constant'])
        Et, Ep = swe_ff.far_field(theta_rad, phi_rad, frequency=freq, normalize=False)
        Eco, Ecx = spherical_to_ludwig3(Et, Ep, phi_rad)
        all_Eco.append(Eco);      all_Ecx.append(Ecx)
        all_Eco_ref.append(cut['data'][:, 0])
        all_Ecx_ref.append(cut['data'][:, 1])

    eco_s, eco_n = scale_and_nrms(np.concatenate(all_Eco), np.concatenate(all_Eco_ref))
    ecx_s, ecx_n = scale_and_nrms(np.concatenate(all_Ecx), np.concatenate(all_Ecx_ref))
    ff_results.append((freq, eco_s, eco_n, ecx_s, ecx_n))
    print(f"  {freq/1e9:.4f}     {eco_s:>10.6f} {eco_n:>10.3e} {ecx_s:>10.6f} {ecx_n:>10.3e}")

print("-" * 72)
eco_scales = [r[1] for r in ff_results]
ecx_scales = [r[3] for r in ff_results]
print(f"  {'Mean':<11} {np.mean(eco_scales):>10.6f} {'':>10} {np.mean(ecx_scales):>10.6f}")
print(f"  {'Min':<11} {np.min(eco_scales):>10.6f} {'':>10} {np.min(ecx_scales):>10.6f}")
print(f"  {'Max':<11} {np.max(eco_scales):>10.6f} {'':>10} {np.max(ecx_scales):>10.6f}")

# ── Near-field comparison ──────────────────────────────────────────────────────
print()
print("=" * 72)
print(f"{'NEAR-FIELD vs TICRA .grd  (z = 0.25 m)':^72}")
print("=" * 72)
print(f"{'Freq (GHz)':<12} {'NMAX_trunc':>10} {'Eco scale':>10} {'Eco nRMS':>10} {'Ecx scale':>10}")
print("-" * 72)

swe_full = SphericalWaveExpansion.from_sph_file(SPH_FILE, normalize=False)

nf_results = []
for i, freq in enumerate(freqs):
    field = grd_data['fields'][i]

    # Same truncation as the test fixture
    kr_min  = swe_full.k(freq) * Z_DISTANCE
    nmax_safe = max(int(kr_min) - 5, swe_full.MMAX(freq))

    q1t = {k: v for k, v in swe_full.Q1_coeffs(freq).items() if k[0] <= nmax_safe}
    q2t = {k: v for k, v in swe_full.Q2_coeffs(freq).items() if k[0] <= nmax_safe}
    swe_t = SphericalWaveExpansion(
        Q1_coeffs={freq: q1t}, Q2_coeffs={freq: q2t},
        NMAX={freq: nmax_safe}, MMAX={freq: swe_full.MMAX(freq)},
    )

    x = np.linspace(field['grid_min_x'], field['grid_max_x'], field['nx'])
    y = np.linspace(field['grid_min_y'], field['grid_max_y'], field['ny'])
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, Z_DISTANCE)
    r, theta, phi = cartesian_to_spherical(X.ravel(), Y.ravel(), Z.ravel())

    (_, Et, Ep), _ = swe_t.near_field(r, theta, phi, frequency=freq, normalize=False)
    Eco = spherical_to_ludwig3(Et, Ep, phi)[0].reshape(X.shape)
    Ecx = spherical_to_ludwig3(Et, Ep, phi)[1].reshape(X.shape)

    eco_s, eco_n = scale_and_nrms(Eco, field['data'][:, :, 0])
    ecx_s, _     = scale_and_nrms(Ecx, field['data'][:, :, 1])
    nf_results.append((freq, nmax_safe, eco_s, eco_n, ecx_s))
    print(f"  {freq/1e9:.4f}     {nmax_safe:>10d} {eco_s:>10.6f} {eco_n:>10.3e} {ecx_s:>10.6f}")

print("-" * 72)
eco_scales_nf = [r[2] for r in nf_results]
ecx_scales_nf = [r[4] for r in nf_results]
print(f"  {'Mean':<22} {np.mean(eco_scales_nf):>10.6f} {'':>10} {np.mean(ecx_scales_nf):>10.6f}")
print(f"  {'Min':<22} {np.min(eco_scales_nf):>10.6f} {'':>10} {np.min(ecx_scales_nf):>10.6f}")
print(f"  {'Max':<22} {np.max(eco_scales_nf):>10.6f} {'':>10} {np.max(ecx_scales_nf):>10.6f}")

print()
print("All comparisons complete.")
