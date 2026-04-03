#!/usr/bin/env python3
"""
Validation: Compare SWE package outputs against TICRA GRASP reference data.

Generates PNG figures in docs/figures/ and prints a Markdown summary table
of accuracy metrics for all 9 frequencies.

Reference data (in tests/):
  example.sph  — SWE coefficients, NMAX=359, MMAX=35, 9 frequencies 8.0–8.5 GHz
  example.cut  — Far-field pattern cuts (Ludwig-3), 37 phi cuts per frequency
  example.grd  — Near-field planar grid, 101×101 points, z=0.25 m, 9 field sets

Usage:
    cd spherical_wave_expansion
    python docs/generate_validation.py
    (or: PYTHONPATH=. python docs/generate_validation.py)
"""

import sys
import os

# Allow running from either the repo root or the docs/ folder
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from swe import SphericalWaveExpansion, cartesian_to_spherical
from swe.ticra_io import read_grasp_cut, read_grasp_grd
from swe.ludwig3 import (
    spherical_to_ludwig3, remap_negative_theta, extract_cut_frequency_set
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_DIR   = os.path.join(_root, "tests")
FIGS_DIR   = os.path.join(_here, "figures")
SPH_FILE   = os.path.join(TEST_DIR, "example.sph")
CUT_FILE   = os.path.join(TEST_DIR, "example.cut")
GRD_FILE   = os.path.join(TEST_DIR, "example.grd")
N_PHI_PER_FREQ = 37   # phi cuts per frequency in .cut
Z_DISTANCE     = 0.25  # metres — z-plane for .grd

os.makedirs(FIGS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scaling_metrics(computed, reference):
    """Return best-fit scale ratio, phase offset (deg), and normalised RMS."""
    c = np.asarray(computed).ravel()
    r = np.asarray(reference).ravel()
    ref_mag = np.abs(r)
    mask = ref_mag > np.max(ref_mag) * 1e-3
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    scale = np.sum(c[mask] * np.conj(r[mask])) / np.sum(np.abs(r[mask]) ** 2)
    nrms  = (np.sqrt(np.mean(np.abs(c - scale * r)[mask] ** 2))
             / np.sqrt(np.mean(np.abs(r[mask]) ** 2)))
    return np.abs(scale), np.degrees(np.angle(scale)), nrms


def _db(field):
    """Convert complex field to dB magnitude."""
    return 20 * np.log10(np.abs(field) + 1e-30)


def _build_nf_grid(field):
    """Build the planar near-field grid from .grd extents."""
    x = np.linspace(field["grid_min_x"], field["grid_max_x"], field["nx"])
    y = np.linspace(field["grid_min_y"], field["grid_max_y"], field["ny"])
    X, Y = np.meshgrid(x, y)
    return X, Y, np.full_like(X, Z_DISTANCE)


def _truncated_swe(swe_full, freq):
    """Return mode-truncated SWE object safe for near-field at Z_DISTANCE."""
    kr_min    = swe_full.k(freq) * Z_DISTANCE
    nmax_safe = max(int(kr_min) - 5, swe_full.MMAX(freq))
    q1 = {k: v for k, v in swe_full.Q1_coeffs(freq).items() if k[0] <= nmax_safe}
    q2 = {k: v for k, v in swe_full.Q2_coeffs(freq).items() if k[0] <= nmax_safe}
    return SphericalWaveExpansion(
        Q1_coeffs={freq: q1},
        Q2_coeffs={freq: q2},
        NMAX={freq: nmax_safe},
        MMAX={freq: swe_full.MMAX(freq)},
    ), nmax_safe


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading reference data …")
swe = SphericalWaveExpansion.from_sph_file(SPH_FILE, normalize=False)
all_cuts = read_grasp_cut(CUT_FILE)
grd_data  = read_grasp_grd(GRD_FILE)

freqs    = swe.frequencies
n_freqs  = len(freqs)
freq_GHz = [f / 1e9 for f in freqs]
print(f"  {n_freqs} frequencies: {[f'{f:.4f} GHz' for f in freq_GHz]}")


# ===========================================================================
# 1.  FAR-FIELD VALIDATION
# ===========================================================================
print("\nComputing far-field validation …")

# We compare at phi = 0°, 45°, 90° — pick the closest available cut in .cut
PHI_CUTS_DEG = [0.0, 45.0, 90.0]

# Storage for metrics
ff_metrics = {f: {} for f in freqs}   # ff_metrics[freq]['Eco'] = (ratio, phase, nrms)

# ---- Figure 1: 3×3 grid, one panel per frequency, phi=0 E-plane cut ----
fig1, axes1 = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
axes1_flat  = axes1.flatten()
fig1.suptitle("Far-Field Validation — E-plane (φ = 0°)\nSWE vs TICRA GRASP reference", fontsize=13)

for freq_i, freq in enumerate(freqs):
    ax = axes1_flat[freq_i]

    cut_freq = extract_cut_frequency_set(all_cuts, freq_i, N_PHI_PER_FREQ)

    # Accumulate comparison metrics across all cuts for this frequency
    all_Eco, all_Ecx, all_ref_Eco, all_ref_Ecx = [], [], [], []

    # E-plane cut for plotting (phi = 0°)
    eplane_cut = None
    for cut in cut_freq["cuts"]:
        if abs(cut["constant"] - 0.0) < 0.1:
            eplane_cut = cut

        theta_deg = cut["v_ini"] + np.arange(cut["v_num"]) * cut["v_inc"]
        theta_rad, phi_rad = remap_negative_theta(theta_deg, cut["constant"])

        Et, Ep = swe.far_field(theta_rad, phi_rad, frequency=freq, normalize=False)
        Eco, Ecx = spherical_to_ludwig3(Et, Ep, phi_rad)

        all_Eco.append(Eco);       all_ref_Eco.append(cut["data"][:, 0])
        all_Ecx.append(Ecx);       all_ref_Ecx.append(cut["data"][:, 1])

    ratio_co, phase_co, nrms_co = _scaling_metrics(
        np.concatenate(all_Eco), np.concatenate(all_ref_Eco))
    ratio_cx, phase_cx, nrms_cx = _scaling_metrics(
        np.concatenate(all_Ecx), np.concatenate(all_ref_Ecx))

    ff_metrics[freq]["Eco"] = (ratio_co, phase_co, nrms_co)
    ff_metrics[freq]["Ecx"] = (ratio_cx, phase_cx, nrms_cx)

    # Plot E-plane cut
    if eplane_cut is not None:
        td = eplane_cut["v_ini"] + np.arange(eplane_cut["v_num"]) * eplane_cut["v_inc"]
        theta_rad_ep, phi_rad_ep = remap_negative_theta(td, eplane_cut["constant"])
        Et_ep, Ep_ep = swe.far_field(theta_rad_ep, phi_rad_ep, frequency=freq, normalize=False)
        Eco_ep, _ = spherical_to_ludwig3(Et_ep, Ep_ep, phi_rad_ep)

        ref_Eco_ep = eplane_cut["data"][:, 0]

        # Normalise both to max of reference for dB plot
        ref_peak = np.max(np.abs(ref_Eco_ep))
        ax.plot(td, _db(ref_Eco_ep / ref_peak), "k-",  lw=1.5, label="GRASP ref.")
        ax.plot(td, _db(Eco_ep    / ref_peak), "r--", lw=1.2, label="SWE")

    ax.set_title(f"{freq/1e9:.4f} GHz\nscale={ratio_co:.4f}, nRMS={nrms_co:.2e}",
                 fontsize=9)
    ax.set_ylim(-50, 3)
    ax.set_xlim(-180, 180)
    ax.grid(True, alpha=0.4)
    ax.tick_params(labelsize=8)
    if freq_i % 3 == 0:
        ax.set_ylabel("Rel. amplitude (dB)", fontsize=8)
    if freq_i >= 6:
        ax.set_xlabel("θ (degrees)", fontsize=8)

# Shared legend
handles, labels = axes1_flat[0].get_legend_handles_labels()
fig1.legend(handles, labels, loc="lower center", ncol=2, fontsize=10,
            bbox_to_anchor=(0.5, 0.01))
fig1.tight_layout(rect=[0, 0.04, 1, 0.97])

out1 = os.path.join(FIGS_DIR, "validation_farfield_eplane.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"  Saved: {out1}")


# ---- Figure 2: 3-cut overlay at 8.0 GHz (phi = 0°, 45°, 90°) ----
freq_ref = freqs[0]
cut_freq0 = extract_cut_frequency_set(all_cuts, 0, N_PHI_PER_FREQ)

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
fig2.suptitle(f"Far-Field Validation at {freq_ref/1e9:.4f} GHz — multiple cuts\n"
              "SWE vs TICRA GRASP reference", fontsize=12)

for ci, target_phi in enumerate(PHI_CUTS_DEG):
    ax = axes2[ci]
    cut_match = None
    for cut in cut_freq0["cuts"]:
        if abs(cut["constant"] - target_phi) < 0.1:
            cut_match = cut
            break

    if cut_match is None:
        ax.set_title(f"φ = {target_phi}° (not found)")
        continue

    td  = cut_match["v_ini"] + np.arange(cut_match["v_num"]) * cut_match["v_inc"]
    tr, pr = remap_negative_theta(td, cut_match["constant"])
    Et, Ep = swe.far_field(tr, pr, frequency=freq_ref, normalize=False)
    Eco, Ecx = spherical_to_ludwig3(Et, Ep, pr)

    ref_Eco = cut_match["data"][:, 0]
    ref_Ecx = cut_match["data"][:, 1]
    ref_peak = max(np.max(np.abs(ref_Eco)), 1e-30)

    ax.plot(td, _db(ref_Eco / ref_peak), "k-",  lw=1.5, label="Eco GRASP")
    ax.plot(td, _db(Eco     / ref_peak), "r--", lw=1.2, label="Eco SWE")
    ax.plot(td, _db(ref_Ecx / ref_peak), "b-",  lw=1.5, label="Ecx GRASP", alpha=0.6)
    ax.plot(td, _db(Ecx     / ref_peak), "m--", lw=1.2, label="Ecx SWE",   alpha=0.8)

    r_co, p_co, n_co = _scaling_metrics(Eco, ref_Eco)
    ax.set_title(f"φ = {target_phi}°\nEco: scale={r_co:.4f}, nRMS={n_co:.2e}", fontsize=9)
    ax.set_xlabel("θ (degrees)", fontsize=10)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 3)
    ax.grid(True, alpha=0.4)

axes2[0].set_ylabel("Rel. amplitude (dB)", fontsize=10)
axes2[1].legend(fontsize=8, loc="lower center", ncol=2)
fig2.tight_layout()

out2 = os.path.join(FIGS_DIR, "validation_farfield_cuts.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved: {out2}")


# ===========================================================================
# 2.  NEAR-FIELD VALIDATION
# ===========================================================================
print("\nComputing near-field validation …")

nf_metrics = {f: {} for f in freqs}

nf_results = {}   # store arrays for 8.0 GHz 2D map

for freq_i, freq in enumerate(freqs):
    field = grd_data["fields"][freq_i]
    X, Y, Z = _build_nf_grid(field)
    r, theta, phi = cartesian_to_spherical(X.ravel(), Y.ravel(), Z.ravel())

    swe_tr, nmax_tr = _truncated_swe(swe, freq)

    (_, E_theta, E_phi), _ = swe_tr.near_field(r, theta, phi,
                                                frequency=freq, normalize=False)

    Eco = spherical_to_ludwig3(E_theta, E_phi, phi)[0].reshape(X.shape)
    Ecx = spherical_to_ludwig3(E_theta, E_phi, phi)[1].reshape(X.shape)

    ref_Eco = field["data"][:, :, 0]
    ref_Ecx = field["data"][:, :, 1]

    ratio_co, phase_co, nrms_co = _scaling_metrics(Eco, ref_Eco)
    ratio_cx, phase_cx, nrms_cx = _scaling_metrics(Ecx, ref_Ecx)

    nf_metrics[freq]["Eco"] = (ratio_co, phase_co, nrms_co)
    nf_metrics[freq]["Ecx"] = (ratio_cx, phase_cx, nrms_cx)
    nf_metrics[freq]["nmax_safe"] = nmax_tr

    if freq_i == 0:
        nf_results = {"X": X, "Y": Y, "Eco": Eco, "ref_Eco": ref_Eco,
                      "Ecx": Ecx, "ref_Ecx": ref_Ecx, "freq": freq}
    print(f"  {freq/1e9:.4f} GHz — Eco scale={ratio_co:.4f}, nRMS={nrms_co:.2e}  "
          f"| Ecx scale={ratio_cx:.4f}, nRMS={nrms_cx:.2e}")


# ---- Figure 3: 2D near-field maps at 8.0 GHz ----
freq_nf = nf_results["freq"]
X0, Y0  = nf_results["X"], nf_results["Y"]
Eco0    = nf_results["Eco"]
ref0    = nf_results["ref_Eco"]

# Normalise to ref peak for display
ref_peak_nf = np.max(np.abs(ref0))
Eco_norm    = Eco0    / ref_peak_nf
ref_norm    = ref0    / ref_peak_nf

# Relative error (amplitude only)
amp_err = (np.abs(Eco_norm) - np.abs(ref_norm)) / (np.abs(ref_norm) + 1e-10)

# Clamp amplitude dB range
vmin_dB, vmax_dB = -40, 0

fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle(f"Near-Field Validation at {freq_nf/1e9:.4f} GHz, z = {Z_DISTANCE} m\n"
              "Co-pol amplitude (Ludwig-3 Eco)", fontsize=12)

ax = axes3[0]
im = ax.pcolormesh(X0 * 100, Y0 * 100,
                   np.clip(_db(ref_norm), vmin_dB, vmax_dB),
                   cmap="viridis", vmin=vmin_dB, vmax=vmax_dB)
plt.colorbar(im, ax=ax, label="dB (rel. peak)")
ax.set_title("GRASP reference |Eco|")
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_aspect("equal")

ax = axes3[1]
im = ax.pcolormesh(X0 * 100, Y0 * 100,
                   np.clip(_db(Eco_norm), vmin_dB, vmax_dB),
                   cmap="viridis", vmin=vmin_dB, vmax=vmax_dB)
plt.colorbar(im, ax=ax, label="dB (rel. peak)")
ax.set_title("SWE computed |Eco|")
ax.set_xlabel("x (cm)")
ax.set_aspect("equal")

ax = axes3[2]
err_pct = amp_err * 100  # percent
clim = max(0.1, np.percentile(np.abs(err_pct), 99))
im = ax.pcolormesh(X0 * 100, Y0 * 100, err_pct,
                   cmap="RdBu_r", vmin=-clim, vmax=clim)
plt.colorbar(im, ax=ax, label="Amplitude error (%)")
ax.set_title("Amplitude error: (SWE − ref) / ref")
ax.set_xlabel("x (cm)")
ax.set_aspect("equal")

fig3.tight_layout()
out3 = os.path.join(FIGS_DIR, "validation_nearfield_2d.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"  Saved: {out3}")


# ---- Figure 4: Horizontal slice through near-field for each frequency ----
fig4, axes4 = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
axes4_flat  = axes4.flatten()
fig4.suptitle("Near-Field Validation — Horizontal slice (y = 0)\n"
              "Co-pol amplitude |Eco| vs TICRA GRASP reference", fontsize=13)

for freq_i, freq in enumerate(freqs):
    ax = axes4_flat[freq_i]
    field = grd_data["fields"][freq_i]
    X0f, Y0f, _ = _build_nf_grid(field)

    # find row closest to y=0
    y_vals = Y0f[:, 0]
    row = np.argmin(np.abs(y_vals))
    x_slice = X0f[row, :] * 100   # cm

    # Recompute or use stored result
    swe_tr, _ = _truncated_swe(swe, freq)
    x_m    = x_slice / 100          # cm → m
    r_sl   = np.sqrt(x_m**2 + Z_DISTANCE**2)
    # theta = angle from +z axis; symmetric in x, only phi flips sign
    theta_sl = np.arctan2(np.abs(x_m), Z_DISTANCE)
    phi_sl   = np.where(x_slice >= 0, 0.0, np.pi)

    (_, Et_sl, Ep_sl), _ = swe_tr.near_field(r_sl, theta_sl, phi_sl,
                                              frequency=freq, normalize=False)
    Eco_sl = spherical_to_ludwig3(Et_sl, Ep_sl, phi_sl)[0]

    ref_Eco_sl = field["data"][row, :, 0]
    ref_pk = max(np.max(np.abs(ref_Eco_sl)), 1e-30)

    ratio_sl, _, nrms_sl = _scaling_metrics(Eco_sl, ref_Eco_sl)

    ax.plot(x_slice, _db(ref_Eco_sl / ref_pk), "k-",  lw=1.5, label="GRASP")
    ax.plot(x_slice, _db(Eco_sl     / ref_pk), "r--", lw=1.2, label="SWE")

    ax.set_title(f"{freq/1e9:.4f} GHz\nscale={ratio_sl:.4f}, nRMS={nrms_sl:.2e}",
                 fontsize=9)
    ax.set_ylim(-50, 3)
    ax.grid(True, alpha=0.4)
    ax.tick_params(labelsize=8)
    if freq_i % 3 == 0:
        ax.set_ylabel("|Eco| rel. (dB)", fontsize=8)
    if freq_i >= 6:
        ax.set_xlabel("x (cm)", fontsize=8)

handles, labels = axes4_flat[0].get_legend_handles_labels()
fig4.legend(handles, labels, loc="lower center", ncol=2, fontsize=10,
            bbox_to_anchor=(0.5, 0.01))
fig4.tight_layout(rect=[0, 0.04, 1, 0.97])

out4 = os.path.join(FIGS_DIR, "validation_nearfield_slice.png")
fig4.savefig(out4, dpi=150, bbox_inches="tight")
plt.close(fig4)
print(f"  Saved: {out4}")


# ===========================================================================
# 3.  COEFFICIENT EXTRACTION VALIDATION (from_far_field round-trip)
# ===========================================================================
print("\nComputing coefficient extraction validation …")

# Use 8.0 GHz.  Compute synthetic far-field on a regular grid from the
# reference SWE, then extract Q coefficients with from_far_field and compare
# against the originals.  This tests the round-trip:  Q → far-field → Q.
freq_ext = freqs[0]

# Build a 181×361 far-field grid
theta_1d = np.linspace(0, np.pi, 181)
phi_1d   = np.linspace(0, 2 * np.pi, 361)
TH, PH   = np.meshgrid(theta_1d, phi_1d, indexing="ij")

print(f"  Forward: computing far-field on {TH.size} pts at {freq_ext/1e9:.4f} GHz …")
Et_syn, Ep_syn = swe.far_field(TH.ravel(), PH.ravel(),
                               frequency=freq_ext, normalize=False)

print(f"  Inverse: running from_far_field (NMAX_initial=30) …")
swe_ext = SphericalWaveExpansion.from_far_field(
    TH.ravel(), PH.ravel(), Et_syn, Ep_syn,
    frequency=freq_ext,
    NMAX_initial=30,
    MMAX_initial=None,      # adaptive
    normalize=False,
    use_multiprocessing=False,
)

# Compare extracted vs reference Q coefficients
q1_ref = swe.Q1_coeffs(freq_ext)
q2_ref = swe.Q2_coeffs(freq_ext)
q1_ext = swe_ext.Q1_coeffs(freq_ext)
q2_ext = swe_ext.Q2_coeffs(freq_ext)

nmax_ref = swe.NMAX(freq_ext)
nmax_ext = swe_ext.NMAX(freq_ext)
print(f"  Reference NMAX={nmax_ref}, extracted NMAX={nmax_ext}")

# Gather shared modes (modes present in both)
shared_modes = sorted(set(q1_ref.keys()) & set(q1_ext.keys()))

q1_ref_arr = np.array([q1_ref[k] for k in shared_modes])
q1_ext_arr = np.array([q1_ext[k] for k in shared_modes])
q2_ref_arr = np.array([q2_ref[k] for k in shared_modes])
q2_ext_arr = np.array([q2_ext.get(k, 0) for k in shared_modes])

# Normalize each set by its own sqrt(total_power) before comparing shapes.
# This removes the .sph absolute-unit convention vs extraction-unit difference,
# and focuses the comparison on whether the mode spectrum shape is correct.
tp_ref = swe.total_power(freq_ext)
tp_ext = swe_ext.total_power(freq_ext)
q1_ref_n = q1_ref_arr / np.sqrt(tp_ref)
q2_ref_n = q2_ref_arr / np.sqrt(tp_ref)
q1_ext_n = q1_ext_arr / np.sqrt(tp_ext)
q2_ext_n = q2_ext_arr / np.sqrt(tp_ext)

# Scale and nRMS on normalized coefficients
s1, _, n1 = _scaling_metrics(q1_ext_n, q1_ref_n)
s2, _, n2 = _scaling_metrics(q2_ext_n, q2_ref_n)
print(f"  Q1 (normalized): scale={s1:.6f}, nRMS={n1:.2e}")
print(f"  Q2 (normalized): scale={s2:.6f}, nRMS={n2:.2e}")

# Power by n — reference vs extracted
def _power_by_n(q1d, q2d, nmax):
    pbn = np.zeros(nmax + 1)
    for (n, m), v in q1d.items():
        if n <= nmax:
            pbn[n] += abs(v)**2
    for (n, m), v in q2d.items():
        if n <= nmax:
            pbn[n] += abs(v)**2
    return pbn

n_plot = min(nmax_ref, 25)   # show reference up to n=25 for context
pbn_ref = _power_by_n(q1_ref, q2_ref, n_plot)
pbn_ext = _power_by_n(q1_ext, q2_ext, n_plot)

# ---- Figure 5: Coefficient extraction validation ----
fig5, axes5 = plt.subplots(1, 3, figsize=(16, 5))
fig5.suptitle(f"Coefficient Extraction Validation at {freq_ext/1e9:.4f} GHz\n"
              "from_far_field round-trip: Q → far-field → Q", fontsize=12)

# Panel A: |Q1| scatter — extracted vs reference (only significant modes)
ax = axes5[0]
mask_sig = np.abs(q1_ref_arr) > np.max(np.abs(q1_ref_arr)) * 1e-3
ax.scatter(np.abs(q1_ref_arr[mask_sig]), np.abs(q1_ext_arr[mask_sig]),
           s=12, alpha=0.6, color="steelblue", label=f"Q1 ({mask_sig.sum()} modes)")
mask_sig2 = np.abs(q2_ref_arr) > np.max(np.abs(q2_ref_arr)) * 1e-3
ax.scatter(np.abs(q2_ref_arr[mask_sig2]), np.abs(q2_ext_arr[mask_sig2]),
           s=12, alpha=0.6, color="tomato", marker="^", label=f"Q2 ({mask_sig2.sum()} modes)")
xlim = max(np.max(np.abs(q1_ref_arr[mask_sig])), np.max(np.abs(q2_ref_arr[mask_sig2])))
ax.plot([0, xlim], [0, xlim], "k--", lw=0.8, label="y = x")
ax.set_xlabel("|Q| reference (.sph)")
ax.set_ylabel("|Q| extracted (from_far_field)")
ax.set_title(f"Q magnitude scatter\nQ1 scale={s1:.5f}, nRMS={n1:.2e}")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)

# Panel B: Mode power by n
# Reference shown up to n=25; extraction shown up to nmax_ext.
# Modes above nmax_ext are not in the extracted set (extraction stopped there).
ax = axes5[1]
ns_ref = np.arange(1, n_plot + 1)
ns_ext = np.arange(1, nmax_ext + 1)
ax.semilogy(ns_ref, pbn_ref[1:n_plot+1], "k-o", ms=5, lw=1.5, label="Reference (.sph)")
ax.semilogy(ns_ext, pbn_ext[1:nmax_ext+1], "r--s", ms=4, lw=1.2,
            label=f"Extracted (NMAX={nmax_ext})")
ax.axvline(nmax_ext, color="red", lw=0.8, ls=":", alpha=0.7, label=f"Extraction limit (n={nmax_ext})")
ax.set_xlabel("Degree n")
ax.set_ylabel("Mode power (|Q1|² + |Q2|²)")
ax.set_title("Power by degree n\n(extraction stops at noise floor)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)
ax.set_xlim(0, n_plot + 1)

# Panel C: Per-mode |Q1| vs n for all significant modes
ax = axes5[2]
n_vals  = np.array([k[0] for k in shared_modes])
m_vals  = np.array([k[1] for k in shared_modes])
sig_any = (np.abs(q1_ref_arr) > np.max(np.abs(q1_ref_arr)) * 1e-4)
sc = ax.scatter(n_vals[sig_any], np.abs(q1_ref_arr[sig_any]),
                c=np.abs(m_vals[sig_any]), cmap="plasma",
                s=15, alpha=0.7, label="Reference |Q1|")
ax.scatter(n_vals[sig_any], np.abs(q1_ext_arr[sig_any]),
           marker="x", s=15, alpha=0.7, color="red", label="Extracted |Q1|")
plt.colorbar(sc, ax=ax, label="|m|")
ax.set_xlabel("Degree n")
ax.set_ylabel("|Q1|")
ax.set_title("Q1 magnitude vs n (colour = |m|)")
ax.set_yscale("log")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)

fig5.tight_layout()
out5 = os.path.join(FIGS_DIR, "validation_coeff_extraction.png")
fig5.savefig(out5, dpi=150, bbox_inches="tight")
plt.close(fig5)
print(f"  Saved: {out5}")


# ---- Figure 6: Scaling ratio summary (bar chart) ----
fig6, axes6 = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig6.suptitle("Validation Accuracy Summary: SWE vs TICRA GRASP\nAll 9 frequencies",
              fontsize=12)

x  = np.arange(n_freqs)
bw = 0.35
freq_labels = [f"{f:.4f}" for f in freq_GHz]

for ax, field_type, metrics_dict in [
    (axes6[0], "Far-field", ff_metrics),
    (axes6[1], "Near-field (z=0.25 m)", nf_metrics),
]:
    ratios_co = [metrics_dict[f]["Eco"][0] for f in freqs]
    ratios_cx = [metrics_dict[f]["Ecx"][0] for f in freqs]

    ax.bar(x - bw/2, ratios_co, bw, label="Eco (co-pol)", color="steelblue")
    ax.bar(x + bw/2, ratios_cx, bw, label="Ecx (cross-pol)", color="tomato")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.axhspan(0.99, 1.01, alpha=0.08, color="green", label="±1% band")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f:.3f}" for f in freq_GHz], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Frequency (GHz)", fontsize=10)
    ax.set_ylabel("Best-fit scale ratio", fontsize=10)
    ax.set_title(field_type, fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(0.9, 1.1)
    ax.grid(axis="y", alpha=0.4)

fig6.tight_layout()
out6 = os.path.join(FIGS_DIR, "validation_scaling_summary.png")
fig6.savefig(out6, dpi=150, bbox_inches="tight")
plt.close(fig6)
print(f"  Saved: {out6}")


# ===========================================================================
# 4.  Print Markdown summary table
# ===========================================================================
print("\n\n" + "=" * 78)
print("VALIDATION SUMMARY — SWE vs TICRA GRASP")
print("=" * 78)

print("\n### Far-Field (all cuts, 37 phi cuts per frequency)\n")
header = (f"{'Freq (GHz)':>12} | {'Eco scale':>10} | {'Eco phase°':>10} | "
          f"{'Eco nRMS':>10} | {'Ecx scale':>10} | {'Ecx nRMS':>10}")
sep    = "-" * len(header)
print(header)
print(sep)
for freq in freqs:
    rc, pc, nc = ff_metrics[freq]["Eco"]
    rx, px, nx_ = ff_metrics[freq]["Ecx"]
    print(f"{freq/1e9:12.4f} | {rc:10.6f} | {pc:10.4f} | {nc:10.2e} | "
          f"{rx:10.6f} | {nx_:10.2e}")

print("\n### Near-Field (planar grid, z = 0.25 m)\n")
print(header)
print(sep)
for freq in freqs:
    rc, pc, nc = nf_metrics[freq]["Eco"]
    rx, px, nx_ = nf_metrics[freq]["Ecx"]
    ns = nf_metrics[freq]["nmax_safe"]
    print(f"{freq/1e9:12.4f} | {rc:10.6f} | {pc:10.4f} | {nc:10.2e} | "
          f"{rx:10.6f} | {nx_:10.2e}   (NMAX_safe={ns})")

print("\n### Coefficient Extraction Round-Trip at 8.0000 GHz (from_far_field)\n")
print(f"  Reference NMAX={nmax_ref}, extracted NMAX={nmax_ext}")
print(f"  {'':12} | {'Scale ratio':>12} | {'nRMS':>12}")
print(f"  {'-'*42}")
print(f"  {'Q1':12} | {s1:12.6f} | {n1:12.2e}")
print(f"  {'Q2':12} | {s2:12.6f} | {n2:12.2e}")

print("\n" + "=" * 78)
print(f"\nFigures saved to: {FIGS_DIR}")
print("  validation_farfield_eplane.png   — E-plane cuts, all 9 frequencies")
print("  validation_farfield_cuts.png     — phi=0/45/90 cuts at 8.0 GHz")
print("  validation_nearfield_2d.png      — 2D amplitude maps at 8.0 GHz")
print("  validation_nearfield_slice.png   — Horizontal slice, all 9 frequencies (fixed)")
print("  validation_coeff_extraction.png  — Q coefficient round-trip validation")
print("  validation_scaling_summary.png   — Scaling ratio bar chart summary")
