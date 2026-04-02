# Spherical Wave Expansion

A Python package for Spherical Wave Expansion (SWE) analysis of electromagnetic
fields, following the Hansen convention as used by TICRA GRASP.

## Features

- **Read/write TICRA file formats** — `.sph` (spherical wave coefficients),
  `.cut` (far-field pattern cuts), `.grd` (near-field grids)
- **Multi-frequency support** — a single `SphericalWaveExpansion` object holds
  all frequencies from a file; every compute method accepts an explicit
  `frequency` argument
- **Far-field pattern calculation** — E_θ and E_φ at absolute field levels (V/m)
  from Q coefficients
- **Near-field calculation** — full E and H vectors at arbitrary points in space
- **Cartesian near-field** — direct (E_x, E_y, E_z) output via coordinate
  conversion
- **Surface current extraction** — equivalent J and M currents on arbitrary
  surfaces using the Surface Equivalence Theorem
- **Ludwig-3 polarization** — convert between spherical (E_θ/E_φ) and
  Ludwig-3 (Eco/Ecx) representations, matching GRASP's ICOMP=3 convention
- **SWE coefficient extraction** — recover Q coefficients from a full-sphere
  far-field grid via orthogonality integration (`from_far_field`)
- **Coordinate utilities** — Cartesian ↔ spherical, field vector rotation

## Installation

```bash
# From the repository root
pip install -e .

# Or set PYTHONPATH if editable install fails
PYTHONPATH=. python your_script.py
```

**Dependencies**: numpy, scipy, matplotlib (optional, for validation plots)

## Quick Start

```python
from swe import SphericalWaveExpansion
import numpy as np

# Load a multi-frequency TICRA .sph file
swe = SphericalWaveExpansion.from_sph_file("antenna.sph", normalize=False)

print(swe.frequencies)           # [8.0e9, 8.0625e9, ...]  (Hz)

freq = swe.frequencies[0]        # pick first frequency
print(f"NMAX = {swe.NMAX(freq)}, MMAX = {swe.MMAX(freq)}")

# Far-field pattern at one frequency
theta = np.linspace(0, np.pi, 181)
phi   = np.zeros_like(theta)
E_theta, E_phi = swe.far_field(theta, phi, frequency=freq, normalize=False)

# Near-field E and H at a point
(E_r, E_t, E_p), (H_r, H_t, H_p) = swe.near_field(
    np.array([0.5]), np.array([np.pi/4]), np.array([0.0]),
    frequency=freq, normalize=False
)
```

## Usage Examples

### Load a multi-frequency .sph file and plot an E-plane cut

```python
import numpy as np
import matplotlib.pyplot as plt
from swe import SphericalWaveExpansion
from swe.ludwig3 import spherical_to_ludwig3

swe = SphericalWaveExpansion.from_sph_file("antenna.sph", normalize=False)

theta = np.linspace(0, np.pi, 361)
phi   = np.zeros_like(theta)

fig, ax = plt.subplots()
for freq in swe.frequencies:
    E_theta, E_phi = swe.far_field(theta, phi, frequency=freq, normalize=False)
    Eco, _         = spherical_to_ludwig3(E_theta, E_phi, phi)
    peak_dB        = 20 * np.log10(np.max(np.abs(Eco)))
    Eco_dB         = 20 * np.log10(np.abs(Eco)) - peak_dB
    ax.plot(np.degrees(theta), Eco_dB, label=f"{freq/1e9:.4f} GHz")

ax.set_xlabel("θ (degrees)")
ax.set_ylabel("Normalised amplitude (dB)")
ax.set_ylim(-50, 3)
ax.legend()
ax.grid(True)
plt.show()
```

### Compute near-field on a planar grid

```python
import numpy as np
from swe import SphericalWaveExpansion, cartesian_to_spherical
from swe.ludwig3 import spherical_to_ludwig3

swe = SphericalWaveExpansion.from_sph_file("antenna.sph", normalize=False)
freq = swe.frequencies[0]

# Build a 0.5 m × 0.5 m planar grid at z = 0.25 m
x = np.linspace(-0.25, 0.25, 101)
y = np.linspace(-0.25, 0.25, 101)
X, Y = np.meshgrid(x, y)
Z    = np.full_like(X, 0.25)

# IMPORTANT: truncate NMAX to avoid Bessel function overflow at finite distances
k        = swe.k(freq)
kr_min   = k * 0.25
nmax_safe = max(int(kr_min) - 5, swe.MMAX(freq))

q1 = {k: v for k, v in swe.Q1_coeffs(freq).items() if k[0] <= nmax_safe}
q2 = {k: v for k, v in swe.Q2_coeffs(freq).items() if k[0] <= nmax_safe}
swe_trunc = SphericalWaveExpansion(
    Q1_coeffs={freq: q1}, Q2_coeffs={freq: q2},
    NMAX={freq: nmax_safe}, MMAX={freq: swe.MMAX(freq)},
)

r, theta, phi = cartesian_to_spherical(X.ravel(), Y.ravel(), Z.ravel())
(_, E_theta, E_phi), _ = swe_trunc.near_field(
    r, theta, phi, frequency=freq, normalize=False
)
Eco, _ = spherical_to_ludwig3(E_theta, E_phi, phi)
Eco    = Eco.reshape(X.shape)
```

### Work with multiple frequencies

```python
swe = SphericalWaveExpansion.from_sph_file("antenna.sph", normalize=False)

for freq in swe.frequencies:
    print(f"\n{freq/1e9:.4f} GHz")
    print(f"  NMAX = {swe.NMAX(freq)}, MMAX = {swe.MMAX(freq)}")
    print(f"  Total power = {swe.total_power(freq):.4f}")
    print(f"  Wavelength  = {swe.wavelength(freq)*1000:.2f} mm")
```

### Read GRASP .cut and .grd files

```python
from swe.ticra_io import read_grasp_cut, read_grasp_grd
from swe.ludwig3 import extract_cut_frequency_set, remap_negative_theta

# .cut file: list of phi cuts, grouped by frequency
cuts = read_grasp_cut("pattern.cut")
print(f"{len(cuts['cuts'])} cuts total")

# Extract one frequency block (37 cuts per frequency)
freq_cuts = extract_cut_frequency_set(cuts, freq_index=0, n_phi=37)
cut0 = freq_cuts["cuts"][0]

theta_deg = cut0["v_ini"] + np.arange(cut0["v_num"]) * cut0["v_inc"]
theta_rad, phi_rad = remap_negative_theta(theta_deg, cut0["constant"])
Eco_ref = cut0["data"][:, 0]   # complex Ludwig-3 Eco

# .grd file: list of 2D near-field grids
grd = read_grasp_grd("nearfield.grd")
field = grd["fields"][0]        # first frequency set
print(f"Grid: {field['nx']} × {field['ny']} points")
Eco_2d = field["data"][:, :, 0]  # complex Eco, shape (ny, nx)
```

### Save to TICRA .sph file

```python
swe.to_sph_file("output.sph")
```

## Validation Against TICRA GRASP

The package is validated against TICRA GRASP reference data for a horn
antenna operating across 9 frequencies (8.0 – 8.5 GHz, 62.5 MHz steps).

### Reference files (in `tests/`)

| File | Format | Description |
|------|--------|-------------|
| `example.sph` | TICRA `.sph` | SWE coefficients; NMAX=359, MMAX=35, 24,299 modes per frequency; 9 frequencies |
| `example.cut` | TICRA `.cut` | Far-field (Ludwig-3 Eco/Ecx); 37 φ cuts × 9 frequencies; θ ∈ [−180°, +180°] |
| `example.grd` | TICRA `.grd` | Near-field (Ludwig-3 Eco/Ecx); 101×101 points; x,y ∈ [−0.5, 0.5] m; z = 0.25 m; 9 sets |

### Accuracy summary

Comparison uses a best-fit complex scale factor between the SWE output and the
GRASP reference. A scale ratio of 1.00 and zero phase offset indicate perfect
agreement.

**Far-field** (37 phi cuts concatenated per frequency):

| Freq (GHz) | Eco scale | Eco phase (°) | Eco nRMS | Ecx scale | Ecx nRMS |
|:----------:|:---------:|:-------------:|:--------:|:---------:|:--------:|
| 8.0000 | 0.9941 | 0.23 | 1.7e-02 | 0.9837 | 8.0e-02 |
| 8.0625 | 0.9939 | 0.26 | 1.8e-02 | 0.9818 | 8.5e-02 |
| 8.1250 | 0.9976 | 0.08 | 9.2e-03 | 0.9938 | 5.0e-02 |
| 8.1875 | 0.9974 | 0.08 | 9.8e-03 | 0.9931 | 5.3e-02 |
| 8.2500 | 0.9968 | 0.09 | 1.1e-02 | 0.9924 | 5.4e-02 |
| 8.3125 | 0.9964 | 0.12 | 1.1e-02 | 0.9918 | 5.4e-02 |
| 8.3750 | 0.9963 | 0.14 | 1.1e-02 | 0.9917 | 5.2e-02 |
| 8.4375 | 0.9961 | 0.15 | 1.0e-02 | 0.9915 | 4.9e-02 |
| 8.5000 | 0.9957 | 0.15 | 1.1e-02 | 0.9908 | 5.0e-02 |

**Near-field** (planar grid, z = 0.25 m; modes truncated to n < kr_min):

| Freq (GHz) | Eco scale | Eco phase (°) | Eco nRMS | Ecx scale | Ecx nRMS |
|:----------:|:---------:|:-------------:|:--------:|:---------:|:--------:|
| 8.0000 | 0.9967 | −0.53 | 6.3e-02 | 1.0115 | 1.5e-01 |
| 8.0625 | 0.9965 | −0.53 | 6.0e-02 | 1.0084 | 1.4e-01 |
| 8.1250 | 0.9955 | −0.53 | 5.7e-02 | 1.0027 | 1.4e-01 |
| 8.1875 | 0.9944 | −0.53 | 5.7e-02 | 1.0011 | 1.5e-01 |
| 8.2500 | 0.9945 | −0.51 | 6.1e-02 | 1.0081 | 1.6e-01 |
| 8.3125 | 0.9956 | −0.49 | 6.3e-02 | 1.0147 | 1.6e-01 |
| 8.3750 | 0.9958 | −0.49 | 6.1e-02 | 1.0126 | 1.6e-01 |
| 8.4375 | 0.9952 | −0.50 | 5.8e-02 | 1.0065 | 1.5e-01 |
| 8.5000 | 0.9947 | −0.50 | 5.8e-02 | 1.0037 | 1.5e-01 |

The near-field nRMS for the co-pol (Eco) is ~6 % after scale correction,
reflecting the mode truncation required to prevent Bessel function overflow
at the relatively close observation plane (z = 0.25 m, kr_min ≈ 42).

### Validation plots

Run the validation script to regenerate all figures:

```bash
python docs/generate_validation.py
```

Figures are saved to `docs/figures/`:

| File | Description |
|------|-------------|
| `validation_farfield_eplane.png` | E-plane (φ = 0°) cuts, all 9 frequencies |
| `validation_farfield_cuts.png` | φ = 0°/45°/90° cuts at 8.0 GHz |
| `validation_nearfield_2d.png` | 2D amplitude maps at 8.0 GHz (ref / SWE / error) |
| `validation_nearfield_slice.png` | Horizontal slice, all 9 frequencies |
| `validation_scaling_summary.png` | Scaling ratio bar chart for all frequencies |

### Running the test suite

```bash
cd spherical_wave_expansion
python -m pytest tests/ -v -s
```

Tests that require the example data files are automatically skipped when the
files are absent.

## Physical Conventions

| Quantity | Convention |
|---------|-----------|
| Time dependence | exp(+jωt) — matches TICRA GRASP |
| Outgoing wave | Hankel function of the second kind h_n^(2) |
| Frequency | Hz |
| Wavenumber | k = 2πf/c (rad/m) |
| Electric field | V/m |
| Magnetic field | A/m |
| Impedance | η₀ = 376.73 Ω |

## API Reference

### `SphericalWaveExpansion`

**Constructor** — typically use a class method instead:

```python
SphericalWaveExpansion(
    Q1_coeffs={freq_Hz: {(n, m): complex}},
    Q2_coeffs={freq_Hz: {(n, m): complex}},
    NMAX={freq_Hz: int},   # optional, auto-detected
    MMAX={freq_Hz: int},   # optional, auto-detected
)
```

**Class methods**:

| Method | Description |
|--------|-------------|
| `from_sph_file(filename, normalize=True)` | Load all frequencies from a TICRA `.sph` file |
| `from_far_field(theta, phi, E_theta, E_phi, frequency, ...)` | Extract coefficients from a full-sphere far-field grid |

**Per-frequency accessors**:

| Method / property | Description |
|-------------------|-------------|
| `frequencies` | Sorted list of loaded frequencies in Hz |
| `Q1_coeffs(freq)` | `{(n,m): complex}` dict of Q₁ (TE) coefficients |
| `Q2_coeffs(freq)` | `{(n,m): complex}` dict of Q₂ (TM) coefficients |
| `NMAX(freq)` | Maximum degree n |
| `MMAX(freq)` | Maximum order m |
| `k(freq)` | Wavenumber in rad/m |
| `wavelength(freq)` | Wavelength in metres |
| `total_power(freq)` | Σ(\|Q1\|²+\|Q2\|²) |

**Compute methods**:

| Method | Description |
|--------|-------------|
| `far_field(theta, phi, frequency, normalize=True)` | E_θ, E_φ in V/m |
| `near_field(r, theta, phi, frequency, normalize=True)` | (E_r, E_θ, E_φ), (H_r, H_θ, H_φ) |
| `near_field_cartesian(x, y, z, frequency, normalize=True)` | (E_x, E_y, E_z), (H_x, H_y, H_z) |
| `currents_on_surface(rr, unr, dSr, frequency, ...)` | Equivalent J and M currents (A, V) |
| `normalize_coefficients(freq=None)` | Normalize Q so total_power = 1 |
| `add_frequency(other)` | Merge another SWE object's frequencies |
| `to_sph_file(filename, ...)` | Write all frequencies to TICRA `.sph` |

### `swe.ludwig3`

| Function | Description |
|----------|-------------|
| `spherical_to_ludwig3(E_theta, E_phi, phi)` | → (Eco, Ecx) |
| `ludwig3_to_spherical(Eco, Ecx, phi)` | → (E_theta, E_phi) |
| `remap_negative_theta(theta_deg, phi_deg)` | Map GRASP negative-θ cuts to (θ, φ+180°) |
| `extract_cut_frequency_set(cut_data, freq_index, n_phi)` | Extract one frequency block from a multi-frequency `.cut` |

### `swe.ticra_io`

| Function | Description |
|----------|-------------|
| `read_grasp_cut(filename)` | Read `.cut` far-field pattern cuts |
| `write_grasp_cut(filename, cuts, ...)` | Write `.cut` file |
| `read_grasp_grd(filename)` | Read `.grd` near-field grid |
| `write_grasp_grd(filename, grd_data)` | Write `.grd` file |
| `cut_to_fields(cut_data)` | Extract (θ, φ, E_θ, E_φ) arrays from cut dict |

### Low-level functions (`swe.core`)

| Function | Description |
|----------|-------------|
| `read_ticra_sph(filename)` | Read `.sph` → list of per-frequency dicts |
| `write_ticra_sph(filename, freq_data_list, ...)` | Write `.sph` from list of dicts |
| `cartesian_to_spherical(x, y, z)` | → (r, θ, φ) |
| `spherical_to_cartesian(r, theta, phi)` | → (x, y, z) |
| `spherical_to_cartesian_field(E_r, E_t, E_p, theta, phi)` | Rotate field vectors |
| `normalized_associated_legendre(n, m, theta)` | Compute P̄_n^m and its θ-derivative |
| `far_field_pattern_functions(n, m, theta, phi, ...)` | K₁, K₂ pattern functions |

## References

1. J. E. Hansen (Ed.), *Spherical Near-Field Antenna Measurements*,
   IEE Electromagnetic Waves Series 26, Peter Peregrinus Ltd., London, 1988.
   *(Primary reference for the Hansen formulation, conventions, and normalization.)*

2. IEEE Std 1720-2012, *IEEE Recommended Practice for Near-Field Antenna
   Measurements*, IEEE, 2012.

3. TICRA, *GRASP Technical Description*, Copenhagen, Denmark.
   *(Specification of the `.sph`, `.cut`, and `.grd` file formats.)*

4. A. C. Ludwig, "The definition of cross polarization,"
   *IEEE Trans. Antennas Propagat.*, vol. AP-21, pp. 116–119, Jan. 1973.
   *(Ludwig-3 polarization definition used in `swe.ludwig3`.)*

## License

MIT License
