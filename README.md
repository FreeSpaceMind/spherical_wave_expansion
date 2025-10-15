# Spherical Wave Expansion

A Python package for spherical wave expansion (SWE) analysis of electromagnetic fields, with applications in antenna pattern measurements and near-field to far-field transformations.

## Features

- **Read/Write TICRA .sph files** - Industry-standard spherical wave coefficient format
- **Far-field pattern calculation** - Compute E_θ and E_φ from Q coefficients
- **Near-field calculations** - Full E and H fields at arbitrary points in space
- **Surface current extraction** - Compute equivalent J and M currents on surfaces
- **Multiple measurement geometries**:
  - Spherical near-field measurements
  - Planar near-field measurements
  - Cylindrical near-field measurements
  - Far-field pattern measurements
- **Coordinate system utilities** - Convert between Cartesian, spherical, and cylindrical
- **Least-squares fitting** - Extract Q coefficients from field measurements

## Installation

### From PyPI (once published)
```bash
pip install spherical-wave-expansion
```

### From source
```bash
git clone https://github.com/yourusername/spherical-wave-expansion.git
cd spherical-wave-expansion
pip install -e .
```

### Development installation
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from swe import SphericalWaveExpansion
import numpy as np

# Load from TICRA .sph file
swe = SphericalWaveExpansion.from_sph_file("antenna.sph")

# Calculate far-field pattern
theta = np.linspace(0, np.pi, 181)
phi = np.linspace(0, 2*np.pi, 361)
theta_grid, phi_grid = np.meshgrid(theta, phi)
E_theta, E_phi = swe.far_field(theta_grid, phi_grid)

# Calculate near-field at specific points
x, y, z = 0.1, 0.2, 0.3  # meters
(E_x, E_y, E_z), (H_x, H_y, H_z) = swe.near_field_cartesian(x, y, z)

# Create SWE from planar near-field measurements
swe_fitted = SphericalWaveExpansion.from_planar_near_field(
    x=x_scan, y=y_scan, z_scan=0.3,
    E_x=E_x_meas, E_y=E_y_meas,
    frequency=10e9, NMAX=15
)

# Save to .sph file
swe_fitted.to_sph_file("output.sph")
```

## Usage Examples

### Example 1: Load and visualize antenna pattern

```python
import matplotlib.pyplot as plt
from swe import SphericalWaveExpansion
import numpy as np

# Load antenna data
swe = SphericalWaveExpansion.from_sph_file("horn_antenna.sph")
print(f"Frequency: {swe.frequency/1e9:.2f} GHz")
print(f"Modes: NMAX={swe.NMAX}, MMAX={swe.MMAX}")

# Calculate far-field pattern
theta = np.linspace(0, np.pi, 181)
phi_cut = 0.0  # E-plane cut
E_theta, E_phi = swe.far_field(theta, np.full_like(theta, phi_cut))

# Plot
E_total_dB = 20 * np.log10(np.abs(E_theta) / np.max(np.abs(E_theta)))
plt.plot(np.degrees(theta), E_total_dB)
plt.xlabel('Theta (degrees)')
plt.ylabel('Normalized Gain (dB)')
plt.title(f'Antenna Pattern at φ={phi_cut}°')
plt.grid(True)
plt.ylim([-40, 0])
plt.show()
```

### Example 2: Near-field to far-field transformation

```python
from swe import SphericalWaveExpansion
import numpy as np

# Spherical near-field measurements on 0.5m radius sphere
r_meas = 0.5
N_meas = 500
theta_meas = np.random.uniform(0, np.pi, N_meas)
phi_meas = np.random.uniform(0, 2*np.pi, N_meas)

# Your measured data
E_theta_meas = ...  # complex array of measurements
E_phi_meas = ...

# Fit spherical wave expansion
swe = SphericalWaveExpansion.from_spherical_near_field(
    r=r_meas,
    theta=theta_meas,
    phi=phi_meas,
    E_theta=E_theta_meas,
    E_phi=E_phi_meas,
    frequency=10e9,
    NMAX=20  # Higher order for better accuracy
)

# Transform to far field
theta_ff = np.linspace(0, np.pi, 361)
phi_ff = np.linspace(0, 2*np.pi, 721)
theta_grid, phi_grid = np.meshgrid(theta_ff, phi_ff)
E_theta_ff, E_phi_ff = swe.far_field(theta_grid, phi_grid)

# Save results
swe.to_sph_file("antenna_swe.sph")
```

### Example 3: Planar scanner data processing

```python
from swe import SphericalWaveExpansion
import numpy as np

# Rectangular planar scan
x_scan = np.linspace(-0.5, 0.5, 51)
y_scan = np.linspace(-0.5, 0.5, 51)
X, Y = np.meshgrid(x_scan, y_scan)
z_scan = 0.3  # 30 cm from antenna

# Your measured data (flattened)
E_x_meas = ...  # shape: (51*51,)
E_y_meas = ...

# Process planar near-field data
swe = SphericalWaveExpansion.from_planar_near_field(
    x=X.flatten(),
    y=Y.flatten(),
    z_scan=z_scan,
    E_x=E_x_meas,
    E_y=E_y_meas,
    frequency=28e9,  # 28 GHz
    NMAX=25,
    origin_offset=(0., 0., -0.1)  # Antenna is 10 cm behind origin
)

# Calculate surface currents on antenna aperture
rho = 0.05  # 5 cm radius
theta_surf = np.linspace(0, np.pi/2, 50)
phi_surf = np.linspace(0, 2*np.pi, 100)
theta_grid, phi_grid = np.meshgrid(theta_surf, phi_surf)
r_grid = np.full_like(theta_grid, rho)

(J_r, J_theta, J_phi), (M_r, M_theta, M_phi) = swe.surface_currents(
    r_grid, theta_grid, phi_grid
)
```

## Physical Conventions

- **Time dependence**: exp(+iωt)
- **Frequency**: Hz
- **Wavenumber**: k = 2πf/c (rad/m)
- **Electric field**: V/m
- **Magnetic field**: A/m
- **Impedance**: η₀ = 376.73 Ω (free space)

## Coordinate Systems

**Spherical (r, θ, φ)**:
- r: radial distance from origin (m)
- θ: polar angle from +z axis (radians, 0 to π)
- φ: azimuthal angle from +x axis (radians, 0 to 2π)

**Cartesian (x, y, z)**:
- Right-handed coordinate system
- x: →, y: ↑ (on page), z: ⊙ (out of page)

**Cylindrical (ρ, φ, z)**:
- ρ: radial distance from z-axis (m)
- φ: azimuthal angle from +x axis (radians)
- z: height along z-axis (m)

## API Reference

### SphericalWaveExpansion Class

**Constructor**:
```python
swe = SphericalWaveExpansion(Q1_coeffs, Q2_coeffs, frequency, NMAX, MMAX)
```

**Class Methods**:
- `from_sph_file(filename)` - Load from TICRA .sph file
- `from_far_field(theta, phi, E_theta, E_phi, frequency, NMAX)` - Fit from far-field
- `from_spherical_near_field(r, theta, phi, E_theta, E_phi, frequency, NMAX)` - Fit from spherical NF
- `from_planar_near_field(x, y, z_scan, E_x, E_y, frequency, NMAX)` - Fit from planar NF
- `from_cylindrical_near_field(rho, phi, z, E_rho, E_z, frequency, NMAX)` - Fit from cylindrical NF

**Instance Methods**:
- `far_field(theta, phi)` - Calculate far-field E_theta, E_phi
- `near_field(r, theta, phi)` - Calculate near-field (E_r, E_θ, E_φ), (H_r, H_θ, H_φ)
- `near_field_cartesian(x, y, z)` - Calculate near-field in Cartesian coordinates
- `surface_currents(r, theta, phi)` - Calculate equivalent surface currents
- `to_sph_file(filename)` - Save to TICRA .sph file

**Properties**:
- `frequency` - Frequency in Hz
- `k` - Wavenumber in rad/m
- `wavelength` - Wavelength in m
- `NMAX` - Maximum degree
- `MMAX` - Maximum order

## File Format Support

The package reads and writes **TICRA .sph files**, which contain:
- Spherical wave coefficients Q₁ₘₙ and Q₂ₘₙ
- Frequency information
- Mode truncation parameters (NMAX, MMAX)

## References

1. J.E. Hansen, "Spherical Near-Field Antenna Measurements" (1988)
2. IEEE Std 1720-2012: "Recommended Practice for Near-Field Antenna Measurements"
3. TICRA documentation for .sph file format

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{spherical_wave_expansion,
  author = {Your Name},
  title = {Spherical Wave Expansion: A Python package for antenna pattern analysis},
  year = {2025},
  url = {https://github.com/yourusername/spherical-wave-expansion}
}
```