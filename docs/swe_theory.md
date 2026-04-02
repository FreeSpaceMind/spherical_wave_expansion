# Spherical Wave Expansion — Theory & Algorithms

A comprehensive technical reference for the `spherical_wave_expansion` package.

---

## 1. Introduction

The Spherical Wave Expansion (SWE) is a rigorous technique for representing
electromagnetic fields radiated or scattered by antennas as a superposition of
orthogonal spherical wave modes. Every mode is uniquely identified by three
indices:

| Index | Symbol | Range | Meaning |
|-------|--------|-------|---------|
| Wave type | $s$ | 1, 2 | 1 = TE (transverse electric), 2 = TM (transverse magnetic) |
| Azimuthal order | $m$ | $-N_{\max} \ldots N_{\max}$ | Angular variation in $\phi$ |
| Polar degree | $n$ | $\max(1,\lvert m\rvert) \ldots N_{\max}$ | Angular variation in $\theta$ |

Any physically realisable field on and outside a minimum sphere enclosing the
antenna can be written as

$$\mathbf{E}(r,\theta,\phi) = \sum_{s=1}^{2}\sum_{n=1}^{N_{\max}}\sum_{m=-n}^{n} Q_{smn}\,\mathbf{F}_{smn}(r,\theta,\phi)$$

where $Q_{smn}$ are complex-valued **spherical wave coefficients** and
$\mathbf{F}_{smn}$ are the vector spherical wave functions.

This package implements the **Hansen formulation** as described in
*Spherical Near-Field Antenna Measurements* (J. E. Hansen, 1988), with
extensions for coefficient extraction, near-field reconstruction, and surface
current computation.

---

## 2. Physical Conventions

All equations and implementations in this package adhere to the following
conventions.

### Time Convention

The suppressed time dependence is

$$e^{j\omega t}$$

This matches the **TICRA** sign convention used in GRASP and EditSph. The
outgoing spherical wave therefore uses the Hankel function of the **second**
kind (see Section 5).

### Units and Constants

| Quantity | Symbol | Unit |
|----------|--------|------|
| Frequency | $f$ | Hz |
| Free-space speed of light | $c$ | $2.99792458 \times 10^{8}\;\text{m/s}$ |
| Wavenumber | $k = 2\pi f / c$ | rad/m |
| Electric field | $\mathbf{E}$ | V/m |
| Magnetic field | $\mathbf{H}$ | A/m |
| Free-space impedance | $\eta_0$ | $376.73\;\Omega$ |

### Coordinate System

The package uses the standard spherical coordinate system:

| Coordinate | Symbol | Range | Definition |
|------------|--------|-------|------------|
| Radial distance | $r$ | $[0, \infty)$ | Distance from origin |
| Polar angle | $\theta$ | $[0, \pi]$ | Angle from $+z$ axis |
| Azimuthal angle | $\phi$ | $[0, 2\pi)$ | Angle from $+x$ axis in $xy$-plane |

The unit vectors $(\hat{r}, \hat{\theta}, \hat{\phi})$ form a right-handed
orthonormal triad at every point.

---

## 3. Normalized Associated Legendre Functions

The angular dependence of every spherical wave mode is governed by the
associated Legendre functions. This package uses **Hansen-normalised** forms,
computed via stable recurrence relations.

### 3.1 Normalization Factor

The Hansen normalization factor is

$$N_n^m = \sqrt{\frac{2n+1}{2} \cdot \frac{(n-m)!}{(n+m)!}}$$

The normalised associated Legendre function is then

$$\bar{P}_n^m(\cos\theta) = N_n^m \, P_n^m(\cos\theta)$$

where $P_n^m$ is the conventional (unnormalised) associated Legendre function.
This normalization ensures that the angular integrals of the pattern functions
yield simple orthogonality relations.

### 3.2 Recurrence Relations

The functions are computed without ever evaluating the factorials explicitly.
Three recurrence stages are used.

**Seed value** ($n = m$):

$$\bar{P}_m^m(\cos\theta) = (-1)^m \, (2m-1)!! \, \sin^m\theta \cdot N_m^m$$

where $(2m-1)!! = 1 \cdot 3 \cdot 5 \cdots (2m-1)$ is the double factorial.

**First step** ($n = m + 1$):

$$\bar{P}_{m+1}^m(\cos\theta) = (2m+1)\cos\theta \cdot \bar{P}_m^m(\cos\theta) \cdot \frac{N_{m+1}^m}{N_m^m}$$

**General recurrence** ($n \ge m + 2$):

$$(n - m)\,\bar{P}_n^m = (2n - 1)\cos\theta \cdot \bar{P}_{n-1}^m - (n + m - 1)\,\bar{P}_{n-2}^m$$

where appropriate normalization ratios are absorbed at each step to maintain
numerical stability.

### 3.3 Derivative Recurrence

The $\theta$-derivative needed by the pattern functions is computed as

$$\frac{d\bar{P}_n^m}{d\theta} = \frac{n\cos\theta \cdot \bar{P}_n^m(\cos\theta) - (n+m)\,\bar{P}_{n-1}^m(\cos\theta)}{\sin\theta}$$

This recurrence avoids forming large intermediate products and remains stable
for high $n$.

### 3.4 Pole Handling ($\theta = 0$ and $\theta = \pi$)

The ratios $\bar{P}_n^m / \sin\theta$ and
$d\bar{P}_n^m / d\theta$ appear explicitly in the pattern functions and become
$0/0$ indeterminate forms at the poles. The package resolves these with
**analytical limits**:

For $|m| = 1$:

$$\lim_{\theta \to 0} \frac{\bar{P}_n^1(\cos\theta)}{\sin\theta} = -\frac{n(n+1)}{2} \cdot N_n^1$$

For $|m| \neq 1$:

$$\lim_{\theta \to 0} \frac{\bar{P}_n^m(\cos\theta)}{\sin\theta} = 0$$

The same structure holds at $\theta = \pi$ with an additional sign factor
$(-1)^{n+1}$.

As an additional safeguard, a small epsilon shift ($\epsilon = 10^{-6}$ rad) is
applied to any $\theta$ value that lies exactly at 0 or $\pi$, ensuring the
numerical recurrence pathway also remains well-conditioned.

---

## 4. Far-Field Pattern Functions

In the far field ($r \to \infty$), the radial Hankel functions reduce to a
simple phase factor, and the electric field can be written as

$$\mathbf{E}(\theta,\phi) = \frac{e^{-jkr}}{kr}\sum_{s,m,n} Q_{smn}\,\mathbf{K}_{smn}(\theta,\phi)$$

where $\mathbf{K}_{smn}(\theta,\phi)$ are the **far-field pattern functions**
containing only the angular dependence.

### 4.1 Common Prefactors

**Amplitude normalization:**

$$A_n = \sqrt{\frac{2}{n(n+1)}}$$

This factor ensures the pattern functions carry unit integrated power per mode.

**Sign factor:**

$$\sigma_m = \begin{cases} \left(\dfrac{-m}{|m|}\right)^{m} & m \neq 0 \\[6pt] 1 & m = 0 \end{cases}$$

**Azimuthal phase:**

$$\Phi_m(\phi) = e^{-im\phi}$$

### 4.2 TE Modes ($s = 1$)

The TE (transverse electric) modes have no radial electric field component. The
pattern function carries an $i$-factor of $i^{n+1}$:

$$K_{1,\theta}(\theta,\phi) = A_n \, \sigma_m \, e^{-im\phi} \cdot i^{n+1} \left(-im\,\frac{\bar{P}_n^m(\cos\theta)}{\sin\theta}\right)$$

$$K_{1,\phi}(\theta,\phi) = A_n \, \sigma_m \, e^{-im\phi} \cdot i^{n+1} \left(-\frac{d\bar{P}_n^m}{d\theta}\right)$$

### 4.3 TM Modes ($s = 2$)

The TM (transverse magnetic) modes have no radial magnetic field component. The
$i$-factor is $i^n$:

$$K_{2,\theta}(\theta,\phi) = A_n \, \sigma_m \, e^{-im\phi} \cdot i^{n} \left(\frac{d\bar{P}_n^m}{d\theta}\right)$$

$$K_{2,\phi}(\theta,\phi) = A_n \, \sigma_m \, e^{-im\phi} \cdot i^{n} \left(-im\,\frac{\bar{P}_n^m(\cos\theta)}{\sin\theta}\right)$$

### 4.4 Physical Interpretation

| Mode type | $E_r$ | Dominant field behaviour |
|-----------|--------|------------------------|
| TE ($s=1$) | 0 | Electric field lies on spherical surfaces; analogous to waveguide TE modes |
| TM ($s=2$) | $\neq 0$ at finite $r$ | Magnetic field lies on spherical surfaces; analogous to waveguide TM modes |

In the far field both types are purely transverse ($E_r = H_r = 0$), and the
distinction appears only through the relationship between $\hat{\theta}$ and
$\hat{\phi}$ components.

---

## 5. Near-Field Computation

At a finite distance $r$ from the origin, the full vector field includes radial
components and the radial dependence is governed by spherical Hankel functions.

### 5.1 Spherical Hankel Functions

With the $e^{j\omega t}$ time convention, outgoing waves use the Hankel
function of the **second** kind:

$$h_n^{(2)}(kr) = j_n(kr) - i\,y_n(kr)$$

where $j_n$ and $y_n$ are the spherical Bessel functions of the first and
second kind respectively.

The derivative combination that appears in the TM field expressions is

$$\frac{d\bigl[kr \cdot h_n^{(2)}(kr)\bigr]}{d(kr)} = kr \cdot h_{n-1}^{(2)}(kr) - n \cdot h_n^{(2)}(kr)$$

This is computed from the Bessel recurrence without numerical differentiation.

### 5.2 Field Scaling

The complete near-field expressions use the following overall scaling:

$$\mathbf{E}(r,\theta,\phi) = \sqrt{4\pi} \sum_{s,m,n} Q_{smn}\,\mathbf{F}^{E}_{smn}(r,\theta,\phi)$$

$$\mathbf{H}(r,\theta,\phi) = \frac{j\sqrt{4\pi}}{\eta_0} \sum_{s,m,n} Q_{smn}\,\mathbf{F}^{H}_{smn}(r,\theta,\phi)$$

where $\mathbf{F}^{E}_{smn}$ and $\mathbf{F}^{H}_{smn}$ are the near-field
pattern functions defined below.

> **Implementation note**: Hansen's textbook expresses the prefactor as
> $k\sqrt{\eta_0}$, but that form applies to a particular normalization of
> the Q coefficients. This package uses the TICRA file convention, in which
> the internal coefficients are related to the file values by
> $Q_\text{internal} = -\overline{Q_\text{file}}$ with no additional
> scaling. The effective prefactor for these coefficients is $\sqrt{4\pi}$
> for $\mathbf{E}$ and $j\sqrt{4\pi}/\eta_0$ for $\mathbf{H}$.
> The far-field K-function normalisation uses the same convention,
> ensuring consistency between far-field and near-field outputs.

### 5.3 TM Mode ($s = 2$) Electric Field Pattern

The TM modes produce all three electric field components:

$$F^{E}_{2,r}(r,\theta,\phi) = A_n \, \frac{n(n+1)}{kr} \, h_n^{(2)}(kr) \, \bar{P}_n^m(\cos\theta) \, e^{-im\phi}$$

$$F^{E}_{2,\theta}(r,\theta,\phi) = A_n \, \frac{\bigl[kr \cdot h_n^{(2)}(kr)\bigr]'}{kr} \, \frac{d\bar{P}_n^m}{d\theta} \, e^{-im\phi}$$

$$F^{E}_{2,\phi}(r,\theta,\phi) = -A_n \, \frac{\bigl[kr \cdot h_n^{(2)}(kr)\bigr]'}{kr} \, \frac{im\,\bar{P}_n^m(\cos\theta)}{\sin\theta} \, e^{-im\phi}$$

Note that $F^{E}_{2,r}$ decays as $1/kr$ relative to the transverse
components, reflecting the near-field radial contribution.

### 5.4 TE Mode ($s = 1$) Electric Field Pattern

TE modes have **no radial electric field**:

$$F^{E}_{1,r}(r,\theta,\phi) = 0$$

$$F^{E}_{1,\theta}(r,\theta,\phi) = -A_n \, h_n^{(2)}(kr) \, \frac{im\,\bar{P}_n^m(\cos\theta)}{\sin\theta} \, e^{-im\phi}$$

$$F^{E}_{1,\phi}(r,\theta,\phi) = -A_n \, h_n^{(2)}(kr) \, \frac{d\bar{P}_n^m}{d\theta} \, e^{-im\phi}$$

### 5.5 Magnetic Field via Duality

The magnetic field pattern functions are obtained by exchanging the role of the
two mode types:

$$\mathbf{F}^{H}_{1} = \mathbf{F}^{E}_{2}$$

$$\mathbf{F}^{H}_{2} = \mathbf{F}^{E}_{1}$$

This duality relation means TE electric field patterns are TM magnetic field
patterns and vice versa. The overall scaling factors in Section 5.2 already
account for the impedance ratio between $\mathbf{E}$ and $\mathbf{H}$.

---

## 6. Coefficient Extraction

Given measured or simulated far-field data $\mathbf{E}(\theta,\phi)$ on a
spherical grid, the spherical wave coefficients $Q_{smn}$ are recovered by
exploiting the orthogonality of the pattern functions.

### 6.1 Orthogonality Integral

The coefficients satisfy

$$Q_{smn} = \frac{1}{4\pi} \int_0^{2\pi}\!\int_0^{\pi} \mathbf{E}(\theta,\phi) \cdot \mathbf{K}^{*}_{smn}(\theta,\phi)\,\sin\theta\;d\theta\;d\phi$$

where $\mathbf{K}^{*}_{smn}$ is the complex conjugate of the far-field pattern
function. The $\sin\theta$ factor is the Jacobian of the spherical coordinate
system.

### 6.2 FFT-Accelerated Azimuthal Integration

When the input data is sampled on a **uniform** $\phi$ grid (constant
$\Delta\phi$), the azimuthal integral reduces to a discrete Fourier transform.

The procedure for each $(\theta_i, s, n)$:

1. Form the integrand
   $g(\phi) = \mathbf{E}(\theta_i, \phi) \cdot \mathbf{K}^{*}_{sn}(\theta_i)$
   (the $\phi$-independent part of $\mathbf{K}^*$ is separated out).
2. Multiply by $e^{+im\phi}$ to shift the desired azimuthal harmonic to DC.
3. Compute the FFT; the DC component (zeroth bin) yields the integral over
   $\phi$.

This reduces the complexity from $O(M \cdot N_\phi)$ to
$O(M \cdot \log N_\phi)$ per $\theta$ sample.

The polar ($\theta$) integral is then evaluated using the **trapezoidal rule**,
which is well-suited to the smooth, periodic-like integrands that arise in SWE.

### 6.3 Adaptive Mode Truncation

In practice, the required $N_{\max}$ is not known a priori. The package
determines it automatically through an iterative algorithm:

1. Begin with an initial guess of $N_{\max}$ (default: 100).
2. Extract all coefficients $Q_{smn}$ up to this $N_{\max}$.
3. Compute the total radiated power in the highest 10% of $n$-indices
   (the "tail").
4. If the tail power fraction exceeds the threshold ($10^{-5}$ of total
   power), increase $N_{\max}$ by 50 and return to step 2.
5. Repeat for up to 10 iterations. If convergence is not reached, a warning
   is issued.

**Azimuthal truncation** ($M_{\max}$) is determined separately by accumulating
the total power per $|m|$ index and finding the $|m|$ beyond which the
cumulative power contribution becomes negligible. This avoids computing
modes with azimuthal orders that carry no significant energy.

---

## 7. Surface Currents

Equivalent surface currents on a closed surface surrounding the antenna can be
computed from the near fields using the **surface equivalence theorem**
(Love's equivalence principle).

### 7.1 Definitions

Given the electric and magnetic fields evaluated at the surface and the outward
unit normal $\hat{n}$:

**Electric surface current density:**

$$\mathbf{J} = \hat{n} \times \mathbf{H}$$

**Magnetic surface current density:**

$$\mathbf{M} = -\hat{n} \times \mathbf{E}$$

These produce the same fields outside the surface as the original sources, and
zero field inside.

### 7.2 Implementation

The computation proceeds as follows:

1. Evaluate $\mathbf{E}$ and $\mathbf{H}$ on the desired surface using the
   near-field equations of Section 5 at each surface point $(r, \theta, \phi)$.
2. Compute the outward normal $\hat{n}$ for the surface geometry.
3. Evaluate the cross products to obtain $\mathbf{J}$ and $\mathbf{M}$ at each
   point.

### 7.3 Coordinate Frame Rotation

The SWE is defined in its own coordinate frame, but the physical surface (e.g.,
a reflector) may be defined in a different frame. The package supports arbitrary
rotation between frames via **ZYZ Euler angles** $(\alpha, \beta, \gamma)$:

1. Rotate by $\alpha$ about the $z$-axis.
2. Rotate by $\beta$ about the new $y$-axis.
3. Rotate by $\gamma$ about the new $z$-axis.

Both position vectors and field vectors are transformed accordingly before
computing the cross products.

---

## 8. TICRA .sph File Format

The `.sph` file is the standard interchange format for spherical wave
coefficients, as used by TICRA tools (GRASP, EditSph). The package provides
full read/write support.

### 8.1 File Structure

A `.sph` file consists of a fixed-length header followed by coefficient data
blocks.

| Record | Content | Example |
|--------|---------|---------|
| 1 | Program identification tag with frequency | `"TICRA-SWE Freq [GHz]: 10.000000"` |
| 2 | Free-form description string | `"My antenna pattern"` |
| 3 | Control integers: `NTHE  NPHI  NMAX  MMAX` | `181  361  45  25` |
| 4 | Rotation angles: $\alpha$, $\beta$, $\gamma$ (degrees) | `0.0  0.0  0.0` |
| 5--8 | Reserved lines (dummy/placeholder data) | `0.0  0.0  0.0  ...` |
| 9+ | Coefficient data grouped by $\lvert m\rvert$ | See below |

### 8.2 Coefficient Data Layout

The coefficients are written in blocks organised by increasing $|m|$.

For each $|m|$ from $0$ to $M_{\max}$:

1. **Header line**: the $m$ value and the total radiated power carried by modes
   with this $|m|$.

2. **Coefficient lines**: for each $n$ from $\max(1, |m|)$ to $N_{\max}$:

   - If $m = 0$: **one line** containing four real values:
     $$\operatorname{Re}(Q_{1,0,n}),\quad \operatorname{Im}(Q_{1,0,n}),\quad \operatorname{Re}(Q_{2,0,n}),\quad \operatorname{Im}(Q_{2,0,n})$$

   - If $m > 0$: **two lines** --
     - First line: coefficients for $-m$:
       $\operatorname{Re}(Q_{1,-m,n}),\; \operatorname{Im}(Q_{1,-m,n}),\; \operatorname{Re}(Q_{2,-m,n}),\; \operatorname{Im}(Q_{2,-m,n})$
     - Second line: coefficients for $+m$:
       $\operatorname{Re}(Q_{1,+m,n}),\; \operatorname{Im}(Q_{1,+m,n}),\; \operatorname{Re}(Q_{2,+m,n}),\; \operatorname{Im}(Q_{2,+m,n})$

### 8.3 Coefficient Convention

TICRA stores coefficients with a sign and conjugation that differs from the
convention used internally. The conversion applied during I/O is:

- **Reading**: $Q_\text{internal} = -\overline{Q_\text{file}}$
  (negate and conjugate the stored value).
- **Writing**: $Q_\text{file} = -\overline{Q_\text{internal}}$
  (the exact inverse operation).

No additional scale factor is applied. This sign/conjugation convention
matches the TICRA time-dependence ($e^{j\omega t}$) and ensures that the
pattern functions produce the correct far-field and near-field amplitudes
when combined with the $\sqrt{4\pi}$ prefactor (Section 5.2).

---

## 9. Performance Optimizations

The SWE computation is inherently $O(N_{\max}^2 \cdot N_\theta \cdot N_\phi)$
in the number of modes and spatial samples. Several strategies are employed to
keep wall-clock times practical.

### 9.1 Numba JIT Acceleration

When [Numba](https://numba.pydata.org/) is installed, the innermost loops over
spatial points are compiled to optimised machine code at first call via
just-in-time (JIT) compilation.

Key features:

- Spherical Bessel function precomputation is JIT-compiled.
- Parallel loops via `prange` distribute work across CPU cores.
- **Fallback**: when Numba is not available, the same code paths execute using
  SciPy's `spherical_jn` / `spherical_yn` functions with NumPy vectorisation.

### 9.2 Caching

Repeated evaluations (e.g., sweeping frequency or distance) benefit from
several caching layers:

| Cache | What is stored | Scope |
|-------|---------------|-------|
| `LegendreCoefficientCache` | Factorial ratios and normalization factors $N_n^m$ | Per session (computed once) |
| `compute_all_modes_legendre()` | All $\bar{P}_n^m(\cos\theta)$ and derivatives for all $(n,m)$ pairs | Per $\theta$-grid |
| Bessel precomputation | All $h_n^{(2)}(kr)$ for $n = 0 \ldots N_{\max}$ | Per set of radial distances |

The Legendre cache avoids recomputing the normalization factors and recurrence
coefficients, which involve factorial ratios that would otherwise dominate
setup time for large $N_{\max}$.

### 9.3 Parallelization

Coefficient extraction (Section 6) distributes work across multiple processes:

- Modes are grouped into batches and distributed across
  $\min(\text{CPU count},\; 8)$ worker processes using Python's
  `multiprocessing` module.
- Each worker independently computes the orthogonality integral for its
  assigned batch of $(s, m, n)$ triples.
- **Fallback**: for problems with fewer than 100 modes, serial execution is
  used to avoid the overhead of process spawning.

---

## References

1. J. E. Hansen (Ed.), *Spherical Near-Field Antenna Measurements*,
   IEE Electromagnetic Waves Series 26, Peter Peregrinus Ltd., London, 1988.

2. IEEE Std 1720-2012, *IEEE Recommended Practice for Near-Field Antenna
   Measurements*, IEEE, 2012.

3. TICRA, *GRASP Technical Description*, Copenhagen, Denmark. Specification of
   the `.sph` file format for spherical wave coefficients.
