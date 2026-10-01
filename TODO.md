# TODO

## High Priority

### Near-field validation against example.grd fails
`tests/test_sph_to_nearfield.py::test_near_field_absolute_vs_grd` now asserts
and is marked `xfail(strict=True)`: the near field synthesized from
`example.sph` on the z = 0.25 m plane of `example.grd` has a normalized RMS
error after best-fit scaling of about 10 (Eco) and 28 (Ecx) at every
frequency, with a best-fit scale near 183. Either the .grd plane is not read
the way the test assumes (axes, orientation, which component is which) or the
near-field synthesis (Hankel functions, prefactors) is wrong. Settle which,
fix it, and remove the xfail (strict mode fails the suite as soon as the
comparison passes, so the marker cannot be forgotten).

### Add frequency sweep/interpolation utilities
Multi-frequency support is now implemented (SphericalWaveExpansion stores all
frequencies in a single object, keyed by frequency in Hz). A useful follow-on
would be interpolation across loaded frequencies for fine frequency resolution.

## Medium Priority

### Far-field sign convention relative to TICRA
`tests/test_sph_to_farfield.py` asserts that the far field synthesized from
`example.sph` matches `example.cut` in shape (normalized RMS after scaling
below 1e-2 co, 3e-2 cross), that the co- and cross-polar scale factors agree,
and that the global phase offset is 180 degrees (within 1 degree). That last
figure is pinned, not explained: the field comes out as minus the TICRA cut.
Decide whether the sign belongs in the pattern functions (`_pattern_function_bases`)
or in the TICRA readers, and change the assertion with the fix.

### Add power_threshold filtering to near_field
The `far_field()` method supports a `power_threshold` parameter (default 0.999) that filters
out weak modes for efficiency. The `near_field()` method lacks this feature.

## Low Priority

### Run and validate test_cut_to_sph.py
The SWE extraction tests (`test_cut_to_sph.py`) have not been fully run due to long computation
time (least-squares fitting over ~26,000 grid points with NMAX=359). Validate that extraction
produces reasonable Q coefficients and reproduces both .cut and .grd.
Extraction is now checked quickly in `tests/test_extraction_roundtrip.py`
(synthesize from known Q, extract, compare): the pattern functions used by
`from_far_field` were a different convention from the synthesis (a factor
j(-1)^m per mode, which rotated extracted patterns by 180 degrees in phi);
both now use `_pattern_function_bases`. The per-mode error on a 1 degree
grid is about 1-2 % at the pole-sensitive modes; worth looking at the
pole handling (`sin_theta_safe`) and the Clenshaw-Curtis weights.

### Performance optimization for SWE extraction
`from_far_field()` with NMAX=359 is very slow. Consider:
- Multiprocessing support (already has parameter, needs testing)
- Numba/JIT acceleration for the K matrix construction
- Sparse matrix techniques for the least-squares solve
