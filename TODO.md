# TODO

## High Priority

### Add frequency sweep/interpolation utilities
Multi-frequency support is now implemented (SphericalWaveExpansion stores all
frequencies in a single object, keyed by frequency in Hz). A useful follow-on
would be interpolation across loaded frequencies for fine frequency resolution.

## Medium Priority

### Resolve K function inconsistency
The `far_field_pattern_functions()` and `from_far_field()` use different sign conventions:
- `far_field_pattern_functions`: `sign_factor = (-m/|m|)^m`, `i_factor_1 = (1j)^(n+1)`, `i_factor_2 = (1j)^n`
- `from_far_field` (K functions): different sign_factor, swapped i_factors
Both produce correct results independently (far-field matches GRASP, roundtrip is perfect),
but the internal inconsistency should be resolved to avoid confusion.

### Add assertions to validation tests
The far-field and near-field comparison tests currently print metrics but don't assert thresholds.
Add `assert` statements for:
- Scaling ratio within tolerance of 1.0 (e.g., 0.99 to 1.01 for far-field)
- Normalized RMS after scaling below threshold (e.g., < 1e-3 for far-field)
- Phase offset within tolerance (e.g., < 1 degree)

### Add power_threshold filtering to near_field
The `far_field()` method supports a `power_threshold` parameter (default 0.999) that filters
out weak modes for efficiency. The `near_field()` method lacks this feature.

## Low Priority

### Fix pyproject.toml for editable install
`pip install -e .` may fail depending on the environment. Current workaround is `PYTHONPATH=.`.
Ensure pyproject.toml works for editable installs across Python 3.9-3.12.

### Run and validate test_cut_to_sph.py
The SWE extraction tests (`test_cut_to_sph.py`) have not been fully run due to long computation
time (least-squares fitting over ~26,000 grid points with NMAX=359). Validate that extraction
produces reasonable Q coefficients and reproduces both .cut and .grd.

### Performance optimization for SWE extraction
`from_far_field()` with NMAX=359 is very slow. Consider:
- Multiprocessing support (already has parameter, needs testing)
- Numba/JIT acceleration for the K matrix construction
- Sparse matrix techniques for the least-squares solve
