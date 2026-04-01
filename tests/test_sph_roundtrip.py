"""
Test .sph file read/write roundtrip.

Validates that reading a .sph file and writing it back produces consistent
coefficients, confirming the normalization (sqrt(8*pi)) and conjugation
operations are properly inverse.
"""

import os
import tempfile
import numpy as np
import pytest

from swe import SphericalWaveExpansion, read_ticra_sph, write_ticra_sph
from tests.conftest import SPH_FILE, requires_sph


@requires_sph
class TestSphRoundtrip:

    def test_read_sph_file(self):
        """Verify .sph file loads without errors and contains expected data."""
        data = read_ticra_sph(SPH_FILE)

        assert data['NMAX'] > 0, "NMAX should be positive"
        assert data['MMAX'] >= 0, "MMAX should be non-negative"
        assert data['MMAX'] <= data['NMAX'], "MMAX should not exceed NMAX"
        assert len(data['Q1_coeffs']) > 0, "Should have Q1 coefficients"
        assert len(data['Q2_coeffs']) > 0, "Should have Q2 coefficients"

        if data['frequency'] is not None:
            assert data['frequency'] > 0, "Frequency should be positive"

        print(f"\nLoaded .sph: NMAX={data['NMAX']}, MMAX={data['MMAX']}, "
              f"freq={data['frequency']} GHz")
        print(f"  Q1 modes: {len(data['Q1_coeffs'])}")
        print(f"  Q2 modes: {len(data['Q2_coeffs'])}")

    def test_sph_roundtrip_coefficients(self):
        """Read .sph, write to temp, read back, compare coefficients."""
        # Read original
        swe = SphericalWaveExpansion.from_sph_file(SPH_FILE)

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix='.sph', delete=False, mode='w') as f:
            tmp_path = f.name

        try:
            swe.to_sph_file(tmp_path)

            # Read back
            swe2 = SphericalWaveExpansion.from_sph_file(tmp_path)

            # Compare
            assert swe2.NMAX == swe.NMAX, f"NMAX mismatch: {swe2.NMAX} vs {swe.NMAX}"
            assert swe2.MMAX == swe.MMAX, f"MMAX mismatch: {swe2.MMAX} vs {swe.MMAX}"

            # Compare Q1 coefficients
            all_keys = set(swe.Q1_coeffs.keys()) | set(swe2.Q1_coeffs.keys())
            max_q1_err = 0
            for key in all_keys:
                q1_orig = swe.Q1_coeffs.get(key, 0.0)
                q1_new = swe2.Q1_coeffs.get(key, 0.0)
                err = abs(q1_orig - q1_new)
                max_q1_err = max(max_q1_err, err)

            # Compare Q2 coefficients
            all_keys = set(swe.Q2_coeffs.keys()) | set(swe2.Q2_coeffs.keys())
            max_q2_err = 0
            for key in all_keys:
                q2_orig = swe.Q2_coeffs.get(key, 0.0)
                q2_new = swe2.Q2_coeffs.get(key, 0.0)
                err = abs(q2_orig - q2_new)
                max_q2_err = max(max_q2_err, err)

            print(f"\nRoundtrip max Q1 error: {max_q1_err:.6e}")
            print(f"Roundtrip max Q2 error: {max_q2_err:.6e}")

            # Normalization roundtrip should be exact to machine precision
            # relative to coefficient magnitudes
            q1_mag = max(abs(v) for v in swe.Q1_coeffs.values()) if swe.Q1_coeffs else 1.0
            q2_mag = max(abs(v) for v in swe.Q2_coeffs.values()) if swe.Q2_coeffs else 1.0

            assert max_q1_err / q1_mag < 1e-10, (
                f"Q1 roundtrip error too large: {max_q1_err / q1_mag:.2e}")
            assert max_q2_err / q2_mag < 1e-10, (
                f"Q2 roundtrip error too large: {max_q2_err / q2_mag:.2e}")

        finally:
            os.unlink(tmp_path)

    def test_from_sph_file_creates_valid_object(self):
        """Verify SphericalWaveExpansion.from_sph_file produces a usable object."""
        swe = SphericalWaveExpansion.from_sph_file(SPH_FILE)

        assert swe.NMAX > 0
        assert swe.frequency is not None or True  # frequency may be None
        assert len(swe.Q1_coeffs) > 0 or len(swe.Q2_coeffs) > 0

        # All mode indices should satisfy n >= 1, |m| <= n
        for n, m in list(swe.Q1_coeffs.keys()) + list(swe.Q2_coeffs.keys()):
            assert n >= 1, f"Invalid n={n}"
            assert abs(m) <= n, f"Invalid mode (n={n}, m={m}): |m| > n"
            assert abs(m) <= swe.MMAX, f"Mode (n={n}, m={m}) exceeds MMAX={swe.MMAX}"
            assert n <= swe.NMAX, f"Mode (n={n}, m={m}) exceeds NMAX={swe.NMAX}"
