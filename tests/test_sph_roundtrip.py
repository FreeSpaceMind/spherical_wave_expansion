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
        """Verify .sph file loads all frequency blocks without errors."""
        all_blocks = read_ticra_sph(SPH_FILE)

        assert len(all_blocks) > 0, "Should have at least one frequency block"

        # Inspect first block
        data = all_blocks[0]
        assert data['NMAX'] > 0, "NMAX should be positive"
        assert data['MMAX'] >= 0, "MMAX should be non-negative"
        assert data['MMAX'] <= data['NMAX'], "MMAX should not exceed NMAX"
        assert len(data['Q1_coeffs']) > 0, "Should have Q1 coefficients"
        assert len(data['Q2_coeffs']) > 0, "Should have Q2 coefficients"

        if data['frequency'] is not None:
            assert data['frequency'] > 0, "Frequency should be positive"

        print(f"\nLoaded .sph: {len(all_blocks)} frequency block(s)")
        print(f"  First block: NMAX={data['NMAX']}, MMAX={data['MMAX']}, "
              f"freq={data['frequency']} GHz")
        print(f"  Q1 modes: {len(data['Q1_coeffs'])}")
        print(f"  Q2 modes: {len(data['Q2_coeffs'])}")

    def test_sph_roundtrip_coefficients(self):
        """Read .sph, write to temp, read back, compare coefficients for all freqs."""
        swe = SphericalWaveExpansion.from_sph_file(SPH_FILE)

        with tempfile.NamedTemporaryFile(suffix='.sph', delete=False, mode='w') as f:
            tmp_path = f.name

        try:
            swe.to_sph_file(tmp_path)
            swe2 = SphericalWaveExpansion.from_sph_file(tmp_path)

            assert swe2.frequencies == swe.frequencies, (
                f"Frequency lists differ: {swe2.frequencies} vs {swe.frequencies}"
            )

            for freq in swe.frequencies:
                assert swe2.NMAX(freq) == swe.NMAX(freq), (
                    f"NMAX mismatch at {freq/1e9:.4f} GHz: "
                    f"{swe2.NMAX(freq)} vs {swe.NMAX(freq)}"
                )
                assert swe2.MMAX(freq) == swe.MMAX(freq), (
                    f"MMAX mismatch at {freq/1e9:.4f} GHz: "
                    f"{swe2.MMAX(freq)} vs {swe.MMAX(freq)}"
                )

                q1_orig = swe.Q1_coeffs(freq)
                q1_new = swe2.Q1_coeffs(freq)
                q2_orig = swe.Q2_coeffs(freq)
                q2_new = swe2.Q2_coeffs(freq)

                all_keys_q1 = set(q1_orig.keys()) | set(q1_new.keys())
                max_q1_err = max(
                    abs(q1_orig.get(k, 0.0) - q1_new.get(k, 0.0))
                    for k in all_keys_q1
                )

                all_keys_q2 = set(q2_orig.keys()) | set(q2_new.keys())
                max_q2_err = max(
                    abs(q2_orig.get(k, 0.0) - q2_new.get(k, 0.0))
                    for k in all_keys_q2
                )

                q1_mag = max(abs(v) for v in q1_orig.values()) if q1_orig else 1.0
                q2_mag = max(abs(v) for v in q2_orig.values()) if q2_orig else 1.0

                print(f"\n  {freq/1e9:.4f} GHz — "
                      f"max Q1 err: {max_q1_err:.6e}, max Q2 err: {max_q2_err:.6e}")

                assert max_q1_err / q1_mag < 1e-10, (
                    f"Q1 roundtrip error too large at {freq/1e9:.4f} GHz: "
                    f"{max_q1_err / q1_mag:.2e}"
                )
                assert max_q2_err / q2_mag < 1e-10, (
                    f"Q2 roundtrip error too large at {freq/1e9:.4f} GHz: "
                    f"{max_q2_err / q2_mag:.2e}"
                )

        finally:
            os.unlink(tmp_path)

    def test_from_sph_file_creates_valid_object(self):
        """Verify SphericalWaveExpansion.from_sph_file produces a valid multi-freq object."""
        swe = SphericalWaveExpansion.from_sph_file(SPH_FILE)

        assert len(swe.frequencies) > 0, "Should have at least one frequency"

        for freq in swe.frequencies:
            assert swe.NMAX(freq) > 0, f"NMAX should be positive at {freq/1e9:.4f} GHz"
            q1 = swe.Q1_coeffs(freq)
            q2 = swe.Q2_coeffs(freq)
            assert len(q1) > 0 or len(q2) > 0, "Should have coefficients"

            for n, m in list(q1.keys()) + list(q2.keys()):
                assert n >= 1, f"Invalid n={n}"
                assert abs(m) <= n, f"Invalid mode (n={n}, m={m}): |m| > n"
                assert abs(m) <= swe.MMAX(freq), (
                    f"Mode (n={n}, m={m}) exceeds MMAX={swe.MMAX(freq)}"
                )
                assert n <= swe.NMAX(freq), (
                    f"Mode (n={n}, m={m}) exceeds NMAX={swe.NMAX(freq)}"
                )

        print(f"\nLoaded {len(swe.frequencies)} frequencies: "
              f"{[f'{f/1e9:.4f} GHz' for f in swe.frequencies]}")
