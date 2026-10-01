"""
Extraction inverts synthesis: coefficients -> far field -> coefficients.

These are the tests that pin the pattern-function convention shared by
far_field() and from_far_field() (see _pattern_function_bases in swe.core).
"""
import numpy as np
import pytest

from swe import SphericalWaveExpansion
from tests.conftest import SPH_FILE, requires_sph

FREQ = 8.0e9
N_KEEP, M_KEEP = 12, 6          # the TICRA example cut down to modes a coarse grid resolves fast


def _grid(n_theta=181, n_phi=180):
    theta = np.radians(np.linspace(0.0, 180.0, n_theta))
    phi = np.radians(np.arange(0.0, 360.0, 360.0 / n_phi))
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    return TH.ravel(), PH.ravel()


def _roundtrip(swe, nmax_initial, mmax_initial):
    theta, phi = _grid()
    e_theta, e_phi = swe.far_field(theta, phi, frequency=FREQ, normalize=False, power_threshold=1.0)
    extracted = SphericalWaveExpansion.from_far_field(
        theta, phi, e_theta, e_phi, frequency=FREQ,
        NMAX_initial=nmax_initial, MMAX_initial=mmax_initial,
        power_threshold=1.0, high_mode_power_threshold=1.0, azimuthal_power_threshold=0.0,
        normalize=False)
    e_theta2, e_phi2 = extracted.far_field(theta, phi, frequency=FREQ, normalize=False,
                                           power_threshold=1.0)
    field_error = (np.linalg.norm(e_theta2 - e_theta) + np.linalg.norm(e_phi2 - e_phi)) / \
                  (np.linalg.norm(e_theta) + np.linalg.norm(e_phi))
    return extracted, field_error


def _mode_ratios(original_q1, original_q2, extracted, power_floor=0.01):
    """(n, m, ratio) of extracted over original for modes above the power floor."""
    e1, e2 = extracted.Q1_coeffs(FREQ), extracted.Q2_coeffs(FREQ)
    total = sum(abs(v) ** 2 for v in original_q1.values()) + sum(abs(v) ** 2 for v in original_q2.values())
    ratios = []
    for key in original_q1:
        for q, e in ((original_q1[key], e1.get(key, 0.0)), (original_q2[key], e2.get(key, 0.0))):
            if abs(q) ** 2 / total >= power_floor:
                ratios.append((key[0], key[1], e / q))
    return ratios


@pytest.fixture(scope='module')
def synthetic():
    rng = np.random.default_rng(0)
    nmax, mmax = 6, 4
    q1, q2 = {}, {}
    for n in range(1, nmax + 1):
        for m in range(-min(n, mmax), min(n, mmax) + 1):
            q1[(n, m)] = complex(*rng.normal(size=2)) * 10 ** (-0.3 * n)
            q2[(n, m)] = complex(*rng.normal(size=2)) * 10 ** (-0.3 * n)
    swe = SphericalWaveExpansion(Q1_coeffs={FREQ: q1}, Q2_coeffs={FREQ: q2},
                                 NMAX={FREQ: nmax}, MMAX={FREQ: mmax})
    return q1, q2, swe


@pytest.fixture(scope='module')
def truncated():
    full = SphericalWaveExpansion.from_sph_file(SPH_FILE, frequencies=[FREQ])
    q1 = {k: v for k, v in full.Q1_coeffs(FREQ).items() if k[0] <= N_KEEP and abs(k[1]) <= M_KEEP}
    q2 = {k: v for k, v in full.Q2_coeffs(FREQ).items() if k[0] <= N_KEEP and abs(k[1]) <= M_KEEP}
    return q1, q2, SphericalWaveExpansion(Q1_coeffs={FREQ: q1}, Q2_coeffs={FREQ: q2},
                                          NMAX={FREQ: N_KEEP}, MMAX={FREQ: M_KEEP})


class TestSyntheticRoundTrip:
    def test_field_is_reproduced(self, synthetic):
        _q1, _q2, swe = synthetic
        _extracted, field_error = _roundtrip(swe, nmax_initial=10, mmax_initial=6)
        assert field_error < 0.03, field_error

    def test_coefficients_come_back_unchanged(self, synthetic):
        """Each significant mode returns with ratio 1: no j(-1)^m factor, no sign flip."""
        q1, q2, swe = synthetic
        extracted, _ = _roundtrip(swe, nmax_initial=10, mmax_initial=6)
        ratios = _mode_ratios(q1, q2, extracted)
        assert ratios, "no mode above the power floor"
        for n, m, ratio in ratios:
            assert abs(ratio - 1.0) < 0.05, f"mode ({n}, {m}) came back scaled by {ratio:.4f}"

    def test_normalized_extraction_keeps_the_shape(self, synthetic):
        """normalize=True only rescales: the unit-power coefficients are a real multiple."""
        q1, q2, swe = synthetic
        theta, phi = _grid()
        e_theta, e_phi = swe.far_field(theta, phi, frequency=FREQ, normalize=False, power_threshold=1.0)
        extracted = SphericalWaveExpansion.from_far_field(
            theta, phi, e_theta, e_phi, frequency=FREQ, NMAX_initial=10, MMAX_initial=6,
            power_threshold=1.0, high_mode_power_threshold=1.0, azimuthal_power_threshold=0.0)
        power = sum(abs(v) ** 2 for v in extracted.Q1_coeffs(FREQ).values()) + \
            sum(abs(v) ** 2 for v in extracted.Q2_coeffs(FREQ).values())
        assert power == pytest.approx(1.0, rel=1e-6)
        ratios = [r for _n, _m, r in _mode_ratios(q1, q2, extracted)]
        scale = np.mean([abs(r) for r in ratios])
        for ratio in ratios:
            assert abs(ratio / scale - 1.0) < 0.05


@requires_sph
class TestTicraRoundTrip:
    """A truncated TICRA expansion survives synthesis and extraction."""

    def test_ticra_coefficients_round_trip(self, truncated):
        q1, q2, swe = truncated
        extracted, field_error = _roundtrip(swe, nmax_initial=N_KEEP + 6, mmax_initial=M_KEEP + 2)
        assert field_error < 0.03, field_error
        for n, m, ratio in _mode_ratios(q1, q2, extracted, power_floor=0.005):
            assert abs(ratio - 1.0) < 0.05, f"mode ({n}, {m}) came back scaled by {ratio:.4f}"
